#!/usr/bin/env python3
# (C) Copyright 2026 QUICK contributors
# All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.

"""Measure paired whole-job BOMD overhead with properties disabled/enabled.

Every pair starts two fresh processes from identical deterministic physical
inputs. The wall clock includes process/launcher startup, setup, all SCFs,
gradients, normal driver/API output, checkpoints and teardown. Input creation,
provenance hashing and Python validation are outside that clock. No filesystem
cache flushing or extra durable-write synchronization is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np

from validate_oeprop_bomd import (
    ENERGY_COLUMNS, HF_GRADIENT, INVARIANT_COLUMNS, PBE0_GRADIENT,
    PROPERTY_COLUMNS, finite, invariant_errors, max_difference,
    property_difference, read_checkpoint, read_csv, read_trajectory,
    sha256, write_control,
)

CASES = {
    "water-hf": ("water.xyz", HF_GRADIENT),
    "trimer-hf": ("water_trimer.xyz", HF_GRADIENT),
    "water-pbe-d3": ("water.xyz", PBE0_GRADIENT.replace("PBE0", "PBE")),
    "trimer-pbe-d3": ("water_trimer.xyz", PBE0_GRADIENT.replace("PBE0", "PBE")),
    "water-pbe0-d3": ("water.xyz", PBE0_GRADIENT),
    "trimer-pbe0-d3": ("water_trimer.xyz", PBE0_GRADIENT),
}
# Fixed absolute gates; changing sampling must not change the physical run.
TOLERANCES = {"energy_au": 1.0e-7, "coordinate_ang": 1.0e-7,
              "gradient_au_per_bohr": 2.0e-7, "velocity_au": 1.0e-7,
              "property_au": 2.0e-7}
PHASE_COLUMNS = ("force_wall_seconds", "probe_wall_seconds", "oeprop_wall_seconds")
COUNT_COLUMNS = ("force_calls", "api_energy_gradient_calls", "property_calls",
                 "probe_generation_calls", "properties_extra_api_calls")
LAUNCH_COLUMNS = ("esp_launches", "efield_launches", "efg_launches")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def capture(command: list[str], *, env: dict[str, str] | None = None) -> dict:
    try:
        process = subprocess.run(command, env=env, text=True, capture_output=True,
                                 check=False, timeout=15)
        return {"argv": command, "returncode": process.returncode,
                "stdout": process.stdout, "stderr": process.stderr}
    except (OSError, subprocess.SubprocessError) as error:
        return {"argv": command, "error": str(error)}


def driver_snapshot(driver: Path, env: dict[str, str]) -> dict:
    linked = capture(["ldd", str(driver)], env=env)
    require(linked.get("returncode") == 0, f"Cannot resolve driver libraries: {linked}")
    require("not found" not in linked["stdout"], "Unresolved driver library in ldd output")
    libraries = {}
    for line in linked["stdout"].splitlines():
        match = re.search(r"(?:=>\s*)?(/\S+)\s+\(", line)
        if match:
            library = Path(match[1]).resolve()
            libraries[str(library)] = sha256(library)
    require(bool(libraries), "ldd did not resolve any library paths")
    # ASLR addresses change on every ldd invocation; dependency paths/hashes do not.
    linked["stdout"] = re.sub(r"\(0x[0-9a-fA-F]+\)", "(address omitted)", linked["stdout"])
    return {"path": str(driver), "sha256": sha256(driver),
            "resolved_libraries_sha256": libraries, "ldd": linked}


def source_snapshot(repo: Path) -> dict:
    paths = ("src", "quick-cmake", "CMakeLists.txt", "validation/oeprop_bomd",
             "validation/oeprop_api")
    def git(*arguments: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(repo), *arguments])
    files = git("ls-files", "-z", "--", *paths).decode().split("\0")
    # Include newly added source before Git tracks it; omit runtime bytecode.
    extensions = {".f", ".f90", ".f95", ".fh", ".c", ".cc", ".cpp", ".cu", ".cuh",
                  ".h", ".hpp", ".inc", ".py", ".cmake", ".in"}
    extra = {str(path.relative_to(repo)) for name in paths if (repo / name).is_dir()
             for path in (repo / name).rglob("*") if path.is_file() and path.suffix.lower() in extensions}
    files = sorted(set(filter(None, files)) | extra)
    return {"revision": git("rev-parse", "HEAD").decode().strip(),
            "scientific_diff_sha256": hashlib.sha256(git("diff", "HEAD", "--", *paths)).hexdigest(),
            "files_sha256": {name: sha256(repo / name) for name in files if (repo / name).is_file()}}


def fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def make_probes(geometry: Path, fixture: Path, count: int) -> str:
    """Retain the four checked-in sites, then add deterministic distant sites."""
    atoms = np.asarray([[finite(v, str(geometry)) for v in line.split()[1:]]
                        for line in geometry.read_text().splitlines()[2:]])
    require(atoms.shape == (int(geometry.read_text().splitlines()[0]), 3), "Invalid geometry fixture")
    records = [[finite(v, str(fixture)) for v in line.split()]
               for line in fixture.read_text().splitlines()[1:]]
    require(len(records) == int(fixture.read_text().splitlines()[0]) and all(len(p) == 3 for p in records),
            "Invalid fixed-probe fixture")
    points = records[:count]
    center = atoms.mean(axis=0)
    radius = max(np.linalg.norm(atoms - center, axis=1)) + 2.0
    extra = count - len(points)
    for index in range(extra):
        # Count-independent sequence: a larger grid contains the smaller grid.
        z = 1.0 - 2.0 * (((index + 0.5) * (math.sqrt(2.0) - 1.0)) % 1.0)
        angle = index * math.pi * (3.0 - math.sqrt(5.0))
        radial = math.sqrt(1.0 - z * z)
        points.append((center + radius * np.array([radial * math.cos(angle),
                                                  radial * math.sin(angle), z])).tolist())
    require(min(np.linalg.norm(np.asarray(points)[:, None, :] - atoms[None, :, :], axis=2).flat) > 0.25,
            "A generated probe is too close to an initial nucleus")
    return str(count) + "\n" + "".join(" ".join(f"{v:.16g}" for v in point) + "\n" for point in points)


def read_gradients(path: Path, natom: int, frames: int) -> np.ndarray:
    """Parse full gradients already printed by QUICK, without extra API calls."""
    sections = path.read_text(errors="replace").split("ANALYTICAL GRADIENT:")[1:]
    require(len(sections) == frames, f"Expected {frames} full gradient tables in {path}, got {len(sections)}")
    arrays = []
    expected = [(atom, axis) for atom in range(1, natom + 1) for axis in "XYZ"]
    pattern = re.compile(r"^\s*(\d+)([XYZ])\s+(\S+)\s+(\S+)\s*$")
    for section in sections:
        rows = []
        for line in section.splitlines():
            match = pattern.match(line)
            if match:
                finite(match[3], str(path))
                rows.append((int(match[1]), match[2], finite(match[4], str(path))))
            elif rows:
                break
        require([(atom, axis) for atom, axis, _ in rows] == expected, f"Malformed gradient table in {path}")
        arrays.append([value for _, _, value in rows])
    return np.asarray(arrays).reshape(frames, natom, 3)


def load_trial(directory: Path, args: argparse.Namespace, stride: int, keyword: str) -> dict:
    summary = json.loads((directory / "bomd_summary.json").read_text())
    for name, value in summary.items():
        if isinstance(value, (int, float)):
            finite(value, f"summary:{name}")
    steps = list(range(args.nsteps + 1))
    sampled = [step for step in steps if stride and step % stride == 0]
    require(summary["start_step"] == 0 and summary["final_step"] == args.nsteps
            and summary["steps_run"] == args.nsteps and summary["oeprop_stride"] == stride
            and abs(summary["dt_fs"] - args.dt_fs) < 1.0e-14, "Wrong summary step/timestep settings")
    expected = dict(zip(COUNT_COLUMNS, (len(steps), len(steps), len(sampled), len(sampled), 0)))
    require(all(summary[name] == value for name, value in expected.items()), "Wrong API/property counters")
    require(summary["mpi_ranks"] == args.ranks, "Driver rank count differs from --ranks")
    ranks = read_csv(directory / "bomd_kernels.csv", ["rank", *COUNT_COLUMNS, *LAUNCH_COLUMNS, *PHASE_COLUMNS])
    require([int(row["rank"]) for row in ranks] == list(range(args.ranks)), "Missing/reordered rank evidence")
    gpu = summary["backend"].lower() in ("gpu", "cuda", "hip")
    require(summary["backend"].lower() in ("cpu", "gpu", "cuda", "hip"), "Unknown driver backend")
    for row in ranks:
        require(all(int(row[name]) == value for name, value in expected.items()), "Rank API/property counters differ")
        require(all(int(row[name]) == (len(sampled) if gpu else 0) for name in LAUNCH_COLUMNS),
                "Wrong property kernel-launch counts")
        require(all(float(row[name]) >= 0 for name in PHASE_COLUMNS), "Negative rank phase timer")
    phase = {name: max(float(row[name]) for row in ranks) for name in PHASE_COLUMNS}
    require(phase["force_wall_seconds"] > 0, "Zero force API elapsed time")
    require(all(float(summary[name]) >= 0 for name in PHASE_COLUMNS), "Negative summary phase timer")
    if not stride:
        require(phase["probe_wall_seconds"] == phase["oeprop_wall_seconds"] == 0,
                "Disabled properties performed timed work")
    energy = read_csv(directory / "bomd_energy.csv", ["step", *ENERGY_COLUMNS])
    require([int(row["step"]) for row in energy] == steps, "Wrong energy step sequence")
    require(all(abs(float(row["time_fs"]) - step * args.dt_fs) <= 1.0e-12
                for step, row in zip(steps, energy)), "Wrong energy timestamps")
    require(all(abs(float(row["total_au"]) - float(row["potential_au"]) - float(row["kinetic_au"])) <= 1.0e-12
                for row in energy), "Inconsistent energy sum")
    trajectory = read_trajectory(directory / "bomd_trajectory.xyz", steps, args.dt_fs)
    checkpoint = read_checkpoint(directory / "bomd.chk")
    require(checkpoint["step"] == args.nsteps and checkpoint["keyword"] == keyword.upper()
            and abs(checkpoint["dt_fs"] - args.dt_fs) < 1.0e-14, "Wrong checkpoint metadata")
    require(max_difference(checkpoint["coordinates"], trajectory[-1]) <= 1.0e-11, "Checkpoint/XYZ endpoint differs")
    properties = read_csv(directory / "bomd_oeprop.csv",
                          ["step", "time_fs", "probe", "probe_mode", "x_ang", "y_ang", "z_ang",
                           *PROPERTY_COLUMNS, *INVARIANT_COLUMNS], allow_empty=not stride)
    require([(int(row["step"]), int(row["probe"])) for row in properties]
            == [(step, probe) for step in sampled for probe in range(1, args.probe_count + 1)],
            "Wrong sampled property/probe row sequence")
    points = np.loadtxt(directory / "probes.probe", skiprows=1).reshape(args.probe_count, 3)
    for row in properties:
        require(row["probe_mode"] == "fixed" and abs(float(row["time_fs"]) - int(row["step"]) * args.dt_fs) <= 1.0e-12,
                "Wrong property mode/timestamp")
        require(max_difference([float(row[f"{axis}_ang"]) for axis in "xyz"], points[int(row["probe"]) - 1]) <= 1.0e-11,
                "Fixed probe coordinates changed")
    invariants = invariant_errors(properties)
    require(all(value <= TOLERANCES["property_au"] for value in invariants.values()), "Wrong EFG/field invariants")
    gradients = read_gradients(directory / "quick_oeprop_bomd.out", trajectory.shape[1], len(steps))
    return {"summary": summary, "rank_evidence": ranks, "phase_max_rank_seconds": phase,
            "energy": energy, "trajectory": trajectory, "checkpoint": checkpoint,
            "properties": properties, "gradients": gradients, "invariant_max_abs": invariants}


def compare_physics(first: dict, second: dict) -> dict:
    errors = {"coordinate_ang": max_difference(first["trajectory"], second["trajectory"]),
              "gradient_au_per_bohr": max_difference(first["gradients"], second["gradients"]),
              "velocity_au": max_difference(first["checkpoint"]["velocities"], second["checkpoint"]["velocities"])}
    energies = {column: max_difference([float(row[column]) for row in first["energy"]],
                                      [float(row[column]) for row in second["energy"]])
                for column in ("potential_au", "kinetic_au", "total_au", "delta_total_au")}
    errors["energy_au"] = max(energies.values())
    require(first["checkpoint"]["atoms"] == second["checkpoint"]["atoms"]
            and first["checkpoint"]["frozen"] == second["checkpoint"]["frozen"], "Checkpoint atom/constraint identity changed")
    require(all(value <= TOLERANCES[name] for name, value in errors.items()), f"Physical trajectory changed: {errors}")
    return {"max_abs": errors, "energy_max_abs_by_column_au": energies,
            "gradient_comparison": "Every printed Cartesian gradient; QUICK output quantized to 1e-10 Eh/bohr."}


def describe(values: list[float]) -> dict:
    return {"values": values, "median": statistics.median(values), "min": min(values),
            "max": max(values), "sample_stdev": statistics.stdev(values)}


def benchmark(args: argparse.Namespace) -> dict:
    repo, workdir, driver = args.repo_root.resolve(), args.workdir.resolve(), args.driver.resolve()
    require(driver.is_file() and os.access(driver, os.X_OK), "--driver must be an executable file")
    require(not workdir.exists() or not any(workdir.iterdir()), "Use an empty --workdir; prior raw trials are never overwritten")
    workdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("QUICK_BASIS", str(repo / "basis"))
    env["QUICK_BASIS"] = str(Path(env["QUICK_BASIS"]).resolve())
    if "LD_LIBRARY_PATH" in env:
        env["LD_LIBRARY_PATH"] = ":".join(str(Path(value or ".").resolve())
                                          for value in env["LD_LIBRARY_PATH"].split(":"))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(args.threads)
    command = shlex.split(args.launcher) + [str(driver), "control.nml"]
    fixture_root = repo / "validation/oeprop_bomd"
    geometry_name, keyword = CASES[args.case]
    geometry, probes = fixture_root / geometry_name, fixture_root / "water_fixed.probe"
    probe_text = make_probes(geometry, probes, args.probe_count)
    input_files = [geometry, probes]
    plans = []
    for pair in range(args.repeats + 1):
        order = ("off", "on") if pair % 2 == 0 else ("on", "off")
        for position, mode in enumerate(order):
            directory = workdir / f"pair_{pair:03d}" / mode
            directory.mkdir(parents=True)
            shutil.copyfile(geometry, directory / "geometry.xyz")
            (directory / "probes.probe").write_text(probe_text)
            stride = args.stride if mode == "on" else 0
            write_control(directory / "control.nml", geometry=Path("geometry.xyz"), probes=Path("probes.probe"),
                          prefix=Path("bomd"), checkpoint=Path("bomd.chk"), mode="fixed", steps=args.nsteps,
                          dt_fs=args.dt_fs, keyword=keyword, stride=stride)
            input_files.extend(directory / name for name in ("geometry.xyz", "probes.probe", "control.nml"))
            plans.append({"pair": pair, "discarded_warmup": pair == 0, "order_in_pair": position,
                          "mode": mode, "stride": stride, "directory": str(directory), "argv": command})
    basis = Path(env["QUICK_BASIS"]).resolve()
    require(basis.is_dir(), "QUICK_BASIS is not a directory")
    input_files.extend(path for path in sorted(basis.rglob("*")) if path.is_file())
    def snapshot() -> dict:
        return {"driver": driver_snapshot(driver, env), "source": source_snapshot(repo),
                "inputs_and_basis_sha256": {str(path): sha256(path) for path in input_files}}
    baseline = snapshot()
    if args.expected_driver_sha256:
        require(baseline["driver"]["sha256"] == args.expected_driver_sha256.lower(), "Driver SHA256 differs from accepted binary")
    result = {"schema_version": 1, "passed": False, "prepared_only": args.prepare_only,
              "configuration": {"case": args.case, "keyword": keyword, "probe_mode": "fixed",
                  "probe_count": args.probe_count, "stride": args.stride, "nsteps": args.nsteps,
                  "dt_fs": args.dt_fs, "temperature_k": 50.0, "ranks": args.ranks,
                  "threads": args.threads, "discarded_pairs": 1, "retained_pairs": args.repeats,
                  "argv": sys.argv, "tolerances": TOLERANCES},
              "timing_boundary": __doc__, "provenance_before": baseline,
              "provenance_fingerprint": fingerprint(baseline), "trials": [], "pairs": [],
              "hardware": {"platform": platform.platform(), "cpu": capture(["lscpu"]),
                  "gpu": capture(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv"]),
                  "python": sys.version, "numpy": np.__version__},
              "environment": {name: env.get(name) for name in ("QUICK_BASIS", "LD_LIBRARY_PATH", "OMP_NUM_THREADS",
                  "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")}}
    output = workdir / "overhead_summary.json"
    def save() -> None:
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.prepare_only:
        result["plans"] = plans
        save()
        return result
    saved = {}
    reference_off = reference_on = None
    try:
        for plan in plans:
            record = dict(plan)
            result["trials"].append(record)
            directory = Path(plan["directory"])
            require(snapshot() == baseline, "Binary, resolved libraries, source or inputs changed before trial")
            record["provenance_before_unchanged"] = True
            with (directory / "run.log").open("w") as log:
                begin = time.perf_counter_ns()
                try:
                    process = subprocess.run(command, cwd=directory, env=env, stdout=log,
                                             stderr=subprocess.STDOUT, check=False, timeout=args.timeout)
                    record["returncode"] = process.returncode
                finally:
                    record["whole_job_wall_seconds"] = (time.perf_counter_ns() - begin) * 1.0e-9
            require(snapshot() == baseline, "Binary, resolved libraries, source or inputs changed during trial")
            record["provenance_after_unchanged"] = True
            require(process.returncode == 0, f"Driver failed; see {directory / 'run.log'}")
            data = load_trial(directory, args, plan["stride"], keyword)
            record.update({name: data[name] for name in ("summary", "rank_evidence", "phase_max_rank_seconds", "invariant_max_abs")})
            record["output_files"] = {str(path.relative_to(directory)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
                                      for path in sorted(directory.rglob("*")) if path.is_file()
                                      and path.name not in ("geometry.xyz", "probes.probe", "control.nml")}
            record["property_rows"] = len(data["properties"])
            phase = data["phase_max_rank_seconds"]
            record["property_api_to_force_api_percent"] = 100.0 * phase["oeprop_wall_seconds"] / phase["force_wall_seconds"]
            if plan["mode"] == "off":
                if reference_off is not None:
                    record["repeat_physics"] = compare_physics(reference_off, data)
                else:
                    reference_off = data
            elif reference_on is None:
                reference_on = data
            else:
                errors = property_difference(reference_on["properties"], data["properties"])
                require(max(errors.values()) <= TOLERANCES["property_au"], "Properties changed between deterministic repeats")
                record["repeat_property_max_abs"] = errors
            record["validated"] = True
            saved[plan["mode"]] = (record, data)
            if plan["order_in_pair"] == 1:
                off, on = saved["off"], saved["on"]
                before, after = off[0]["whole_job_wall_seconds"], on[0]["whole_job_wall_seconds"]
                require(before > 0 and after > 0, "Invalid whole-job elapsed time")
                require(off[1]["summary"]["backend"] == on[1]["summary"]["backend"], "Paired backends differ")
                result["pairs"].append({"pair": plan["pair"], "discarded_warmup": plan["discarded_warmup"],
                    "execution_order": [p["mode"] for p in plans if p["pair"] == plan["pair"]],
                    "off_wall_seconds": before, "on_wall_seconds": after,
                    "delta_wall_seconds": after - before, "whole_job_overhead_percent": 100.0 * (after / before - 1.0),
                    "matched_physics": compare_physics(off[1], on[1])})
            save()
        result["provenance_after"] = snapshot()
        require(result["provenance_after"] == baseline, "Provenance changed after benchmark")
        retained = [pair for pair in result["pairs"] if not pair["discarded_warmup"]]
        require(len(retained) == args.repeats, "Missing retained trial pair")
        result["whole_job"] = {column: describe([pair[column] for pair in retained]) for column in (
            "off_wall_seconds", "on_wall_seconds", "delta_wall_seconds", "whole_job_overhead_percent")}
        result["api_phase_ratio"] = {"scope": "Maximum-rank accumulated synchronous OEPROP API / force API; excludes driver IO; not whole-job overhead.",
            "property_api_to_force_api_percent": describe([trial["property_api_to_force_api_percent"]
                for trial in result["trials"] if trial["mode"] == "on" and not trial["discarded_warmup"]])}
        result["passed"] = True
    except (ValueError, OSError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
        result["error"] = str(error)
        try:
            result["provenance_after"] = snapshot()
        except (OSError, ValueError, subprocess.SubprocessError) as snapshot_error:
            result["provenance_after_error"] = str(snapshot_error)
    finally:
        save()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path, required=True, help="Accepted installed BOMD driver")
    parser.add_argument("--workdir", type=Path, required=True, help="Empty directory; retains every raw trial")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--case", choices=CASES, default="water-hf")
    parser.add_argument("--probe-count", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1, help="ON sampling stride; OFF always uses stride 0")
    parser.add_argument("--nsteps", type=int, default=100)
    parser.add_argument("--dt-fs", type=float, default=0.05)
    parser.add_argument("--repeats", type=int, default=3, help="Retained pairs, after one discarded pair; minimum 3")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--ranks", type=int, default=1, help="Must agree with launcher/driver MPI rank evidence")
    parser.add_argument("--launcher", default="", help="Optional launcher, e.g. 'mpirun -np 1'")
    parser.add_argument("--timeout", type=float, default=3600, help="Per-process timeout in seconds")
    parser.add_argument("--expected-driver-sha256", help="Optional accepted binary hash gate")
    parser.add_argument("--prepare-only", action="store_true", help="Write input plan/provenance without running the driver")
    args = parser.parse_args()
    for name in ("probe_count", "stride", "nsteps", "threads", "ranks"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if args.repeats < 3:
        parser.error("At least three retained pairs are required")
    if any(not math.isfinite(value) or value <= 0 for value in (args.dt_fs, args.timeout)):
        parser.error("dt-fs and timeout must be finite and positive")
    if args.expected_driver_sha256 and not re.fullmatch(r"[0-9a-fA-F]{64}", args.expected_driver_sha256):
        parser.error("expected-driver-sha256 must contain 64 hexadecimal digits")
    try:
        result = benchmark(args)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Benchmark setup failed: {error}\n")
    print(json.dumps({"passed": result["passed"], "prepared_only": result["prepared_only"],
                      "summary": str(args.workdir.resolve() / "overhead_summary.json"),
                      "whole_job": result.get("whole_job"), "error": result.get("error")}, indent=2))
    return 0 if result["passed"] or result["prepared_only"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
