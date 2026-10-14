#!/usr/bin/env python3
# (C) Copyright 2026 QUICK contributors
# All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.

"""Validate short QUICK BOMD trajectories and on-the-fly electrostatic properties."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import numpy as np

PROPERTY_COLUMNS = ["esp", "efield_x", "efield_y", "efield_z"] + [
    f"efg_{first}{second}" for first in "xyz" for second in "xyz"
]
HF_GRADIENT = (
    "HF BASIS=STO-3G CUTOFF=1.0D-12 GRADCUTOFF=1.0D-12 "
    "DENSERMS=1.0D-10 GRADIENT CHARGE=0"
)
HF_ENERGY = HF_GRADIENT.replace("GRADIENT", "ENERGY")
PBE0_GRADIENT = (
    "DFT PBE0 D3 BASIS=DEF2-SVPD CUTOFF=1.0D-12 GRADCUTOFF=1.0D-12 "
    "XCCUTOFF=1.0D-10 BASISCUTOFF=1.0D-10 DENSERMS=1.0D-9 GRADIENT CHARGE=0"
)
ENERGY_COLUMNS = [
    "time_fs", "potential_au", "kinetic_au", "total_au", "delta_total_au",
    "temperature_K", "max_gradient_au_per_bohr", "net_gradient_au_per_bohr",
    "max_gradient_change_au_per_bohr",
]
INVARIANT_COLUMNS = [
    "efield_mag", "efg_trace", "efg_frobenius", "efg_eigenvalue_1",
    "efg_eigenvalue_2", "efg_eigenvalue_3", "efg_anisotropy",
]


def finite(value: str | float, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Nonfinite {label}: {value}")
    return number


def read_csv(path: Path, required: list[str], allow_empty: bool = False) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not set(required).issubset(fields):
            raise ValueError(f"Invalid CSV header in {path}: {fields}")
        rows = list(reader)
    if not rows and not allow_empty:
        raise ValueError(f"Empty CSV: {path}")
    for row_index, row in enumerate(rows):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Malformed CSV row {row_index} in {path}")
        for key, value in row.items():
            if key != "probe_mode":
                finite(value, f"{path.name}:{row_index}:{key}")
    return rows


def read_checkpoint(path: Path) -> dict[str, object]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] not in (
        "QUICK_OEPROP_BOMD_CHECKPOINT_V1", "QUICK_OEPROP_BOMD_CHECKPOINT_V2"
    ):
        raise ValueError(f"Invalid checkpoint header: {path}")
    atom_offset = 4 if lines[0].endswith("V2") else 3
    keyword = lines[3] if atom_offset == 4 else None
    natom, step = map(int, lines[1].split())
    dt, reference = (finite(value, str(path)) for value in lines[2].split())
    if natom < 1 or step < 0 or dt <= 0 or len(lines) != natom + atom_offset:
        raise ValueError(f"Invalid checkpoint dimensions: {path}")
    atoms, frozen, coords, velocities = [], [], [], []
    for line in lines[atom_offset:]:
        values = line.split()
        if len(values) != 8:
            raise ValueError(f"Invalid checkpoint atom row: {path}")
        atom, flag = map(int, values[:2])
        if atom < 1 or flag not in (0, 1):
            raise ValueError(f"Invalid checkpoint atom/constraint: {path}")
        numbers = [finite(value, str(path)) for value in values[2:]]
        atoms.append(atom)
        frozen.append(flag)
        coords.append(numbers[:3])
        velocities.append(numbers[3:])
    return {"step": step, "dt_fs": dt, "reference_total_au": reference,
            "keyword": keyword, "atoms": atoms, "frozen": frozen, "coordinates": coords, "velocities": velocities}


def read_trajectory(path: Path, steps: list[int], dt_fs: float) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    frames, atom_order, offset = [], None, 0
    for step in steps:
        if offset + 2 > len(lines):
            raise ValueError(f"Missing XYZ frame {step}: {path}")
        natom = int(lines[offset])
        comment = lines[offset + 1]
        stored_step = re.search(r"\bstep=(\d+)", comment)
        stored_time = re.search(r"\btime_fs=\s*([+\-\d.eE]+)", comment)
        if natom < 1 or stored_step is None or int(stored_step[1]) != step:
            raise ValueError(f"Invalid XYZ frame identity {step}: {path}")
        if stored_time is None or abs(finite(stored_time[1], str(path)) - step * dt_fs) > 5.1e-7:
            raise ValueError(f"Invalid XYZ time at step {step}: {path}")
        records = [line.split() for line in lines[offset + 2:offset + 2 + natom]]
        if len(records) != natom or any(len(row) != 4 for row in records):
            raise ValueError(f"Invalid XYZ atom records at step {step}: {path}")
        atoms = [row[0] for row in records]
        if atom_order is None:
            atom_order = atoms
        elif atoms != atom_order:
            raise ValueError(f"Changed XYZ atom identity: {path}")
        frames.append([[finite(value, str(path)) for value in row[1:]] for row in records])
        offset += natom + 2
    if offset != len(lines):
        raise ValueError(f"Unexpected trailing XYZ records: {path}")
    return np.asarray(frames)


def max_difference(first: object, second: object) -> float:
    left, right = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if left.shape != right.shape or not left.size:
        raise ValueError(f"Array shape mismatch or empty comparison: {left.shape}, {right.shape}")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("Nonfinite comparison array")
    return float(np.max(np.abs(left - right)))


def property_difference(first: list[dict[str, str]], second: list[dict[str, str]]) -> dict[str, float]:
    if [(int(row["step"]), int(row["probe"])) for row in first] != [
        (int(row["step"]), int(row["probe"])) for row in second
    ] or not first:
        raise ValueError("Property step/probe identity mismatch or empty comparison")
    return {name: max_difference([float(row[name]) for row in first], [float(row[name]) for row in second])
            for name in PROPERTY_COLUMNS}


def invariant_errors(rows: list[dict[str, str]]) -> dict[str, float]:
    errors = {name: 0.0 for name in ("symmetry", "trace", "frobenius", "eigenvalues", "anisotropy", "field_magnitude")}
    for row in rows:
        matrix = np.asarray([float(row[f"efg_{a}{b}"]) for a in "xyz" for b in "xyz"]).reshape(3, 3)
        eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) * 0.5)
        advertised = [float(row[f"efg_eigenvalue_{index}"]) for index in (1, 2, 3)]
        differences = {
            "symmetry": float(np.max(np.abs(matrix - matrix.T))),
            "trace": abs(float(row["efg_trace"]) - float(np.trace(matrix))),
            "frobenius": abs(float(row["efg_frobenius"]) - float(np.linalg.norm(matrix))),
            "eigenvalues": max_difference(advertised, eigenvalues),
            "anisotropy": abs(float(row["efg_anisotropy"]) - math.sqrt(0.5 * sum(
                (eigenvalues[i] - eigenvalues[j]) ** 2 for i, j in ((0, 1), (1, 2), (2, 0))))),
            "field_magnitude": abs(float(row["efield_mag"]) - math.sqrt(sum(
                float(row[f"efield_{axis}"]) ** 2 for axis in "xyz"))),
        }
        for name, value in differences.items():
            errors[name] = max(errors[name], value)
    return errors


def write_control(path: Path, *, geometry: Path | None, probes: Path | None,
                  prefix: Path, mode: str, steps: int, dt_fs: float = 0.05,
                  keyword: str = HF_GRADIENT, restart: Path | None = None,
                  checkpoint: Path | None = None, coordinate_atoms: tuple[int, int] = (2, 3),
                  stride: int = 1) -> None:
    values = ["&quick_bomd"]
    if geometry is not None:
        values.append(f" geometry_file='{geometry}'")
    if probes is not None:
        values.append(f" probe_file='{probes}'")
    values.extend([
        f" output_prefix='{prefix}'", f" checkpoint_file='{checkpoint or Path(str(prefix) + '.chk')}'",
        f" keywd='{keyword}'", f" probe_mode='{mode}'", f" nsteps={steps}",
        f" dt_fs={dt_fs:.12g}", " temperature_k=50.0",
        f" coordinate_atoms={coordinate_atoms[0]},{coordinate_atoms[1]}", f" oeprop_stride={stride}",
    ])
    if restart is not None:
        values.append(f" restart_file='{restart}'")
    if mode == "molecule":
        values.append(" anchor_atoms=1,2,3")
    if mode == "density":
        values.extend([" density_isovalue=1.0D-3", " density_spacing_ang=1.0", " density_max_radius_ang=8.0"])
    path.write_text("\n".join(values + ["/"]) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def driver_metadata(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    process = subprocess.run(["ldd", str(resolved)], text=True, capture_output=True, check=False)
    libraries = {}
    for line in process.stdout.splitlines():
        match = re.search(r"libquick\S*\s+=>\s+(/\S+)", line)
        if match:
            library = Path(match[1]).resolve()
            libraries[str(library)] = sha256(library)
    return {"path": str(resolved), "sha256": sha256(resolved), "loaded_quick_libraries": libraries,
            "ldd_returncode": process.returncode, "ldd_output": process.stdout}


def source_metadata(repo: Path) -> dict[str, object]:
    def git(*arguments: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(repo), *arguments])
    files = git("ls-files", "src", "validation/oeprop_bomd", "validation/oeprop_api").decode().splitlines()
    return {"revision": git("rev-parse", "HEAD").decode().strip(),
            "status": git("status", "--short").decode(),
            "diff_sha256": hashlib.sha256(git("diff", "HEAD", "--")).hexdigest(),
            "source_sha256": {name: sha256(repo / name) for name in files if (repo / name).is_file()}}


def validate(args: argparse.Namespace) -> dict[str, object]:
    workdir, repo = args.workdir.resolve(), args.repo_root.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("QUICK_BASIS", str(repo / "basis"))
    backends = {"cpu": [str(args.bomd_driver.resolve())]}
    if args.mpi_driver:
        backends["mpi"] = shlex.split(args.launcher) + [str(args.mpi_driver.resolve())]
    if args.gpu_driver:
        backends["gpu"] = [str(args.gpu_driver.resolve())]
    if args.gpu_mpi_driver:
        backends["gpu_mpi"] = shlex.split(args.gpu_launcher) + [str(args.gpu_mpi_driver.resolve())]
    fixture_root = repo / "validation/oeprop_bomd"
    probes = {mode: fixture_root / f"water_{mode}.probe" for mode in ("fixed", "molecule", "atom")}
    fixtures = {"water": fixture_root / "water.xyz", "trimer": fixture_root / "water_trimer.xyz", **probes}
    for path in fixtures.values():
        if not path.is_file():
            raise ValueError(f"Missing required fixture: {path}")
    probe_records = {}
    for mode, path in probes.items():
        lines = path.read_text(encoding="utf-8").splitlines()
        count = int(lines[0])
        records = [[finite(value, str(path)) for value in line.split()] for line in lines[1:]]
        if count < 1 or len(records) != count or any(len(row) != (4 if mode == "atom" else 3) for row in records):
            raise ValueError(f"Invalid probe fixture: {path}")
        probe_records[mode] = records
    source_before = source_metadata(repo)
    drivers_before = {name: driver_metadata(Path(command[-1])) for name, command in backends.items()}
    drivers_before["postprocessor_cpu"] = driver_metadata(args.oeprop_driver)
    result: dict[str, object] = {
        "schema_version": 2, "passed": False, "checks": {}, "cases": {}, "comparisons": {}, "commands": [],
        "metadata": {"source": source_before, "drivers": drivers_before,
                     "fixtures_sha256": {str(path): sha256(path) for path in fixtures.values()},
                     "numpy_version": np.__version__, "argv": sys.argv,
                     "environment": {name: env.get(name) for name in ("QUICK_BASIS", "OMP_NUM_THREADS",
                         "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")},
                     "tolerances": {name: value for name, value in vars(args).items() if name.endswith("tolerance")},
                     "scope": "Short trajectory smoke tests. Same-geometry postprocessing uses independently converged CPU SCFs, not an identical imported density. Separate trajectories can diverge at longer times."},
    }
    checks = result["checks"]
    cases = {}

    def check(name: str, value: bool) -> None:
        checks[name] = bool(value)

    def run(command: list[str], log: Path) -> None:
        record = {"argv": command, "cwd": str(workdir), "log": str(log)}
        result["commands"].append(record)
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.run(command, cwd=workdir, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
        record["returncode"] = process.returncode
        if process.returncode:
            raise RuntimeError(f"Driver exited {process.returncode}; see {log}")

    def run_case(backend: str, label: str, *, mode: str = "fixed", steps: int = 4,
                 dt_fs: float = 0.05, stride: int = 1, keyword: str = HF_GRADIENT,
                 geometry: Path | None = None, restart: Path | None = None,
                 coordinate_atoms: tuple[int, int] = (2, 3)) -> dict[str, object]:
        name = f"{backend}_{label}"
        prefix, control = workdir / name, workdir / f"{name}.nml"
        start = int(read_checkpoint(restart)["step"]) if restart else 0
        all_steps = list(range(start, start + steps + 1))
        sampled_steps = [step for step in all_steps if stride > 0 and step % stride == 0]
        write_control(control, geometry=None if restart else (geometry or fixtures["water"]),
                      probes=probes.get(mode), prefix=prefix, mode=mode, steps=steps,
                      dt_fs=dt_fs, stride=stride, keyword=keyword, restart=restart,
                      coordinate_atoms=coordinate_atoms)
        run(backends[backend] + [str(control)], workdir / f"{name}.log")
        summary = json.loads(Path(str(prefix) + "_summary.json").read_text(encoding="utf-8"))
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                finite(value, f"{name}:summary:{key}")
        required_summary = {"start_step", "final_step", "steps_run", "dt_fs", "oeprop_stride", "force_calls",
                            "api_energy_gradient_calls", "property_calls", "probe_generation_calls",
                            "properties_extra_api_calls", "force_wall_seconds", "probe_wall_seconds",
                            "oeprop_wall_seconds", "backend", "mpi_ranks", "restart_hamiltonian_verified"}
        if not required_summary.issubset(summary):
            raise ValueError(f"Missing summary fields for {name}: {sorted(required_summary - summary.keys())}")
        check(f"{name}:step_counts", summary["start_step"] == start and summary["final_step"] == start + steps
              and summary["steps_run"] == steps and summary["oeprop_stride"] == stride
              and abs(float(summary["dt_fs"]) - dt_fs) < 1.0e-14)
        check(f"{name}:one_gradient_call_per_step", summary["force_calls"] == steps + 1
              and summary["api_energy_gradient_calls"] == steps + 1)
        check(f"{name}:property_stride_and_no_extra_scf", summary["property_calls"] == len(sampled_steps)
              and summary["probe_generation_calls"] == len(sampled_steps) and summary["properties_extra_api_calls"] == 0)
        check(f"{name}:phase_times", all(float(summary[key]) >= 0.0 for key in (
            "force_wall_seconds", "probe_wall_seconds", "oeprop_wall_seconds")))
        check(f"{name}:restart_hamiltonian", not restart or summary["restart_hamiltonian_verified"] is True)
        if backend.startswith("gpu"):
            check(f"{name}:gpu_backend", any(token in str(summary["backend"]).upper() for token in ("GPU", "CUDA", "HIP")))
        rank_rows = read_csv(Path(str(prefix) + "_kernels.csv"), ["rank", "force_calls",
                             "api_energy_gradient_calls", "property_calls", "probe_generation_calls",
                             "properties_extra_api_calls", "esp_launches", "efield_launches", "efg_launches",
                             "force_wall_seconds", "probe_wall_seconds", "oeprop_wall_seconds"])
        rank_count = int(summary["mpi_ranks"])
        if rank_count < 1 or [int(row["rank"]) for row in rank_rows] != list(range(rank_count)):
            raise ValueError(f"Wrong MPI rank evidence: {name}")
        command = backends[backend]
        expected_ranks = 1 if backend in ("cpu", "gpu") else None
        for option in ("-np", "-n", "--np"):
            if option in command:
                expected_ranks = int(command[command.index(option) + 1])
        check(f"{name}:mpi_rank_count", expected_ranks is None or expected_ranks == rank_count)
        check(f"{name}:all_rank_api_counts", all(
            int(row["force_calls"]) == steps + 1 and int(row["api_energy_gradient_calls"]) == steps + 1
            and int(row["property_calls"]) == len(sampled_steps)
            and int(row["probe_generation_calls"]) == len(sampled_steps)
            and int(row["properties_extra_api_calls"]) == 0 for row in rank_rows))
        expected_launches = len(sampled_steps) if backend.startswith("gpu") else 0
        check(f"{name}:actual_property_kernel_launches", all(
            int(row[column]) == expected_launches for row in rank_rows
            for column in ("esp_launches", "efield_launches", "efg_launches")))
        phase_maxima = {column: max(float(row[column]) for row in rank_rows)
                        for column in ("force_wall_seconds", "probe_wall_seconds", "oeprop_wall_seconds")}
        check(f"{name}:all_rank_finite_nonnegative_phase_times", all(
            float(row[column]) >= 0.0 for row in rank_rows for column in phase_maxima))
        energy = read_csv(Path(str(prefix) + "_energy.csv"), ["step"] + ENERGY_COLUMNS)
        if [int(row["step"]) for row in energy] != all_steps:
            raise ValueError(f"Wrong energy step sequence: {name}")
        for row in energy:
            if abs(float(row["time_fs"]) - int(row["step"]) * dt_fs) > 1.0e-12:
                raise ValueError(f"Wrong energy timestamp: {name}")
        trajectory = read_trajectory(Path(str(prefix) + "_trajectory.xyz"), all_steps, dt_fs)
        checkpoint = read_checkpoint(Path(str(prefix) + ".chk"))
        check(f"{name}:checkpoint_hamiltonian", checkpoint["keyword"] == keyword.upper())
        check(f"{name}:checkpoint_endpoint", checkpoint["step"] == start + steps
              and abs(float(checkpoint["dt_fs"]) - dt_fs) < 1.0e-14
              and max_difference(checkpoint["coordinates"], trajectory[-1]) <= 1.0e-11)
        properties = read_csv(Path(str(prefix) + "_oeprop.csv"), ["step", "time_fs", "probe", "probe_mode",
                             "x_ang", "y_ang", "z_ang"] + PROPERTY_COLUMNS + INVARIANT_COLUMNS,
                             allow_empty=stride == 0)
        actual_steps = list(dict.fromkeys(int(row["step"]) for row in properties))
        if actual_steps != sampled_steps:
            raise ValueError(f"Wrong property step sequence: {name}: {actual_steps}, expected {sampled_steps}")
        probe_error = 0.0
        for step in sampled_steps:
            frame_rows = [row for row in properties if int(row["step"]) == step]
            expected_count = len(probe_records[mode]) if mode != "density" else len(frame_rows)
            if expected_count < 1 or [int(row["probe"]) for row in frame_rows] != list(range(1, expected_count + 1)):
                raise ValueError(f"Wrong property probe sequence at step {step}: {name}")
            if any(row["probe_mode"] != mode or abs(float(row["time_fs"]) - step * dt_fs) > 1.0e-12 for row in frame_rows):
                raise ValueError(f"Wrong property mode/time: {name}")
            coordinates = np.asarray([[float(row[f"{axis}_ang"]) for axis in "xyz"] for row in frame_rows])
            if mode != "density":
                reference = np.asarray(probe_records[mode])
                atoms = trajectory[step - start]
                if mode == "fixed":
                    expected_coordinates = reference
                elif mode == "atom":
                    indices = reference[:, 0].astype(int) - 1
                    if np.any(reference[:, 0] != indices + 1) or np.any(indices < 0) or np.any(indices >= len(atoms)):
                        raise ValueError("Invalid atom probe index")
                    expected_coordinates = atoms[indices] + reference[:, 1:]
                else:
                    e1 = atoms[1] - atoms[0]
                    e1 /= np.linalg.norm(e1)
                    e3 = np.cross(e1, atoms[2] - atoms[0])
                    e3 /= np.linalg.norm(e3)
                    e2 = np.cross(e3, e1)
                    expected_coordinates = atoms[0] + reference @ np.asarray([e1, e2, e3])
                probe_error = max(probe_error, max_difference(coordinates, expected_coordinates))
        invariants = invariant_errors(properties)
        check(f"{name}:probe_coordinate_transform", probe_error <= 1.0e-10)
        check(f"{name}:independent_efg_invariants", all(value <= args.property_tolerance for value in invariants.values()))
        drift = max(abs(float(row["delta_total_au"])) for row in energy)
        check(f"{name}:summary_drift_matches_rows", abs(drift - float(summary["max_abs_drift_au"])) <= 1.0e-12)
        check(f"{name}:short_nve_drift", drift <= args.drift_tolerance)
        check(f"{name}:energy_identity", all(abs(float(row["total_au"]) - float(row["potential_au"])
              - float(row["kinetic_au"])) <= 1.0e-12 for row in energy))
        data = {"prefix": str(prefix), "summary": summary, "energy": energy, "properties": properties,
                "trajectory": trajectory, "checkpoint": checkpoint, "start_step": start, "dt_fs": dt_fs}
        cases[name] = data
        result["cases"][name] = {"prefix": str(prefix), "control_sha256": sha256(control), "summary": summary,
                                 "energy_rows": len(energy), "property_rows": len(properties),
                                 "probe_transform_max_abs_ang": probe_error, "invariant_max_abs": invariants,
                                 "rank_evidence": rank_rows, "phase_max_rank_seconds": phase_maxima}
        return data

    def compare_trajectory(name: str, first: dict[str, object], second: dict[str, object], backend: str) -> None:
        geo_error = max_difference(first["trajectory"], second["trajectory"])
        energy_errors = {column: max_difference([float(row[column]) for row in first["energy"]],
                                              [float(row[column]) for row in second["energy"]])
                         for column in ("potential_au", "kinetic_au", "total_au")}
        property_errors = property_difference(first["properties"], second["properties"])
        gpu = backend.startswith("gpu")
        check(f"{name}:short_trajectory_coordinates", geo_error <= (args.backend_coordinate_tolerance if gpu else args.restart_tolerance))
        check(f"{name}:short_trajectory_energy", max(energy_errors.values()) <= (args.backend_energy_tolerance if gpu else args.property_tolerance))
        check(f"{name}:short_trajectory_properties", max(property_errors.values()) <= (args.backend_property_tolerance if gpu else args.property_tolerance))
        result["comparisons"][name] = {"scope": "Separate short trajectories from identical starting inputs; not identical geometries/densities after integration",
                                      "coordinate_max_abs_ang": geo_error, "energy_max_abs_au": energy_errors,
                                      "property_max_abs": property_errors}

    def postprocess(name: str, data: dict[str, object], keyword: str, backend: str) -> None:
        prefix = Path(data["prefix"])
        output = workdir / f"{name}_postprocessed.csv"
        run([str(args.oeprop_driver.resolve()), "trajectory", str(Path(str(prefix) + "_trajectory.xyz")),
             str(probes["fixed"]), str(output), keyword.replace("GRADIENT", "ENERGY"), f"post_{name}"],
            workdir / f"{name}_postprocessed.log")
        rows = read_csv(output, ["frame", "probe", "energy_hartree", "x_ang", "y_ang", "z_ang"] + PROPERTY_COLUMNS)
        expected = [(frame, probe) for frame in range(1, len(data["energy"]) + 1)
                    for probe in range(1, len(probe_records["fixed"]) + 1)]
        if [(int(row["frame"]), int(row["probe"])) for row in rows] != expected:
            raise ValueError(f"Invalid postprocessed frame/probe sequence: {name}")
        for row in rows:
            row["step"] = str(int(row["frame"]) - 1 + data["start_step"])
        errors = property_difference(data["properties"], rows)
        coord_error = max_difference([[float(row[f"{axis}_ang"]) for axis in "xyz"] for row in data["properties"]],
                                    [[float(row[f"{axis}_ang"]) for axis in "xyz"] for row in rows])
        energy_error = max(abs(float(row["energy_hartree"]) - float(data["energy"][int(row["frame"]) - 1]["potential_au"]))
                           for row in rows)
        gpu = backend.startswith("gpu")
        check(f"{name}:same_geometry_cpu_scf_property", max(errors.values()) <= (args.backend_property_tolerance if gpu else args.property_tolerance))
        check(f"{name}:same_geometry_cpu_scf_energy", energy_error <= (args.backend_energy_tolerance if gpu else args.property_tolerance))
        check(f"{name}:postprocessed_probe_coordinates", coord_error <= 1.0e-12)
        result["comparisons"][f"{name}_same_geometry_cpu_postprocessing"] = {
            "scope": "Online density versus independently converged CPU SCFs at the identical saved geometries; not a fixed-density backend comparison",
            "property_max_abs": errors, "energy_max_abs_au": energy_error, "probe_coordinate_max_abs_ang": coord_error,
            "csv": str(output)}

    try:
        for backend in backends:
            full = run_case(backend, "full")
            postprocess(f"{backend}_full", full, HF_GRADIENT, backend)
            split_a = run_case(backend, "split_a", steps=2)
            split_b = run_case(backend, "split_b", steps=2, restart=Path(split_a["prefix"] + ".chk"))
            restart_errors = {"coordinate_max_abs_ang": max_difference(full["checkpoint"]["coordinates"], split_b["checkpoint"]["coordinates"]),
                              "velocity_max_abs_au": max_difference(full["checkpoint"]["velocities"], split_b["checkpoint"]["velocities"])}
            final_rows = [row for row in full["properties"] if int(row["step"]) == 4]
            split_rows = [row for row in split_b["properties"] if int(row["step"]) == 4]
            restart_errors["property_max_abs"] = property_difference(final_rows, split_rows)
            check(f"{backend}:restart_coordinates", restart_errors["coordinate_max_abs_ang"] <= args.restart_tolerance)
            check(f"{backend}:restart_velocities", restart_errors["velocity_max_abs_au"] <= args.restart_tolerance)
            check(f"{backend}:restart_properties", max(restart_errors["property_max_abs"].values()) <= args.property_tolerance)
            result["comparisons"][f"{backend}_restart"] = restart_errors
            for stride in (2, 0):
                decimated = run_case(backend, f"stride{stride}", stride=stride)
                check(f"{backend}:stride{stride}_trajectory_unchanged", max_difference(full["trajectory"], decimated["trajectory"]) <= args.restart_tolerance)
                check(f"{backend}:stride{stride}_energy_unchanged", max_difference(
                    [float(row["total_au"]) for row in full["energy"]], [float(row["total_au"]) for row in decimated["energy"]]) <= args.property_tolerance)
                if stride:
                    selected = [row for row in full["properties"] if int(row["step"]) % stride == 0]
                    differences = property_difference(selected, decimated["properties"])
                    check(f"{backend}:stride2_properties", max(differences.values()) <= args.property_tolerance)
                    result["comparisons"][f"{backend}_stride2"] = differences
            # Restart at an unsampled odd step: the resumed initial frame must remain absent.
            stride_a = run_case(backend, "stride2_split_a", steps=1, stride=2)
            stride_b = run_case(backend, "stride2_split_b", steps=3, stride=2,
                                restart=Path(stride_a["prefix"] + ".chk"))
            expected_rows = [row for row in full["properties"] if int(row["step"]) in (2, 4)]
            differences = property_difference(expected_rows, stride_b["properties"])
            check(f"{backend}:stride2_restart_properties", max(differences.values()) <= args.property_tolerance)
            check(f"{backend}:stride2_restart_endpoint", max_difference(
                full["checkpoint"]["coordinates"], stride_b["checkpoint"]["coordinates"]) <= args.restart_tolerance
                and max_difference(full["checkpoint"]["velocities"], stride_b["checkpoint"]["velocities"]) <= args.restart_tolerance)
            result["comparisons"][f"{backend}_stride2_restart"] = differences
            for mode in ("molecule", "atom", "density"):
                run_case(backend, mode, mode=mode, steps=1)
            trimer = run_case(backend, "trimer", geometry=fixtures["trimer"], steps=2, dt_fs=0.025, coordinate_atoms=(1, 4))
            postprocess(f"{backend}_trimer", trimer, HF_GRADIENT, backend)
            if args.run_pbe0:
                pbe0 = run_case(backend, "pbe0", steps=1, dt_fs=0.025, keyword=PBE0_GRADIENT)
                postprocess(f"{backend}_pbe0", pbe0, PBE0_GRADIENT, backend)
            if backend != "cpu":
                for label in ("full", "trimer") + (("pbe0",) if args.run_pbe0 else ()):
                    compare_trajectory(f"cpu_vs_{backend}_{label}", cases[f"cpu_{label}"], cases[f"{backend}_{label}"], backend)
        drivers_after = {name: driver_metadata(Path(command[-1])) for name, command in backends.items()}
        drivers_after["postprocessor_cpu"] = driver_metadata(args.oeprop_driver)
        source_after = source_metadata(repo)
        check("provenance:driver_and_loaded_libraries_unchanged", all(
            drivers_before[name][key] == drivers_after[name][key] for name in drivers_before
            for key in ("path", "sha256", "loaded_quick_libraries")))
        check("provenance:source_blobs_unchanged", source_before["source_sha256"] == source_after["source_sha256"])
        result["metadata"]["source_after"] = source_after
        result["metadata"]["drivers_after"] = drivers_after
        result["passed"] = all(checks.values())
    except (ValueError, OSError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
        result["error"] = str(error)
    (workdir / "validation_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bomd-driver", type=Path, required=True)
    parser.add_argument("--oeprop-driver", type=Path, required=True, help="CPU API trajectory postprocessor")
    parser.add_argument("--mpi-driver", type=Path)
    parser.add_argument("--gpu-driver", type=Path)
    parser.add_argument("--gpu-mpi-driver", type=Path)
    parser.add_argument("--launcher", default="mpirun -np 2")
    parser.add_argument("--gpu-launcher", default="mpirun -np 1", help="Use ranks appropriate to available physical GPUs")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/quick-oeprop-bomd-validation"))
    parser.add_argument("--run-pbe0", action="store_true")
    parser.add_argument("--drift-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--property-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--restart-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--backend-property-tolerance", type=float, default=2.0e-7)
    parser.add_argument("--backend-coordinate-tolerance", type=float, default=1.0e-7)
    parser.add_argument("--backend-energy-tolerance", type=float, default=1.0e-7)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if name.endswith("tolerance") and (not math.isfinite(value) or value < 0):
            parser.error(f"{name} must be finite and nonnegative")
    result = validate(args)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
