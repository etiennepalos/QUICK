#!/usr/bin/env python3
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
"""Matched CPU/CUDA V/E/G/H API timings on one retained density per case."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BOHR_ANGSTROM = .52917720859
GATES = {"V": (1e-10, 1e-9), "E": (1e-10, 1e-9),
         "G": (1e-10, 1e-9), "H": (1e-10, 1e-9)}
ROUTES = {"V": [1, 0, 0, 0], "E": [0, 1, 0, 0],
          "G": [0, 0, 1, 0], "H": [0, 0, 0, 1],
          "VEG": [1, 1, 1, 0], "VEGH": [1, 1, 1, 1]}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime(driver, env):
    process = subprocess.run(["ldd", str(driver)], env=env, text=True,
                             capture_output=True, check=True)
    if "not found" in process.stdout:
        raise ValueError("Unresolved runtime dependency")
    paths = {}
    for line in process.stdout.splitlines():
        match = re.search(r"(?:=>\s*)?(/\S+)\s+\(", line)
        if match:
            path = Path(match[1]).resolve(strict=True)
            paths[str(path)] = sha(path)
    quick = [p for p in paths if re.fullmatch(r"libquick(?:_cuda)?\.so(?:\.[\d.]+)?", Path(p).name)]
    if len(quick) != 1:
        raise ValueError("Expected exactly one serial QUICK host library")
    return {"driver": str(driver), "driver_sha256": sha(driver),
            "quick_library": quick[0], "libraries_sha256": paths}


def fixture(case):
    name = "water_trimer.xyz" if case.startswith("trimer") else "water.xyz"
    path = ROOT / "validation/oeprop_bomd" / name
    lines = path.read_text().splitlines()
    atoms = []
    for line in lines[2:]:
        words = line.split()
        atoms.append(({"H": 1, "O": 8}[words[0]], list(map(float, words[1:]))))
    if len(atoms) != int(lines[0]):
        raise ValueError("Invalid geometry")
    basis = {"water-sto3g": "STO-3G", "water-tzvpd": "DEF2-TZVPD",
             "trimer-svp": "DEF2-SVP"}[case]
    return atoms, basis, path


def input_text(atoms, basis, count):
    geometry = np.asarray([row[1] for row in atoms]) / BOHR_ANGSTROM
    center = geometry.mean(axis=0)
    radius = np.max(np.linalg.norm(geometry - center, axis=1))
    # Nested deterministic exterior boxes, held fixed for each frozen density.
    box_center = center + np.array([0., 0., radius + 3.])
    points = box_center + .5 * np.random.default_rng(50271).uniform(-1., 1., (count, 3))
    if np.min(np.linalg.norm(points[:, None] - geometry[None, :], axis=2)) <= .25:
        raise ValueError("Query too close to a nucleus")
    card = f"HF BASIS={basis} ENERGY SCF=400 DENSERMS=1e-10 CUTOFF=1e-12"
    text = card + f"\n{len(atoms)} 0 {count}\n"
    text += "".join(str(z) + " " + " ".join(f"{v:.17g}" for v in xyz) + "\n" for z, xyz in atoms)
    text += "".join(" ".join(f"{v:.17g}" for v in point) + "\n" for point in points)
    return text, points, {"center_bohr": box_center.tolist(), "half_width_bohr": .5, "seed": 50271}


def result_data(folder, backend, count, repeats, outputs="VEGH"):
    result = {}
    for line in (folder / "result.dat").read_text().splitlines():
        words = line.split()
        result[words[0]] = words[1] == "T" if words[1] in ("T", "F") else float(words[1])
    if (not result["scf_converged"] or result["points"] != count or result["repeats"] != repeats
            or any(result[k] != 0 for k in ("density_error", "fock_error", "extra_API_calls"))):
        raise ValueError(f"State/convergence/count violation: {folder}")
    with (folder / "timings.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    routes = {name: launches for name, launches in ROUTES.items()
              if outputs == "VEGH" or name not in ("H", "VEGH")}
    if result.get("hessian_requested", True) != (outputs == "VEGH"):
        raise ValueError("Requested outputs differ from native evidence")
    if len(rows) != len(routes) * (repeats + 1):
        raise ValueError("Missing timing rows")
    phases = {}
    for route, launches in routes.items():
        selected = [row for row in rows if row["route"] == route]
        if [int(row["iteration"]) for row in selected] != list(range(repeats + 1)):
            raise ValueError("Reordered iteration evidence")
        expected = launches if backend == "cuda" else [0, 0, 0, 0]
        for row in selected:
            observed = [int(row[name + "_launches"]) for name in ("esp", "efield", "efg", "hessian")]
            if observed != expected or not np.isfinite(float(row["seconds"])) or float(row["seconds"]) <= 0:
                raise ValueError(f"Invalid native launch/time evidence: {row}")
        values = [float(row["seconds"]) for row in selected[1:]]
        phases[route] = {"first_route_call_seconds": float(selected[0]["seconds"]),
                         "samples_seconds": values, "median_seconds": statistics.median(values),
                         "min_seconds": min(values), "max_seconds": max(values)}
    data = np.atleast_2d(np.loadtxt(folder / "properties.dat", skiprows=1))
    if data.shape != (count, 44) or not np.isfinite(data[:, :17]).all():
        raise ValueError("Malformed/nonfinite property output")
    if outputs == "VEGH" and not np.isfinite(data[:, 17:]).all():
        raise ValueError("Missing requested field Hessian")
    if outputs == "VEG" and not np.isnan(data[:, 17:]).all():
        raise ValueError("Unrequested field Hessian must be explicitly unavailable (NaN)")
    return result, phases, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for backend in ("cpu", "cuda"):
        parser.add_argument(f"--{backend}-driver", required=True, type=Path)
        parser.add_argument(f"--{backend}-basis", required=True, type=Path)
    parser.add_argument("--cases", nargs="+", choices=("water-sto3g", "water-tzvpd", "trimer-svp"),
                        default=["water-sto3g", "water-tzvpd"])
    parser.add_argument("--counts", nargs="+", type=int, default=[1000, 10000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--outputs", choices=("VEG", "VEGH"), default="VEGH")
    parser.add_argument("--timeout", type=float, default=1800.)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, default=ROOT,
                        help="Immutable production source used by the selected libraries")
    parser.add_argument("--build-report", type=Path,
                        help="Optional retained build/library/source-manifest provenance")
    args = parser.parse_args()
    if (len(set(args.counts)) != len(args.counts) or any(n < 1 or n > 100000 for n in args.counts)
            or not 1 <= args.repeats <= 100 or not np.isfinite(args.timeout) or args.timeout <= 0):
        parser.error("invalid counts/repeats/timeout")
    output = args.workdir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    reproduction = output / "reproduction"
    reproduction.mkdir()
    local_sources = {p.name: sha(p) for p in HERE.iterdir() if p.is_file()}
    for name in local_sources:
        shutil.copy2(HERE / name, reproduction / name)
    snapshot = args.source_snapshot.resolve(strict=True)
    source_names = ["CMakeLists.txt"] + sorted({str(p.relative_to(snapshot))
        for folder in ("src", "quick-cmake") for p in (snapshot/folder).rglob("*") if p.is_file()})
    production_before = {name: sha(snapshot / name) for name in source_names}
    report = {"passed": False, "started_utc": datetime.now(timezone.utc).isoformat(),
              "source_revision": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
              "scope": "Frozen-density serial native CPU/CUDA property API; separate V/E/G routes, no whole-BOMD claim",
              "gates": GATES, "outputs": args.outputs, "runs": [], "comparisons": [],
              "source_files_sha256": local_sources,
              "production_source_files_sha256": production_before,
              "production_source_root": str(snapshot),
              "limitations": ["Single GPU and CPU thread; isolated all-electron water cases",
                              "External/reaction backgrounds excluded; no SCF changes during properties",
                              "All property call costs included; initial SCF excluded from property timings",
                              "Exterior-box workload; this does not establish speed for other regions"]}
    envs, drivers, before = {}, {}, {}
    built_libraries = None
    if args.build_report:
        build_path = args.build_report.resolve(strict=True)
        build = json.loads(build_path.read_text())
        manifest_path = build_path.parent/"source-manifest.json"
        if not build.get("passed") or sha(manifest_path) != build["source_manifest_sha256"]:
            raise ValueError("Invalid retained build/source-manifest provenance")
        manifest = json.loads(manifest_path.read_text())
        if any(manifest.get(name) != digest for name,digest in production_before.items()):
            raise ValueError("Selected source snapshot does not match the compiled build manifest")
        built_libraries = {str(Path(path).resolve()): digest for backend in build["backends"].values()
                           for path,digest in backend["libraries_sha256"].items()}
        report["build_report"] = {"path":str(build_path),"sha256":sha(build_path),
                                  "source_manifest_sha256":sha(manifest_path)}
    for backend in ("cpu", "cuda"):
        driver = getattr(args, backend + "_driver").resolve(strict=True)
        basis = getattr(args, backend + "_basis").resolve(strict=True)
        env = os.environ.copy()
        env.update(QUICK_BASIS=str(basis), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        env["LD_LIBRARY_PATH"] = str(basis.parent / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
        if backend == "cuda":
            env["CUDA_VISIBLE_DEVICES"] = "0"
        drivers[backend], envs[backend] = driver, env
        compile_record = driver.with_suffix(".compile.json")
        if compile_record.is_file():
            compiled = json.loads(compile_record.read_text())
            if (compiled.get("binary_sha256") != sha(driver)
                    or compiled.get("source_sha256") != sha(HERE / "quick_electrostatics_benchmark.f90")):
                raise ValueError("Benchmark binary does not match its retained compile/source record")
        before[backend] = runtime(driver, env)
        if built_libraries is not None:
            actual = before[backend]["quick_library"]
            if built_libraries.get(actual) != before[backend]["libraries_sha256"][actual]:
                raise ValueError("Loaded QUICK library is absent from the selected build record")
    report["runtime_before"] = before
    try:
        for case in args.cases:
            atoms, basis_name, geometry = fixture(case)
            basis_filename = {"STO-3G": "STO-3G.BAS", "DEF2-TZVPD": "DEF2-TZVPD.BAS",
                              "DEF2-SVP": "DEF2-SVP.BAS"}[basis_name]
            basis_files = {backend: getattr(args, backend + "_basis").resolve() / basis_filename
                           for backend in ("cpu", "cuda")}
            if sha(basis_files["cpu"]) != sha(basis_files["cuda"]):
                raise ValueError("CPU/CUDA Gaussian basis files differ")
            density = None
            ao_basis_hash = None
            overlap_reference = None
            for count in args.counts:
                text, points, box = input_text(atoms, basis_name, count)
                observed = {}
                for backend in ("cpu", "cuda"):
                    folder = output / f"{case}-{count}-{backend}"
                    folder.mkdir()
                    (folder / "input.dat").write_text(text)
                    frozen = str(density) if density else "-"
                    command = [str(drivers[backend]), "input.dat", frozen, str(args.repeats), args.outputs]
                    row = {"case": case, "backend": backend, "points": count, "folder": str(folder),
                           "argv": command, "density_input": frozen, "geometry_sha256": sha(geometry),
                           "basis_name": basis_name, "basis_file": str(basis_files[backend]),
                           "basis_file_sha256": sha(basis_files[backend]), "box": box}
                    report["runs"].append(row)
                    print("RUN", case, count, backend, flush=True)
                    start = time.monotonic()
                    with (folder / "execution.log").open("w") as log:
                        process = subprocess.run(command, cwd=folder, env=envs[backend], stdout=log,
                                                 stderr=subprocess.STDOUT, timeout=args.timeout)
                    row.update(returncode=process.returncode, process_wall_seconds=time.monotonic()-start)
                    if process.returncode:
                        raise RuntimeError(f"Native client failed: {folder}")
                    result, phases, data = result_data(folder, backend, count, args.repeats, args.outputs)
                    row.update(result=result, phases=phases, files_sha256={p.name: sha(p) for p in folder.iterdir() if p.is_file()})
                    if np.max(np.abs(data[:, 1:4] - points)) > 1e-14:
                        raise ValueError("Native output query coordinates changed")
                    if density is None:
                        density = folder / "density.dat"
                        ao_basis_hash = sha(folder / "ao_basis.dat")
                        overlap_reference = np.loadtxt(folder / "overlap.dat", skiprows=1)
                    if sha(folder / "density.dat") != sha(density):
                        raise ValueError("Frozen density changed between runs")
                    if sha(folder / "ao_basis.dat") != ao_basis_hash:
                        raise ValueError("AO definition/order/normalization changed between runs")
                    if np.max(np.abs(np.loadtxt(folder / "overlap.dat", skiprows=1) - overlap_reference)) > 2e-12:
                        raise ValueError("AO overlap changed between runs")
                    observed[backend] = data
                    (output / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
                comparisons = {}
                for name, columns in {"V": slice(4,5), "E": slice(5,8), "G": slice(8,17), "H": slice(17,44)}.items():
                    if name == "H" and args.outputs == "VEG":
                        continue
                    reference = observed["cpu"][:, columns]
                    delta = observed["cuda"][:, columns] - reference
                    absolute, relative = GATES[name]
                    passed = np.all(np.abs(delta) <= absolute + relative * np.abs(reference))
                    comparisons[name] = {"max_abs": float(np.max(np.abs(delta))),
                                         "rms": float(np.sqrt(np.mean(delta**2))), "passed": bool(passed)}
                report["comparisons"].append({"case": case, "points": count, "same_density": True,
                                               "properties": comparisons})
        after = {backend: runtime(drivers[backend], envs[backend]) for backend in drivers}
        report["runtime_after"] = after
        if before != after:
            raise ValueError("Runtime files changed during benchmark")
        if production_before != {name: sha(snapshot / name) for name in source_names}:
            raise ValueError("Production source changed during benchmark")
        report["passed"] = all(row["passed"] for pair in report["comparisons"] for row in pair["properties"].values())
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "report": str(output / "report.json")}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
