#!/usr/bin/env python3
"""Fixed-density OEPROP acceptance and warmed property-only timing (stdlib only).

Run binaries built from this worktree. Default acceptance and benchmarks use
one initialization SCF; the optional repeated-gradient lifecycle mode uses two.
MPI/GPU load the CPU reference density for the pointwise property checks.
"""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
from __future__ import annotations

import argparse
import csv
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

FIELDS = {
    "esp": ["esp"],
    "efield": [f"efield_{a}" for a in "xyz"],
    "efg": [f"efg_{a}{b}" for a in "xyz" for b in "xyz"],
}
SIZES = [64, 256, 1024, 4096, 16384, 65536]
CUTOFFS = "CUTOFF=1.0D-12 DENSERMS=4.0D-8 ENERGY"
ROOT = Path(__file__).resolve().parents[2]


def geometry(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if not line.strip()) + 1
    atoms = []
    for line in lines[start:]:
        if not line.strip():
            if atoms:
                break
            continue
        atoms.append(line.strip())
    return atoms


def cases(root: Path) -> dict[str, tuple[str, list[str]]]:
    return {
        "water": (f"HF BASIS=STO-3G CHARGE=0 {CUTOFFS}", geometry(root / "test/ene_H2O_rhf_sto3g.in")),
        "acetone": (f"PBE0 D3BJ BASIS=cc-pVDZ CHARGE=0 XCCUTOFF=1.0D-12 BASISCUTOFF=1.0D-12 {CUTOFFS}",
                    geometry(root / "test/esp_grid_acetone_pbe0_d3bj_ccpvdz.in")),
        "ammonium": (f"HF BASIS=STO-3G CHARGE=1 {CUTOFFS}", geometry(root / "test/ene_NH4_rhf_sto3g.in")),
        "oh": (f"UHF BASIS=STO-3G CHARGE=0 MULT=2 {CUTOFFS}", ["O 0.0 0.0 0.0", "H 0.0 0.0 0.97"]),
    }


def grid(n: int) -> list[tuple[float, float, float]]:
    # Nested deterministic sequence: all backend/size comparisons share points.
    # Radii 3.5--4.4 A avoid molecular nuclei for these compact molecules.
    points = []
    for i in range(n):
        z = 2.0 * ((i * 0.6180339887498949 + 0.5) % 1.0) - 1.0
        phi = 2.0 * math.pi * ((i * 0.7548776662466927) % 1.0)
        radius = 3.5 + (i % 10) * 0.1
        xy = radius * math.sqrt(max(0.0, 1.0 - z*z))
        points.append((xy*math.cos(phi), xy*math.sin(phi), radius*z))
    return points


def write_points(path: Path, points: list[tuple[float, float, float]]) -> None:
    path.write_text(str(len(points)) + "\n" + "".join(" ".join(f"{x:.17g}" for x in p) + "\n" for p in points))


def rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as handle:
        parsed = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(handle)]
    if not parsed or any(not math.isfinite(v) for row in parsed for v in row.values()):
        raise ValueError(f"Empty or nonfinite numeric output: {path}")
    return parsed


def components(path: Path) -> dict[str, list[float]]:
    parsed = rows(path)
    return {kind: [row[field] for row in parsed for field in names] for kind, names in FIELDS.items()}


def file_components(path: Path, count: int) -> list[float]:
    values = []
    for line in path.read_text().splitlines():
        words = line.split()
        if len(words) != count + 3:
            continue
        try:
            numbers = [float(word.replace("D", "E").replace("d", "e")) for word in words]
        except ValueError:
            continue
        values.extend(numbers[3:])
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError(f"Empty or nonfinite property file: {path}")
    return values


def metric(a: list[float], b: list[float], tolerance: float) -> dict[str, object]:
    if len(a) != len(b):
        raise ValueError(f"Property value count differs: {len(a)} versus {len(b)}")
    differences = [x-y for x, y in zip(a, b)]
    maximum = max(map(abs, differences), default=0.0)
    return {"nvalues": len(a), "max_abs": maximum,
            "rms": math.sqrt(sum(d*d for d in differences)/len(differences)),
            "tolerance": tolerance, "passed": maximum <= tolerance}


def benchmark_checks(timings: list[dict], expected: list[tuple], repeats: int,
                     tolerance: float) -> dict[str, dict]:
    """Compare sums per component; cancellation prevents pointwise inference."""
    groups: dict[tuple, list[dict]] = {}
    for row in timings:
        key = (row["case"], row["backend"], row["npoints"], row["property"])
        groups.setdefault(key, []).append(row)
    checks = {}
    for case, backend, npoints, prop in expected:
        key = (case, backend, npoints, prop)
        values = groups.get(key, [])
        label = f"{case}/{backend}/benchmark/{npoints}/{prop}"
        checks[f"{label}/trials"] = {
            "passed": sorted(row["repeat"] for row in values) == list(range(1, repeats+1))
                      and all(row["scf_calls"] == 1 and math.isfinite(row["seconds"])
                              and row["seconds"] >= 0 and math.isfinite(row["checksum"]) for row in values),
            "observed_trials": len(values), "expected_trials": repeats,
            "expected_scf_calls": 1}
        if not values:
            continue
        scale = npoints * (13 if prop == "all" else len(FIELDS[prop]))
        cpu = groups.get((case, "cpu", npoints, prop), [])
        sums = [row["checksum"]/scale for row in values]
        for name, reference in [("checksum_repeat_consistency", values), ("checksum_cpu_parity", cpu)]:
            if reference:
                result = metric(sums, [reference[0]["checksum"]/scale]*len(sums), tolerance)
                result["normalization"] = scale
                result["scope"] = "aggregate sum per component; not pointwise accuracy"
                checks[f"{label}/{name}"] = result
            else:
                checks[f"{label}/{name}"] = {"passed": False, "reason": "CPU timing reference missing"}
    return checks


def probe_command(command: list[str], timeout: int = 15) -> str:
    if not shutil.which(command[0]):
        return "unavailable"
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return (result.stdout + result.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return str(error)


def build_metadata(driver: Path) -> dict[str, object]:
    cache = driver.parent.parent / "CMakeCache.txt"
    if not cache.is_file():
        return {"cmake_cache": "unavailable; retain the build configuration separately"}
    wanted = {"COMPILER", "CUDA", "HIP", "MPI", "ENABLEF", "QUICK_USER_ARCH", "CMAKE_BUILD_TYPE",
              "CMAKE_Fortran_COMPILER", "CMAKE_CXX_COMPILER", "CUDA_NVCC_EXECUTABLE", "CUDA_VERSION",
              "CMAKE_Fortran_FLAGS", "CMAKE_Fortran_FLAGS_RELEASE", "CMAKE_CXX_FLAGS", "CMAKE_CXX_FLAGS_RELEASE",
              "CUDA_NVCC_FLAGS", "CUDA_NVCC_FLAGS_RELEASE"}
    options = {}
    for line in cache.read_text().splitlines():
        declaration, separator, value = line.partition("=")
        name = declaration.partition(":")[0]
        if separator and name in wanted:
            options[name] = value
    compiler = options.get("CMAKE_Fortran_COMPILER", "gfortran")
    return {"cmake_cache": str(cache), "cmake_cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
            "cmake_options": options, "fortran_compiler_version": probe_command([compiler, "--version"])}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def driver_metadata(command: list[str]) -> dict[str, object]:
    driver = Path(command[-1])
    libraries = [{"path": str(path), "resolved_path": str(path.resolve()),
                  "sha256": sha256(path)}
                 for path in sorted(driver.parent.glob("libquick*.so*")) if path.is_file()]
    process = subprocess.run(["ldd", str(driver)], text=True, capture_output=True,
                             check=False, timeout=15)
    loaded_libraries = {}
    for line in process.stdout.splitlines():
        match = re.search(r"libquick\S*\s+=>\s+(/\S+)", line)
        if match:
            library = Path(match[1]).resolve()
            loaded_libraries[str(library)] = sha256(library)
    return {"command": command, "resolved_path": str(driver.resolve()), "sha256": sha256(driver),
            "adjacent_quick_shared_libraries": libraries,
            "loaded_quick_libraries": loaded_libraries,
            "ldd_returncode": process.returncode, "ldd_output": process.stdout + process.stderr}


def metadata(root: Path, backends: dict[str, list[str]]) -> dict[str, object]:
    diff = subprocess.run(["git", "-C", str(root), "diff", "--binary", "HEAD", "--"],
                          capture_output=True, check=True, timeout=15).stdout
    status = probe_command(["git", "-C", str(root), "status", "--porcelain"])
    validation_sources = [root / "validation/oeprop_api/quick_oeprop_api_driver.f90", Path(__file__).resolve(), Path(__file__).with_name("README.md")]
    return {"source_status": status, "source_dirty": bool(status),
            "tracked_source_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "validation_source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in validation_sources},
            "environment": {key: os.environ.get(key) for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_LAUNCH_BLOCKING", "CUDA_CACHE_DISABLE", "LD_LIBRARY_PATH", "LD_PRELOAD"]},
            "pointwise_acceptance_scope": "16 deterministic external probes and each generated density-surface point; large timing grids compare sums normalized per component across backends and repeats, not pointwise validation",
            "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "cpu": probe_command(["lscpu"]), "gpu": probe_command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
            "cuda": probe_command(["nvcc", "--version"]), "fortran": probe_command(["gfortran", "--version"]),
            "mpi": probe_command(["mpirun", "--version"]), "python": sys.version,
            "revision": probe_command(["git", "-C", str(root), "rev-parse", "HEAD"]),
            "drivers": {name: driver_metadata(cmd) for name, cmd in backends.items()},
            "builds": {name: build_metadata(Path(cmd[-1])) for name, cmd in backends.items()},
            "quick_basis": os.environ.get("QUICK_BASIS", str(root/"basis")),
            "units": {"coordinates": "angstrom", "api_coordinates": "bohr", "properties": "atomic units", "timings": "wall seconds"},
            "conventions": {"field": "E_i=-dV/dC_i", "field_gradient": "G_ij=dE_i/dC_j", "csv_tensor_order": "xx,xy,xz,yx,yy,yz,zx,zy,zz", "api_tensor_layout": "Fortran component-fast (3,3,npoints)"},
            "timing_method": "one initialization SCF per process; one warm call per grid size/property; all recorded trials retained; synchronous host API including density/grid transfer, kernel, nuclear terms, and MPI reductions; MPI maximum rank wall time"}


def execute(command: list[str], work: Path, env: dict[str, str], log: str) -> None:
    with (work / log).open("w") as handle:
        subprocess.run(command, cwd=work, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-driver", type=Path, required=True)
    parser.add_argument("--mpi-driver", type=Path)
    parser.add_argument("--gpu-driver", type=Path)
    parser.add_argument("--gpu-mpi-driver", type=Path)
    parser.add_argument("--mpi-launch", default="mpirun -np 2")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=["water", "acetone", "ammonium", "oh"], default=["water", "acetone", "ammonium", "oh"])
    parser.add_argument("--tolerance", type=float, default=2.0e-7)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--gradient", action="store_true", help="Initialize acceptance through one analytic-gradient SCF")
    modes.add_argument("--lifecycle", action="store_true", help="Check two gradient SCFs at identical geometry with OEPROP between them")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-cases", nargs="+", choices=["water", "acetone"], default=["water", "acetone"])
    parser.add_argument("--benchmark-max-points", type=int, choices=SIZES, default=max(SIZES),
                        help="Largest benchmark grid; smaller supported sizes are also timed (default: %(default)s)")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gpu-evidence", type=Path, help="Profiler kernel-summary text containing getOEPROP* kernel names")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    root, work = args.root.resolve(), args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    backends = {"cpu": [str(args.cpu_driver.resolve())]}
    for name, path in [("mpi2", args.mpi_driver), ("gpu", args.gpu_driver), ("gpu_mpi2", args.gpu_mpi_driver)]:
        if path:
            backends[name] = (shlex.split(args.mpi_launch) if "mpi" in name else []) + [str(path.resolve())]
    env = os.environ.copy()
    env.setdefault("QUICK_BASIS", str(root/"basis"))
    case_specs = cases(root)
    summary: dict[str, object] = {"schema_version": 1, "metadata": metadata(root, backends), "cases": {}, "comparisons": {}, "timings": []}
    benchmark_sizes = [size for size in SIZES if size <= args.benchmark_max_points]
    summary["benchmark"] = {"enabled": args.benchmark, "max_points": args.benchmark_max_points,
                            "sizes": benchmark_sizes if args.benchmark else [], "repeats": args.repeats}
    mode = "lifecycle-gradient" if args.lifecycle else "acceptance-gradient" if args.gradient else "acceptance"
    summary["initialization"] = {"mode": mode, "scf_calls_per_case": 2 if args.lifecycle else 1,
                                 "initial_scf_energy_parity_tolerance": 4.0e-5}
    comparisons = summary["comparisons"]
    commands = []
    for name in args.cases:
        keyword, atoms = case_specs[name]
        if args.gradient or args.lifecycle:
            keyword = keyword.replace(" ENERGY", " GRADIENT") + " GRADCUTOFF=1.0D-12"
        case_dir = work / name
        case_dir.mkdir(exist_ok=True)
        xyz = case_dir / "molecule.xyz"
        xyz.write_text(f"{len(atoms)}\n{name}; coordinates in angstrom\n" + "\n".join(atoms) + "\n")
        probes = case_dir / "probes.dat"
        write_points(probes, grid(16))
        cpu_density = case_dir / "cpu" / "results.density.dat"
        reference = None
        reference_energy = None
        surface_reference = None
        summary["cases"][name] = {"keyword": keyword, "natoms": len(atoms), "probes": str(probes), "reference_density": str(cpu_density)}
        for backend, launcher in backends.items():
            run_dir = case_dir / backend
            run_dir.mkdir(exist_ok=True)
            command = launcher + [mode, str(xyz), str(probes), "results", keyword,
                                  "-" if backend == "cpu" else str(cpu_density)]
            commands.append({"cwd": str(run_dir), "argv": command})
            execute(command, run_dir, env, "acceptance.log")
            api_rows = rows(run_dir / "results.api.csv")
            initial_energy = api_rows[0]["energy_hartree"]
            if reference_energy is None:
                reference_energy = initial_energy
            comparisons[f"{name}/{backend}/initial_scf_energy_cpu_parity"] = metric([reference_energy], [initial_energy], 4.0e-5)
            lifecycle = rows(run_dir / "results.lifecycle.csv")[0]
            summary["cases"][name].setdefault("lifecycle", {})[backend] = lifecycle
            comparisons[f"{name}/{backend}/api_lifecycle"] = {"passed": lifecycle["post_delete_ierr"] == 47
                                                            and lifecycle["property_calls_preserve_scf"] == 1
                                                            and lifecycle["scf_calls"] == (2 if args.lifecycle else 1),
                                                            "observed": lifecycle}
            api = components(run_dir / "results.api.csv")
            if reference is None:
                reference = api
            for prop, values in api.items():
                comparisons[f"{name}/{backend}/{prop}_cpu_parity"] = metric(reference[prop], values, args.tolerance)
                printed = file_components(run_dir / f"results.{prop}", len(FIELDS[prop]))
                comparisons[f"{name}/{backend}/{prop}_api_file"] = metric(values, printed, args.tolerance)
            surface_rows = rows(run_dir / "results.surface.api.csv")
            trace_error = max(abs(row["efg_trace"] + 4.0*math.pi*1.0e-3) for row in surface_rows)
            comparisons[f"{name}/{backend}/surface_trace_density"] = {"max_abs": trace_error, "tolerance": args.tolerance,
                                                                       "passed": trace_error <= args.tolerance,
                                                                       "equation": "trace(G)=-4*pi*rho_e", "rho_e": 1.0e-3}
            surface_api = components(run_dir / "results.surface.api.csv")
            if surface_reference is None:
                surface_reference = surface_api
            for prop, values in surface_api.items():
                comparisons[f"{name}/{backend}/{prop}_surface_cpu_parity"] = metric(surface_reference[prop], values, args.tolerance)
                printed = file_components(run_dir / f"results.surface.{prop}", len(FIELDS[prop]))
                comparisons[f"{name}/{backend}/{prop}_surface_api_file"] = metric(values, printed, args.tolerance)
            with (run_dir / "results.checks.csv").open(newline="") as handle:
                for row in csv.DictReader(handle):
                    error = float(row["max_abs"])
                    comparisons[f"{name}/{backend}/{row['check']}"] = {"max_abs": error, "tolerance": args.tolerance,
                                                                          "passed": math.isfinite(error) and error <= args.tolerance}
            kernel_rows = rows(run_dir / "results.kernels.csv")
            observed = all(row[field] > 0 for row in kernel_rows for field in ["esp_launches", "efield_launches", "efg_launches"])
            summary["cases"][name].setdefault("kernel_launches", {})[backend] = kernel_rows
            if "gpu" in backend:
                comparisons[f"{name}/{backend}/gpu_kernels_executed"] = {"passed": observed, "counts": kernel_rows}
            # The text round-trip must reproduce each AO density component exactly.
            comparisons[f"{name}/{backend}/fixed_density_snapshot"] = {"passed": cpu_density.read_bytes() == (run_dir/"results.density.dat").read_bytes(), "reference": str(cpu_density)}
        if args.benchmark and name in args.benchmark_cases:
            timing_points = case_dir / "benchmark_probes.dat"
            write_points(timing_points, grid(max(benchmark_sizes)))
            summary["cases"][name]["benchmark_sizes"] = benchmark_sizes
            for backend, launcher in backends.items():
                run_dir = case_dir / backend
                command = launcher + ["benchmark-case", str(xyz), str(timing_points), "benchmark", keyword, str(cpu_density), str(args.repeats)]
                commands.append({"cwd": str(run_dir), "argv": command})
                execute(command, run_dir, env, "benchmark.log")
                with (run_dir / "benchmark.timing.csv").open(newline="") as handle:
                    for row in csv.DictReader(handle):
                        summary["timings"].append({"case": name, "backend": backend, "npoints": int(row["npoints"]), "property": row["property"],
                                                  "repeat": int(row["repeat"]), "seconds": float(row["seconds"]), "checksum": float(row["checksum"]), "scf_calls": int(row["scf_calls"])})
    expected_timings = [(name, backend, size, prop) for name in args.cases
                        if args.benchmark and name in args.benchmark_cases
                        for backend in backends for size in benchmark_sizes
                        for prop in ["esp", "efield", "efg", "all"]]
    comparisons.update(benchmark_checks(summary["timings"], expected_timings, args.repeats, args.tolerance))
    drivers_after = {name: driver_metadata(command) for name, command in backends.items()}
    summary["metadata"]["drivers_after"] = drivers_after
    drivers_before = summary["metadata"]["drivers"]
    comparisons["provenance/driver_and_loaded_libraries_unchanged"] = {
        "passed": all(drivers_before[name][key] == drivers_after[name][key]
                      for name in backends for key in ["resolved_path", "sha256", "loaded_quick_libraries"])
                  and all(data["ldd_returncode"] == 0 and bool(data["loaded_quick_libraries"])
                          for snapshots in [drivers_before, drivers_after] for data in snapshots.values()),
        "scope": "Executable and ldd-resolved QUICK shared-library paths and SHA-256 hashes before/after the complete run"}
    summary["commands"] = commands
    gpu_names = ["getOEPROP_kernel", "getOEPROP_efield_kernel", "getOEPROP_efg_kernel"]
    evidence = args.gpu_evidence.read_text() if args.gpu_evidence else ""
    all_gpu_counts = [data["kernel_launches"][backend] for data in summary["cases"].values() for backend in backends if "gpu" in backend]
    counted = bool(all_gpu_counts) and all(row[field] > 0 for rank_counts in all_gpu_counts for row in rank_counts for field in ["esp_launches", "efield_launches", "efg_launches"])
    summary["gpu_execution_evidence"] = {"successful_launch_counts_all_cases_and_ranks": counted, "path": str(args.gpu_evidence) if args.gpu_evidence else None,
                                         "kernel_names_found": [name for name in gpu_names if name in evidence],
                                         "all_property_kernels_observed": all(name in evidence for name in gpu_names)}
    summary["passed"] = all(item["passed"] for item in comparisons.values())
    groups: dict[tuple, list[float]] = {}
    for row in summary["timings"]:
        groups.setdefault((row["case"], row["backend"], row["npoints"], row["property"]), []).append(row["seconds"])
    summary["timing_summary"] = [{"case": key[0], "backend": key[1], "npoints": key[2], "property": key[3],
                                   "median_seconds": statistics.median(values), "min_seconds": min(values), "max_seconds": max(values)} for key, values in groups.items()]
    (work/"acceptance.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if summary["timings"]:
        with (work/"timings.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary["timings"][0]))
            writer.writeheader()
            writer.writerows(summary["timings"])
    failures = {name: result for name, result in comparisons.items() if not result["passed"]}
    print(json.dumps({"passed": summary["passed"], "comparisons": len(comparisons), "failures": failures,
                      "artifacts": str(work), "gpu_execution_evidence": summary["gpu_execution_evidence"]}, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
