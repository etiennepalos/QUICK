#!/usr/bin/env python3
"""Exercise standalone analytic and numerical-only EFG grid input parsing.

The unchanged acetone B3LYP/def2-SVP and PBE0-D3BJ/cc-pVDZ input pairs differ
only in EFG_GRID versus EFG_GRID_NUMERICAL. Every input probe must be present
in order, both methods must converge, and tensors must match each other and
their checked-in references at the existing absolute regression tolerance.
"""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

from validate_master_integration import ENVIRONMENT_KEYS, identities, number, sha256, source_paths

ROOT = Path(__file__).resolve().parents[2]
CASES = ("acetone_b3lyp_def2svp", "acetone_pbe0_d3bj_ccpvdz")
BOHRS_TO_A = 0.52917720859
ENERGY_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"


def runtime(quick: Path, env: dict[str, str]) -> dict:
    result = subprocess.run(["ldd", str(quick)], env=env, text=True,
                            capture_output=True, check=True)
    if "not found" in result.stdout:
        raise ValueError("Unresolved runtime library")
    libraries = []
    for line in result.stdout.splitlines():
        match = re.search(r"(?:=>\s*)?(/[^\s]+)", line)
        if match:
            libraries.append(Path(match.group(1)).resolve())
    hosts = [path for path in libraries if re.fullmatch(
        r"libquick(?:_mpi)?(?:_cuda|_hip)?\.so(?:\.[\d.]+)?", path.name)]
    if len(hosts) != 1:
        raise ValueError("Expected exactly one resolved QUICK host library")
    if hosts[0].parent != quick.parent:
        raise ValueError("QUICK host library resolves outside the supplied executable directory")
    cache = quick.parent.parent / "CMakeCache.txt"
    home = re.search(r"(?m)^CMAKE_HOME_DIRECTORY:INTERNAL=(.*)$", cache.read_text())
    if home is None or Path(home.group(1)).resolve() != ROOT:
        raise ValueError("CMake build source does not match this harness checkout")
    return {"executable": str(quick), "quick_library": str(hosts[0]),
            "files": identities([quick, cache, *libraries]),
            "build_provenance_limit": "Compilation is established by separate build logs."}


def points_from_input(path: Path) -> list[list[float]]:
    blocks = re.split(r"\n\s*\n", path.read_text().strip())
    if len(blocks) != 3:
        raise ValueError(f"Expected keyword, molecule, and probe blocks: {path}")
    points = [[number(word) / BOHRS_TO_A for word in line.split()]
              for line in blocks[2].splitlines()]
    if not points or any(len(point) != 3 for point in points):
        raise ValueError(f"Malformed input probe block: {path}")
    return points


def grid_rows(path: Path) -> list[list[float]]:
    rows = []
    table = False
    for line in path.read_text().splitlines():
        words = line.split()
        if words[:3] == ["X", "Y", "Z"] and "EFG_XX" in words:
            table = True
            continue
        if not table or not words:
            continue
        if len(words) != 12:
            raise ValueError(f"Malformed EFG table row: {path}")
        rows.append([number(word) for word in words])
    if not table or not rows:
        raise ValueError(f"Missing or empty EFG table: {path}")
    return rows


def check_points(rows: list[list[float]], points: list[list[float]]) -> float:
    if len(rows) != len(points):
        raise ValueError(f"Probe count differs: output {len(rows)}, input {len(points)}")
    maximum = 0.0
    for row, point in zip(rows, points):
        for actual, expected in zip(row[:3], point):
            # Coordinates are printed with ES14.6, so compare to the known
            # seven-significant-digit output precision, retaining input order.
            rounding = 0.0 if expected == 0 else 0.500001 * 10.0 ** (math.floor(math.log10(abs(expected))) - 6)
            tolerance = rounding + 64.0 * sys.float_info.epsilon * max(1.0, abs(expected))
            error = abs(actual - expected)
            maximum = max(maximum, error)
            if error > tolerance:
                raise ValueError("Output coordinate/order does not match the original input probe")
    return maximum


def tensor_difference(left: list[list[float]], right: list[list[float]], tolerance: float) -> dict:
    if len(left) != len(right):
        raise ValueError("Compared EFG tables have different point counts")
    if any(a[:3] != b[:3] for a, b in zip(left, right)):
        raise ValueError("Compared EFG tables have different coordinate order")
    maximum = max(abs(x-y) for a, b in zip(left, right) for x, y in zip(a[3:], b[3:]))
    if maximum > tolerance:
        raise ValueError(f"EFG maximum absolute error {maximum:.9g} exceeds {tolerance:.9g}")
    return {"passed": True, "points": len(left), "components": 9 * len(left), "max_abs": maximum}


def read_result(folder: Path, stem: str, numerical: bool, points: list[list[float]]) -> tuple[dict, list[list[float]]]:
    text = (folder / f"{stem}.out").read_text(errors="replace")
    cycles = re.findall(r"REACH CONVERGENCE AFTER\s+(\d+)\s+CYCLES", text)
    if len(cycles) != 1 or re.search(r"NO CONVERGENCE|RAN OUT OF CYCLES|FAILED TO CONVERGE", text, re.I):
        raise ValueError("Expected exactly one explicitly converged SCF")
    if re.search(r"(?i)(?<![a-z])(?:nan|[-+]?inf(?:inity)?)(?![a-z])", text):
        raise ValueError("Nonfinite token in QUICK output")
    energies = re.findall(r"(?m)^\s*TOTAL ENERGY\s*=\s*(" + ENERGY_NUMBER + r")\s*$", text)
    if len(energies) != 1:
        raise ValueError("Expected exactly one total energy")
    if ("NUMERICAL EFG FINITE DIFFERENCE" in text) != numerical:
        raise ValueError("Selected numerical/analytic EFG dispatch differs from the input keyword")
    if len(list(folder.glob("*.efg"))) != 1:
        raise ValueError("Expected exactly one EFG output table")
    rows = grid_rows(folder / f"{stem}.efg")
    return {"scf_converged": True, "scf_cycles": int(cycles[0]), "energy_hartree": number(energies[0]),
            "points": len(rows), "finite_tensor_components": 9 * len(rows),
            "input_coordinate_max_abs_bohr": check_points(rows, points)}, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", type=Path, required=True)
    parser.add_argument("--launcher", default="", help="For example 'mpirun -np 2' for QUICK MPI")
    parser.add_argument("--basis", type=Path, default=ROOT / "basis")
    parser.add_argument("--workdir", type=Path, required=True, help="New evidence directory")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=2.0e-7)
    parser.add_argument("--energy-tolerance", type=float, default=4.0e-5)
    args = parser.parse_args()
    if args.threads < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("threads and timeout must be positive and finite")
    if any(not math.isfinite(value) or value < 0 for value in (args.tolerance, args.energy_tolerance)):
        parser.error("tolerances must be finite and nonnegative")
    args.quick = args.quick.resolve(strict=True)
    args.basis = args.basis.resolve(strict=True)
    args.workdir = args.workdir.resolve()
    args.workdir.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update({"QUICK_BASIS": str(args.basis), "OMP_NUM_THREADS": str(args.threads),
                "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    report = {"passed": False, "started_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Standalone numerical-only grid parser and analytic/numerical EFG parity",
              "tolerance_au": args.tolerance, "energy_tolerance_hartree": args.energy_tolerance,
              "limitations": "Existing Cartesian d-basis fixtures; separate SCFs, no performance claim.",
              "processes": [], "comparisons": {}}
    paths = []
    try:
        paths = source_paths(args.basis) + [Path(__file__).resolve()]
        for case in args.cases:
            for prefix in ("efg_grid_", "efg_grid_numerical_"):
                paths.extend([ROOT / "test" / f"{prefix}{case}.in", ROOT / "test/saved" / f"{prefix}{case}.efg"])
        report["sources_before"] = identities(paths)
        report["runtime_before"] = runtime(args.quick, env)
        for case in args.cases:
            analytic_input = ROOT / "test" / f"efg_grid_{case}.in"
            numerical_input = ROOT / "test" / f"efg_grid_numerical_{case}.in"
            if analytic_input.read_bytes().replace(b"EFG_GRID", b"EFG_GRID_NUMERICAL", 1) != numerical_input.read_bytes():
                raise ValueError("Fixture pair differs in more than the single EFG keyword")
            points = points_from_input(analytic_input)
            outputs, metadata = {}, {}
            for mode, source in (("analytic", analytic_input), ("numerical", numerical_input)):
                run_dir = args.workdir / f"{case}-{mode}"
                run_dir.mkdir()
                shutil.copyfile(source, run_dir / source.name)
                command = [*shlex.split(args.launcher), str(args.quick), source.name]
                result = run_process_in_existing(command, run_dir, env, args.timeout)
                report["processes"].append(result)
                if result["returncode"] != 0 or result["timed_out"]:
                    raise ValueError(f"QUICK process failed or timed out: {case}/{mode}")
                if sha256(run_dir / source.name) != sha256(source):
                    raise ValueError("QUICK input bytes changed")
                metadata[mode], outputs[mode] = read_result(run_dir, source.stem, mode == "numerical", points)
                reference = grid_rows(ROOT / "test/saved" / f"{source.stem}.efg")
                check_points(reference, points)
                metadata[mode]["saved_reference"] = tensor_difference(outputs[mode], reference, args.tolerance)
            comparison = tensor_difference(outputs["analytic"], outputs["numerical"], args.tolerance)
            comparison["energy_abs_hartree"] = abs(metadata["analytic"]["energy_hartree"] - metadata["numerical"]["energy_hartree"])
            if comparison["energy_abs_hartree"] > args.energy_tolerance:
                raise ValueError("Analytic/numerical initialization energies differ")
            report["comparisons"][case] = {"parity": comparison, "runs": metadata}
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            report["sources_after"] = identities(paths)
            report["runtime_after"] = runtime(args.quick, env)
            report["immutable_sources"] = report.get("sources_before") == report["sources_after"]
            report["immutable_runtime"] = report.get("runtime_before") == report["runtime_after"]
            if not report["immutable_sources"] or not report["immutable_runtime"]:
                report["passed"] = False
                report["identity_error"] = "Scientific source/input/runtime identities changed"
        except Exception as exc:
            report["passed"] = False
            report["identity_error"] = f"{type(exc).__name__}: {exc}"
        report["outputs"] = identities([path for path in args.workdir.rglob("*")
                                        if path.is_file() and path.name != "report.json"])
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (args.workdir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "report": str(args.workdir / "report.json"),
                      "error": report.get("error"), "identity_error": report.get("identity_error")}))
    return 0 if report["passed"] else 1


def run_process_in_existing(command: list[str], folder: Path, env: dict[str, str], timeout: float) -> dict:
    start = time.perf_counter()
    record = {"argv": command, "cwd": str(folder), "timeout_seconds": timeout,
              "environment": {key: env.get(key) for key in ENVIRONMENT_KEYS}}
    with (folder / "process.log").open("wb") as handle:
        process = subprocess.Popen(command, cwd=folder, env=env, stdout=handle,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            record["returncode"] = process.wait(timeout=timeout)
            record["timed_out"] = False
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            record["returncode"] = process.returncode
            record["timed_out"] = True
    record["elapsed_seconds_diagnostic_only"] = time.perf_counter() - start
    (folder / "invocation.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    sys.exit(main())
