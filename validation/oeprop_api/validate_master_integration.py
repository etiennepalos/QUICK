#!/usr/bin/env python3
"""CPU parser and four-rank split-communicator integration regression.

Two interleaved MPI groups run different molecules, probe counts, surface
isovalues, and property-call counts. Each group's outputs must agree with its
serial reference. A timeout also detects accidental world collectives.
"""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PARSER_CASES = [
    "analytic_grid", "numerical_implies_grid", "reject_property_substrings",
    "esp_surface", "efield_surface", "efg_surface", "density_value_implies_mode",
    "density_spacing_implies_mode", "density_radius_implies_mode",
    "reject_density_substring", "external_field_origin", "finite_field_alias",
    "finite_alias_precedence", "reject_field_substrings", "cutoff_boundaries",
    "rejected_then_valid_token", "origin_only_enables_field", "tab_boundaries",
]
PROPERTY_FIELDS = [
    "esp", "efield_x", "efield_y", "efield_z", "efield_mag",
    *[f"efg_{a}{b}" for a in "xyz" for b in "xyz"],
    "efg_trace", "efg_frobenius", "efg_anisotropy",
]
COORDINATE_FIELDS = ["x_ang", "y_ang", "z_ang"]
NUMERIC_FIELDS = ["frame", "probe", "energy_hartree", *COORDINATE_FIELDS, *PROPERTY_FIELDS]
ENVIRONMENT_KEYS = [
    "QUICK_BASIS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH", "LD_PRELOAD", "OMPI_ALLOW_RUN_AS_ROOT",
    "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identities(paths: list[Path]) -> dict[str, str]:
    return {str(path.resolve()): sha256(path) for path in sorted(set(paths))}


def source_paths(basis: Path) -> list[Path]:
    # Git's inventory excludes build products; the untracked new harness is
    # included explicitly until the integration commit records it.
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "src"], cwd=ROOT,
        check=True, capture_output=True,
    )
    paths = [ROOT / os.fsdecode(name) for name in result.stdout.split(b"\0") if name]
    paths.extend(path for path in basis.rglob("*") if path.is_file())
    paths.extend([
        Path(__file__).resolve(),
        ROOT / "validation/oeprop_api/quick_oeprop_api_driver.f90",
        ROOT / "validation/oeprop_bomd/quick_oeprop_bomd_driver.f90",
    ])
    return paths


def runtime(driver: Path, env: dict[str, str]) -> dict:
    result = subprocess.run(
        ["ldd", str(driver)], env=env, check=True, capture_output=True, text=True,
    )
    libraries = []
    for line in result.stdout.splitlines():
        if "not found" in line:
            raise ValueError(f"Unresolved library for {driver}: {line.strip()}")
        match = re.search(r"(?:=>\s*)?(/[^\s]+)", line)
        if match:
            libraries.append(Path(match.group(1)).resolve())
    paths = [driver.resolve(), *libraries]
    if any(re.search(r"(?:cuda|cudart|cublas|cusolver|hip|rocblas|quickgpu)", str(path), re.I)
           for path in paths):
        raise ValueError("This regression requires CPU serial and CPU MPI binaries")
    return {"driver": str(driver.resolve()), "files": identities(paths)}


def number(raw: str) -> float:
    value = raw.strip().replace("D", "E").replace("d", "e")
    # ES24.16 omits E when the exponent requires three digits.
    if not re.search(r"[eE]", value):
        value = re.sub(r"(?<=\d)([+-]\d{3,})$", r"E\1", value)
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite output: {raw!r}")
    return result


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Invalid CSV header: {path}")
        rows = list(reader)
    if not rows or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Empty or malformed CSV: {path}")
    return rows


def check_parser(path: Path) -> dict:
    rows = csv_rows(path)
    if [row["case"] for row in rows] != PARSER_CASES:
        raise ValueError("Parser case order/count differs from the explicit 18-case fixture")
    flags = ["passed", "esp", "efield", "efg", "numerical", "density", "external",
             "extgrid", "angstrom"]
    for row in rows:
        if row["passed"].strip() != "T":
            raise ValueError(f"Parser contract failed: {row['case']}")
        if any(row[key].strip() not in ("T", "F") for key in flags):
            raise ValueError(f"Invalid parser flag: {row['case']}")
        for key, value in row.items():
            if key not in ["case", *flags]:
                number(value)
    return {"passed": True, "cases": len(rows), "raw": str(path), "sha256": sha256(path)}


def check_state(path: Path, group: int, ranks: int, surface_count: int) -> dict:
    rows = csv_rows(path)
    if len(rows) != 1:
        raise ValueError(f"Expected one state row: {path}")
    row = rows[0]
    if (int(row["group"]) != group or int(row["ranks"]) != ranks
            or int(row["api_step_before"]) != int(row["api_step_after"])
            or int(row["api_step_before"]) < 1
            or row["scf_converged"].strip() != "T"
            or int(row["surface_points"]) != surface_count):
        raise ValueError(f"SCF/lifecycle/subgroup state failed: {path}")
    return row


def compare_properties(reference: Path, candidate: Path, args: argparse.Namespace) -> dict:
    expected = csv_rows(reference)
    actual = csv_rows(candidate)
    if len(expected) != len(actual):
        raise ValueError(f"Probe count differs: {reference} vs {candidate}")
    maxima = {field: 0.0 for field in NUMERIC_FIELDS}
    for index, (left, right) in enumerate(zip(expected, actual), start=1):
        if set(left) != set(NUMERIC_FIELDS) or set(right) != set(NUMERIC_FIELDS):
            raise ValueError("Property CSV schema changed")
        for field in NUMERIC_FIELDS:
            a, b = number(left[field]), number(right[field])
            if field in ("frame", "probe"):
                target = 1 if field == "frame" else index
                if a != target or b != target:
                    raise ValueError(f"Frame/probe order differs: {candidate}, row {index}")
                tolerance = 0.0
            elif field == "energy_hartree":
                tolerance = args.energy_atol
            elif field in COORDINATE_FIELDS:
                tolerance = args.coordinate_atol
            else:
                tolerance = args.property_atol
            error = abs(a - b)
            maxima[field] = max(maxima[field], error)
            if error > tolerance:
                raise ValueError(f"{candidate.name}, row {index}, {field}: {error:.6g} > {tolerance:.6g}")
    return {"passed": True, "points": len(actual), "max_abs_by_field": maxima,
            "reference": str(reference), "candidate": str(candidate)}


def run_process(command: list[str], cwd: Path, env: dict[str, str], timeout: float) -> dict:
    cwd.mkdir()
    start = time.perf_counter()
    record = {"argv": command, "cwd": str(cwd), "timeout_seconds": timeout,
              "environment": {key: env.get(key) for key in ENVIRONMENT_KEYS}}
    with (cwd / "process.log").open("wb") as handle:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=handle,
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
    (cwd / "invocation.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial-driver", type=Path, required=True)
    parser.add_argument("--mpi-driver", type=Path, required=True)
    parser.add_argument("--launcher", default="mpirun -np 4")
    parser.add_argument("--workdir", type=Path, required=True, help="New evidence directory")
    parser.add_argument("--basis", type=Path, default=ROOT / "basis")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--energy-atol", type=float, default=1.0e-10)
    parser.add_argument("--coordinate-atol", type=float, default=1.0e-8)
    parser.add_argument("--property-atol", type=float, default=1.0e-9)
    args = parser.parse_args()
    if args.threads < 1 or args.timeout <= 0:
        parser.error("threads and timeout must be positive")
    if any(not math.isfinite(value) or value < 0 for value in
           (args.energy_atol, args.coordinate_atol, args.property_atol)):
        parser.error("comparison tolerances must be finite and nonnegative")
    args.workdir = args.workdir.resolve()
    args.serial_driver = args.serial_driver.resolve(strict=True)
    args.mpi_driver = args.mpi_driver.resolve(strict=True)
    args.basis = args.basis.resolve(strict=True)
    args.workdir.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update({"QUICK_BASIS": str(args.basis), "OMP_NUM_THREADS": str(args.threads),
                "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "",
                "ROCR_VISIBLE_DEVICES": ""})
    report = {"passed": False, "started_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "CPU parser contract and two independent two-rank subcommunicators",
              "limitations": "Small HF/STO-3G regression; no performance, GPU, or large-rank claim.",
              "tolerances": {key: getattr(args, key) for key in
                             ("energy_atol", "coordinate_atol", "property_atol")},
              "processes": [], "checks": {}}
    paths = []
    try:
        paths = source_paths(args.basis)
        report["sources_before"] = identities(paths)
        report["runtime_before"] = {
            "serial": runtime(args.serial_driver, env), "mpi": runtime(args.mpi_driver, env)}
        jobs = [
            ("parser", [str(args.serial_driver), "parser-contract", "parser.csv"]),
            ("serial-group0", [str(args.serial_driver), "subcommunicator-reference", "result", "0"]),
            ("serial-group1", [str(args.serial_driver), "subcommunicator-reference", "result", "1"]),
            ("mpi-split", [*shlex.split(args.launcher), str(args.mpi_driver), "mpi-subcommunicator", "result"]),
        ]
        for name, command in jobs:
            result = run_process(command, args.workdir / name, env, args.timeout)
            report["processes"].append(result)
            if result["returncode"] != 0 or result["timed_out"]:
                raise ValueError(f"Process failed or timed out: {name}")
        report["checks"]["parser"] = check_parser(args.workdir / "parser/parser.csv")
        for group in (0, 1):
            serial = args.workdir / f"serial-group{group}" / f"result.group{group}"
            mpi = args.workdir / "mpi-split" / f"result.group{group}"
            checks = {}
            for kind in ("properties", "surface"):
                checks[kind] = compare_properties(Path(f"{serial}.{kind}.csv"), Path(f"{mpi}.{kind}.csv"), args)
            if checks["properties"]["points"] != (4 if group == 0 else 3):
                raise ValueError(f"Unexpected fixed-probe count for group {group}")
            checks["serial_state"] = check_state(Path(f"{serial}.state.csv"), group, 1, checks["surface"]["points"])
            checks["mpi_state"] = check_state(Path(f"{mpi}.state.csv"), group, 2, checks["surface"]["points"])
            report["checks"][f"group{group}"] = checks
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            report["sources_after"] = identities(paths)
            report["runtime_after"] = {
                "serial": runtime(args.serial_driver, env), "mpi": runtime(args.mpi_driver, env)}
            report["immutable_sources"] = report.get("sources_before") == report["sources_after"]
            report["immutable_runtime"] = report.get("runtime_before") == report["runtime_after"]
            if not report["immutable_sources"] or not report["immutable_runtime"]:
                report["passed"] = False
                report["identity_error"] = "Sources or resolved executable/library identities changed"
        except Exception as exc:
            report["passed"] = False
            report["identity_error"] = f"{type(exc).__name__}: {exc}"
        report["outputs"] = identities([
            path for path in args.workdir.rglob("*") if path.is_file() and path.name != "report.json"])
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (args.workdir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "report": str(args.workdir / "report.json"),
                      "error": report.get("error"), "identity_error": report.get("identity_error")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
