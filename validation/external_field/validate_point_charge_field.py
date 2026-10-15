#!/usr/bin/env python3
"""Validate QM and external-charge analytic gradients under a uniform field.

The separately compiled quick_external_field_api_driver.f90 is a plain QUICK
API client. All three QM atoms and both signed external charges are displaced;
SCF-energy differences test each returned gradient component independently.
"""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location("external_field_validation", HERE / "validate_external_field.py")
reference = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(reference)


def data_text(case: dict, charges: list[list[float]], field: list[float] | None,
              origin: list[float], gradient: bool) -> str:
    card = (f"{case['method']} BASIS={case['basis']} {'GRADIENT' if gradient else 'ENERGY'} "
            "SCF=250 DENSERMS=1e-10 CUTOFF=1e-12 GRADCUTOFF=1e-12 EXTCHARGES")
    if case["charge"] != 0:
        card += f" CHARGE={case['charge']}"
    if case["mult"] != 1:
        card += f" MULT={case['mult']}"
    if "DFT" in case["method"]:
        card += " XCCUTOFF=1e-12 BASISCUTOFF=1e-12"
    if field is not None:
        for axis, value, center in zip("XYZ", field, origin):
            card += f" FINITE_FIELD_{axis}={value:.12g}"
            if center != 0.0:
                card += f" FINITE_FIELD_ORIGIN_{axis}={center:.12g}"
    if len(card) > 256:
        raise ValueError("Point-charge API keyword card exceeds QUICK's 256-character API buffer")
    lines = [card, f"{len(case['atoms'])} {len(charges)}"]
    atomic_numbers = {"H": 1, "O": 8}
    lines.extend(str(atomic_numbers[element]) + " " + " ".join(f"{value:.16f}" for value in point)
                 for element, point in case["atoms"])
    lines.extend(" ".join(f"{value:.16f}" for value in charge) for charge in charges)
    return "\n".join(lines) + "\n"


def read_result(path: Path, natoms: int, ncharges: int, gradient: bool) -> dict:
    result = {"qm_gradient": [], "charge_gradient": []}
    lines = [line.split() for line in path.read_text().splitlines()]
    if len(lines) != 2 + (natoms + ncharges if gradient else 0):
        raise ValueError("Unexpected API result row count")
    if len(lines[0]) != 2 or lines[0][0] != "energy" or lines[1] != ["scf_converged", "T"]:
        raise ValueError("Expected converged finite-energy API result")
    result["energy_hartree"] = reference.number(lines[0][1])
    result["scf_converged"] = True
    if gradient:
        for kind, count, rows in (("qm_gradient", natoms, lines[2:2 + natoms]),
                                  ("charge_gradient", ncharges, lines[2 + natoms:])):
            if len(rows) != count:
                raise ValueError("Missing gradient rows")
            for index, words in enumerate(rows, 1):
                if len(words) != 5 or words[:2] != [kind, str(index)]:
                    raise ValueError("Gradient label/index order differs")
                result[kind].extend(reference.number(value) for value in words[2:])
    return result


def runtime(driver: Path, env: dict[str, str]) -> dict:
    loaded = subprocess.run(["ldd", str(driver)], env=env, text=True, capture_output=True, check=True)
    if "not found" in loaded.stdout:
        raise ValueError("Unresolved API client runtime library")
    paths = [driver]
    for line in loaded.stdout.splitlines():
        match = re.search(r"(?:=>\s*)?(/[^\s]+)", line)
        if match:
            paths.append(Path(match.group(1)).resolve())
    libraries = [path for path in paths if re.fullmatch(r"libquick(?:_mpi)?(?:_cuda|_hip)?\.so(?:\.[\d.]+)?", path.name)]
    if len(libraries) != 1:
        raise ValueError("API client must resolve exactly one QUICK host library")
    return {"driver": str(driver), "quick_library": str(libraries[0]), "files": reference.identities(paths)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--launcher", default="")
    parser.add_argument("--basis", type=Path, default=ROOT / "basis")
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--case", choices=("water_hf_sto3g", "water_hf_631gss", "water_cation_uhf_sto3g", "water_pbe0_sto3g"),
                        default="water_hf_sto3g")
    parser.add_argument("--steps", type=float, nargs="+", default=[1.0e-3, 5.0e-4])
    parser.add_argument("--gradient-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--invariance-energy-tolerance", type=float, default=2.0e-8)
    parser.add_argument("--zero-energy-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--zero-gradient-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    if args.threads < 1 or any(not math.isfinite(value) or value <= 0 for value in [*args.steps, args.timeout]):
        parser.error("steps/timeout and threads must be positive and finite")
    if any(not math.isfinite(value) or value < 0 for value in [args.gradient_tolerance, args.invariance_energy_tolerance,
                                                             args.zero_energy_tolerance, args.zero_gradient_tolerance]):
        parser.error("tolerances must be finite and nonnegative")
    args.driver, args.basis = args.driver.resolve(strict=True), args.basis.resolve(strict=True)
    args.workdir = args.workdir.resolve()
    args.workdir.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update({"QUICK_BASIS": str(args.basis), "OMP_NUM_THREADS": str(args.threads),
                "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    report = {"passed": False, "started_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "API QM and external point-charge gradients in a static uniform field",
              "case": args.case, "steps_bohr": args.steps, "gradient_tolerance_hartree_per_bohr": args.gradient_tolerance,
              "limitations": ["QM nuclear and fixed external-charge derivatives; no polarizable embedding or solvent",
                              "External-external charge pair self-energy is outside QUICK's returned QM energy",
                              "SCF finite differences, no performance claim; GPU field-specific term initially host assembled"],
              "runs": [], "checks": []}
    case = copy.deepcopy(reference.CASES[args.case])
    charges = [[2.8, 0.7, -0.4, 0.35], [-2.5, -0.6, 1.3, -0.20]]
    field, origin = [2.0e-4, -1.5e-4, 1.0e-4], [0.3, 0.0, 0.0]
    total_charge = case["charge"] + sum(row[3] for row in charges)
    report.update({"molecular_charge": case["charge"], "external_charges": charges,
                   "total_charge": total_charge, "field_au": field, "origin_bohr": origin})
    cache = {}

    def check(name: str, error: float, tolerance: float, **details) -> None:
        report["checks"].append({"name": name, "passed": math.isfinite(error) and error <= tolerance,
                                  "max_abs_error": error, "tolerance": tolerance, **details})

    def run(label: str, molecule: dict, point_charges: list, applied: list | None,
            center: list, gradient: bool) -> dict:
        text = data_text(molecule, point_charges, applied, center, gradient)
        if text in cache:
            return cache[text]
        folder = args.workdir / "runs" / f"{len(report['runs']):04d}-{label}"
        folder.mkdir(parents=True, exist_ok=False)
        input_path = folder / "field.dat"
        input_path.write_text(text)
        command = shlex.split(args.launcher) + [str(args.driver), "gradient" if gradient else "energy", input_path.name]
        entry = {"label": label, "command": command, "workdir": str(folder), "input_sha256": reference.sha256(input_path)}
        report["runs"].append(entry)
        print(f"RUN {folder.name}", flush=True)
        start = time.monotonic()
        with (folder / "execution.log").open("w") as log:
            process = subprocess.Popen(command, cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       text=True, start_new_session=True)
            try:
                entry["returncode"] = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError(f"API client timeout: {folder}")
        entry["wall_seconds"] = time.monotonic() - start
        if entry["returncode"] != 0:
            raise RuntimeError(f"API client returned {entry['returncode']}: {folder}")
        result = read_result(folder / "result.dat", len(molecule["atoms"]), len(point_charges), gradient)
        entry["result_sha256"] = reference.sha256(folder / "result.dat")
        entry["result"] = result
        cache[text] = result
        return result

    paths = [Path(__file__).resolve(), HERE / "validate_external_field.py", HERE / "quick_external_field_api_driver.f90",
             ROOT / "src/modules/quick_oei_module.f90", ROOT / "src/modules/include/gradient.fh",
             ROOT / "src/modules/quick_method_module.f90", ROOT / "src/modules/quick_api_module.f90",
             *[path for path in args.basis.rglob("*") if path.is_file()]]
    try:
        report["source_inputs_before"] = reference.identities(paths)
        report["runtime_before"] = runtime(args.driver, env)
        analytic = run("analytic-field", case, charges, field, origin, True)
        for step in args.steps:
            gradients = {"qm_gradient": [], "charge_gradient": []}
            for kind, count in (("qm_gradient", len(case["atoms"])), ("charge_gradient", len(charges))):
                for atom in range(count):
                    for axis in range(3):
                        values = []
                        for sign, suffix in ((1, "plus"), (-1, "minus")):
                            molecule, point_charges = copy.deepcopy(case), copy.deepcopy(charges)
                            coordinates = molecule["atoms"][atom][1] if kind == "qm_gradient" else point_charges[atom]
                            coordinates[axis] += sign * step * reference.BOHRS_TO_A
                            label = f"h{step:g}-{kind}-{atom + 1}{'xyz'[axis]}-{suffix}"
                            values.append(run(label, molecule, point_charges, field, origin, False)["energy_hartree"])
                        gradients[kind].append((values[0] - values[1]) / (2 * step))
            for kind in gradients:
                check(f"{kind}:finite_difference:h={step:g}", reference.max_difference(analytic[kind], gradients[kind]),
                      args.gradient_tolerance, units="hartree/bohr", finite_difference=gradients[kind], step_bohr=step)
        summed = [sum(analytic["qm_gradient"][axis::3]) + sum(analytic["charge_gradient"][axis::3]) for axis in range(3)]
        check("all_centers_sum_gradient_minus_QF", reference.max_difference(summed, [-total_charge * x for x in field]),
              args.gradient_tolerance, units="hartree/bohr", summed_gradient=summed)
        offset = [0.27, 0.0, 0.0]
        shifted_origin = [a + b for a, b in zip(origin, offset)]
        shifted = run("origin-shift", case, charges, field, shifted_origin, True)
        expected = total_charge * sum(a * b for a, b in zip(field, offset))
        check("origin_energy_covariance", abs(shifted["energy_hartree"] - analytic["energy_hartree"] - expected),
              args.invariance_energy_tolerance, units="hartree", expected_energy_shift=expected)
        for kind in ("qm_gradient", "charge_gradient"):
            check(f"{kind}:origin_invariance", reference.max_difference(shifted[kind], analytic[kind]),
                  args.gradient_tolerance, units="hartree/bohr")
        translated, translated_charges = copy.deepcopy(case), copy.deepcopy(charges)
        translation = [-0.21, 0.17, 0.29]
        for _, point in translated["atoms"]:
            for axis in range(3):
                point[axis] += translation[axis] * reference.BOHRS_TO_A
        for point in translated_charges:
            for axis in range(3):
                point[axis] += translation[axis] * reference.BOHRS_TO_A
        moved = run("all-centers-translate", translated, translated_charges, field, origin, True)
        expected = -total_charge * sum(a * b for a, b in zip(field, translation))
        check("translation_energy_covariance", abs(moved["energy_hartree"] - analytic["energy_hartree"] - expected),
              args.invariance_energy_tolerance, units="hartree", expected_energy_shift=expected)
        for kind in ("qm_gradient", "charge_gradient"):
            check(f"{kind}:translation_invariance", reference.max_difference(moved[kind], analytic[kind]),
                  args.gradient_tolerance, units="hartree/bohr")
        legacy = run("no-field", case, charges, None, origin, True)
        zero = run("explicit-zero-field", case, charges, [0.0] * 3, origin, True)
        check("zero_field_energy", abs(legacy["energy_hartree"] - zero["energy_hartree"]),
              args.zero_energy_tolerance, units="hartree")
        for kind in ("qm_gradient", "charge_gradient"):
            check(f"{kind}:zero_field", reference.max_difference(legacy[kind], zero[kind]),
                  args.zero_gradient_tolerance, units="hartree/bohr")
        report["source_inputs_after"] = reference.identities(paths)
        report["runtime_after"] = runtime(args.driver, env)
        check("source_inputs_unchanged", float(report["source_inputs_before"] != report["source_inputs_after"]), 0.0)
        check("runtime_unchanged", float(report["runtime_before"] != report["runtime_after"]), 0.0)
        report["passed"] = all(row["passed"] for row in report["checks"])
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["completed_runs"] = len(report["runs"])
    report["passed_checks"] = sum(row["passed"] for row in report["checks"])
    report["failed_checks"] = sum(not row["passed"] for row in report["checks"])
    path = args.workdir / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": report["passed"], "runs": report["completed_runs"], "error": report.get("error"),
                      "passed_checks": report["passed_checks"], "failed_checks": report["failed_checks"], "report": str(path)}))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
