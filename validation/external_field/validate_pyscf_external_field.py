#!/usr/bin/env python3
"""Compare accepted QUICK field-force runs with independent PySCF SCF energies.

The orchestrator uses only the Python standard library. --python selects a
separate environment containing NumPy and PySCF. A worker recomputes each
unique mixed-field molecule once, differentiates its SCF energy at several
displacement sizes, and supplies an unrounded AO dipole. Several QUICK backend
reports can share those independently evaluated references.
"""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

BOHRS_TO_A = 0.52917720859
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value: str) -> float:
    result = float(value.replace("D", "E").replace("d", "e"))
    if not math.isfinite(result):
        raise ValueError("Nonfinite numeric result")
    return result


def keyword(card: str, name: str, default: str | None = None) -> str:
    matches = re.findall(r"(?:^|\s)" + re.escape(name) + r"\s*=\s*([^\s]+)(?=\s|$)", card, re.I)
    if len(matches) > 1:
        raise ValueError(f"Repeated {name} keyword")
    if matches:
        return matches[0]
    if default is None:
        raise ValueError(f"Missing {name} keyword")
    return default


def parse_model(path: Path) -> dict:
    blocks = re.split(r"\n\s*\n", path.read_text().strip())
    if len(blocks) != 2:
        raise ValueError("Expected exactly keyword and molecule blocks")
    card = " ".join(blocks[0].splitlines()).upper()
    if "EXTCHARGES" in card:
        raise ValueError("This independent oracle supports isolated molecules, without point charges")
    atoms = []
    for line in blocks[1].splitlines():
        words = line.split()
        if len(words) != 4 or not re.fullmatch(r"[A-Za-z]{1,2}", words[0]):
            raise ValueError("Malformed molecule row")
        atoms.append([words[0], [number(word) / BOHRS_TO_A for word in words[1:]]])
    method = "PBE0" if re.search(r"(?:^|\s)PBE0(?:\s|$)", card) else "HF"
    if method == "HF" and not re.search(r"(?:^|\s)U?HF(?:\s|$)", card):
        raise ValueError("Only HF/UHF and PBE0 are supported by this oracle")
    field, origin = [], []
    for axis in "XYZ":
        field.append(number(keyword(card, f"FINITE_FIELD_{axis}",
                                    keyword(card, f"EXTERNAL_EFIELD_{axis}", "0"))))
        origin.append(number(keyword(card, f"FINITE_FIELD_ORIGIN_{axis}",
                                     keyword(card, f"EXTERNAL_EFIELD_ORIGIN_{axis}", "0"))))
    return {"atoms_bohr": atoms, "method": method, "basis": keyword(card, "BASIS"),
            "charge": int(keyword(card, "CHARGE", "0")), "spin": int(keyword(card, "MULT", "1")) - 1,
            "unrestricted": bool(re.search(r"(?:^|\s)(?:UHF|UDFT)(?:\s|$)", card)),
            "field_au": field, "origin_bohr": origin}


def worker(input_path: Path, output_path: Path) -> int:
    # Imported only in the explicit external scientific environment.
    import numpy as np
    import pyscf
    from pyscf import dft, gto, lib, scf

    request = json.loads(input_path.read_text())
    lib.num_threads(request["threads"])
    output = {"passed": False, "pyscf_version": pyscf.__version__, "numpy_version": np.__version__,
              "python": sys.executable, "settings": request["settings"], "models": {}}
    try:
        for key, model in request["models"].items():
            atoms = model["atoms_bohr"]
            field = np.array(model["field_au"], dtype=float)
            origin = np.array(model["origin_bohr"], dtype=float)

            def evaluate(coordinates: list, density: bool = False) -> dict:
                molecule = gto.M(atom=coordinates, basis=model["basis"], unit="Bohr", cart=True,
                                 charge=model["charge"], spin=model["spin"], verbose=0)
                if model["method"] == "HF":
                    mean_field = scf.UHF(molecule) if model["unrestricted"] else scf.RHF(molecule)
                else:
                    mean_field = dft.UKS(molecule) if model["unrestricted"] else dft.RKS(molecule)
                    mean_field.xc = "PBE0"
                    mean_field.grids.level = request["settings"]["dft_grid_level"]
                mean_field.conv_tol = request["settings"]["energy_convergence_hartree"]
                mean_field.conv_tol_grad = request["settings"]["orbital_gradient_convergence"]
                mean_field.max_cycle = 250
                overlap = mean_field.get_ovlp()
                with molecule.with_common_orig(origin):
                    moments = molecule.intor("int1e_r", comp=3, hermi=1)
                core = mean_field.get_hcore() + np.einsum("x,xij->ij", field, moments)
                classical = -np.einsum("a,ax,x->", molecule.atom_charges(), molecule.atom_coords() - origin, field)
                nuclear = molecule.energy_nuc() + classical
                mean_field.get_hcore = lambda *args: core
                mean_field.energy_nuc = lambda *args: nuclear
                energy = float(mean_field.kernel())
                if not mean_field.converged or not math.isfinite(energy):
                    raise RuntimeError("Independent PySCF field SCF did not converge")
                result = {"energy_hartree": energy, "converged": True}
                if density:
                    matrix = mean_field.make_rdm1()
                    if matrix.ndim == 3:
                        matrix = matrix.sum(axis=0)
                    dipole = (np.einsum("a,ax->x", molecule.atom_charges(), molecule.atom_coords() - origin)
                              - np.einsum("ij,xji->x", matrix, moments))
                    result.update({"dipole_au_about_field_origin": dipole.tolist(),
                                   "density_cartesian_ao": matrix.tolist(), "basis_functions": molecule.nao_nr(),
                                   "electron_count": float(np.einsum("ij,ji->", matrix, overlap)),
                                   "spin_square": float(mean_field.spin_square()[0]) if model["unrestricted"] else 0.0})
                return result

            print(f"PYSCF {key} {model['method']}/{model['basis']}", flush=True)
            result = evaluate(atoms, density=True)
            result["finite_difference_gradients"] = []
            result["scf_calculations"] = 1
            output["models"][key] = result
            for step in request["steps_bohr"]:
                gradient = []
                for atom in range(len(atoms)):
                    for axis in range(3):
                        energies = []
                        for sign in (1, -1):
                            displaced = [[element, point.copy()] for element, point in atoms]
                            displaced[atom][1][axis] += sign * step
                            energies.append(evaluate(displaced)["energy_hartree"])
                            result["scf_calculations"] += 1
                        gradient.append((energies[0] - energies[1]) / (2 * step))
                result["finite_difference_gradients"].append({"step_bohr": step, "gradient_hartree_per_bohr": gradient})
            # Field derivatives check the independent oracle's sign/origin,
            # using its full-precision variational dipole as a separate gate.
            result["energy_field_derivatives"] = []
            saved_field = field.copy()
            for axis in range(3):
                energies = []
                for sign in (1, -1):
                    field[:] = saved_field
                    field[axis] += sign * request["field_step_au"]
                    energies.append(evaluate(atoms)["energy_hartree"])
                    result["scf_calculations"] += 1
                result["energy_field_derivatives"].append((energies[0] - energies[1]) / (2 * request["field_step_au"]))
            field[:] = saved_field
        output["passed"] = True
    except Exception as error:
        output["error"] = f"{type(error).__name__}: {error}"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return 0 if output["passed"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick-reports", type=Path, nargs="+", required=True)
    parser.add_argument("--python", type=Path, required=True, help="Python executable with PySCF and NumPy installed")
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", help="Case names to select from the QUICK reports")
    parser.add_argument("--steps", type=float, nargs="+", default=[1.0e-3, 5.0e-4])
    parser.add_argument("--field-step", type=float, default=1.0e-4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--dft-grid-level", type=int, default=5)
    parser.add_argument("--hf-energy-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--dft-energy-tolerance", type=float, default=5.0e-5)
    parser.add_argument("--hf-gradient-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--dft-gradient-tolerance", type=float, default=5.0e-5)
    parser.add_argument("--oracle-dipole-tolerance", type=float, default=2.0e-6)
    parser.add_argument("--timeout", type=float, default=7200.0)
    args = parser.parse_args()
    if args.threads < 1 or args.dft_grid_level < 0:
        parser.error("threads must be positive and DFT grid level nonnegative")
    if any(not math.isfinite(value) or value <= 0 for value in [*args.steps, args.field_step, args.timeout]):
        parser.error("steps and timeout must be positive and finite")
    if any(not math.isfinite(value) or value < 0 for value in [args.hf_energy_tolerance, args.dft_energy_tolerance,
            args.hf_gradient_tolerance, args.dft_gradient_tolerance, args.oracle_dipole_tolerance]):
        parser.error("tolerances must be finite and nonnegative")
    args.python = args.python.resolve(strict=True)
    args.quick_reports = [path.resolve(strict=True) for path in args.quick_reports]
    args.workdir = args.workdir.resolve()
    args.workdir.mkdir(parents=True, exist_ok=False)
    report = {"passed": False, "started_utc": datetime.now(timezone.utc).isoformat(), "checks": [],
              "scope": "Independent PySCF finite-field SCF energies, energy-difference gradients and variational dipole",
              "limitations": ["QUICK and PySCF DFT quadratures differ; DFT comparison is not a bitwise-density claim",
                              "Isolated molecules, static uniform field, Cartesian basis; no solvent/electrode model",
                              "QUICK CLI energies have printed decimal precision; oracle energies retain full precision"],
              "quick_reports": [str(path) for path in args.quick_reports], "comparisons": []}

    def check(name: str, error: float, tolerance: float, **details) -> None:
        report["checks"].append({"name": name, "passed": math.isfinite(error) and error <= tolerance,
                                  "max_abs_error": error, "tolerance": tolerance, **details})

    source_paths = [Path(__file__).resolve(), args.python, *args.quick_reports]
    try:
        models, targets = {}, []
        selected = set(args.cases or [])
        found_cases = set()
        for report_path in args.quick_reports:
            quick_report = json.loads(report_path.read_text())
            check(f"{report_path}:quick_campaign_accepted", float(not quick_report.get("passed", False)), 0.0)
            for entry in quick_report["runs"]:
                if entry["reference"] or not entry["analytic_gradient_requested"] or not entry["label"].endswith("-mixed_plus"):
                    continue
                case = entry["label"][:-len("-mixed_plus")]
                if selected and case not in selected:
                    continue
                found_cases.add(case)
                input_path = Path(entry["workdir"]) / "field.in"
                if sha256(input_path) != entry["input_sha256"]:
                    raise ValueError("QUICK input fingerprint no longer matches its campaign")
                key = entry["input_sha256"]
                model = parse_model(input_path)
                if key in models and models[key] != model:
                    raise ValueError("Equal input fingerprints gave different parsed models")
                models[key] = model
                source_paths.append(input_path)
                targets.append({"key": key, "case": case, "report": str(report_path), "result": entry["result"]})
        if not targets or selected - found_cases:
            raise ValueError(f"No matching complete QUICK cases; missing {sorted(selected - found_cases)}")
        source_paths = sorted(set(source_paths))
        report["source_inputs_before"] = {str(path): sha256(path) for path in source_paths}
        request = {"models": models, "steps_bohr": args.steps, "field_step_au": args.field_step,
                   "threads": args.threads, "settings": {"energy_convergence_hartree": 1.0e-12,
                        "orbital_gradient_convergence": 1.0e-9, "dft_grid_level": args.dft_grid_level,
                        "cartesian_basis": True, "geometry": "Exact QUICK input coordinates divided by QUICK BOHRS_TO_A"}}
        request_path, output_path = args.workdir / "oracle-input.json", args.workdir / "oracle-output.json"
        request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
        command = [str(args.python), "-B", str(Path(__file__).resolve()), "--worker", str(request_path), str(output_path)]
        report["command"] = command
        env = os.environ.copy()
        env.update({"OMP_NUM_THREADS": str(args.threads), "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        start = time.monotonic()
        with (args.workdir / "oracle-execution.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                       cwd=args.workdir, start_new_session=True, text=True)
            try:
                report["returncode"] = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError("Independent PySCF oracle timed out")
        report["wall_seconds"] = time.monotonic() - start
        if report["returncode"] != 0:
            raise RuntimeError("Independent PySCF worker failed; inspect oracle-execution.log and oracle-output.json")
        oracle = json.loads(output_path.read_text())
        if not oracle["passed"]:
            raise RuntimeError("Independent PySCF worker did not complete")
        report["oracle_versions"] = {name: oracle[name] for name in ("pyscf_version", "numpy_version", "python")}
        for key, result in oracle["models"].items():
            dipole_error = max(abs(a + b) for a, b in zip(result["energy_field_derivatives"], result["dipole_au_about_field_origin"]))
            check(f"{key}:oracle_dE_dF_minus_mu", dipole_error, args.oracle_dipole_tolerance, units="atomic_dipole")
            model = models[key]
            electrons = sum({"H": 1, "HE": 2, "O": 8}[atom[0].upper()] for atom in model["atoms_bohr"]) - model["charge"]
            check(f"{key}:oracle_electron_count", abs(result["electron_count"] - electrons), 1.0e-8)
        for target in targets:
            key, quick, model = target["key"], target["result"], models[target["key"]]
            reference = oracle["models"][key]
            dft = model["method"] == "PBE0"
            energy_tolerance = args.dft_energy_tolerance if dft else args.hf_energy_tolerance
            gradient_tolerance = args.dft_gradient_tolerance if dft else args.hf_gradient_tolerance
            label = f"{target['report']}:{target['case']}"
            check(f"{label}:independent_energy", abs(quick["energy_hartree"] - reference["energy_hartree"]),
                  energy_tolerance, units="hartree")
            comparison = {"case": target["case"], "quick_report": target["report"], "oracle_model": key,
                          "quick_energy_hartree": quick["energy_hartree"],
                          "oracle_energy_hartree": reference["energy_hartree"], "gradient_comparisons": []}
            for finite_difference in reference["finite_difference_gradients"]:
                errors = [abs(a - b) for a, b in zip(quick["gradient_hartree_per_bohr"],
                                                    finite_difference["gradient_hartree_per_bohr"])]
                if len(errors) != 3 * len(model["atoms_bohr"]):
                    raise ValueError("Gradient component count differs")
                check(f"{label}:independent_gradient:h={finite_difference['step_bohr']:g}", max(errors),
                      gradient_tolerance, units="hartree/bohr", step_bohr=finite_difference["step_bohr"])
                comparison["gradient_comparisons"].append({"step_bohr": finite_difference["step_bohr"],
                                                           "maximum_absolute_error": max(errors)})
            report["comparisons"].append(comparison)
        report["source_inputs_after"] = {str(path): sha256(path) for path in source_paths}
        check("source_inputs_unchanged", float(report["source_inputs_before"] != report["source_inputs_after"]), 0.0)
        report["oracle_input_sha256"], report["oracle_output_sha256"] = sha256(request_path), sha256(output_path)
        report["passed"] = bool(report["checks"]) and all(row["passed"] for row in report["checks"])
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["passed_checks"] = sum(row["passed"] for row in report["checks"])
    report["failed_checks"] = sum(not row["passed"] for row in report["checks"])
    path = args.workdir / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": report["passed"], "passed_checks": report["passed_checks"],
                      "failed_checks": report["failed_checks"], "error": report.get("error"), "report": str(path)}))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        raise SystemExit(worker(Path(sys.argv[2]), Path(sys.argv[3])))
    raise SystemExit(main())
