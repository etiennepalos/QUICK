#!/usr/bin/env python3
"""Validate uniform-field analytic forces against independent SCF energies.

Each invocation exercises one executable/launcher. Run the same suite with
CPU, CUDA, MPI and CUDA/MPI executables, using separate new work directories.
No fixture, baseline, tolerance or executable is modified. The optional old
executable checks legacy zero-field results. Output precision is included in
the documented dipole/finite-difference error budget.
"""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
from __future__ import annotations

import argparse
import copy
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
# QUICK's coordinate-input conversion, not its older Amber print conversion.
BOHRS_TO_A = 0.52917720859
DEBYE_PER_AU = 2.541765
MASSES = {"H": 1.0079, "He": 4.0026, "O": 15.99994}
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
WATER = [("O", [-0.33840, 0.00380, 0.23923]),
         ("H", [-0.33510, -0.00190, -0.83277]),
         ("H", [0.67350, -0.00190, 0.59353])]
CASES = {
    "water_hf_sto3g": {"method": "HF", "basis": "STO-3G", "charge": 0, "mult": 1, "atoms": WATER},
    "water_hf_631gss": {"method": "HF", "basis": "6-31G**", "charge": 0, "mult": 1, "atoms": WATER},
    "water_cation_uhf_sto3g": {"method": "UHF", "basis": "STO-3G", "charge": 1, "mult": 2, "atoms": WATER},
    "water_pbe0_sto3g": {"method": "DFT PBE0", "basis": "STO-3G", "charge": 0, "mult": 1, "atoms": WATER},
    "helium_hf_sto3g": {"method": "HF", "basis": "STO-3G", "charge": 0, "mult": 1,
                         "atoms": [("He", [0.31, -0.22, 0.19])]},
}


def number(value: str) -> float:
    result = float(value.replace("D", "E").replace("d", "e"))
    if not math.isfinite(result):
        raise ValueError("Nonfinite numeric result")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identities(paths: list[Path]) -> dict[str, str]:
    return {str(path.resolve()): sha256(path) for path in sorted(set(paths))}


def runtime(quick: Path, env: dict[str, str]) -> dict:
    loaded = subprocess.run(["ldd", str(quick)], env=env, capture_output=True, text=True, check=True)
    if "not found" in loaded.stdout:
        raise ValueError("Unresolved runtime library")
    libraries = []
    for line in loaded.stdout.splitlines():
        match = re.search(r"(?:=>\s*)?(/[^\s]+)", line)
        if match:
            libraries.append(Path(match.group(1)).resolve())
    quick_libraries = [path for path in libraries if re.fullmatch(
        r"libquick(?:_mpi)?(?:_cuda|_hip)?\.so(?:\.[\d.]+)?", path.name)]
    if len(quick_libraries) != 1 or quick_libraries[0].parent != quick.parent:
        raise ValueError("Expected one QUICK host library beside the supplied executable")
    cache = quick.parent.parent / "CMakeCache.txt"
    files = [quick, *libraries]
    source = None
    if cache.exists():
        files.append(cache)
        match = re.search(r"(?m)^CMAKE_HOME_DIRECTORY:INTERNAL=(.*)$", cache.read_text())
        if match:
            source = match.group(1)
    return {"executable": str(quick), "quick_library": str(quick_libraries[0]),
            "cmake_source_directory": source, "files": identities(files)}


def input_text(case: dict, field: list[float] | None, origin: list[float], gradient: bool) -> str:
    mode = "GRADIENT" if gradient else "ENERGY"
    keywords = (f"{case['method']} BASIS={case['basis']} CHARGE={case['charge']} MULT={case['mult']} "
                f"{mode} DIPOLE SCF=250 DENSERMS=1e-10 CUTOFF=1e-12 "
                "GRADCUTOFF=1e-12 XCCUTOFF=1e-12 BASISCUTOFF=1e-12")
    if field is not None:
        for axis, value, center in zip("XYZ", field, origin):
            keywords += f" FINITE_FIELD_{axis}={value:.12g} FINITE_FIELD_ORIGIN_{axis}={center:.12g}"
    if len(keywords) > 299:
        raise ValueError("Generated keyword card exceeds QUICK's 300-character buffer")
    geometry = "\n".join(f"{element:2s} " + " ".join(f"{x:.16f}" for x in point)
                         for element, point in case["atoms"])
    return keywords + "\n\n" + geometry + "\n\n"


def read_result(path: Path, natoms: int, gradient: bool) -> dict:
    text = path.read_text(errors="replace")
    cycles = re.findall(r"REACH CONVERGENCE AFTER\s+(\d+)\s+CYCLES", text)
    if len(cycles) != 1 or re.search(r"NO CONVERGENCE|RAN OUT OF CYCLES|FAILED TO CONVERGE", text, re.I):
        raise ValueError(f"Expected exactly one explicitly converged SCF: {path}")
    if re.search(r"(?i)(?<![a-z])(?:nan|[-+]?inf(?:inity)?)(?![a-z])", text):
        raise ValueError(f"Nonfinite token in output: {path}")
    energies = re.findall(r"(?m)^\s*TOTAL ENERGY\s*=\s*(" + NUMBER + r")\s*$", text)
    if len(energies) != 1:
        raise ValueError(f"Expected exactly one total energy: {path}")
    result = {"energy_hartree": number(energies[0]), "scf_cycles": int(cycles[0]), "scf_converged": True}
    # Preserve decimal precision rather than assume every QUICK build prints
    # the same number of places. This bounds energy-derived numeric noise.
    mantissa, _, exponent = energies[0].upper().replace("D", "E").partition("E")
    decimals = len(mantissa.partition(".")[2])
    result["energy_print_half_ulp"] = 0.5 * 10.0 ** ((int(exponent) if exponent else 0) - decimals)
    if gradient:
        blocks = text.split("ANALYTICAL GRADIENT:")
        if len(blocks) != 2:
            raise ValueError(f"Expected one analytic-gradient block: {path}")
        rows = re.findall(r"(?m)^\s*(\d+)([XYZ])\s+(" + NUMBER + r")\s+(" + NUMBER + r")\s*$",
                          blocks[1].split("POINT CHARGE GRADIENT:")[0])
        if len(rows) != 3 * natoms:
            raise ValueError(f"Expected {3 * natoms} gradient rows, found {len(rows)}: {path}")
        expected = [(str(atom + 1), axis) for atom in range(natoms) for axis in "XYZ"]
        if [(atom, axis) for atom, axis, _, _ in rows] != expected:
            raise ValueError(f"Gradient atom/component order differs: {path}")
        result["gradient_hartree_per_bohr"] = [number(value) for _, _, _, value in rows]
    blocks = text.split("DIPOLE (DEBYE)")
    if len(blocks) != 2:
        raise ValueError(f"Expected one dipole block: {path}")
    vectors = re.findall(r"(?m)^\s*(" + NUMBER + r")\s+(" + NUMBER + r")\s+(" + NUMBER +
                         r")\s+(" + NUMBER + r")\s*$", blocks[1])
    if not vectors:
        raise ValueError(f"Missing printed dipole: {path}")
    result["dipole_debye_about_center_of_mass"] = [number(value) for value in vectors[0][:3]]
    return result


def center_of_mass_bohr(case: dict) -> list[float]:
    mass = sum(MASSES[element] for element, _ in case["atoms"])
    return [sum(MASSES[element] * point[axis] for element, point in case["atoms"]) / mass / BOHRS_TO_A
            for axis in range(3)]


def max_difference(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Compared result lengths differ")
    return max((abs(a - b) for a, b in zip(left, right)), default=0.0)


class Campaign:
    def __init__(self, args: argparse.Namespace, report: dict, env: dict[str, str]):
        self.args, self.report, self.env = args, report, env
        self.cache: dict[tuple, dict] = {}

    def check(self, name: str, error: float, tolerance: float, **details) -> None:
        self.report["checks"].append({"name": name, "passed": math.isfinite(error) and error <= tolerance,
                                      "max_abs_error": error, "tolerance": tolerance, **details})

    def run(self, label: str, case: dict, field: list[float] | None, origin: list[float],
            gradient: bool, reference: bool = False) -> dict:
        text = input_text(case, field, origin, gradient)
        key = (reference, text)
        if key in self.cache:
            return self.cache[key]
        binary = self.args.reference_quick if reference else self.args.quick
        launcher = self.args.reference_launcher if reference else self.args.launcher
        if binary is None:
            raise ValueError("Reference executable was not supplied")
        folder = self.args.workdir / "runs" / f"{len(self.report['runs']):04d}-{label}"
        folder.mkdir(parents=True, exist_ok=False)
        input_path = folder / "field.in"
        input_path.write_text(text)
        # QUICK's legacy filename buffer is shorter than a typical evidence
        # directory path; the process already runs in this job directory.
        command = shlex.split(launcher) + [str(binary), input_path.name]
        print(f"RUN {folder.name}", flush=True)
        start = time.monotonic()
        entry = {"label": label, "reference": reference, "command": command, "workdir": str(folder),
                 "input_sha256": sha256(input_path), "field_au": field, "origin_bohr": origin,
                 "analytic_gradient_requested": gradient, "returncode": None}
        self.report["runs"].append(entry)
        with (folder / "execution.log").open("w") as log:
            process = subprocess.Popen(command, cwd=folder, env=self.env, stdout=log, stderr=subprocess.STDOUT,
                                       text=True, start_new_session=True)
            try:
                entry["returncode"] = process.wait(timeout=self.args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError(f"QUICK timeout in {folder}")
        entry["wall_seconds"] = time.monotonic() - start
        if entry["returncode"] != 0:
            raise RuntimeError(f"QUICK returned {entry['returncode']} in {folder}")
        output = folder / "field.out"
        result = read_result(output, len(case["atoms"]), gradient)
        entry["output_sha256"] = sha256(output)
        entry["result"] = result
        self.cache[key] = result
        return result

    def finite_difference(self, name: str, case: dict, field: list[float], origin: list[float],
                          analytic: dict, steps: list[float], tolerance: float) -> None:
        for step in steps:
            numeric = []
            rounding = 0.0
            for atom in range(len(case["atoms"])):
                for axis in range(3):
                    results = []
                    for sign, suffix in ((1, "plus"), (-1, "minus")):
                        displaced = copy.deepcopy(case)
                        displaced["atoms"][atom][1][axis] += sign * step * BOHRS_TO_A
                        results.append(self.run(f"{name}-h{step:g}-a{atom + 1}{'xyz'[axis]}-{suffix}",
                                                displaced, field, origin, False))
                    numeric.append((results[0]["energy_hartree"] - results[1]["energy_hartree"]) / (2 * step))
                    rounding = max(rounding, sum(row["energy_print_half_ulp"] for row in results) / (2 * step))
            self.check(f"{name}:gradient_fd:h={step:g}",
                       max_difference(analytic["gradient_hartree_per_bohr"], numeric), tolerance,
                       units="hartree/bohr", step_bohr=step, finite_difference=numeric,
                       printed_energy_rounding_bound=rounding)

    def case(self, name: str, case: dict) -> None:
        origin = [0.30, -0.20, 0.40]
        strength = self.args.field_strength
        fields = {"mixed_plus": [strength, -0.75 * strength, 0.5 * strength]}
        if self.args.suite == "full":
            fields["mixed_minus"] = [-strength, 0.75 * strength, -0.5 * strength]
            for axis in range(3):
                for sign in (1, -1):
                    fields[f"{'xyz'[axis]}_{'plus' if sign > 0 else 'minus'}"] = [
                        sign * strength if component == axis else 0.0 for component in range(3)]
        is_dft = "DFT" in case["method"]
        tolerance = self.args.dft_gradient_tolerance if is_dft else self.args.hf_gradient_tolerance
        steps = self.args.steps or ([1.0e-3, 5.0e-4] if self.args.suite == "smoke" else [2.0e-3, 1.0e-3, 5.0e-4])
        for label, field in fields.items():
            full_name = f"{name}-{label}"
            result = self.run(full_name, case, field, origin, True)
            # Full vectors receive a convergence scan in displacement size;
            # individual field axes/signs receive the finest displacement.
            self.finite_difference(full_name, case, field, origin, result,
                                   steps if label.startswith("mixed") else [min(steps)], tolerance)
            summed = [sum(result["gradient_hartree_per_bohr"][axis::3]) for axis in range(3)]
            self.check(f"{full_name}:sum_gradient_minus_QF", max_difference(
                summed, [-case["charge"] * value for value in field]), tolerance,
                units="hartree/bohr", summed_gradient=summed)

        field = fields["mixed_plus"]
        result = self.run(f"{name}-mixed_plus", case, field, origin, True)
        # Check origin covariance of charged energy and invariance of force.
        offset = [0.27, -0.19, 0.11]
        shifted_origin = [a + b for a, b in zip(origin, offset)]
        origin_result = self.run(f"{name}-origin-shift", case, field, shifted_origin, True)
        predicted = case["charge"] * sum(a * b for a, b in zip(field, offset))
        self.check(f"{name}:origin_energy_covariance", abs(
            origin_result["energy_hartree"] - result["energy_hartree"] - predicted),
            self.args.invariance_energy_tolerance, units="hartree", predicted_energy_shift=predicted)
        self.check(f"{name}:origin_gradient_invariance", max_difference(
            origin_result["gradient_hartree_per_bohr"], result["gradient_hartree_per_bohr"]), tolerance,
            units="hartree/bohr")

        translation_bohr = [-0.21, 0.17, 0.29]
        translated = copy.deepcopy(case)
        for _, point in translated["atoms"]:
            for axis in range(3):
                point[axis] += translation_bohr[axis] * BOHRS_TO_A
        translated_result = self.run(f"{name}-translate", translated, field, origin, True)
        predicted = -case["charge"] * sum(a * b for a, b in zip(field, translation_bohr))
        self.check(f"{name}:translation_energy_covariance", abs(
            translated_result["energy_hartree"] - result["energy_hartree"] - predicted),
            self.args.invariance_energy_tolerance, units="hartree", predicted_energy_shift=predicted)
        self.check(f"{name}:translation_gradient_invariance", max_difference(
            translated_result["gradient_hartree_per_bohr"], result["gradient_hartree_per_bohr"]), tolerance,
            units="hartree/bohr")

        # QUICK prints its dipole about the mass center. Convert to the
        # imposed-field origin before testing dE/dF = -mu for charged cases.
        com = center_of_mass_bohr(case)
        dipole = [value / DEBYE_PER_AU + case["charge"] * (com[axis] - origin[axis])
                  for axis, value in enumerate(result["dipole_debye_about_center_of_mass"])]
        for axis in range(3):
            plus, minus = field.copy(), field.copy()
            plus[axis] += self.args.field_step
            minus[axis] -= self.args.field_step
            forward = self.run(f"{name}-field-{'xyz'[axis]}-plus", case, plus, origin, False)
            backward = self.run(f"{name}-field-{'xyz'[axis]}-minus", case, minus, origin, False)
            derivative = (forward["energy_hartree"] - backward["energy_hartree"]) / (2 * self.args.field_step)
            rounding = (forward["energy_print_half_ulp"] + backward["energy_print_half_ulp"]) / (2 * self.args.field_step)
            self.check(f"{name}:dE_dF_minus_mu:{'xyz'[axis]}", abs(derivative + dipole[axis]),
                       self.args.dipole_tolerance + rounding, units="atomic_dipole",
                       field_step_au=self.args.field_step, derivative=derivative,
                       dipole_about_field_origin=dipole[axis], energy_rounding_bound=rounding,
                       printed_dipole_rounding_bound=0.00005 / DEBYE_PER_AU)

        legacy = self.run(f"{name}-legacy-zero", case, None, origin, True)
        explicit = self.run(f"{name}-explicit-zero", case, [0.0] * 3, origin, True)
        self.check(f"{name}:zero_field_energy", abs(
            legacy["energy_hartree"] - explicit["energy_hartree"]),
            self.args.zero_field_energy_tolerance, units="hartree")
        self.check(f"{name}:zero_field_gradient", max_difference(
            legacy["gradient_hartree_per_bohr"], explicit["gradient_hartree_per_bohr"]),
            self.args.zero_field_gradient_tolerance,
            units="hartree/bohr")
        if self.args.reference_quick is not None:
            reference = self.run(f"{name}-legacy-reference", case, None, origin, True, reference=True)
            self.check(f"{name}:legacy_reference_energy", abs(
                legacy["energy_hartree"] - reference["energy_hartree"]),
                self.args.legacy_energy_tolerance, units="hartree")
            self.check(f"{name}:legacy_reference_gradient", max_difference(
                legacy["gradient_hartree_per_bohr"], reference["gradient_hartree_per_bohr"]),
                self.args.legacy_gradient_tolerance, units="hartree/bohr")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", type=Path, required=True)
    parser.add_argument("--launcher", default="", help="For example 'mpirun -np 2'")
    parser.add_argument("--basis", type=Path, default=ROOT / "basis")
    parser.add_argument("--workdir", type=Path, required=True, help="New evidence directory")
    parser.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES))
    parser.add_argument("--steps", nargs="+", type=float, help="Central nuclear-displacement sizes in bohr")
    parser.add_argument("--field-strength", type=float, default=2.0e-4)
    parser.add_argument("--field-step", type=float, default=1.0e-4)
    parser.add_argument("--hf-gradient-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--dft-gradient-tolerance", type=float, default=5.0e-5)
    parser.add_argument("--invariance-energy-tolerance", type=float, default=2.0e-8)
    parser.add_argument("--dipole-tolerance", type=float, default=3.0e-5)
    parser.add_argument("--zero-field-energy-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--zero-field-gradient-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--legacy-energy-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--legacy-gradient-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--reference-quick", type=Path)
    parser.add_argument("--reference-launcher", default="")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    positive = [args.timeout, args.field_strength, args.field_step, *(args.steps or [])]
    tolerances = [args.hf_gradient_tolerance, args.dft_gradient_tolerance, args.invariance_energy_tolerance,
                  args.dipole_tolerance, args.zero_field_energy_tolerance, args.zero_field_gradient_tolerance,
                  args.legacy_energy_tolerance, args.legacy_gradient_tolerance]
    if args.threads < 1 or any(not math.isfinite(x) or x <= 0 for x in positive):
        parser.error("timeouts, steps, field strength and threads must be positive and finite")
    if any(not math.isfinite(x) or x < 0 for x in tolerances):
        parser.error("tolerances must be finite and nonnegative")
    args.quick = args.quick.resolve(strict=True)
    args.basis = args.basis.resolve(strict=True)
    if args.reference_quick is not None:
        args.reference_quick = args.reference_quick.resolve(strict=True)
    args.workdir = args.workdir.resolve()
    args.workdir.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update({"QUICK_BASIS": str(args.basis), "OMP_NUM_THREADS": str(args.threads),
                "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    cases = args.cases or (["water_hf_sto3g"] if args.suite == "smoke" else list(CASES))
    source_paths = [Path(__file__).resolve(), ROOT / "CMakeLists.txt", ROOT / "QUICKConfig.cmake.in",
                    *[path for directory in (ROOT / "src", ROOT / "quick-cmake", ROOT / "tools", args.basis)
                      for path in directory.rglob("*") if path.is_file()]]
    report = {"passed": False, "started_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Uniform static field analytic nuclear gradients and electrostatic energy identities",
              "suite": args.suite, "cases": cases, "arguments": {key: str(value) if isinstance(value, Path) else value
                                                                    for key, value in vars(args).items()},
              "environment": {key: env.get(key) for key in ("QUICK_BASIS", "LD_LIBRARY_PATH", "OMP_NUM_THREADS",
                                                          "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES")},
              "conventions": {"coordinates": "Angstrom input; FD displacements converted using QUICK BOHRS_TO_A",
                              "gradient": "dE/dR in hartree/bohr; forces have the opposite sign",
                              "field": "atomic units; h_field=+F dot (r-origin), E_classical=-sum(q F dot (R-origin))",
                              "dipole": "Legacy printed COM dipole converted to field origin for charged molecules"},
              "limitations": ["Finite clusters and static uniform fields; no constant-potential electrode or continuum solvent",
                              "CLI energies/gradients have printed decimal precision; no new performance claim",
                              "GPU execution is established separately by the selected binary and runtime/profile evidence",
                              "SCF energy finite differences are independent of the analytic moment derivative implementation"],
              "runs": [], "checks": []}
    try:
        report["source_inputs_before"] = identities(source_paths)
        report["runtime_before"] = runtime(args.quick, env)
        if args.reference_quick is not None:
            report["reference_runtime_before"] = runtime(args.reference_quick, env)
        campaign = Campaign(args, report, env)
        for name in cases:
            campaign.case(name, copy.deepcopy(CASES[name]))
        report["source_inputs_after"] = identities(source_paths)
        report["runtime_after"] = runtime(args.quick, env)
        campaign.check("source_inputs_unchanged", float(report["source_inputs_before"] != report["source_inputs_after"]), 0.0)
        campaign.check("runtime_unchanged", float(report["runtime_before"] != report["runtime_after"]), 0.0)
        if args.reference_quick is not None:
            report["reference_runtime_after"] = runtime(args.reference_quick, env)
            campaign.check("reference_runtime_unchanged", float(
                report["reference_runtime_before"] != report["reference_runtime_after"]), 0.0)
        report["passed"] = bool(report["checks"]) and all(row["passed"] for row in report["checks"])
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["completed_runs"] = len(report["runs"])
    report["passed_checks"] = sum(row["passed"] for row in report["checks"])
    report["failed_checks"] = sum(not row["passed"] for row in report["checks"])
    path = args.workdir / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": report["passed"], "runs": report["completed_runs"],
                      "passed_checks": report["passed_checks"], "failed_checks": report["failed_checks"],
                      "error": report.get("error"), "report": str(path)}), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
