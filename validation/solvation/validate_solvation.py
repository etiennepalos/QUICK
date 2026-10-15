#!/usr/bin/env python3
"""Validate continuous-density QUICK/ddX energies, forces and AO derivatives.

Use one separately compiled quick_solvation_api_driver per backend. The smoke
suite uses explicitly coarse engineering grids; it is not final quadrature
accuracy evidence. The full suite uses default grids and tests denser grids.
All SCF jobs run sequentially in separate new evidence directories.
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
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BOHRS_TO_A = 0.52917720859
WATER = [[8, [-0.33840, 0.00380, 0.23923]], [1, [-0.33510, -0.00190, -0.83277]],
         [1, [0.67350, -0.00190, 0.59353]]]
CASES = {
    "helium": {"method": "HF", "basis": "STO-3G", "charge": 0, "mult": 1, "atoms": [[2, [0.31, -0.22, 0.19]]]},
    "water": {"method": "HF", "basis": "STO-3G", "charge": 0, "mult": 1, "atoms": WATER},
    "water_spd": {"method": "HF", "basis": "6-31G**", "charge": 0, "mult": 1, "atoms": WATER},
    "water_cation": {"method": "UHF", "basis": "STO-3G", "charge": 1, "mult": 2, "atoms": WATER},
    "water_pbe0": {"method": "DFT PBE0", "basis": "STO-3G", "charge": 0, "mult": 1, "atoms": WATER},
    "water_pbe0_spd": {"method": "DFT PBE0 D3", "basis": "DEF2-SVP", "charge": 0, "mult": 1, "atoms": WATER},
}


def number(value: str) -> float:
    result = float(value.replace("D", "E").replace("d", "e"))
    if not math.isfinite(result):
        raise ValueError("Nonfinite result")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identities(paths: list[Path]) -> dict[str, str]:
    return {str(path.resolve()): sha256(path) for path in sorted(set(paths))}


def difference(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Compared array lengths differ")
    return max((abs(a - b) for a, b in zip(left, right)), default=0.0)


def runtime(driver: Path, env: dict[str, str]) -> dict:
    loaded = subprocess.run(["ldd", str(driver)], env=env, capture_output=True, text=True, check=True)
    if "not found" in loaded.stdout:
        raise ValueError("Unresolved runtime library")
    paths = [driver]
    for line in loaded.stdout.splitlines():
        match = re.search(r"(?:=>\s*)?(/[^\s]+)", line)
        if match:
            paths.append(Path(match.group(1)).resolve())
    quick_libraries = [path for path in paths if re.fullmatch(
        r"libquick(?:_mpi)?(?:_cuda|_hip)?\.so(?:\.[\d.]+)?", path.name)]
    if len(quick_libraries) != 1:
        raise ValueError("Validation client must resolve exactly one QUICK host library")
    return {"driver": str(driver), "quick_library": str(quick_libraries[0]), "files": identities(paths)}


def data_text(case: dict, model: str | None, mode: str, grid: dict,
              epsilon: float, applied_field: bool) -> str:
    card = (f"{case['method']} BASIS={case['basis']} "
            f"{'GRADIENT' if mode == 'gradient' else 'ENERGY'} SCF=250 DENSERMS=1e-10 "
            "CUTOFF=1e-12 GRADCUTOFF=1e-12")
    if case["charge"] != 0:
        card += f" CHARGE={case['charge']}"
    if case["mult"] != 1:
        card += f" MULT={case['mult']}"
    if "DFT" in case["method"]:
        card += " XCCUTOFF=1e-12 BASISCUTOFF=1e-12"
    if model is not None:
        card += (f" DDX={model} DDX_EPS={epsilon:.12g} DDX_LMAX={grid['lmax']} "
                 f"DDX_NLEBEDEV={grid['nlebedev']} DDX_NRAD={grid['nrad']} DDX_NANG={grid['nang']} DDX_TOL=1e-10")
    if applied_field:
        card += " FINITE_FIELD_X=0.0002"
        if model is not None:
            card += " DDX_FIELD=SOLUTE"
    if len(card) > 256:
        raise ValueError(f"Keyword card length {len(card)} exceeds QUICK's 256-character API buffer")
    lines = [card, str(len(case["atoms"]))]
    lines.extend(str(z) + " " + " ".join(f"{x:.16f}" for x in point) for z, point in case["atoms"])
    return "\n".join(lines) + "\n"


def read_result(path: Path, natoms: int, mode: str) -> dict:
    result = {"gradient_hartree_per_bohr": [], "density_derivatives": []}
    integers = {"basis_functions", "nominal_electrons", "solvation_updates", "api_steps", "final_solvation_updates"}
    arrays = {"gpu_volume_counters", "gpu_coulomb_cache", "frozen_api_steps"}
    for line in path.read_text().splitlines():
        words = line.split()
        if not words:
            continue
        key = words[0]
        if key == "gradient":
            expected = len(result["gradient_hartree_per_bohr"]) // 3 + 1
            if len(words) != 5 or int(words[1]) != expected:
                raise ValueError("Unexpected gradient atom/component order")
            result["gradient_hartree_per_bohr"].extend(number(word) for word in words[2:])
        elif key == "density_derivative":
            if len(words) != 6:
                raise ValueError("Malformed density derivative row")
            result["density_derivatives"].append({"spin": int(words[1]), "direction": int(words[2]),
                "step": number(words[3]), "finite_difference": number(words[4]), "trace_V_deltaP": number(words[5])})
        elif key == "scf_converged":
            if words != ["scf_converged", "T"]:
                raise ValueError("SCF convergence flag is false or malformed")
            result[key] = True
        elif key in arrays:
            result[key] = [int(word) for word in words[1:]]
        elif len(words) == 2:
            if key in result:
                raise ValueError("Repeated result key")
            result[key] = int(words[1]) if key in integers else number(words[1])
        else:
            raise ValueError(f"Unexpected result row: {line}")
    required = {"energy", "scf_converged", "basis_functions", "nominal_electrons", "volume_electrons",
                "solvation_energy", "solvation_updates", "gpu_volume_counters", "gpu_coulomb_cache", "api_steps"}
    if (required - result.keys() or len(result["gpu_volume_counters"]) != 3
            or len(result["gpu_coulomb_cache"]) != 7):
        raise ValueError("Incomplete continuum result")
    if len(result["gradient_hartree_per_bohr"]) != (3 * natoms if mode == "gradient" else 0):
        raise ValueError("Wrong gradient component count")
    if mode == "frozen":
        required = {"frozen_base_energy", "potential_symmetry_error", "density_restored_energy_error",
                    "density_restored_potential_error", "fock_unchanged_error", "energy_state_unchanged_error",
                    "frozen_api_steps", "final_solvation_updates"}
        if required - result.keys() or not result["density_derivatives"]:
            raise ValueError("Incomplete frozen-density result")
    return result


class Campaign:
    def __init__(self, args: argparse.Namespace, report: dict, env: dict[str, str]):
        self.args, self.report, self.env = args, report, env
        self.cache = {}
        self.gpu = "_cuda" in report["runtime_before"]["quick_library"] or "_hip" in report["runtime_before"]["quick_library"]

    def check(self, name: str, error: float, tolerance: float, **details) -> None:
        self.report["checks"].append({"name": name, "passed": math.isfinite(error) and error <= tolerance,
                                      "max_abs_error": error, "tolerance": tolerance, **details})

    def run(self, name: str, case: dict, model: str | None, mode: str, grid: dict,
            epsilon: float = 78.3553, field: bool = False) -> dict:
        text = data_text(case, model, mode, grid, epsilon, field)
        cache_key = mode, text
        if cache_key in self.cache:
            return self.cache[cache_key]
        folder = self.args.workdir / "runs" / f"{len(self.report['runs']):04d}-{name}"
        folder.mkdir(parents=True, exist_ok=False)
        input_path = folder / "continuum.dat"
        input_path.write_text(text)
        command = shlex.split(self.args.launcher) + [str(self.args.driver), mode, input_path.name]
        entry = {"label": name, "mode": mode, "model": model, "epsilon": epsilon, "grid": grid.copy(),
                 "field_solute": field, "command": command, "workdir": str(folder), "input_sha256": sha256(input_path)}
        self.report["runs"].append(entry)
        print(f"RUN {folder.name}", flush=True)
        start = time.monotonic()
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
                raise RuntimeError(f"Continuum client timeout: {folder}")
        entry["wall_seconds"] = time.monotonic() - start
        if entry["returncode"] != 0:
            raise RuntimeError(f"Continuum client returned {entry['returncode']}: {folder}")
        result = read_result(folder / "result.dat", len(case["atoms"]), mode)
        entry["result_sha256"] = sha256(folder / "result.dat")
        entry["result"] = result
        output = folder / "continuum_api.out"
        output_text = output.read_text(errors="replace")
        cycles = re.findall(r"REACH CONVERGENCE AFTER\s+(\d+)\s+CYCLES", output_text)
        if len(cycles) != 1 or re.search(r"NO CONVERGENCE|RAN OUT OF CYCLES|FAILED TO CONVERGE", output_text, re.I):
            raise ValueError("Expected one explicitly converged SCF and no extra SCFs")
        entry["quick_output_sha256"] = sha256(output)
        result["scf_cycles"] = int(cycles[0])
        self.check(f"{name}:one_api_scf", abs(result["api_steps"] - 2), 0.0,
                   api_step_convention="Starts at1; one API calculation returns2", explicit_scf_sections=len(cycles))
        expected_electrons = sum(z for z, _ in case["atoms"]) - case["charge"]
        self.check(f"{name}:nominal_electrons", abs(result["nominal_electrons"] - expected_electrons), 0.0)
        active = model is not None and epsilon > 1.0
        if active:
            self.check(f"{name}:volume_electrons", abs(result["volume_electrons"] - expected_electrons),
                       self.args.electron_count_tolerance, units="electrons")
            self.check(f"{name}:solvent_nonpositive_energy", max(0.0, result["solvation_energy"]), 1.0e-8, units="hartree")
            self.check(f"{name}:solvent_updated", float(result["solvation_updates"] < 1), 0.0)
            counters = result["gpu_volume_counters"]
            if self.gpu:
                self.check(f"{name}:gpu_volume_executed", float(counters[0] < 1 or counters[1] < 1 or
                           (mode == "gradient" and counters[2] < 1)), 0.0, counters=counters)
                hits, builds, tbytes, gemvs, direct, fallbacks, allocated = result["gpu_coulomb_cache"]
                self.check(f"{name}:gpu_coulomb_cache_reused", max(abs(builds-1),abs(gemvs-result["solvation_updates"]),
                    abs(hits-(gemvs-builds)),abs(direct),abs(fallbacks),float(tbytes <= 0 or allocated < tbytes)), 0.0,
                    counters=result["gpu_coulomb_cache"],
                    columns=["hits","builds","T_bytes","gemvs","direct_calls","fallbacks","allocated_bytes"])
            else:
                self.check(f"{name}:cpu_gpu_counters_zero", max(abs(x) for x in counters), 0.0)
                self.check(f"{name}:cpu_gpu_cache_zero", max(abs(x) for x in result["gpu_coulomb_cache"]), 0.0)
        else:
            self.check(f"{name}:inactive_solvent_energy", abs(result["solvation_energy"]), 0.0)
            self.check(f"{name}:inactive_solvent_counters", max([abs(result["solvation_updates"]),
                       *[abs(x) for x in result["gpu_volume_counters"]]]), 0.0)
        self.cache[cache_key] = result
        return result

    def frozen(self, name: str, case: dict, model: str, grid: dict) -> None:
        result = self.run(name + "-frozen", case, model, "frozen", grid)
        rows = result["density_derivatives"]
        expected_spins = 2 if case["method"].startswith("U") else 1
        tuples = [(row["spin"], row["direction"], row["step"]) for row in rows]
        expected = [(spin, direction, step) for spin in range(1, expected_spins + 1)
                    for direction in (1, 2) for step in (1.0e-3, 3.0e-4, 1.0e-4)]
        self.check(f"{name}:frozen_direction_spin_step_coverage", float(tuples != expected), 0.0)
        for row in rows:
            self.check(f"{name}:density_variational:spin{row['spin']}:dir{row['direction']}:h{row['step']:g}",
                       abs(row["finite_difference"] - row["trace_V_deltaP"]), self.args.density_tolerance,
                       units="hartree/per_normalized_density_perturbation", **row)
        for key, tolerance in (("potential_symmetry_error", 1.0e-12), ("density_restored_energy_error", 1.0e-10),
                               ("density_restored_potential_error", 1.0e-9), ("fock_unchanged_error", 0.0),
                               ("energy_state_unchanged_error", 0.0)):
            self.check(f"{name}:{key}", result[key], tolerance)
        self.check(f"{name}:frozen_no_extra_api_scf", abs(result["frozen_api_steps"][1] - result["frozen_api_steps"][0]), 0.0)
        self.check(f"{name}:frozen_expected_updates", abs(
            result["final_solvation_updates"] - result["solvation_updates"] - (2 + 2 * len(rows))), 0.0)

    def nuclear_fd(self, name: str, case: dict, model: str, grid: dict, base: dict,
                   tolerance: float, field: bool = False) -> None:
        for step in self.args.steps:
            numeric = []
            for atom in range(len(case["atoms"])):
                for axis in range(3):
                    values = []
                    for sign, suffix in ((1, "plus"), (-1, "minus")):
                        displaced = copy.deepcopy(case)
                        displaced["atoms"][atom][1][axis] += sign * step * BOHRS_TO_A
                        label = f"{name}-h{step:g}-a{atom + 1}{'xyz'[axis]}-{suffix}"
                        values.append(self.run(label, displaced, model, "energy", grid, field=field)["energy"])
                    numeric.append((values[0] - values[1]) / (2 * step))
            self.check(f"{name}:nuclear_gradient_fd:h={step:g}", difference(base["gradient_hartree_per_bohr"], numeric),
                       tolerance, units="hartree/bohr", finite_difference=numeric, step_bohr=step)

    def case(self, case_name: str, model: str, grid: dict) -> None:
        case = copy.deepcopy(CASES[case_name])
        name = f"{case_name}-{model.lower()}"
        needed_gradient = any(key in self.args.checks for key in ("forces", "invariance", "eps1", "grid"))
        mode = "gradient" if needed_gradient else "energy"
        base = self.run(name + "-base", case, model, mode, grid)
        if "frozen" in self.args.checks:
            self.frozen(name, case, model, grid)
        tolerance = self.args.dft_gradient_tolerance if "DFT" in case["method"] else self.args.hf_gradient_tolerance
        if "forces" in self.args.checks:
            self.nuclear_fd(name,case,model,grid,base,tolerance)
        if "invariance" in self.args.checks:
            translated = copy.deepcopy(case)
            for _, point in translated["atoms"]:
                for axis, shift in enumerate((-0.21, 0.17, 0.29)):
                    point[axis] += shift * BOHRS_TO_A
            moved = self.run(name + "-translated", translated, model, "gradient", grid)
            self.check(f"{name}:translation_energy", abs(moved["energy"] - base["energy"]),
                       self.args.invariance_energy_tolerance, units="hartree")
            self.check(f"{name}:translation_gradient", difference(moved["gradient_hartree_per_bohr"],
                       base["gradient_hartree_per_bohr"]), tolerance, units="hartree/bohr")
            summed = [sum(base["gradient_hartree_per_bohr"][axis::3]) for axis in range(3)]
            self.check(f"{name}:net_force_zero", max(abs(value) for value in summed), tolerance,
                       units="hartree/bohr", summed_gradient=summed)
        if "eps1" in self.args.checks:
            vacuum = self.run(name + "-eps1", case, model, "gradient", grid, epsilon=1.0)
            gas = self.run(case_name + "-gas", case, None, "gradient", grid)
            self.check(f"{name}:eps1_gas_energy", abs(vacuum["energy"] - gas["energy"]),
                       self.args.eps1_energy_tolerance, units="hartree")
            self.check(f"{name}:eps1_gas_gradient", difference(vacuum["gradient_hartree_per_bohr"],
                       gas["gradient_hartree_per_bohr"]), self.args.eps1_gradient_tolerance, units="hartree/bohr")
        if "grid" in self.args.checks:
            volume_results = []
            for nrad, nang in ((50, 110), (75, 302), (100, 434)):
                varied = {**grid, "nrad": nrad, "nang": nang}
                result = self.run(f"{name}-volume{nrad}x{nang}", case, model, "gradient", varied)
                volume_results.append({"nrad": nrad, "nang": nang, "energy": result["energy"],
                                       "solvation_energy": result["solvation_energy"],
                                       "volume_electrons": result["volume_electrons"],
                                       "gradient": result["gradient_hartree_per_bohr"]})
            self.report["grid_convergence"].append({"case": case_name, "model": model,
                                                    "fixed_cavity_grid": {key: grid[key] for key in ("lmax", "nlebedev")},
                                                    "volume_grids": volume_results})
            default, dense = volume_results[1:]
            self.check(f"{name}:default_dense_volume_energy", abs(default["energy"] - dense["energy"]),
                       self.args.grid_energy_tolerance, units="hartree")
            self.check(f"{name}:default_dense_volume_gradient", difference(default["gradient"], dense["gradient"]),
                       self.args.grid_gradient_tolerance, units="hartree/bohr")
        if self.args.applied_field:
            applied = self.run(name + "-field-solute", case, model, "gradient", grid, field=True)
            summed = [sum(applied["gradient_hartree_per_bohr"][axis::3]) for axis in range(3)]
            self.check(f"{name}:field_solute_net_gradient", difference(summed, [-case["charge"] * 2e-4, 0.0, 0.0]),
                       tolerance, units="hartree/bohr", summed_gradient=summed)
            if "forces" in self.args.checks:
                self.nuclear_fd(name+"-field-solute",case,model,grid,applied,tolerance,field=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--launcher", default="")
    parser.add_argument("--basis", type=Path, default=ROOT / "basis")
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES))
    parser.add_argument("--models", nargs="+", choices=("COSMO", "PCM"), default=["COSMO", "PCM"])
    parser.add_argument("--checks", nargs="+", choices=("energy", "frozen", "forces", "invariance", "eps1", "grid"))
    parser.add_argument("--steps", nargs="+", type=float, default=[1.0e-3, 5.0e-4])
    parser.add_argument("--lmax", type=int)
    parser.add_argument("--nlebedev", type=int)
    parser.add_argument("--nrad", type=int)
    parser.add_argument("--nang", type=int)
    parser.add_argument("--applied-field", action="store_true", help="Also exercise the explicit DDX_FIELD=SOLUTE convention")
    parser.add_argument("--hf-gradient-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--dft-gradient-tolerance", type=float, default=5.0e-5)
    parser.add_argument("--density-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--electron-count-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--invariance-energy-tolerance", type=float, default=2.0e-8)
    parser.add_argument("--eps1-energy-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--eps1-gradient-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--grid-energy-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--grid-gradient-tolerance", type=float, default=5.0e-5)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1 or any(not math.isfinite(value) or value <= 0 for value in [*args.steps, args.timeout]):
        parser.error("threads, steps and timeouts must be positive and finite")
    tolerances = [value for key, value in vars(args).items() if key.endswith("tolerance")]
    if any(not math.isfinite(value) or value < 0 for value in tolerances):
        parser.error("tolerances must be finite and nonnegative")
    args.cases = args.cases or (["helium", "water"] if args.suite == "smoke" else list(CASES))
    args.checks = args.checks or ["energy", "frozen", "forces", "invariance", "eps1"]
    if args.suite == "full" and "grid" not in args.checks:
        args.checks.append("grid")
    default_grid = (4, 110, 50, 110) if args.suite == "smoke" else (6, 302, 75, 302)
    grid = {key: getattr(args, key) if getattr(args, key) is not None else default
            for key, default in zip(("lmax", "nlebedev", "nrad", "nang"), default_grid)}
    if grid["lmax"] < 0 or any(grid[key] < 1 for key in ("nlebedev", "nrad", "nang")):
        parser.error("invalid grid dimensions")
    args.driver, args.basis = args.driver.resolve(strict=True), args.basis.resolve(strict=True)
    args.workdir = args.workdir.resolve()
    args.workdir.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update({"QUICK_BASIS": str(args.basis), "OMP_NUM_THREADS": str(args.threads),
                "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    # Cover all source/build inputs, including CUDA helpers and generated
    # integral tables. This is an immutability check for this invocation;
    # a separate build manifest ties these identities to compiled objects.
    source_paths = [Path(__file__).resolve(), HERE / "quick_solvation_api_driver.f90",
                    ROOT / "CMakeLists.txt", ROOT / "QUICKConfig.cmake.in",
                    *[path for directory in (ROOT / "src", ROOT / "quick-cmake", ROOT / "tools", args.basis)
                      for path in directory.rglob("*") if path.is_file()]]
    report = {"passed": False, "started_utc": datetime.now(timezone.utc).isoformat(), "suite": args.suite,
              "cases": args.cases, "models": args.models, "requested_checks": args.checks, "grid": grid,
              "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "limitations": ["Continuous electronic density, electrostatic ddCOSMO/ddPCM only; no SMD/non-electrostatic terms",
                              "Smoke grids are engineering checks, not quadrature-converged final accuracy evidence",
                              "Explicit DDX_FIELD=SOLUTE polarizes the QM solute; no dielectric external-field boundary condition",
                              "Finite molecules, one CLI backend per invocation; no performance or long-trajectory claim"],
              "environment": {key: env.get(key) for key in ("QUICK_BASIS", "LD_LIBRARY_PATH", "OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES")},
              "runs": [], "checks": [], "grid_convergence": []}
    try:
        report["source_inputs_before"] = identities(source_paths)
        report["runtime_before"] = runtime(args.driver, env)
        campaign = Campaign(args, report, env)
        for case in args.cases:
            for model in args.models:
                campaign.case(case, model, grid)
        report["source_inputs_after"] = identities(source_paths)
        report["runtime_after"] = runtime(args.driver, env)
        campaign.check("source_inputs_unchanged", float(report["source_inputs_before"] != report["source_inputs_after"]), 0.0)
        campaign.check("runtime_unchanged", float(report["runtime_before"] != report["runtime_after"]), 0.0)
        report["passed"] = bool(report["checks"]) and all(row["passed"] for row in report["checks"])
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["completed_runs"] = len(report["runs"])
    report["passed_checks"] = sum(row["passed"] for row in report["checks"])
    report["failed_checks"] = sum(not row["passed"] for row in report["checks"])
    path = args.workdir / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": report["passed"], "runs": len(report["runs"]), "passed_checks": report["passed_checks"],
                      "failed_checks": report["failed_checks"], "error": report.get("error"), "report": str(path)}), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
