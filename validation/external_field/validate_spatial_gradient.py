#!/usr/bin/env python3
"""Validate a static affine field using installed QUICK and independent PySCF.

New evidence directories are mandatory. This script does not build or modify
QUICK, basis files, reference outputs, tolerances, or existing evidence. The
independent worker uses exact installed Gaussian basis definitions, Cartesian
AOs, full-precision energies, and native libcint dipole/quadrupole integrals.
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
import subprocess
import sys
import time

BOHR_ANGSTROM = 0.52917720859  # QUICK's input conversion.
WATER = [[8, [-.33840, .00380, .23923]], [1, [-.33510, -.00190, -.83277]],
         [1, [.67350, -.00190, .59353]]]
FIELD = [.0002, -.0001, .0003]
ORIGIN = [.2, -.3, .1]
TENSOR = [[.00002, .00003, -.00002], [.00003, -.00001, .00001], [-.00002, .00001, -.00001]]
COMPONENTS = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
SYMBOLS = {1: "H", 2: "He", 8: "O"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(value: float) -> str:
    candidates = [f"{value:.12g}"]
    mantissa, exponent = f"{value:.11e}".split("e")
    candidates.append(mantissa.rstrip("0").rstrip(".") + "e" + str(int(exponent)))
    candidates = [re.sub(r"^(-?)0\.", r"\1.", text) for text in candidates]
    candidates = [re.sub(r"e([+-])0+(\d+)", r"e\1\2", text) for text in candidates]
    return min(candidates, key=len)


def card(model: dict, zero_tensor: bool = False, gradient: bool = True) -> str:
    text = (f"HF BASIS={model['basis']}" + (" GRADIENT" if gradient else "")
            + f" DENSERMS={compact(model.get('density_rms', 1e-11))} CUTOFF=1e-12 GRADCUTOFF=1e-12"
            + (f" SCF={model['scf_max']}" if model.get("scf_max") else "")
            + (" EXTCHARGES" if model["charges"] else "")
            + (f" CHARGE={model['charge']}" if model['charge'] else ""))
    for prefix, vector in (("F", model["field"]), ("O", model["origin"])):
        for axis, value in zip("XYZ", vector):
            if value:
                text += f" EFIELD_{prefix}{axis}={compact(value)}"
    for left, right in COMPONENTS:
        value = model["tensor"][left][right]
        if value or zero_tensor:
            text += f" EFIELD_G{'XYZ'[left]}{'XYZ'[right]}={compact(value)}"
    if len(text) > 256:
        raise ValueError(f"API keyword card exceeds 256 characters ({len(text)}): {text}")
    return text


def data(model: dict, zero_tensor: bool = False, gradient: bool = True) -> str:
    return (card(model, zero_tensor, gradient) + f"\n{len(model['atoms'])} {len(model['charges'])}\n"
            + "".join(str(z) + " " + " ".join(f"{v:.17g}" for v in xyz) + "\n" for z, xyz in model["atoms"])
            + "".join(" ".join(f"{v:.17g}" for v in row) + "\n" for row in model["charges"]))


def parse(path: Path) -> dict:
    result = {"qm_gradient": [], "charge_gradient": [], "external_tensor": []}
    for line in path.read_text().splitlines():
        words = line.split()
        key = words[0]
        if key in ("qm_gradient", "charge_gradient"):
            result[key].extend(map(float, words[2:]))
        elif key == "external_tensor":
            if int(words[1]) != len(result[key]) + 1:
                raise ValueError("Nonsequential tensor row")
            result[key].append(list(map(float, words[2:])))
        elif key in ("external_field_active", "scf_converged", "analytic_gradient_active"):
            result[key] = words[1] == "T"
        elif key in ("external_field", "external_origin"):
            result[key] = list(map(float, words[1:]))
        elif key == "native_gpu_field_counts":
            result[key] = list(map(int, words[1:]))
        elif key in ("basis_functions", "api_steps"):
            result[key] = int(words[1])
        elif key == "energy":
            result[key] = float(words[1])
    values = [*result.get("external_field", []), *result.get("external_origin", []),
              *[v for row in result["external_tensor"] for v in row], *result["qm_gradient"],
              *result["charge_gradient"], result.get("energy", 0.)]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Nonfinite result")
    if len(result["external_tensor"]) != 3 or any(len(row) != 3 for row in result["external_tensor"]):
        raise ValueError("Incomplete tensor")
    return result


def difference(left: list, right: list) -> float:
    if len(left) != len(right):
        raise ValueError("Mismatched comparison sizes")
    return max((abs(a - b) for a, b in zip(left, right)), default=0.)


def runtime(driver: Path, env: dict) -> dict:
    completed = subprocess.run(["ldd", str(driver)], env=env, capture_output=True, text=True, check=True)
    if "not found" in completed.stdout:
        raise ValueError("Unresolved runtime library")
    paths = [driver]
    quick = []
    for line in completed.stdout.splitlines():
        match = re.search(r"(?:=>\s*)?(/[^\s]+)", line)
        if match:
            path = Path(match.group(1)).resolve(strict=True)
            paths.append(path)
            if re.fullmatch(r"libquick(?:_cuda)?\.so(?:\.[\d.]+)?", path.name):
                quick.append(path)
    if len(quick) != 1:
        raise ValueError("Expected exactly one QUICK runtime")
    return {"quick_library": str(quick[0]), "files_sha256": {str(p): sha(p) for p in paths}}


def oracle_worker(request_path: Path, output_path: Path) -> int:
    import numpy as np
    import pyscf
    from pyscf import gto, lib, qmmm, scf
    from pyscf.gto.basis import parse_gaussian

    request = json.loads(request_path.read_text())
    lib.num_threads(1)
    output = {"passed": False, "pyscf_version": pyscf.__version__, "numpy_version": np.__version__,
              "python": sys.executable, "models": {}, "new_scf_calculations": 0, "checks": []}
    oracle_sources = [Path(__file__).resolve(), request_path.resolve(), Path(sys.executable).resolve(),
                      Path(pyscf.__file__).resolve(), Path(np.__file__).resolve()]
    oracle_sources.extend(Path(item["basis_file"]).resolve() for item in request["models"].values())
    output["source_runtime_before"] = {str(p): sha(p) for p in set(oracle_sources)}

    def check(name, error, tolerance):
        output["checks"].append({"name": name, "error": float(error), "tolerance": tolerance,
                                 "passed": bool(np.isfinite(error) and error <= tolerance)})

    try:
        for name, item in request["models"].items():
            model = item["model"]
            basis = {SYMBOLS[z]: parse_gaussian.load(item["basis_file"], SYMBOLS[z], optimize=False)
                     for z in set(z for z, _ in model["atoms"])}
            field, origin, tensor = (np.asarray(model[k], dtype=float) for k in ("field", "origin", "tensor"))
            atoms = [[SYMBOLS[z], (np.asarray(xyz) / BOHR_ANGSTROM).tolist()] for z, xyz in model["atoms"]]

            def evaluate(coordinates, save=False):
                mol = gto.M(atom=coordinates, basis=basis, unit="Bohr", cart=True,
                            charge=model["charge"], spin=0, verbose=0)
                mf = scf.RHF(mol)
                if model["charges"]:
                    mm = np.asarray(model["charges"], dtype=float)
                    mf = qmmm.mm_charge(mf, mm[:, :3] / BOHR_ANGSTROM, mm[:, 3], unit="Bohr")
                mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-10, 250
                with mol.with_common_orig(origin):
                    r = mol.intor("int1e_r", comp=3, hermi=1)
                    rr = mol.intor("int1e_rr", comp=9, hermi=1).reshape(3, 3, mol.nao_nr(), mol.nao_nr())
                perturbation = np.einsum("x,xij->ij", field, r) + .5 * np.einsum("xy,xyij->ij", tensor, rr)
                h = mf.get_hcore() + perturbation
                nuclear_coords = mol.atom_coords() - origin
                charge_coords = [(float(z), point) for z, point in zip(mol.atom_charges(), nuclear_coords)]
                charge_coords.extend((float(row[3]), np.asarray(row[:3]) / BOHR_ANGSTROM - origin)
                                     for row in model["charges"])
                classical = -sum(q * (np.dot(field, point) + .5 * point @ tensor @ point)
                                 for q, point in charge_coords)
                nuclear = mf.energy_nuc() + classical
                mf.get_hcore = lambda *args: h
                mf.energy_nuc = lambda *args: nuclear
                energy = float(mf.kernel())
                output["new_scf_calculations"] += 1
                if not mf.converged or not math.isfinite(energy):
                    raise ValueError("Independent PySCF SCF failed")
                if save:
                    return energy, mol, mf, r, rr, perturbation
                return energy

            energy, mol, mf, r, rr, hfield = evaluate(atoms, True)
            target = item["target"]
            folder = Path(item["folder"])
            check(name + ":energy", abs(energy - target["energy"]), 1e-7)
            s = mol.intor("int1e_ovlp", hermi=1)
            normalization = np.sqrt(np.diag(s))
            native_unit_s = s / normalization[:, None] / normalization[None, :]
            # Match QUICK Cartesian functions by atom, powers and primitive
            # radial signature, rather than assume the two AO orders coincide.
            lines = (folder / "ao_basis.dat").read_text().splitlines()
            nao, pointer, quick_aos = int(lines[0]), 1, []
            for _ in range(nao):
                descriptor = list(map(int, lines[pointer].split())); pointer += 1
                count = descriptor[-1]
                primitives = np.array([list(map(float, line.split())) for line in lines[pointer:pointer + count]])
                pointer += count
                quick_aos.append((descriptor, primitives))
            candidates = []
            for shell in range(mol.nbas):
                angular = mol.bas_angular(shell)
                powers = [(x, y, angular - x - y) for x in range(angular, -1, -1)
                          for y in range(angular - x, -1, -1)]
                exponents = mol.bas_exp(shell)
                coefficients = mol._libcint_ctr_coeff(shell)
                for contraction in range(mol.bas_nctr(shell)):
                    for powers_xyz in powers:
                        candidates.append((mol.bas_atom(shell) + 1, powers_xyz, exponents, coefficients[:, contraction]))
            mapping, phases = [], []
            for descriptor, primitives in quick_aos:
                choices = []
                for index, (center, powers, exponents, coefficients) in enumerate(candidates):
                    if center != descriptor[1] or tuple(descriptor[2:5]) != powers or len(exponents) != len(primitives):
                        continue
                    order = np.argsort(exponents); qorder = np.argsort(primitives[:, 0])
                    if not np.allclose(exponents[order], primitives[qorder, 0], rtol=1e-11, atol=1e-13):
                        continue
                    a = coefficients[order] / np.linalg.norm(coefficients)
                    b = primitives[qorder, 1] / np.linalg.norm(primitives[:, 1])
                    phase = 1 if np.dot(a, b) >= 0 else -1
                    if np.max(np.abs(a - phase * b)) < 1e-9:
                        choices.append((index, phase))
                if len(choices) != 1:
                    raise ValueError(f"AO mapping not unique for {descriptor}: {choices}")
                mapping.append(choices[0][0]); phases.append(choices[0][1])
            if len(set(mapping)) != nao or nao != mol.nao_nr():
                raise ValueError("Nonbijective AO mapping")
            phase = np.asarray(phases)

            def transform(matrix):
                unit = matrix / normalization[:, None] / normalization[None, :]
                return unit[np.ix_(mapping, mapping)] * phase[:, None] * phase[None, :]

            rows = np.loadtxt(folder / "operators.dat", skiprows=1)
            if rows.shape != (nao * nao, 5):
                raise ValueError("Incomplete operator export")
            quick_s, quick_d, quick_h = (rows[:, column].reshape(nao, nao) for column in (2, 3, 4))
            check(name + ":exact_cartesian_basis_overlap", np.max(np.abs(quick_s - transform(s))), 2e-10)
            check(name + ":full_AO_affine_operator", np.max(np.abs(quick_h - transform(hfield))), 2e-11)
            six = np.zeros((6, nao, nao))
            for component, mu, nu, value in np.loadtxt(folder / "tensor_operators.dat", skiprows=1):
                component, mu, nu = int(component) - 1, int(mu) - 1, int(nu) - 1
                six[component, mu, nu] = six[component, nu, mu] = value
            # Fixed PSD density includes off-diagonal AO terms, without using
            # either code's converged density to cancel a moment error.
            v = np.sin(np.arange(nao * 3).reshape(nao, 3) + .37)
            fixed_d = v @ v.T
            fixed_d /= np.einsum("ij,ji->", fixed_d, quick_s)
            component_errors = []
            for index, (left, right) in enumerate(COMPONENTS):
                oracle = transform(rr[left, right] * (.5 if left == right else 1.))
                error = float(np.max(np.abs(six[index] - oracle)))
                component_errors.append(error)
                check(name + f":unit_tensor_{'XYZ'[left]}{'XYZ'[right]}_AO", error, 2e-10)
                check(name + f":unit_tensor_{'XYZ'[left]}{'XYZ'[right]}_fixed_density",
                      abs(np.einsum("ij,ji->", fixed_d, six[index] - oracle)), 2e-10)
            check(name + ":electron_count", abs(np.einsum("ij,ji->", quick_d, quick_s) - mol.nelectron), 1e-8)
            nuclear_dipole = np.einsum("a,ax->x", mol.atom_charges(), mol.atom_coords() - origin)
            electronic_dipole = np.array([np.einsum("ij,ji->", quick_d, transform(moment)) for moment in r])
            dipole = nuclear_dipole - electronic_dipole
            for row in model["charges"]:
                dipole += row[3] * (np.asarray(row[:3]) / BOHR_ANGSTROM - origin)
            total_charge = model["charge"] + sum(row[3] for row in model["charges"])
            net_gradient = np.asarray(target["qm_gradient"]).reshape(-1, 3).sum(axis=0)
            if target["charge_gradient"]:
                net_gradient += np.asarray(target["charge_gradient"]).reshape(-1, 3).sum(axis=0)
            expected_net = -total_charge * field - tensor @ dipole
            check(name + ":total_force_in_affine_field", np.max(np.abs(net_gradient - expected_net)), 2e-7)
            result = {"energy": energy, "basis_functions": nao, "AO_mapping_0based": mapping,
                      "AO_phases": phases, "all_six_operator_max_abs_error": component_errors,
                      "dipole_au_about_field_origin": dipole.tolist(),
                      "summed_energy_gradient": net_gradient.tolist(),
                      "expected_summed_energy_gradient": expected_net.tolist(),
                      "finite_difference_gradients": []}
            if item.get("full_fd", False):
                for step in request["oracle_steps"]:
                    gradient = []
                    for atom in range(len(atoms)):
                        for axis in range(3):
                            displaced_energies = []
                            for sign in (1, -1):
                                moved = copy.deepcopy(atoms)
                                moved[atom][1][axis] += sign * step
                                displaced_energies.append(evaluate(moved))
                            gradient.append((displaced_energies[0] - displaced_energies[1]) / (2 * step))
                    check(name + f":independent_full_nuclear_gradient_h{step:g}",
                          difference(gradient, target["qm_gradient"]), 1e-5)
                    result["finite_difference_gradients"].append({"step_bohr": step, "gradient": gradient})
            output["models"][name] = result
        output["source_runtime_after"] = {str(p): sha(p) for p in set(oracle_sources)}
        check("independent_oracle_inputs_unchanged", int(output["source_runtime_before"] != output["source_runtime_after"]), 0.)
        if Path("/proc/self/maps").exists():
            paths = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                            if len(line.split()) >= 6 and line.split()[-1].startswith("/")
                            and Path(line.split()[-1]).is_file()})
            output["loaded_files_sha256"] = {path: sha(Path(path)) for path in paths}
        output["passed"] = all(item["passed"] for item in output["checks"])
    except Exception as error:
        import traceback
        output["error"] = f"{type(error).__name__}: {error}"
        output["traceback"] = traceback.format_exc()
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return 0 if output["passed"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-driver", type=Path)
    parser.add_argument("--cuda-driver", type=Path)
    parser.add_argument("--cpu-basis", type=Path)
    parser.add_argument("--cuda-basis", type=Path)
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--pyscf-python", type=Path)
    parser.add_argument("--steps", type=float, nargs="+", default=[1e-3, 5e-4])
    parser.add_argument("--oracle-steps", type=float, nargs="+", default=[5e-4])
    parser.add_argument("--timeout", type=float, default=1800.)
    parser.add_argument("--skip-f-basis", action="store_true")
    parser.add_argument("--oracle-request", type=Path)
    parser.add_argument("--oracle-output", type=Path)
    args = parser.parse_args()
    if args.oracle_request:
        return oracle_worker(args.oracle_request, args.oracle_output)
    if not args.workdir or not args.cpu_driver or not args.cpu_basis:
        parser.error("--workdir, --cpu-driver and --cpu-basis are required")
    if bool(args.cuda_driver) != bool(args.cuda_basis):
        parser.error("CUDA driver and basis directory must be provided together")
    if any(not math.isfinite(value) or value <= 0 for value in [*args.steps, *args.oracle_steps, args.timeout]):
        parser.error("Steps and timeout must be positive finite values")
    args.workdir = args.workdir.resolve()
    args.workdir.mkdir(parents=True, exist_ok=False)
    backends = {"cpu": (args.cpu_driver.resolve(strict=True), args.cpu_basis.resolve(strict=True))}
    if args.cuda_driver:
        backends["cuda"] = (args.cuda_driver.resolve(strict=True), args.cuda_basis.resolve(strict=True))
    report = {"passed": False, "started_utc": datetime.now(timezone.utc).isoformat(), "runs": [], "checks": [],
              "scope": "Static affine applied field: parser, exact Cartesian AO operator, independent HF energy, nuclear/charge forces and covariance",
              "fixed_gates": {"CPU_GPU_energy_hartree": 1e-7, "CPU_GPU_gradient_hartree_per_bohr": 1e-7,
                              "energy_FD_gradient_hartree_per_bohr": 1e-5, "covariance_energy_hartree": 1e-7,
                              "zero_tensor_AO_operator": 0., "zero_tensor_energy_hartree": 1e-10,
                              "zero_tensor_gradient_hartree_per_bohr": 1e-9, "independent_PySCF_energy_hartree": 1e-7,
                              "independent_PySCF_AO_operator": 2e-11, "independent_unit_tensor_AO_operator": 2e-10},
              "limitations": ["Finite isolated closed-shell HF molecules, static fields; this harness does not establish production sampling",
                              "No continuum solvent, DFT grid convergence, PBC, time-dependent field, or electrode calculation",
                              "Nonzero trace is permitted but represents a prescribed source density; source-free tensors have zero trace",
                              "Property grids keep their existing source-only convention; no direct external field is added here"]}

    def check(name, error, tolerance, **details):
        report["checks"].append({"name": name, "error": error, "tolerance": tolerance,
                                 "passed": bool(math.isfinite(error) and error <= tolerance), **details})

    envs = {}
    for backend, (driver, basis) in backends.items():
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", QUICK_BASIS=str(basis))
        envs[backend] = env
    cache, oracle_models = {}, {}

    def save_progress():
        (args.workdir / "progress.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    def run(backend, label, model, gradient=True, explicit_zero=False):
        text = data(model, explicit_zero, gradient)
        key = (backend, text, gradient)
        if key in cache:
            return cache[key]
        folder = args.workdir / f"{len(report['runs']):04d}-{backend}-{label}"
        folder.mkdir(exist_ok=False)
        path = folder / "field.dat"
        path.write_text(text)
        command = [str(backends[backend][0]), "gradient" if gradient else "energy", path.name]
        row = {"backend": backend, "label": label, "model": model, "folder": str(folder),
               "command": command, "input_sha256": sha(path), "gradient_requested": gradient}
        report["runs"].append(row)
        print(f"RUN {backend} {label}", flush=True)
        started = time.monotonic()
        with (folder / "execution.log").open("w") as log:
            process = subprocess.run(command, cwd=folder, env=envs[backend], stdout=log,
                                     stderr=subprocess.STDOUT, timeout=args.timeout)
        row.update(returncode=process.returncode, wall_seconds=time.monotonic() - started)
        if process.returncode:
            raise ValueError(f"QUICK failed: {folder}")
        result = parse(folder / "result.dat")
        if gradient:
            if not result.get("analytic_gradient_active", False):
                raise ValueError("Gradient API fixture did not activate analytic gradients")
            # Every accepted geometry in this bounded campaign is explicitly
            # nonstationary; an all-zero return indicates a fixture/API error.
            if max(abs(value) for value in result["qm_gradient"]) < 1e-5:
                raise ValueError("Nonstationary fixture unexpectedly returned zero gradients")
        output = folder / "spatial_gradient_api.out"
        cycles = re.findall(r"REACH CONVERGENCE AFTER\s+(\d+)\s+CYCLES", output.read_text(errors="replace"))
        check(f"{backend}:{label}:explicit_converged_SCF", int(len(cycles) != 1 or not result["scf_converged"]), 0.)
        check(f"{backend}:{label}:all_nuclear_gradient_components", abs(len(result["qm_gradient"]) - (3 * len(model["atoms"]) if gradient else 0)), 0.)
        check(f"{backend}:{label}:all_charge_gradient_components", abs(len(result["charge_gradient"]) - (3 * len(model["charges"]) if gradient else 0)), 0.)
        for name, expected in (("external_field", model["field"]), ("external_origin", model["origin"])):
            check(f"{backend}:{label}:{name}", difference(result[name], expected), 1e-18)
        check(f"{backend}:{label}:symmetric_tensor", difference([x for row_ in result["external_tensor"] for x in row_],
                                                               [x for row_ in model["tensor"] for x in row_]), 1e-18)
        active = any(model["field"]) or any(x for row_ in model["tensor"] for x in row_)
        counts = result["native_gpu_field_counts"]
        expected_counts = [1, int(gradient)] if backend == "cuda" and active else [0, 0]
        check(f"{backend}:{label}:native_GPU_field_matrix_force_dispatch", int(counts != expected_counts), 0.)
        row.update(result=result, files_sha256={p.name: sha(p) for p in folder.iterdir() if p.is_file()})
        cache[key] = (result, folder)
        save_progress()
        return result, folder

    def fd(backend, model, analytic, steps, components=None, kind="qm_gradient", prefix="FD"):
        indices = components if components is not None else list(range(len(analytic[kind])))
        for step in steps:
            values = []
            for index in indices:
                atom, axis = divmod(index, 3)
                energies = []
                for sign in (1, -1):
                    moved = copy.deepcopy(model)
                    point = moved["atoms"][atom][1] if kind == "qm_gradient" else moved["charges"][atom]
                    point[axis] += sign * step * BOHR_ANGSTROM
                    energies.append(run(backend, f"{prefix}-{kind}-{index}-h{step:g}-{sign:+d}", moved, False)[0]["energy"])
                values.append((energies[0] - energies[1]) / (2 * step))
            check(f"{backend}:{prefix}:{kind}:h{step:g}", difference(values, [analytic[kind][i] for i in indices]), 1e-5,
                  component_indices_0based=indices, step_bohr=step, finite_difference=values)

    try:
        protected = [Path(__file__).resolve(), Path(__file__).with_name("quick_spatial_gradient_api_driver.f90")]
        protected.extend(basis / name for _, basis in backends.values() for name in ("STO-3G.BAS", "DEF2-TZVPD.BAS")
                         if (basis / name).exists())
        report["inputs_before"] = {str(p): sha(p) for p in set(protected)}
        report["runtime_before"] = {name: runtime(driver, envs[name]) for name, (driver, _) in backends.items()}
        for backend, (_, basis) in backends.items():
            expected_library = (basis.parent / "lib" / ("libquick_cuda.so" if backend == "cuda" else "libquick.so")).resolve(strict=True)
            actual_library = Path(report["runtime_before"][backend]["quick_library"])
            if actual_library != expected_library:
                raise ValueError(f"Wrong {backend} QUICK runtime: {actual_library}; expected {expected_library}")
        # Read/check tests use no SCF, and each starts from fresh defaults.
        parser_cases = [("default", "HF", True, None),
                        ("boundary", "HF NOT_EFIELD_GXX=.2 MY_EXTERNAL_EFIELD_GRADIENT_XY=.4", True, None),
                        ("origin_only", "HF EFIELD_OX=.2", True, None),
                        ("nonzero_trace", "HF EFIELD_GXX=.1 EFIELD_GYY=.2 EFIELD_GZZ=.3", True, "nonzero"),
                        ("aliases", "HF " + " ".join(f"EFIELD_G{'XYZ'[i]}{'XYZ'[j]}={k + 1}" for k, (i, j) in enumerate(COMPONENTS)), True, "all")]
        for reverse in ("YX", "ZX", "ZY"):
            parser_cases.append(("reject_reverse_" + reverse, "HF EFIELD_G" + reverse + "=.1", False, None))
        parser_cases.append(("reject_conflicting_aliases", "HF EFIELD_GXX=.1 EXTERNAL_EFIELD_GRADIENT_XX=.2", False, None))
        for left, right in COMPONENTS:
            component = "XYZ"[left] + "XYZ"[right]
            parser_cases.append(("canonical_" + component, "HF EXTERNAL_EFIELD_GRADIENT_" + component + "=.125", True, (left, right)))
            for value in ("NaN", "Inf", "1e400", "bad"):
                parser_cases.append(("reject_" + component + "_" + value, "HF EFIELD_G" + component + "=" + value, False, None))
        for index, (label, text, success, expectation) in enumerate(parser_cases):
            folder = args.workdir / "parser" / f"{index:03d}-{label}"
            folder.mkdir(parents=True, exist_ok=False)
            (folder / "options.dat").write_text(text + "\n")
            process = subprocess.run([str(backends["cpu"][0]), "options", "options.dat"], cwd=folder,
                                     env=envs["cpu"], capture_output=True, text=True, timeout=args.timeout)
            (folder / "execution.log").write_text(process.stdout + process.stderr)
            check("parser:" + label, int((process.returncode == 0) != success), 0.)
            if success and process.returncode == 0:
                result = parse(folder / "result.dat")
                tensor = [[0.] * 3 for _ in range(3)]
                if expectation == "all":
                    for k, (i, j) in enumerate(COMPONENTS):
                        tensor[i][j] = tensor[j][i] = k + 1
                elif expectation == "nonzero":
                    for i in range(3):
                        tensor[i][i] = [.1, .2, .3][i]
                elif isinstance(expectation, tuple):
                    i, j = expectation
                    tensor[i][j] = tensor[j][i] = .125
                check("parser:" + label + ":all_six_tensor_values",
                      difference(sum(result["external_tensor"], []), sum(tensor, [])), 0.)
                expected_active = label == "origin_only" or bool(any(sum(tensor, [])))
                check("parser:" + label + ":active", int(result["external_field_active"] != expected_active), 0.)

        model = {"atoms": copy.deepcopy(WATER), "charges": [], "charge": 0, "basis": "STO-3G",
                 "field": FIELD.copy(), "origin": ORIGIN.copy(), "tensor": copy.deepcopy(TENSOR)}
        central = {}
        for backend in backends:
            analytic, folder = run(backend, "mixed-all6", model)
            central[backend] = analytic
            check(backend + ":STO3G_Cartesian_AOs", abs(analytic["basis_functions"] - 7), 0.)
            fd(backend, model, analytic, args.steps)
            oracle_models[backend + "-mixed-all6"] = {"model": model, "target": analytic, "folder": str(folder),
                                                       "basis_file": str(backends[backend][1] / "STO-3G.BAS"),
                                                       "full_fd": backend == "cuda" or "cuda" not in backends}
            # A true zero G must preserve the existing uniform-field path.
            uniform = copy.deepcopy(model); uniform["tensor"] = [[0.] * 3 for _ in range(3)]
            old, old_folder = run(backend, "uniform-omitted-G", uniform)
            zero, zero_folder = run(backend, "uniform-explicit-zero-G", uniform, explicit_zero=True)
            old_h = [float(line.split()[-1]) for line in (old_folder / "operators.dat").read_text().splitlines()[1:]]
            zero_h = [float(line.split()[-1]) for line in (zero_folder / "operators.dat").read_text().splitlines()[1:]]
            check(backend + ":zero_G_uniform_AO_operator", difference(old_h, zero_h), 0.)
            check(backend + ":zero_G_uniform_energy", abs(old["energy"] - zero["energy"]), 1e-10)
            check(backend + ":zero_G_uniform_gradient", difference(old["qm_gradient"], zero["qm_gradient"]), 1e-9)
        if "cuda" in backends:
            check("CPU_CUDA:mixed_energy", abs(central["cpu"]["energy"] - central["cuda"]["energy"]), 1e-7)
            check("CPU_CUDA:mixed_full_nuclear_gradient", difference(central["cpu"]["qm_gradient"], central["cuda"]["qm_gradient"]), 1e-7)

        # Closed-shell charged water avoids the unrelated unrestricted-f
        # gradient restriction. Gauge-origin change also changes F so that
        # E(r) is unchanged; the remaining constant shifts energy by Q*C.
        charged = copy.deepcopy(model); charged["charge"] = 2
        charged["tensor"] = [[.00002, .00003, 0.], [.00003, 0., 0.], [0., 0., 0.]]
        shift = [.3, -.2, .4]
        for backend in backends:
            initial, folder = run(backend, "charged-central", charged)
            oracle_models[backend + "-charged"] = {"model": charged, "target": initial, "folder": str(folder),
                                                      "basis_file": str(backends[backend][1] / "STO-3G.BAS"),
                                                      "full_fd": backend == "cuda" or "cuda" not in backends}
            moved = copy.deepcopy(charged)
            moved["origin"] = [a + b for a, b in zip(charged["origin"], shift)]
            moved["field"] = [charged["field"][i] + sum(charged["tensor"][i][j] * shift[j] for j in range(3)) for i in range(3)]
            other = run(backend, "charged-origin-gauge", moved)[0]
            constant = sum(charged["field"][i] * shift[i] for i in range(3))
            constant += .5 * sum(shift[i] * charged["tensor"][i][j] * shift[j] for i in range(3) for j in range(3))
            check(backend + ":charged_gauge_energy", abs(other["energy"] - initial["energy"] - charged["charge"] * constant), 1e-7,
                  expected_energy_shift=charged["charge"] * constant)
            check(backend + ":charged_gauge_force", difference(other["qm_gradient"], initial["qm_gradient"]), 1e-7)
            translated = copy.deepcopy(charged)
            translated["origin"] = [a + b for a, b in zip(charged["origin"], shift)]
            for _, xyz in translated["atoms"]:
                for i in range(3):
                    xyz[i] += shift[i] * BOHR_ANGSTROM
            other = run(backend, "charged-all-centers-and-origin-translate", translated)[0]
            check(backend + ":charged_translation_energy", abs(other["energy"] - initial["energy"]), 1e-7)
            check(backend + ":charged_translation_force", difference(other["qm_gradient"], initial["qm_gradient"]), 1e-7)

        # Signed external charge coordinates also receive -q*(F+G*s) in the
        # energy gradient. Test each charge coordinate against SCF energies.
        mm_model = copy.deepcopy(model)
        mm_model["origin"] = [0., 0., 0.]
        mm_model["charges"] = [[2.2, -.7, 1.6, .35], [-1.8, 1.9, -.8, -.20]]
        mm_central = {}
        for backend in backends:
            result, folder = run(backend, "signed-MM-charges", mm_model)
            mm_central[backend] = result
            if backend == "cuda" or "cuda" not in backends:
                fd(backend, mm_model, result, [min(args.steps)], kind="charge_gradient", prefix="MM-FD")
            oracle_models[backend + "-MM-charges"] = {"model": mm_model, "target": result, "folder": str(folder),
                                                       "basis_file": str(backends[backend][1] / "STO-3G.BAS")}
        if "cuda" in backends:
            check("CPU_CUDA:MM_energy", abs(mm_central["cpu"]["energy"] - mm_central["cuda"]["energy"]), 1e-7)
            for kind in ("qm_gradient", "charge_gradient"):
                check("CPU_CUDA:MM_" + kind, difference(mm_central["cpu"][kind], mm_central["cuda"][kind]), 1e-7)

        if not args.skip_f_basis:
            f_model = copy.deepcopy(model); f_model["basis"] = "DEF2-TZVPD"
            f_model["origin"] = [0., 0., 0.]
            f_model["scf_max"], f_model["density_rms"] = 400, 1e-10
            f_central = {}
            for backend in backends:
                result, folder = run(backend, "Cartesian-f64", f_model)
                f_central[backend] = result
                check(backend + ":f64_basis_functions", abs(result["basis_functions"] - 64), 0.)
                oracle_models[backend + "-f64"] = {"model": f_model, "target": result, "folder": str(folder),
                                                   "basis_file": str(backends[backend][1] / "DEF2-TZVPD.BAS")}
                if backend == "cuda" or "cuda" not in backends:
                    fd(backend, f_model, result, args.steps, components=[2], prefix="f64-Oz-FD")
            if "cuda" in backends:
                check("CPU_CUDA:f64_energy", abs(f_central["cpu"]["energy"] - f_central["cuda"]["energy"]), 1e-7)
                check("CPU_CUDA:f64_full_nuclear_gradient", difference(f_central["cpu"]["qm_gradient"], f_central["cuda"]["qm_gradient"]), 1e-7)

        if args.pyscf_python:
            request = args.workdir / "oracle-request.json"
            output = args.workdir / "oracle-report.json"
            request.write_text(json.dumps({"models": oracle_models, "oracle_steps": args.oracle_steps}, indent=2) + "\n")
            command = [str(args.pyscf_python.resolve(strict=True)), str(Path(__file__).resolve()),
                       "--oracle-request", str(request), "--oracle-output", str(output)]
            with (args.workdir / "oracle-execution.log").open("w") as log:
                process = subprocess.run(command, env=envs["cpu"], stdout=log, stderr=subprocess.STDOUT,
                                         timeout=max(args.timeout, 7200.))
            oracle = json.loads(output.read_text()) if output.exists() else {}
            report["independent_oracle"] = {"command": command, "returncode": process.returncode,
                                            "report": str(output), "report_sha256": sha(output) if output.exists() else None}
            check("independent_PySCF_campaign", int(process.returncode != 0 or not oracle.get("passed", False)), 0.)
            report["checks"].extend({**item, "name": "PySCF:" + item["name"]} for item in oracle.get("checks", []))
        else:
            report["limitations"].append("Independent oracle omitted; supply --pyscf-python to establish cross-code accuracy")
        report["runtime_after"] = {name: runtime(driver, envs[name]) for name, (driver, _) in backends.items()}
        report["inputs_after"] = {str(p): sha(p) for p in set(protected)}
        check("runtime_unchanged", int(report["runtime_before"] != report["runtime_after"]), 0.)
        check("harness_basis_sources_unchanged", int(report["inputs_before"] != report["inputs_after"]), 0.)
        report["passed"] = all(item["passed"] for item in report["checks"])
    except Exception as error:
        import traceback
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["completed_runs"] = len(report["runs"])
    report["failed_checks"] = sum(not item["passed"] for item in report["checks"])
    path = args.workdir / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": report["passed"], "checks": len(report["checks"]),
                      "failed_checks": report["failed_checks"], "runs": report["completed_runs"],
                      "error": report.get("error"), "report": str(path)}), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
