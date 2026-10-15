#!/usr/bin/env python3
"""Independent libcint third-attraction derivatives and analytic references.

Run this worker with PySCF, NumPy and SciPy. No QUICK recurrence or generated
derivative expression is imported. Observation derivatives follow translation
invariance: d/dC = -(d/dA+d/dB); AO-center derivatives are minus the electron
coordinate derivatives provided by libcint. The three signs therefore cancel.
The third derivative of an AO product has eight bra/ket product-rule terms.
"""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.special import factorial2, hyp1f1
from pyscf import gto
from pyscf.gto.basis import parse_gaussian
import pyscf

BOHR_ANGSTROM = .52917720859
SYMBOLS = {1: "H", 2: "He", 8: "O"}


def point_charge_hessian(points, sources):
    """H_ijk = q*(15RiRjRk/r^7-3(deltaij Rk+...)/r^5)."""
    points = np.asarray(points, dtype=float)
    out = np.zeros((len(points), 3, 3, 3))
    for xyz, charge in sources:
        delta = points - np.asarray(xyz)
        r2 = np.einsum("pi,pi->p", delta, delta)
        if np.any(r2 == 0):
            raise ValueError("Classical singularity in independent reference")
        inv5 = r2 ** -2.5
        inv7 = r2 ** -3.5
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    out[:, i, j, k] += charge * (15 * delta[:, i] * delta[:, j] * delta[:, k] * inv7
                        - 3 * ((i == j) * delta[:, k] + (i == k) * delta[:, j]
                               + (j == k) * delta[:, i]) * inv5)
    return out


def gaussian_hessian(points, ao_exponent, center=(0., 0., 0.)):
    """Closed radial formula for a normalized s AO with D=1, electron q=-1."""
    delta = np.asarray(points, dtype=float) - np.asarray(center)
    beta = 2 * ao_exponent
    t = beta * np.einsum("pi,pi->p", delta, delta)
    f2 = hyp1f1(2.5, 3.5, -t) / 5
    f3 = hyp1f1(3.5, 4.5, -t) / 7
    prefactor = 4 * beta ** 1.5 / math.sqrt(math.pi)
    out = np.zeros((len(delta), 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                out[:, i, j, k] = (2 * prefactor * beta * f2 *
                    ((i == j) * delta[:, k] + (i == k) * delta[:, j] + (j == k) * delta[:, i])
                    - 4 * prefactor * beta ** 2 * delta[:, i] * delta[:, j] * delta[:, k] * f3)
    return out


def attraction_third(molecule, point):
    """All 27 derivatives of positive AO attraction I(C), in native AO order."""
    n = molecule.nao_nr()
    with molecule.with_rinv_origin(point):
        aaa = molecule.intor("int1e_ipipiprinv", comp=27).reshape(3, 3, 3, n, n)
        aab = molecule.intor("int1e_ipiprinvip", comp=27).reshape(3, 3, 3, n, n)
    mixed = aab + aab.transpose(0, 2, 1, 3, 4) + aab.transpose(2, 0, 1, 3, 4)
    return aaa + aaa.swapaxes(-1, -2) + mixed + mixed.swapaxes(-1, -2)


def attraction_second(molecule, point):
    n = molecule.nao_nr()
    with molecule.with_rinv_origin(point):
        aa = molecule.intor("int1e_ipiprinv", comp=9).reshape(3, 3, n, n)
        ab = molecule.intor("int1e_iprinvip", comp=9).reshape(3, 3, n, n)
    return aa + aa.swapaxes(-1, -2) + ab + ab.swapaxes(-1, -2)


def ao_mapping(molecule, descriptors):
    candidates = []
    for shell in range(molecule.nbas):
        l = molecule.bas_angular(shell)
        powers = [(x, y, l-x-y) for x in range(l, -1, -1) for y in range(l-x, -1, -1)]
        exponents = molecule.bas_exp(shell)
        coefficients = molecule._libcint_ctr_coeff(shell)
        for contraction in range(molecule.bas_nctr(shell)):
            for angular in powers:
                candidates.append((molecule.bas_atom(shell) + 1, angular, exponents, coefficients[:, contraction]))
    mapping, phases = [], []
    for descriptor, primitives in descriptors:
        choices = []
        for index, (center, angular, exponents, coefficients) in enumerate(candidates):
            if center != descriptor[1] or angular != tuple(descriptor[2:5]) or len(exponents) != len(primitives):
                continue
            order, qorder = np.argsort(exponents), np.argsort(primitives[:, 0])
            if not np.allclose(exponents[order], primitives[qorder, 0], rtol=1e-11, atol=1e-13):
                continue
            a = coefficients[order] / np.linalg.norm(coefficients)
            b = primitives[qorder, 1] / np.linalg.norm(primitives[:, 1])
            phase = 1 if np.dot(a, b) >= 0 else -1
            if np.max(np.abs(a - phase * b)) < 1e-9:
                choices.append((index, phase))
        if len(choices) != 1:
            raise ValueError(f"AO mapping is not unique: {descriptor}, candidates {choices}")
        mapping.append(choices[0][0]); phases.append(choices[0][1])
    if len(set(mapping)) != len(mapping) or len(mapping) != molecule.nao_nr():
        raise ValueError("Nonbijective exact Cartesian AO mapping")
    return np.asarray(mapping), np.asarray(phases)


def read_descriptors(path):
    lines = Path(path).read_text().splitlines()
    n, pointer, descriptors = int(lines[0]), 1, []
    for _ in range(n):
        values = list(map(int, lines[pointer].split())); pointer += 1
        count = values[-1]
        primitives = np.asarray([list(map(float, line.split())) for line in lines[pointer:pointer+count]])
        pointer += count
        descriptors.append((values, primitives))
    return descriptors


def molecular_reference(item):
    model = item["model"]
    basis = {SYMBOLS[z]: parse_gaussian.load(item["basis_file"], SYMBOLS[z], optimize=False)
             for z in {z for z, _ in model["atoms"]}}
    molecule = gto.M(atom=[[SYMBOLS[z], (np.asarray(xyz) / BOHR_ANGSTROM).tolist()] for z, xyz in model["atoms"]],
                     basis=basis, unit="Bohr", cart=True, charge=model["charge"], spin=model["spin"], verbose=0)
    folder = Path(item["folder"])
    mapping, phases = ao_mapping(molecule, read_descriptors(folder / "ao_basis.dat"))
    overlap = molecule.intor("int1e_ovlp", hermi=1)
    normalization = np.sqrt(np.diag(overlap))

    def transform(matrix):
        scaled = matrix / normalization[:, None] / normalization[None, :]
        return scaled[..., mapping[:, None], mapping] * phases[:, None] * phases[None, :]

    lines = (folder / "density.dat").read_text().splitlines()
    n, unrestricted = lines[0].split()
    n, unrestricted = int(n), unrestricted == "T"
    flat = np.asarray([float(v) for line in lines[1:] for v in line.split()])
    density = flat[:n*n].reshape(n, n, order="F")
    if unrestricted:
        density += flat[n*n:].reshape(n, n, order="F")
    data = np.loadtxt(folder / "properties.dat", skiprows=1)
    data = np.atleast_2d(data)
    points = data[:, 1:4]
    electronic = np.asarray([np.einsum("mn,ijknm->ijk", density, transform(attraction_third(molecule, point)))
                             for point in points])
    sources = [(np.asarray(xyz) / BOHR_ANGSTROM, z) for z, xyz in model["atoms"]]
    sources.extend((np.asarray(row[:3]) / BOHR_ANGSTROM, row[3]) for row in model["charges"])
    expected = electronic + point_charge_hessian(points, sources)
    # Each point occupies a row; reshape each 27-component row separately.
    observed = np.asarray([row[17:].reshape(3, 3, 3, order="F") for row in data])
    quick_s = np.loadtxt(folder / "overlap.dat", skiprows=1).reshape(n, n, order="F")
    values = molecule.eval_gto("GTOval_cart", points) / normalization
    derivatives = molecule.eval_gto("GTOval_ip_cart", points) / normalization
    values = values[:, mapping] * phases
    derivatives = derivatives[:, :, mapping] * phases
    density_gradient = 2 * np.einsum("pm,mn,xpn->px", values, density, derivatives)
    trace = np.einsum("pijj->pi", observed)
    error = np.max(np.abs(observed - expected))
    return {"max_abs_error": float(error), "expected": expected.tolist(), "observed": observed.tolist(),
            "basis_functions": n, "unrestricted": unrestricted, "AO_mapping_0based": mapping.tolist(),
            "overlap_error": float(np.max(np.abs(quick_s - transform(overlap)))),
            "electron_count": float(np.einsum("ij,ji->", density, quick_s)),
            "Poisson_trace_error": float(np.max(np.abs(trace + 4 * np.pi * density_gradient)))}


def raw_reference(model):
    if model["reference"] == "gaussian":
        return gaussian_hessian(model["points"], model["ao_exponent"])
    basis = {"He": [[l, [a, 1.]] for l, a in enumerate([.7, .5, .4, .3])]}
    molecule = gto.M(atom="He 0 0 0", basis=basis, cart=True, unit="Bohr", verbose=0)
    overlap = molecule.intor("int1e_ovlp", hermi=1)
    norm = np.sqrt(np.diag(overlap))
    density = np.asarray(model["density"])
    # Raw kernels define their tolerance-compatible symmetric matrix using
    # the upper triangle; a small admitted lower-triangle discrepancy is
    # deliberately exercised by the caller.
    density = np.triu(density) + np.triu(density, 1).T
    owned = np.asarray(model["owned"])
    partitioned = density * owned[np.maximum.outer(np.arange(len(owned)), np.arange(len(owned)))]
    output = []
    for point in model["points"]:
        integral = attraction_third(molecule, point) / norm[:, None] / norm[None, :]
        output.append(np.einsum("mn,ijknm->ijk", partitioned, integral))
    return np.asarray(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refuse to overwrite independent oracle output")
    request = json.loads(args.request.read_text())
    report = {"passed": False, "pyscf_version": pyscf.__version__, "python": sys.executable,
              "scope": "Analytic independent libcint third derivatives; no independent SCFs; exact QUICK density and basis",
              "molecular": {}, "raw": {}, "checks": []}

    def check(name, error, tolerance):
        report["checks"].append({"name": name, "error": float(error), "tolerance": tolerance,
                                 "passed": bool(np.isfinite(error) and error <= tolerance)})

    try:
        # Independent radial and displaced-second-integral checks establish
        # derivative signs, index ordering and all eight product-rule terms.
        molecule = gto.M(atom="He 0 0 0", basis={"He": [[0, [.7, 1.]]]}, cart=True, unit="Bohr", verbose=0)
        point = np.array([.41, -.33, .19])
        analytic = attraction_third(molecule, point)[:, :, :, 0, 0]
        check("libcint_vs_normalized_sGaussian", np.max(np.abs(analytic - gaussian_hessian([point], .7)[0])), 2e-12)
        for step in [1e-3, 5e-4]:
            for direction in range(3):
                offset = np.eye(3)[direction] * step
                fd = (attraction_second(molecule, point+offset) - attraction_second(molecule, point-offset)) / (2*step)
                check(f"libcint_vs_second_integral_FD:h{step}:axis{direction}",
                      np.max(np.abs(fd[:, :, 0, 0] - analytic[:, :, direction])), 3e-6)
        for name, item in request.get("molecular", {}).items():
            result = molecular_reference(item)
            report["molecular"][name] = result
            check(name + ":exact_cartesian_overlap", result["overlap_error"], 2e-10)
            check(name + ":all27_independent_H", result["max_abs_error"], 2e-8)
            check(name + ":Poisson_density_gradient_trace", result["Poisson_trace_error"], 2e-8)
        for name, item in request.get("raw", {}).items():
            expected = raw_reference(item["model"])
            observed = np.asarray(item["observed"])
            seed = item["model"].get("seed", 0.)
            error = float(np.max(np.abs(observed - seed - expected)))
            scale = float(np.max(np.abs(expected)))
            tolerance = max(2e-12 * scale, 2e-13) if item["model"]["reference"] != "gaussian" else max(2e-11 * scale, 1e-300)
            check(name + ":independent_raw", error, tolerance)
            if item["model"]["reference"] == "gaussian":
                for point in range(len(expected)):
                    point_scale = float(np.max(np.abs(expected[point])))
                    point_error = float(np.max(np.abs(observed[point] - seed - expected[point])))
                    check(name + f":Gaussian_point{point}", point_error, max(2e-11 * point_scale, 1e-300))
            report["raw"][name] = {"max_abs_error": error, "reference_scale": scale, "tolerance": tolerance}
        report["passed"] = all(item["passed"] for item in report["checks"])
    except Exception as error:
        import traceback
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    report["input_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [Path(__file__), args.request, Path(pyscf.__file__), Path(gto.__file__)]}
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": report["passed"], "checks": len(report["checks"]), "error": report.get("error")}), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
