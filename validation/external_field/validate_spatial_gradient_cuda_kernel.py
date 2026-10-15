#!/usr/bin/env python3
"""Validate native affine-field CUDA interfaces against Gaussian quadrature.

No QUICK job, SCF, saved fixture or existing runtime is changed. Every Cartesian
s/p/d/f monomial occurs on each of two centers. Reference integrals evaluate
the original Cartesian polynomials at Gauss--Hermite nodes; they do not use
the CPU xmoment routine or the CUDA moment recurrence. Nuclear-center reference
derivatives are central differences of density-contracted quadrature energies.

Run in a fresh work directory, with the target install on LD_LIBRARY_PATH::

    python validate_spatial_gradient_cuda_kernel.py \\
      --library /path/to/install/lib/libquick_cuda.so --workdir /new/evidence

The same command can be prefixed with ``compute-sanitizer --tool memcheck
--error-exitcode 86``. Atomic gradient additions can have different summation
orders; zero-tensor matrix parity is bitwise, gradient parity is numerical.
"""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
from __future__ import annotations

import argparse
import ctypes as ct
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

IP = ct.POINTER(ct.c_int)
DP = ct.POINTER(ct.c_double)
ERROR_CODE = 53
NODES = 16
MATRIX_TOLERANCE = 3.0e-12
GRADIENT_TOLERANCE = 3.0e-8
STEPS = (2.0e-4, 1.0e-4)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ptr(array: np.ndarray):
    return array.ctypes.data_as(IP if array.dtype == np.int32 else DP)


def double_factorial(value: int) -> int:
    return math.prod(range(value, 0, -2))


class Basis:
    def __init__(self):
        monomials = [power for degree in range(4)
                     for power in itertools.product(range(degree + 1), repeat=3)
                     if sum(power) == degree]
        self.angular = np.asfortranarray(np.array(monomials * 2, dtype=np.int32).T)
        self.nb, self.na, self.mp = self.angular.shape[1], 2, 2
        self.centers = np.array([1] * len(monomials) + [2] * len(monomials), dtype=np.int32)
        self.nc = np.full(self.nb, self.mp, dtype=np.int32)
        self.xyz = np.asfortranarray([[-0.53, 0.83], [0.24, -0.47], [-0.31, 0.62]])
        self.exponents = np.asfortranarray(np.array([
            [0.47 + 0.013 * i, 1.21 + 0.017 * i] for i in range(self.nb)]).T)
        self.coefficients = np.empty_like(self.exponents, order="F")
        for ao in range(self.nb):
            powers = self.angular[:, ao]
            degree = int(sum(powers))
            contraction = np.array([0.71 + 0.002 * ao, -0.19 - 0.001 * ao])
            alpha = self.exponents[:, ao]
            primitive_overlap = (2 * np.sqrt(alpha[:, None] * alpha[None, :]) /
                                 (alpha[:, None] + alpha[None, :])) ** (degree + 1.5)
            norm = math.sqrt(float(contraction @ primitive_overlap @ contraction))
            denominator = math.prod(double_factorial(2 * int(power) - 1) for power in powers)
            primitive_norm = ((2 * alpha / math.pi) ** 0.75 *
                              np.sqrt((4 * alpha) ** degree / denominator))
            self.coefficients[:, ao] = contraction * primitive_norm / norm


def quadrature_features(basis: Basis, xyz: np.ndarray):
    """Return overlap, three first moments, and six second-moment operators.

    Primitive Gaussian products are integrated numerically in each Cartesian
    direction. Six quadratic features multiply (Gxx,Gyy,Gzz,Gxy,Gxz,Gyz):
    diagonal features include one half; mixed features have coefficient one.
    """
    nodes, weights = np.polynomial.hermite.hermgauss(NODES)
    exponent = basis.exponents.T.reshape(-1)
    coefficient = basis.coefficients.T.reshape(-1)
    angular = np.repeat(basis.angular.T, basis.mp, axis=0)
    centers = np.repeat(xyz[:, basis.centers - 1].T, basis.mp, axis=0)
    alpha, beta = exponent[:, None], exponent[None, :]
    g = alpha + beta
    product_center = (alpha[:, :, None] * centers[:, None, :] +
                      beta[:, :, None] * centers[None, :, :]) / g[:, :, None]
    r = product_center[:, :, :, None] + nodes[None, None, None, :] / np.sqrt(g[:, :, None, None])
    polynomial = ((r - centers[:, None, :, None]) ** angular[:, None, :, None] *
                  (r - centers[None, :, :, None]) ** angular[None, :, :, None])
    s = r - ORIGIN[None, None, :, None]
    moments = [np.einsum("ijak,k->ija", polynomial * s ** degree, weights) /
               np.sqrt(g[:, :, None]) for degree in range(3)]
    distance2 = np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    prefactor = (np.exp(-alpha * beta / g * distance2) *
                 coefficient[:, None] * coefficient[None, :])

    def contract(primitive):
        return (primitive * prefactor).reshape(basis.nb, basis.mp, basis.nb, basis.mp).sum(axis=(1, 3))

    overlap = contract(np.prod(moments[0], axis=2))
    features = []
    for axis in range(3):
        values = moments[0].copy()
        values[:, :, axis] = moments[1][:, :, axis]
        features.append(contract(np.prod(values, axis=2)))
    for axis in range(3):
        values = moments[0].copy()
        values[:, :, axis] = moments[2][:, :, axis]
        features.append(0.5 * contract(np.prod(values, axis=2)))
    for i, j in ((0, 1), (0, 2), (1, 2)):
        values = moments[0].copy()
        values[:, :, i] = moments[1][:, :, i]
        values[:, :, j] = moments[1][:, :, j]
        features.append(contract(np.prod(values, axis=2)))
    return overlap, np.array(features)


ORIGIN = np.array([0.27, -0.31, 0.14], dtype=np.float64)


def feature_coefficients(field, tensor):
    return np.array([*field, tensor[0, 0], tensor[1, 1], tensor[2, 2],
                     tensor[0, 1], tensor[0, 2], tensor[1, 2]])


class Native:
    def __init__(self, library: Path, basis: Basis):
        self.library = ct.CDLL(str(library))
        self.basis = basis
        self.matrix = self.library.gpu_external_field_tensor_matrix_
        self.gradient = self.library.gpu_external_field_tensor_gradient_
        self.old_matrix = self.library.gpu_external_field_matrix_
        self.old_gradient = self.library.gpu_external_field_gradient_
        common = [IP, IP, IP, IP, IP, IP, DP, DP, DP, DP, DP]
        self.matrix.argtypes = common + [DP, IP, DP, DP, IP]
        self.gradient.argtypes = common + [DP, IP, DP, DP, DP, IP]
        self.old_matrix.argtypes = common + [IP, DP, DP, IP]
        self.old_gradient.argtypes = common + [IP, DP, DP, DP, IP]
        for function in (self.matrix, self.gradient, self.old_matrix, self.old_gradient):
            function.restype = None
        self.counter = self.library.gpu_external_field_launch_counts_
        self.counter.argtypes, self.counter.restype = [IP, IP], None

    def counts(self):
        matrix, gradient = ct.c_int(), ct.c_int()
        self.counter(ct.byref(matrix), ct.byref(gradient))
        return [matrix.value, gradient.value]

    def call(self, field, tensor, owned, initial, density=None, old=False, cutoff=0.0):
        basis = self.basis
        nb, na, mp, ierr = ct.c_int(basis.nb), ct.c_int(basis.na), ct.c_int(basis.mp), ct.c_int(-1)
        cutoff_c = ct.c_double(cutoff)
        field = np.asfortranarray(field, dtype=np.float64)
        tensor = np.asfortranarray(tensor, dtype=np.float64)
        owned = np.asfortranarray(owned, dtype=np.int32)
        output = np.array(initial, dtype=np.float64, order="F", copy=True)
        args = [ct.byref(nb), ct.byref(na), ct.byref(mp), ptr(basis.nc), ptr(basis.angular),
                ptr(basis.centers), ptr(basis.exponents), ptr(basis.coefficients), ptr(basis.xyz),
                ptr(field), ptr(ORIGIN)]
        if not old:
            args.append(ptr(tensor))
        args.extend([ptr(owned), ct.byref(cutoff_c)])
        if density is not None:
            density = np.asfortranarray(density, dtype=np.float64)
            args.append(ptr(density))
        args.extend([ptr(output), ct.byref(ierr)])
        function = ((self.old_gradient if old else self.gradient) if density is not None else
                    (self.old_matrix if old else self.matrix))
        function(*args)
        return ierr.value, output


def run(args, report):
    basis = Basis()
    native = Native(args.library, basis)
    rng = np.random.default_rng(20261015)
    matrix_initial = np.asfortranarray(rng.normal(scale=0.03, size=(basis.nb, basis.nb)))
    gradient_initial = np.asfortranarray(rng.normal(scale=0.03, size=(3, basis.na)))
    density = rng.normal(scale=0.025, size=(basis.nb, basis.nb))
    density = np.asfortranarray((density + density.T) / 2)
    owned = np.ones(basis.nb, dtype=np.int32)
    zero = np.zeros((3, 3), order="F")
    field = np.array([0.13, -0.09, 0.07])
    report["basis"] = {"nbasis": basis.nb, "ncenters": basis.na, "primitives_per_ao": basis.mp,
                       "cartesian_powers_per_center": basis.angular[:, :20].T.tolist(),
                       "coordinates_bohr": basis.xyz.T.tolist(), "origin_bohr": ORIGIN.tolist(),
                       "contractions_normalized": True, "quadrature_nodes_per_axis": NODES}

    def check(name, error=0.0, tolerance=0.0, **details):
        report["checks"].append({"name": name, "passed": bool(math.isfinite(error) and error <= tolerance),
                                 "max_abs_error": float(error) if math.isfinite(error) else None,
                                 "tolerance": tolerance, **details})

    def error(left, right):
        return float(np.max(np.abs(left - right)))

    overlap, central = quadrature_features(basis, basis.xyz)
    check("independent_contraction_normalization", error(np.diag(overlap), np.ones(basis.nb)), 2e-13)
    check("independent_quadrature_hermiticity", error(central, central.transpose(0, 2, 1)), 2e-13)
    # Reuse geometry-dependent quadrature features across every tensor case.
    numerical = {}
    ownership_numerical = {}
    ownership_masks = {"none": np.zeros(basis.nb, dtype=np.int32),
                       "even": (np.arange(basis.nb) % 2 == 0).astype(np.int32),
                       "odd": (np.arange(basis.nb) % 2 == 1).astype(np.int32)}
    lower_index = np.minimum.outer(np.arange(basis.nb), np.arange(basis.nb))
    for step in STEPS:
        derivatives = np.empty((3, basis.na, 9))
        ownership_derivatives = {label: np.empty_like(derivatives) for label in ("even", "odd")}
        for atom in range(basis.na):
            for axis in range(3):
                displaced = []
                for sign in (1, -1):
                    xyz = basis.xyz.copy()
                    xyz[axis, atom] += sign * step
                    _, features = quadrature_features(basis, xyz)
                    displaced.append(features)
                derivative = (displaced[0] - displaced[1]) / (2 * step)
                derivatives[axis, atom] = np.einsum("ij,kij->k", density, derivative)
                for label in ownership_derivatives:
                    weights = density * ownership_masks[label][lower_index]
                    ownership_derivatives[label][axis, atom] = np.einsum("ij,kij->k", weights, derivative)
        numerical[step] = derivatives
        ownership_numerical[step] = ownership_derivatives

    before = native.counts()
    for label, invalid in (("nan", np.full((3, 3), np.nan)),
                           ("inf", np.full((3, 3), np.inf)),
                           ("asymmetric", np.array([[0., 1e-30, 0.], [0., 0., 0.], [0., 0., 0.]]))):
        for gradient in (False, True):
            initial = gradient_initial if gradient else matrix_initial
            ierr, actual = native.call(field, invalid, owned, initial, density if gradient else None)
            suffix = "gradient" if gradient else "matrix"
            check(f"reject_{label}_{suffix}", float(ierr != ERROR_CODE), 0., ierr=ierr)
            check(f"transaction_{label}_{suffix}", float(not np.array_equal(actual, initial)))
    check("invalid_tensor_rejected_before_launch", error(np.array(native.counts()), np.array(before)))

    for gradient in (False, True):
        initial = gradient_initial if gradient else matrix_initial
        ierr, actual = native.call(field, zero, owned, initial, density if gradient else None)
        old_ierr, previous = native.call(field, zero, owned, initial, density if gradient else None, old=True)
        suffix = "gradient" if gradient else "matrix"
        check(f"zero_tensor_status_{suffix}", float(ierr != 0 or old_ierr != 0), ierr=ierr, old_ierr=old_ierr)
        check(f"zero_tensor_uniform_parity_{suffix}", error(actual, previous), 2e-12 if gradient else 0.,
              bitwise_equal=bool(np.array_equal(actual, previous)))

    cases = []
    for label, (i, j), strength in zip(("xx", "yy", "zz", "xy", "xz", "yz"),
                                      ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)),
                                      (0.07, -0.04, 0.03, 0.05, -0.06, 0.02)):
        tensor = zero.copy()
        tensor[i, j] = tensor[j, i] = strength
        cases.append(("pure_" + label, np.zeros(3), tensor))
    mixed = np.array([[0.07, 0.05, -0.06], [0.05, -0.04, 0.02], [-0.06, 0.02, 0.03]])
    cases.append(("mixed_traceful", field, mixed))
    source_free = mixed.copy()
    source_free[2, 2] = -source_free[0, 0] - source_free[1, 1]
    cases.append(("mixed_source_free", field, source_free))
    lower = np.tril(np.ones((basis.nb, basis.nb), dtype=bool))
    report["cases"] = []
    for label, applied, tensor in cases:
        coefficients = feature_coefficients(applied, tensor)
        reference = np.einsum("k,kij->ij", coefficients, central)
        before = native.counts()
        ierr, actual = native.call(applied, tensor, owned, matrix_initial)
        check(label + "_matrix_status", float(ierr != 0), ierr=ierr)
        expected = matrix_initial.copy()
        expected[lower] += reference[lower]
        check(label + "_matrix_quadrature", error(actual, expected), MATRIX_TOLERANCE)
        check(label + "_upper_triangle_unchanged", error(actual[~lower], matrix_initial[~lower]))
        ierr, actual_gradient = native.call(applied, tensor, owned, gradient_initial, density)
        check(label + "_gradient_status", float(ierr != 0), ierr=ierr)
        electronic = actual_gradient - gradient_initial
        for step in STEPS:
            reference_gradient = np.einsum("iak,k->ia", numerical[step], coefficients)
            check(label + f"_gradient_quadrature_fd_h{step:g}", error(electronic, reference_gradient),
                  GRADIENT_TOLERANCE, step_bohr=step)
        check(label + "_native_matrix_and_gradient_launch", error(np.array(native.counts()) - before, np.ones(2)))
        report["cases"].append({"name": label, "field_au": applied.tolist(), "tensor_au": tensor.tolist(),
                                "gradient_au": electronic.T.tolist()})

    # The complete triangular AO-pair set partitions by its first AO index.
    partial_gradients = []
    reference = np.einsum("k,kij->ij", feature_coefficients(field, mixed), central)
    for label, mask in ownership_masks.items():
        before = native.counts()
        ierr, actual = native.call(field, mixed, mask, matrix_initial)
        expected = matrix_initial.copy()
        selected = lower & mask[None, :].astype(bool)
        expected[selected] += reference[selected]
        check(label + "_owned_matrix_status", float(ierr != 0), ierr=ierr)
        check(label + "_owned_matrix_quadrature", error(actual, expected), MATRIX_TOLERANCE)
        ierr, actual_gradient = native.call(field, mixed, mask, gradient_initial, density)
        check(label + "_owned_gradient_status", float(ierr != 0), ierr=ierr)
        if label == "none":
            check("no_owned_gradient_unchanged", error(actual_gradient, gradient_initial))
            check("no_owned_no_launches", error(np.array(native.counts()), np.array(before)))
        else:
            partial_gradients.append(actual_gradient - gradient_initial)
            for step in STEPS:
                expected_gradient = np.einsum("iak,k->ia", ownership_numerical[step][label],
                                              feature_coefficients(field, mixed))
                check(label + f"_owned_gradient_fd_h{step:g}",
                      error(actual_gradient - gradient_initial, expected_gradient), GRADIENT_TOLERANCE,
                      step_bohr=step)
    coefficients = feature_coefficients(field, mixed)
    reference_gradient = np.einsum("iak,k->ia", numerical[STEPS[-1]], coefficients)
    check("owned_gradient_partition_quadrature_fd", error(sum(partial_gradients), reference_gradient),
          GRADIENT_TOLERANCE)

    for gradient in (False, True):
        initial = gradient_initial if gradient else matrix_initial
        ierr, actual = native.call(field, mixed, owned, initial, density if gradient else None, cutoff=1e99)
        suffix = "gradient" if gradient else "matrix"
        check("screened_status_" + suffix, float(ierr != 0), ierr=ierr)
        check("screened_no_contribution_" + suffix, error(actual, initial))

    # A device arithmetic failure must leave caller storage unchanged as well.
    overflow = np.eye(3) * 1e308
    for gradient in (False, True):
        initial = gradient_initial if gradient else matrix_initial
        ierr, actual = native.call(np.zeros(3), overflow, owned, initial, density if gradient else None)
        suffix = "gradient" if gradient else "matrix"
        check("device_overflow_status_" + suffix, float(ierr != ERROR_CODE), ierr=ierr)
        check("device_overflow_transaction_" + suffix, float(not np.array_equal(actual, initial)))
    report["launch_counts_final"] = native.counts()
    report["passed"] = all(check["passed"] for check in report["checks"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    args.library = args.library.resolve(strict=True)
    args.workdir = args.workdir.resolve()
    args.workdir.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[2] / "src/gpu/gpu_external_field.cu"
    report = {"schema": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
              "library": str(args.library), "library_sha256": sha256(args.library),
              "validator_sha256": sha256(Path(__file__)),
              "source_sha256": sha256(source), "checks": [], "passed": False,
              "scope": "Raw electronic AO matrix/center gradient; no SCF, Pulay or classical charge terms"}
    start = time.monotonic()
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                             text=True, capture_output=True, check=False)
        report["gpu_identity"] = gpu.stdout.strip()
        run(args, report)
    except Exception:
        report["exception"] = traceback.format_exc()
    report["wall_seconds"] = time.monotonic() - start
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (args.workdir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    failures = [check["name"] for check in report["checks"] if not check["passed"]]
    print(json.dumps({"passed": report["passed"], "checks": len(report["checks"]),
                      "failed_checks": failures, "wall_seconds": report["wall_seconds"],
                      "report": str(args.workdir / "report.json")}, indent=2))
    if report.get("exception"):
        print(report["exception"], file=sys.stderr)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
