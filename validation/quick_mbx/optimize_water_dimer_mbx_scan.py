#!/usr/bin/env python3
"""Constrained MB-pol water-dimer optimizations for QUICK-MBX scans.

MBX provides fast analytic gradients but its bundled `optimize` executable is
unconstrained. For a PES scan we need to keep the scanned O--O distance fixed,
so this script uses MBX gradients with a small projected L-BFGS optimizer. The
constraint is |O(4)-O(1)| = R; all other Cartesian degrees of freedom relax.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


Atom = tuple[str, float, float, float]

MONOMER_A_REF: list[Atom] = [
    ("O", -1.58972425, 1.04337922, -0.08780840),
    ("H", -0.63591971, 0.97898520, 0.00000000),
    ("H", -1.90066280, 1.74501050, -0.66454990),
]
MONOMER_B_REF: list[Atom] = [
    ("O", 1.64924507, 1.08594656, 0.00000000),
    ("H", 2.60878026, 1.09587704, -0.02817115),
    ("H", 1.33830653, 1.78757784, 0.57674150),
]
DEFAULT_R_VALUES = [
    2.50, 2.60, 2.70, 2.80, 2.90,
    3.00, 3.10, 3.20, 3.30, 3.40,
    3.50, 3.60, 3.70, 3.80, 3.90,
    4.00, 4.20, 4.40, 4.70, 5.00,
]


@dataclass
class OptResult:
    point: int
    target_r: float
    actual_r: float
    energy_kcal: float
    initial_energy_kcal: float
    steps: int
    converged: bool
    max_force: float
    rms_force: float
    coords: np.ndarray


def import_mbx_binding(mbx_source: Path):
    binding_dir = mbx_source / "plugins" / "ase" / "ase_mbx"
    if not (binding_dir / "mbx_binding.py").exists():
        raise FileNotFoundError(f"Could not find MBX Python binding under {binding_dir}")
    sys.path.insert(0, str(binding_dir))
    from mbx_binding import MBXLibrary  # type: ignore

    return MBXLibrary


def atoms_to_array(atoms: list[Atom]) -> np.ndarray:
    return np.array([[x, y, z] for _, x, y, z in atoms], dtype=float)


def array_to_atoms(coords: np.ndarray) -> list[Atom]:
    symbols = ["O", "H", "H", "O", "H", "H"]
    return [(sym, float(x), float(y), float(z)) for sym, (x, y, z) in zip(symbols, coords)]


def oo_distance(coords: np.ndarray) -> float:
    return float(np.linalg.norm(coords[3] - coords[0]))


def reference_dimer_at_distance(distance: float) -> np.ndarray:
    atoms = MONOMER_A_REF + MONOMER_B_REF
    coords = atoms_to_array(atoms)
    return set_oo_distance(coords, distance)


def set_oo_distance(coords: np.ndarray, distance: float) -> np.ndarray:
    coords = np.array(coords, dtype=float, copy=True)
    rvec = coords[3] - coords[0]
    norm = np.linalg.norm(rvec)
    if norm < 1.0e-12:
        raise ValueError("Cannot set O-O distance from coincident oxygen atoms.")
    unit = rvec / norm
    delta = distance - norm
    coords[0] -= 0.5 * delta * unit
    coords[3] += 0.5 * delta * unit
    return coords


def recenter(coords: np.ndarray) -> np.ndarray:
    return coords - coords.mean(axis=0)


def project_constraint_vector(vec: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Project a Cartesian vector onto the tangent space of fixed O--O distance."""

    out = np.array(vec, dtype=float, copy=True)
    rvec = coords[3] - coords[0]
    norm = np.linalg.norm(rvec)
    if norm < 1.0e-12:
        return out
    unit = rvec / norm
    # Constraint gradient: dC/dO1 = -u, dC/dO2 = +u.
    component = np.dot(out[3] - out[0], unit) / 2.0
    out[0] += component * unit
    out[3] -= component * unit
    return out


def remove_translation(vec: np.ndarray) -> np.ndarray:
    return vec - vec.mean(axis=0)


class MBXWaterDimer:
    def __init__(self, mbx_home: Path, mbx_source: Path, mbx_json: Path):
        MBXLibrary = import_mbx_binding(mbx_source)
        self.lib = MBXLibrary(mbx_home=str(mbx_home))
        self.lib.initialize_system(
            np.zeros((6, 3), dtype=float),
            [3, 3],
            ["O", "H", "H", "O", "H", "H"],
            ["h2o", "h2o"],
            str(mbx_json),
        )

    def energy_grad(self, coords: np.ndarray) -> tuple[float, np.ndarray]:
        energy, grad = self.lib.get_energy_forces(np.ascontiguousarray(coords, dtype=float))
        return float(energy), np.array(grad, dtype=float)

    def finalize(self) -> None:
        self.lib.finalize()


def lbfgs_direction(grad: np.ndarray, history: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    if not history:
        return -grad

    q = grad.copy()
    alphas: list[float] = []
    rhos: list[float] = []
    for s, y in reversed(history):
        sy = float(np.dot(s, y))
        if sy <= 1.0e-14:
            alphas.append(0.0)
            rhos.append(0.0)
            continue
        rho = 1.0 / sy
        alpha = rho * float(np.dot(s, q))
        q -= alpha * y
        alphas.append(alpha)
        rhos.append(rho)

    s_last, y_last = history[-1]
    yy = float(np.dot(y_last, y_last))
    gamma = float(np.dot(s_last, y_last)) / yy if yy > 1.0e-14 else 1.0
    r = gamma * q

    for (s, y), alpha, rho in zip(history, reversed(alphas), reversed(rhos)):
        if rho == 0.0:
            continue
        beta = rho * float(np.dot(y, r))
        r += s * (alpha - beta)
    return -r


def optimize_point(
    model: MBXWaterDimer,
    coords0: np.ndarray,
    target_r: float,
    point: int,
    max_steps: int,
    force_tol: float,
    max_disp: float,
    history_size: int,
) -> OptResult:
    coords = recenter(set_oo_distance(coords0, target_r))
    energy, grad = model.energy_grad(coords)
    initial_energy = energy
    grad = remove_translation(project_constraint_vector(grad, coords))
    history: list[tuple[np.ndarray, np.ndarray]] = []

    converged = False
    max_force = float(np.max(np.linalg.norm(grad, axis=1)))
    rms_force = float(np.sqrt(np.mean(grad * grad)))

    for step in range(1, max_steps + 1):
        max_force = float(np.max(np.linalg.norm(grad, axis=1)))
        rms_force = float(np.sqrt(np.mean(grad * grad)))
        if max_force < force_tol or rms_force < 0.5 * force_tol:
            converged = True
            break

        flat_grad = grad.reshape(-1)
        direction = lbfgs_direction(flat_grad, history).reshape((6, 3))
        direction = remove_translation(project_constraint_vector(direction, coords))
        dnorms = np.linalg.norm(direction, axis=1)
        dmax = float(np.max(dnorms))
        if dmax < 1.0e-14:
            direction = -grad
            dmax = float(np.max(np.linalg.norm(direction, axis=1)))
        direction *= min(1.0, max_disp / max(dmax, 1.0e-14))

        slope = float(np.dot(flat_grad, direction.reshape(-1)))
        if slope >= 0.0:
            history.clear()
            direction = -grad
            direction = remove_translation(project_constraint_vector(direction, coords))
            dmax = float(np.max(np.linalg.norm(direction, axis=1)))
            direction *= min(1.0, max_disp / max(dmax, 1.0e-14))
            slope = float(np.dot(flat_grad, direction.reshape(-1)))

        accepted = False
        step_scale = 1.0
        for _ in range(24):
            trial = recenter(set_oo_distance(coords + step_scale * direction, target_r))
            trial_energy, trial_grad = model.energy_grad(trial)
            if trial_energy <= energy + 1.0e-4 * step_scale * slope:
                trial_grad = remove_translation(project_constraint_vector(trial_grad, trial))
                s = (trial - coords).reshape(-1)
                y = (trial_grad - grad).reshape(-1)
                if np.dot(s, y) > 1.0e-12:
                    history.append((s, y))
                    history = history[-history_size:]
                coords, energy, grad = trial, trial_energy, trial_grad
                accepted = True
                break
            step_scale *= 0.5

        if not accepted:
            history.clear()
            max_disp *= 0.5
            if max_disp < 1.0e-5:
                break

    return OptResult(
        point=point,
        target_r=target_r,
        actual_r=oo_distance(coords),
        energy_kcal=energy,
        initial_energy_kcal=initial_energy,
        steps=step if "step" in locals() else 0,
        converged=converged,
        max_force=max_force,
        rms_force=rms_force,
        coords=coords,
    )


def parse_r_values(text: str | None) -> list[float]:
    if not text:
        return DEFAULT_R_VALUES
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("No R values parsed.")
    return values


def write_xyz(path: Path, results: list[OptResult]) -> None:
    lines: list[str] = []
    for result in results:
        lines.append("6")
        status = "converged" if result.converged else "not_converged"
        lines.append(
            f"point={result.point} target_R_OO_ang={result.target_r:.5f} "
            f"actual_R_OO_ang={result.actual_r:.5f} status={status} "
            f"steps={result.steps} mbpol_energy_kcal={result.energy_kcal:.12f} "
            f"max_projected_force={result.max_force:.6e} rms_projected_force={result.rms_force:.6e}"
        )
        for sym, x, y, z in array_to_atoms(result.coords):
            lines.append(f"{sym:2s} {x:18.10f} {y:18.10f} {z:18.10f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(path: Path, results: list[OptResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "point",
                "target_oo_ang",
                "actual_oo_ang",
                "converged",
                "steps",
                "initial_mbpol_energy_kcal",
                "final_mbpol_energy_kcal",
                "max_projected_force_kcal_mol_a",
                "rms_projected_force_kcal_mol_a",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "point": result.point,
                    "target_oo_ang": f"{result.target_r:.8f}",
                    "actual_oo_ang": f"{result.actual_r:.8f}",
                    "converged": int(result.converged),
                    "steps": result.steps,
                    "initial_mbpol_energy_kcal": f"{result.initial_energy_kcal:.12f}",
                    "final_mbpol_energy_kcal": f"{result.energy_kcal:.12f}",
                    "max_projected_force_kcal_mol_a": f"{result.max_force:.8e}",
                    "rms_projected_force_kcal_mol_a": f"{result.rms_force:.8e}",
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mbx-home", type=Path, default=Path("/private/tmp/mbx-quick-install"))
    parser.add_argument("--mbx-source", type=Path, default=Path("/private/tmp/MBX-quick-plan"))
    parser.add_argument("--mbx-json", type=Path, default=Path("/private/tmp/MBX-quick-plan/examples/PEFs/001_mbpol/C++/mbx.json"))
    parser.add_argument("--workdir", type=Path, default=Path("/private/tmp/quick-mbx-water-mbpol-opt"))
    parser.add_argument("--r-values", help="Comma-separated O-O distances in Angstrom.")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--force-tol", type=float, default=2.0e-4)
    parser.add_argument("--max-disp", type=float, default=0.08)
    parser.add_argument("--history-size", type=int, default=8)
    parser.add_argument("--reuse-workdir", action="store_true")
    parser.add_argument(
        "--xyz",
        type=Path,
        default=Path("validation/quick_mbx/water_dimer_mbpol_optimized_20pt.xyz"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("validation/quick_mbx/water_dimer_mbpol_optimized_20pt_summary.csv"),
    )
    args = parser.parse_args()

    if args.workdir.exists() and not args.reuse_workdir:
        shutil.rmtree(args.workdir)
    args.workdir.mkdir(parents=True, exist_ok=True)

    r_values = parse_r_values(args.r_values)
    model = MBXWaterDimer(args.mbx_home, args.mbx_source, args.mbx_json)
    results: list[OptResult] = []
    previous: np.ndarray | None = None

    try:
        for point, target_r in enumerate(r_values, start=1):
            coords0 = reference_dimer_at_distance(target_r) if previous is None else set_oo_distance(previous, target_r)
            result = optimize_point(
                model,
                coords0,
                target_r=target_r,
                point=point,
                max_steps=args.max_steps,
                force_tol=args.force_tol,
                max_disp=args.max_disp,
                history_size=args.history_size,
            )
            results.append(result)
            previous = result.coords
            print(
                f"point={point:02d} R={target_r:.3f} E={result.energy_kcal:.8f} "
                f"steps={result.steps} maxF={result.max_force:.3e} rmsF={result.rms_force:.3e} "
                f"{'converged' if result.converged else 'not_converged'}",
                flush=True,
            )
    finally:
        model.finalize()

    write_xyz(args.xyz, results)
    write_summary(args.summary_csv, results)
    print(f"Wrote {args.xyz}")
    print(f"Wrote {args.summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
