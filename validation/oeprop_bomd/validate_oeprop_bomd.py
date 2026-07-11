#!/usr/bin/env python3
# (C) Copyright 2026 QUICK contributors
# All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.

"""Validate deterministic QUICK BOMD with on-the-fly ESP, EFIELD, and EFG."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shlex
import subprocess


PROPERTY_COLUMNS = [
    "esp",
    "efield_x",
    "efield_y",
    "efield_z",
    "efg_xx",
    "efg_xy",
    "efg_xz",
    "efg_yx",
    "efg_yy",
    "efg_yz",
    "efg_zx",
    "efg_zy",
    "efg_zz",
]
HF_GRADIENT = (
    "HF BASIS=STO-3G CUTOFF=1.0D-12 GRADCUTOFF=1.0D-12 "
    "DENSERMS=1.0D-10 GRADIENT CHARGE=0"
)
HF_ENERGY = "HF BASIS=STO-3G CUTOFF=1.0D-12 DENSERMS=1.0D-10 ENERGY CHARGE=0"
PBE0_GRADIENT = (
    "DFT PBE0 D3 BASIS=DEF2-SVPD CUTOFF=1.0D-12 GRADCUTOFF=1.0D-12 "
    "XCCUTOFF=1.0D-10 BASISCUTOFF=1.0D-10 DENSERMS=1.0D-9 GRADIENT CHARGE=0"
)


def run(command: list[str], cwd: Path, env: dict[str, str], log: Path) -> None:
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def write_control(
    path: Path,
    *,
    geometry: Path | None,
    probes: Path | None,
    prefix: Path,
    mode: str,
    steps: int,
    dt_fs: float = 0.05,
    keyword: str = HF_GRADIENT,
    restart: Path | None = None,
    checkpoint: Path | None = None,
    coordinate_atoms: tuple[int, int] = (2, 3),
) -> None:
    values = ["&quick_bomd"]
    if geometry is not None:
        values.append(f" geometry_file='{geometry}'")
    if probes is not None:
        values.append(f" probe_file='{probes}'")
    values.extend(
        [
            f" output_prefix='{prefix}'",
            f" checkpoint_file='{checkpoint or Path(str(prefix) + '.chk')}'",
            f" keywd='{keyword}'",
            f" probe_mode='{mode}'",
            f" nsteps={steps}",
            f" dt_fs={dt_fs:.12g}",
            " temperature_k=50.0",
            f" coordinate_atoms={coordinate_atoms[0]},{coordinate_atoms[1]}",
        ]
    )
    if restart is not None:
        values.append(f" restart_file='{restart}'")
    if mode == "molecule":
        values.append(" anchor_atoms=1,2,3")
    if mode == "density":
        values.extend(
            [
                " density_isovalue=1.0D-3",
                " density_spacing_ang=1.0",
                " density_max_radius_ang=8.0",
            ]
        )
    values.append("/")
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def max_property_difference(first: list[dict[str, str]], second: list[dict[str, str]]) -> float:
    if len(first) != len(second):
        raise RuntimeError(f"Property row mismatch: {len(first)} vs {len(second)}")
    return max(
        abs(float(row_a[name]) - float(row_b[name]))
        for row_a, row_b in zip(first, second)
        for name in PROPERTY_COLUMNS
    )


def read_checkpoint(path: Path) -> tuple[list[float], list[float]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "QUICK_OEPROP_BOMD_CHECKPOINT_V1":
        raise RuntimeError(f"Invalid checkpoint: {path}")
    coordinates: list[float] = []
    velocities: list[float] = []
    for line in lines[3:]:
        fields = line.split()
        coordinates.extend(float(value) for value in fields[2:5])
        velocities.extend(float(value) for value in fields[5:8])
    return coordinates, velocities


def max_difference(first: list[float], second: list[float]) -> float:
    if len(first) != len(second):
        raise RuntimeError("Vector length mismatch")
    return max(abs(a - b) for a, b in zip(first, second))


def validate(args: argparse.Namespace) -> dict[str, object]:
    workdir = args.workdir.resolve()
    repo = args.repo_root.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("QUICK_BASIS", str(repo / "basis"))
    water = repo / "validation/oeprop_bomd/water.xyz"
    trimer = repo / "validation/oeprop_bomd/water_trimer.xyz"
    fixed_probes = repo / "validation/oeprop_bomd/water_fixed_probes.dat"
    molecule_probes = repo / "validation/oeprop_bomd/water_molecule_probes.dat"
    atom_probes = repo / "validation/oeprop_bomd/water_atom_probes.dat"

    metrics: dict[str, float | int | bool] = {}
    serial_driver = str(args.bomd_driver.resolve())

    full_prefix = workdir / "full"
    full_control = workdir / "full.nml"
    write_control(full_control, geometry=water, probes=fixed_probes, prefix=full_prefix, mode="fixed", steps=4)
    run([serial_driver, str(full_control)], workdir, env, workdir / "full.log")
    full_summary = json.loads(Path(str(full_prefix) + "_summary.json").read_text(encoding="utf-8"))
    metrics["fixed_nve_max_drift_au"] = full_summary["max_abs_drift_au"]

    postprocessed = workdir / "postprocessed.csv"
    run(
        [
            str(args.oeprop_driver.resolve()),
            "trajectory",
            str(Path(str(full_prefix) + "_trajectory.xyz")),
            str(fixed_probes),
            str(postprocessed),
            HF_ENERGY,
            "oeprop_postprocessed",
        ],
        workdir,
        env,
        workdir / "postprocessed.log",
    )
    online_rows = read_csv(Path(str(full_prefix) + "_oeprop.csv"))
    post_rows = read_csv(postprocessed)
    metrics["on_the_fly_vs_post_max_abs"] = max_property_difference(online_rows, post_rows)

    split_a = workdir / "split_a"
    split_checkpoint = workdir / "split.chk"
    split_a_control = workdir / "split_a.nml"
    write_control(
        split_a_control,
        geometry=water,
        probes=fixed_probes,
        prefix=split_a,
        mode="fixed",
        steps=2,
        checkpoint=split_checkpoint,
    )
    run([serial_driver, str(split_a_control)], workdir, env, workdir / "split_a.log")
    split_b = workdir / "split_b"
    split_final = workdir / "split_final.chk"
    split_b_control = workdir / "split_b.nml"
    write_control(
        split_b_control,
        geometry=None,
        probes=fixed_probes,
        prefix=split_b,
        mode="fixed",
        steps=2,
        restart=split_checkpoint,
        checkpoint=split_final,
    )
    run([serial_driver, str(split_b_control)], workdir, env, workdir / "split_b.log")
    full_coords, full_velocities = read_checkpoint(Path(str(full_prefix) + ".chk"))
    split_coords, split_velocities = read_checkpoint(split_final)
    metrics["restart_coordinate_max_abs_ang"] = max_difference(full_coords, split_coords)
    metrics["restart_velocity_max_abs_au"] = max_difference(full_velocities, split_velocities)
    full_final = [row for row in online_rows if int(row["step"]) == 4]
    split_final_rows = [row for row in read_csv(Path(str(split_b) + "_oeprop.csv")) if int(row["step"]) == 4]
    metrics["restart_oeprop_max_abs"] = max_property_difference(full_final, split_final_rows)

    for mode, probes in (("molecule", molecule_probes), ("atom", atom_probes), ("density", None)):
        prefix = workdir / mode
        control = workdir / f"{mode}.nml"
        write_control(control, geometry=water, probes=probes, prefix=prefix, mode=mode, steps=1)
        run([serial_driver, str(control)], workdir, env, workdir / f"{mode}.log")
        rows = read_csv(Path(str(prefix) + "_oeprop.csv"))
        metrics[f"{mode}_rows"] = len(rows)
        if mode != "density":
            first = [row for row in rows if int(row["step"]) == 0 and int(row["probe"]) == 1][0]
            second = [row for row in rows if int(row["step"]) == 1 and int(row["probe"]) == 1][0]
            displacement = math.sqrt(sum((float(first[name]) - float(second[name])) ** 2 for name in ("x_ang", "y_ang", "z_ang")))
            metrics[f"{mode}_probe_displacement_ang"] = displacement

    trimer_prefix = workdir / "trimer"
    trimer_control = workdir / "trimer.nml"
    write_control(
        trimer_control,
        geometry=trimer,
        probes=fixed_probes,
        prefix=trimer_prefix,
        mode="fixed",
        steps=2,
        dt_fs=0.025,
        coordinate_atoms=(1, 4),
    )
    run([serial_driver, str(trimer_control)], workdir, env, workdir / "trimer.log")
    trimer_summary = json.loads(Path(str(trimer_prefix) + "_summary.json").read_text(encoding="utf-8"))
    metrics["trimer_nve_max_drift_au"] = trimer_summary["max_abs_drift_au"]

    if args.run_pbe0:
        pbe0_prefix = workdir / "pbe0"
        pbe0_control = workdir / "pbe0.nml"
        write_control(
            pbe0_control,
            geometry=water,
            probes=fixed_probes,
            prefix=pbe0_prefix,
            mode="fixed",
            steps=1,
            dt_fs=0.025,
            keyword=PBE0_GRADIENT,
        )
        run([serial_driver, str(pbe0_control)], workdir, env, workdir / "pbe0.log")
        pbe0_summary = json.loads(Path(str(pbe0_prefix) + "_summary.json").read_text(encoding="utf-8"))
        metrics["pbe0_nve_max_drift_au"] = pbe0_summary["max_abs_drift_au"]

    if args.mpi_driver:
        mpi_prefix = workdir / "mpi"
        mpi_control = workdir / "mpi.nml"
        write_control(mpi_control, geometry=water, probes=fixed_probes, prefix=mpi_prefix, mode="fixed", steps=2)
        run(
            shlex.split(args.launcher) + [str(args.mpi_driver.resolve()), str(mpi_control)],
            workdir,
            env,
            workdir / "mpi.log",
        )
        serial_step_two = [row for row in online_rows if int(row["step"]) <= 2]
        mpi_rows = read_csv(Path(str(mpi_prefix) + "_oeprop.csv"))
        metrics["serial_vs_mpi_oeprop_max_abs"] = max_property_difference(serial_step_two, mpi_rows)

    efg_asymmetry = max(
        abs(float(row[first]) - float(row[second]))
        for row in online_rows
        for first, second in (("efg_xy", "efg_yx"), ("efg_xz", "efg_zx"), ("efg_yz", "efg_zy"))
    )
    eigen_trace_error = max(
        abs(
            float(row["efg_trace"])
            - sum(float(row[f"efg_eigenvalue_{index}"]) for index in (1, 2, 3))
        )
        for row in online_rows
    )
    metrics["efg_symmetry_max_abs"] = efg_asymmetry
    metrics["efg_eigenvalue_trace_max_abs"] = eigen_trace_error

    checks = {
        "fixed_nve": float(metrics["fixed_nve_max_drift_au"]) <= args.drift_tolerance,
        "on_the_fly_parity": float(metrics["on_the_fly_vs_post_max_abs"]) <= args.property_tolerance,
        "restart_coordinates": float(metrics["restart_coordinate_max_abs_ang"]) <= args.restart_tolerance,
        "restart_velocities": float(metrics["restart_velocity_max_abs_au"]) <= args.restart_tolerance,
        "restart_oeprop": float(metrics["restart_oeprop_max_abs"]) <= args.property_tolerance,
        "moving_probes": float(metrics["molecule_probe_displacement_ang"]) > 0.0
        and float(metrics["atom_probe_displacement_ang"]) > 0.0,
        "density_surface": int(metrics["density_rows"]) > 0,
        "trimer_nve": float(metrics["trimer_nve_max_drift_au"]) <= args.drift_tolerance,
        "efg_symmetry": efg_asymmetry <= args.property_tolerance,
        "efg_eigensystem": eigen_trace_error <= args.property_tolerance,
    }
    if args.run_pbe0:
        checks["pbe0_nve"] = float(metrics["pbe0_nve_max_drift_au"]) <= args.drift_tolerance
    if args.mpi_driver:
        checks["serial_mpi"] = float(metrics["serial_vs_mpi_oeprop_max_abs"]) <= args.property_tolerance

    result: dict[str, object] = {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "workdir": str(workdir),
    }
    (workdir / "validation_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bomd-driver", type=Path, required=True)
    parser.add_argument("--oeprop-driver", type=Path, required=True)
    parser.add_argument("--mpi-driver", type=Path)
    parser.add_argument("--launcher", default="mpirun -np 2")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--workdir", type=Path, default=Path("/private/tmp/quick_oeprop_bomd_validation"))
    parser.add_argument("--run-pbe0", action="store_true")
    parser.add_argument("--drift-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--property-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--restart-tolerance", type=float, default=1.0e-10)
    args = parser.parse_args()
    result = validate(args)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
