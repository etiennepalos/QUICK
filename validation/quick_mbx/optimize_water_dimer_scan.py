#!/usr/bin/env python3
"""Constrained B3LYP-D3/aug-cc-pVTZ water-dimer optimizations for a PES scan.

Each point fixes the O(1)--O(4) distance at the starting value with QUICK's
DL-FIND `CONSTRAIN` block and relaxes the remaining geometry. The optimized
frames are written as a repeated XYZ stack for the fixed-geometry QUICK-MBX
validation driver, plus a CSV with one row per atom.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
from pathlib import Path


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
DEFAULT_R_VALUES = [2.55, 2.70, 2.85, 3.00, 3.15, 3.30, 3.60, 4.00, 4.50, 5.00]
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?")


def oo_distance(atoms: list[Atom]) -> float:
    _, ax, ay, az = atoms[0]
    _, bx, by, bz = atoms[3]
    return ((bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2) ** 0.5


def shift_second_water_to_distance(atoms: list[Atom], distance: float) -> list[Atom]:
    ax, ay, az = atoms[0][1:]
    bx, by, bz = atoms[3][1:]
    vx, vy, vz = bx - ax, by - ay, bz - az
    norm = (vx * vx + vy * vy + vz * vz) ** 0.5
    ux, uy, uz = vx / norm, vy / norm, vz / norm
    target = (ax + distance * ux, ay + distance * uy, az + distance * uz)
    sx, sy, sz = target[0] - bx, target[1] - by, target[2] - bz
    shifted = atoms[:3]
    shifted += [(sym, x + sx, y + sy, z + sz) for sym, x, y, z in atoms[3:]]
    return shifted


def reference_dimer_at_distance(distance: float) -> list[Atom]:
    return shift_second_water_to_distance(MONOMER_A_REF + MONOMER_B_REF, distance)


def write_quick_input(path: Path, atoms: list[Atom], max_cycles: int) -> None:
    keywd = (
        "DFT B3LYP D3 BASIS=AUG-CC-PVTZ CONSTRAIN "
        "CUTOFF=1.0D-9 XCCUTOFF=1.0D-8 BASISCUTOFF=1.0D-8 "
        f"DENSERMS=1.0D-6 OPTIMIZE={max_cycles} ETOL=1.0D-5 GTOL=3.0D-4 DIPOLE"
    )
    lines = [keywd, ""]
    lines += [f"{sym:2s} {x:18.10f} {y:18.10f} {z:18.10f}" for sym, x, y, z in atoms]
    lines += ["", "DISTANCE 1 4", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_quick(quick_exe: Path, input_path: Path, cwd: Path, env: dict[str, str]) -> tuple[str, Path]:
    result = subprocess.run(
        [str(quick_exe), input_path.name],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    out_path = find_quick_output(cwd, input_path)
    stdout_path = cwd / f"{input_path.stem}.stdout"
    stderr_path = cwd / f"{input_path.stem}.stderr"
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"QUICK failed for {input_path.name} with exit code {result.returncode}. "
            f"See {stdout_path} and {stderr_path}."
        )
    if out_path.exists():
        return out_path.read_text(encoding="utf-8", errors="replace"), out_path
    return result.stdout, stdout_path


def find_quick_output(cwd: Path, input_path: Path) -> Path:
    """Return QUICK's output path, accounting for stems containing periods."""

    candidates = [cwd / f"{input_path.stem}.out"]
    if "." in input_path.stem:
        candidates.append(cwd / f"{input_path.stem.split('.')[0]}.out")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    outputs = sorted(cwd.glob("*.out"), key=lambda item: item.stat().st_mtime, reverse=True)
    if outputs:
        return outputs[0]
    return candidates[0]


def parse_optimized_geometry(text: str, natom: int = 6) -> tuple[list[Atom], bool, int | None, float | None]:
    marker = "OPTIMIZED GEOMETRY IN CARTESIAN"
    pos = text.rfind(marker)
    if pos < 0:
        raise RuntimeError("Could not find optimized geometry block in QUICK output.")
    tail = text[pos:].splitlines()[1:]
    atoms: list[Atom] = []
    atom_re = re.compile(r"^\s*([A-Za-z]{1,2})\s+(" + NUMBER_RE.pattern + r")\s+(" + NUMBER_RE.pattern + r")\s+(" + NUMBER_RE.pattern + r")")
    for line in tail:
        if line.strip().upper().startswith("FORCE"):
            break
        match = atom_re.match(line)
        if match:
            atoms.append((match.group(1), float(match.group(2)), float(match.group(3)), float(match.group(4))))
            if len(atoms) == natom:
                break
    if len(atoms) != natom:
        raise RuntimeError(f"Expected {natom} optimized atoms, parsed {len(atoms)}.")

    converged = "GEOMETRY OPTIMIZED AFTER" in text
    cycles_match = re.search(r"GEOMETRY OPTIMIZED AFTER\s+(\d+)\s+CYCLES", text)
    energy_match = re.search(r"MINIMIZED ENERGY\s*=\s*(" + NUMBER_RE.pattern + r")", text)
    cycles = int(cycles_match.group(1)) if cycles_match else None
    energy = float(energy_match.group(1)) if energy_match else None
    return atoms, converged, cycles, energy


def write_xyz_stack(path: Path, frames: list[tuple[int, float, list[Atom], bool, int | None, float | None]]) -> None:
    lines: list[str] = []
    for point, target_r, atoms, converged, cycles, energy in frames:
        lines.append(str(len(atoms)))
        status = "converged" if converged else "not_converged"
        cycle_text = "NA" if cycles is None else str(cycles)
        energy_text = "NA" if energy is None else f"{energy:.12f}"
        lines.append(
            f"point={point} target_R_OO_ang={target_r:.5f} actual_R_OO_ang={oo_distance(atoms):.5f} "
            f"status={status} cycles={cycle_text} energy_au={energy_text}"
        )
        lines += [f"{sym:2s} {x:18.10f} {y:18.10f} {z:18.10f}" for sym, x, y, z in atoms]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_coordinates_csv(path: Path, frames: list[tuple[int, float, list[Atom], bool, int | None, float | None]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "point",
                "target_oo_ang",
                "actual_oo_ang",
                "converged",
                "cycles",
                "energy_au",
                "atom_index",
                "symbol",
                "x_ang",
                "y_ang",
                "z_ang",
            ],
        )
        writer.writeheader()
        for point, target_r, atoms, converged, cycles, energy in frames:
            actual_r = oo_distance(atoms)
            for iatom, (sym, x, y, z) in enumerate(atoms, start=1):
                writer.writerow(
                    {
                        "point": point,
                        "target_oo_ang": f"{target_r:.8f}",
                        "actual_oo_ang": f"{actual_r:.8f}",
                        "converged": int(converged),
                        "cycles": "" if cycles is None else cycles,
                        "energy_au": "" if energy is None else f"{energy:.12f}",
                        "atom_index": iatom,
                        "symbol": sym,
                        "x_ang": f"{x:.10f}",
                        "y_ang": f"{y:.10f}",
                        "z_ang": f"{z:.10f}",
                    }
                )


def write_summary_csv(path: Path, records: list[dict[str, str | int | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def parse_r_values(text: str | None) -> list[float]:
    if not text:
        return DEFAULT_R_VALUES
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("No R values were parsed.")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick-exe", type=Path, default=Path("/private/tmp/quick-build-mbx-focktest/src/quick"))
    parser.add_argument("--basis", type=Path, default=Path("basis"))
    parser.add_argument("--workdir", type=Path, default=Path("/private/tmp/quick-mbx-water-b3lyp-d3-opt"))
    parser.add_argument(
        "--xyz",
        type=Path,
        default=Path("validation/quick_mbx/water_dimer_b3lyp_d3_augccpvtz_constrained_10pt.xyz"),
    )
    parser.add_argument(
        "--coords-csv",
        type=Path,
        default=Path("validation/quick_mbx/water_dimer_b3lyp_d3_augccpvtz_constrained_10pt_coords.csv"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("validation/quick_mbx/water_dimer_b3lyp_d3_augccpvtz_constrained_10pt_summary.csv"),
    )
    parser.add_argument("--r-values", help="Comma-separated O-O distances in Angstrom.")
    parser.add_argument("--max-cycles", type=int, default=40)
    parser.add_argument("--reuse-workdir", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write QUICK inputs but do not run optimizations.")
    args = parser.parse_args()

    r_values = parse_r_values(args.r_values)
    workdir = args.workdir.resolve()
    if workdir.exists() and not args.reuse_workdir:
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["QUICK_BASIS"] = str(args.basis.resolve())

    if not args.dry_run and not args.quick_exe.exists():
        raise FileNotFoundError(f"QUICK executable not found: {args.quick_exe}")

    frames: list[tuple[int, float, list[Atom], bool, int | None, float | None]] = []
    records: list[dict[str, str | int | float]] = []
    previous_atoms: list[Atom] | None = None

    for point, target_r in enumerate(r_values, start=1):
        point_dir = workdir / f"point_{point:02d}_R_{target_r:.3f}"
        point_dir.mkdir(parents=True, exist_ok=True)
        start_atoms = reference_dimer_at_distance(target_r)
        if previous_atoms is not None:
            start_atoms = shift_second_water_to_distance(previous_atoms, target_r)

        input_path = point_dir / f"water_dimer_b3lyp_d3_opt_R_{target_r:.3f}.in"
        write_quick_input(input_path, start_atoms, args.max_cycles)

        if args.dry_run:
            frames.append((point, target_r, start_atoms, False, None, None))
            records.append(
                {
                    "point": point,
                    "target_oo_ang": f"{target_r:.8f}",
                    "actual_oo_ang": f"{oo_distance(start_atoms):.8f}",
                    "converged": 0,
                    "cycles": "",
                    "energy_au": "",
                    "input_path": str(input_path),
                    "output_path": "",
                }
            )
            previous_atoms = start_atoms
            continue

        output_text, output_path = run_quick(args.quick_exe.resolve(), input_path, point_dir, env)
        atoms, converged, cycles, energy = parse_optimized_geometry(output_text)
        frames.append((point, target_r, atoms, converged, cycles, energy))
        records.append(
            {
                "point": point,
                "target_oo_ang": f"{target_r:.8f}",
                "actual_oo_ang": f"{oo_distance(atoms):.8f}",
                "converged": int(converged),
                "cycles": "" if cycles is None else cycles,
                "energy_au": "" if energy is None else f"{energy:.12f}",
                "input_path": str(input_path),
                "output_path": str(output_path),
            }
        )
        previous_atoms = atoms

    write_xyz_stack(args.xyz, frames)
    write_coordinates_csv(args.coords_csv, frames)
    write_summary_csv(args.summary_csv, records)

    print(f"Wrote {args.xyz}")
    print(f"Wrote {args.coords_csv}")
    print(f"Wrote {args.summary_csv}")
    if args.dry_run:
        print(f"Dry run wrote QUICK inputs under {workdir}")
    else:
        nconv = sum(1 for _, _, _, converged, _, _ in frames if converged)
        print(f"Converged {nconv}/{len(frames)} constrained optimizations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
