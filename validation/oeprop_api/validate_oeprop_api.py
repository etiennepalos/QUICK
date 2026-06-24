#!/usr/bin/env python3
"""Validate QUICK no-I/O OEPROP API output against file-based OEPROP output."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


WATER_GEOM = """O  -0.33840000   0.00380000   0.23923000
H  -0.33510000  -0.00190000  -0.83277000
H   0.67350000  -0.00190000   0.59353000
"""

PROBES = [
    (0.00000000, 0.00000000, 2.00000000),
    (1.50000000, 0.00000000, 0.00000000),
    (0.00000000, 1.50000000, 0.00000000),
    (-1.00000000, -1.00000000, 0.50000000),
]

KEYWORD = "HF BASIS=STO-3G CUTOFF=1.0D-10 DENSERMS=1.0D-8 ENERGY CHARGE=0"
FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?")


def run(cmd: list[str], cwd: Path, env: dict[str, str], log: Path) -> None:
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def write_file_based_input(path: Path) -> None:
    probe_lines = "\n".join(f"{x: .8f} {y: .8f} {z: .8f}" for x, y, z in PROBES)
    path.write_text(
        f"{KEYWORD} ESP_GRID EFIELD_GRID EFG_GRID\n\n"
        f"{WATER_GEOM}\n"
        f"{probe_lines}\n",
        encoding="utf-8",
    )


def floats_from_line(line: str) -> list[float]:
    return [float(tok.replace("D", "E").replace("d", "e")) for tok in FLOAT_RE.findall(line)]


def parse_property_file(path: Path, expected_values: int) -> list[list[float]]:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        vals = floats_from_line(line)
        if len(vals) == 3 + expected_values:
            rows.append(vals[3:])
    if not rows:
        raise RuntimeError(f"No property rows parsed from {path}")
    return rows


def parse_api_csv(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({key: float(value) for key, value in row.items()})
    if not rows:
        raise RuntimeError(f"No API rows parsed from {path}")
    return rows


def flat(rows: list[list[float]]) -> list[float]:
    return [value for row in rows for value in row]


def metrics(a: list[float], b: list[float]) -> dict[str, float]:
    if len(a) != len(b):
        raise RuntimeError(f"Length mismatch: {len(a)} vs {len(b)}")
    diffs = [x - y for x, y in zip(a, b)]
    max_abs = max(abs(x) for x in diffs) if diffs else 0.0
    rms = math.sqrt(sum(x * x for x in diffs) / len(diffs)) if diffs else 0.0
    return {"nvalues": len(diffs), "max_abs": max_abs, "rms": rms}


def extract_api_components(rows: list[dict[str, float]]) -> dict[str, list[float]]:
    esp = [row["esp"] for row in rows]
    efield = flat([[row["efield_x"], row["efield_y"], row["efield_z"]] for row in rows])
    efg_names = ["efg_xx", "efg_xy", "efg_xz", "efg_yx", "efg_yy", "efg_yz", "efg_zx", "efg_zy", "efg_zz"]
    efg = flat([[row[name] for name in efg_names] for row in rows])
    return {"esp": esp, "efield": efield, "efg": efg}


def validate(quick: Path, api_driver: Path, workdir: Path, repo_root: Path, tolerance: float) -> dict[str, object]:
    workdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if "QUICK_BASIS" not in env:
        basis = repo_root / "basis"
        if basis.exists():
            env["QUICK_BASIS"] = str(basis)

    input_name = "oeprop_file_reference.in"
    write_file_based_input(workdir / input_name)
    run([str(quick), input_name], workdir, env, workdir / "quick_file_reference.log")

    api_csv = workdir / "oeprop_api_validation.csv"
    run([str(api_driver), "validate", str(api_csv)], workdir, env, workdir / "api_validation.log")

    file_esp = flat(parse_property_file(workdir / "oeprop_file_reference.esp", 1))
    file_efield = flat(parse_property_file(workdir / "oeprop_file_reference.efield", 3))
    file_efg = flat(parse_property_file(workdir / "oeprop_file_reference.efg", 9))
    api = extract_api_components(parse_api_csv(api_csv))

    comparisons = {
        "esp_file_vs_api": metrics(file_esp, api["esp"]),
        "efield_file_vs_api": metrics(file_efield, api["efield"]),
        "efg_file_vs_api": metrics(file_efg, api["efg"]),
    }

    trajectory_csv = workdir / "oeprop_trajectory.csv"
    run(
        [
            str(api_driver),
            "trajectory",
            str(repo_root / "validation/oeprop_api/water_trajectory_3frames.xyz"),
            str(repo_root / "validation/oeprop_api/water_probe_points_ang.dat"),
            str(trajectory_csv),
        ],
        workdir,
        env,
        workdir / "trajectory.log",
    )
    traj_rows = parse_api_csv(trajectory_csv)
    if len(traj_rows) != 12:
        raise RuntimeError(f"Expected 12 trajectory rows, found {len(traj_rows)}")
    first_frame = [row for row in traj_rows if int(row["frame"]) == 1]
    trajectory_first_frame = extract_api_components(first_frame)
    comparisons["trajectory_frame1_esp_vs_validation"] = metrics(api["esp"], trajectory_first_frame["esp"])
    comparisons["trajectory_frame1_efield_vs_validation"] = metrics(api["efield"], trajectory_first_frame["efield"])
    comparisons["trajectory_frame1_efg_vs_validation"] = metrics(api["efg"], trajectory_first_frame["efg"])

    passed = all(item["max_abs"] <= tolerance for item in comparisons.values())
    summary: dict[str, object] = {
        "passed": passed,
        "tolerance": tolerance,
        "quick": str(quick),
        "api_driver": str(api_driver),
        "workdir": str(workdir),
        "comparisons": comparisons,
        "outputs": {
            "api_validation_csv": str(api_csv),
            "trajectory_csv": str(trajectory_csv),
            "file_reference_input": str(workdir / input_name),
        },
    }
    (workdir / "oeprop_api_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", type=Path, required=True, help="Path to QUICK executable")
    parser.add_argument("--api-driver", type=Path, required=True, help="Path to test-api-oeprop executable")
    parser.add_argument("--workdir", type=Path, default=Path("/private/tmp/quick_oeprop_api_validation"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--tolerance", type=float, default=2.0e-7)
    args = parser.parse_args()

    summary = validate(args.quick.resolve(), args.api_driver.resolve(), args.workdir.resolve(), args.repo_root.resolve(), args.tolerance)
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
