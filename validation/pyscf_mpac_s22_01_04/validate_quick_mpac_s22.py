#!/usr/bin/env python3
"""Validate QUICK MPAC/HFAC scalar models against PySCF RHF/MP2.

The comparison is intentionally conventional and closed-shell:
no density fitting, no frozen core, STO-3G, and the S22 entries 01-04.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYSCF_MPAC_SRC = Path(
    "/Users/etiennepalos/Documents/Research/UNIFR/summer_mpac/pyscf-mpac/src"
)
S22_ROOT = Path(
    "/Users/etiennepalos/Documents/Research/UNIFR/MPAC_THINKSWISS/"
    "perspective_NCIs/b30/S22"
)
ENTRIES = ("01", "02", "03", "04")

QUICK_LABELS = {
    "e_hf": "TOTAL ENERGY",
    "e_mp2_os": "MPAC MP2 OPPOSITE-SPIN",
    "e_mp2_ss": "MPAC MP2 SAME-SPIN",
    "e_mp2_corr": "MPAC MP2 CORRELATION",
    "e_x_hf": "MPAC EXACT EXCHANGE",
    "rho_4_3": "MPAC RHO^(4/3)",
    "grad_square_over_rho_4_3": "MPAC GRAD2/RHO^(4/3)",
    "pc": "MPAC PC MODEL",
    "rho_3_2": "HFAC RHO^(3/2)",
    "grad_square_over_rho_7_6": "HFAC GRAD2/RHO^(7/6)",
    "hfac_e_el": "HFAC E_EL^GGA",
    "w_inf": "HFAC W_C,INF",
    "w_half": "HFAC W_1/2",
    "w_three_quarter": "HFAC W_3/4",
    "e_spl2_corr": "SPL2 CORRELATION",
    "e_spl2_total": "ESPL2",
    "e_os_spl2_corr": "OS-SPL2 CORRELATION",
    "e_os_spl2_total": "EOS-SPL2",
    "e_mpac25_corr": "MPAC25 CORRELATION",
    "e_mpac25_total": "EMPAC25",
    "e_os_mpac25_corr": "OS-MPAC25 CORRELATION",
    "e_os_mpac25_total": "EOS-MPAC25",
    "e_ueg_ihf": "HFAC24 UEGIHF CORRELATION",
    "e_hfac24_corr": "HFAC24 CORRELATION",
    "e_hfac24_total": "EHFAC24",
}

COMPARE_KEYS = (
    "e_hf",
    "e_mp2_os",
    "e_mp2_ss",
    "e_mp2_corr",
    "e_x_hf",
    "pc",
    "w_inf",
    "w_half",
    "w_three_quarter",
    "e_spl2_corr",
    "e_spl2_total",
    "e_os_spl2_corr",
    "e_os_spl2_total",
    "e_mpac25_corr",
    "e_mpac25_total",
    "e_os_mpac25_corr",
    "e_os_mpac25_total",
    "e_hfac24_corr",
    "e_hfac24_total",
)


def load_pyscf_modules():
    sys.path.insert(0, str(PYSCF_MPAC_SRC))
    from pyscf import gto, mp, scf
    from pyscf_mpac.functionals import mpac25_correlation, pc_model, spl2_correlation
    from pyscf_mpac.hfac24 import (
        hfac24_correlation,
        strong_interaction_ingredients_from_mf,
        ueg_ihf_correlation,
    )
    from pyscf_mpac.scf import compute_grid_terms, hf_coulomb_exchange_components

    return {
        "gto": gto,
        "mp": mp,
        "scf": scf,
        "compute_grid_terms": compute_grid_terms,
        "hf_components": hf_coulomb_exchange_components,
        "mpac25": mpac25_correlation,
        "pc_model": pc_model,
        "spl2": spl2_correlation,
        "strong": strong_interaction_ingredients_from_mf,
        "hfac24": hfac24_correlation,
        "ueg": ueg_ihf_correlation,
    }


def s22_atom_block(entry: str) -> str:
    lines = S22_ROOT.joinpath(entry, "struc.xyz").read_text().splitlines()[2:]
    return "\n".join(line.strip() for line in lines if line.strip())


def quick_input_path(entry: str) -> Path:
    return REPO_ROOT / "test" / f"ene_s22_{entry}_mpac25_hfac24_sto3g.in"


def quick_output_path(entry: str) -> Path:
    return REPO_ROOT / "test" / f"ene_s22_{entry}_mpac25_hfac24_sto3g.out"


def run_quick(entry: str, quick_exe: Path, quick_basis: Path) -> None:
    env = os.environ.copy()
    env["QUICK_BASIS"] = str(quick_basis)
    input_arg = quick_input_path(entry).relative_to(REPO_ROOT)
    subprocess.run(
        [str(quick_exe), str(input_arg)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )


def parse_quick_output(entry: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in quick_output_path(entry).read_text().splitlines():
        stripped = line.strip()
        for key, label in QUICK_LABELS.items():
            if stripped.startswith(label):
                values[key] = float(stripped.split("=")[-1])
    missing = [key for key in QUICK_LABELS if key not in values]
    if missing:
        raise RuntimeError(f"{quick_output_path(entry)} missing {missing}")
    return values


def pyscf_reference(entry: str, modules: dict[str, object]) -> dict[str, float]:
    gto = modules["gto"]
    scf = modules["scf"]
    mp = modules["mp"]

    mol = gto.M(
        atom=s22_atom_block(entry),
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1.0e-10
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"PySCF RHF did not converge for S22 {entry}")

    mp2 = mp.MP2(mf)
    e_mp2_corr, _ = mp2.kernel()
    dm = mf.make_rdm1()
    dm_total, _, e_x_hf = modules["hf_components"](mf, dm)
    grid = modules["compute_grid_terms"](
        mol, dm_total, grids_level=4, rho_trunc=1.0e-14
    )
    pc = modules["pc_model"](
        grid["rho_4_3"], grid["grad_square_over_rho_4_3"]
    )
    e_mpac25_corr = modules["mpac25"](
        e_x_hf, e_mp2_corr, grid["rho_4_3"], grid["grad_square_over_rho_4_3"]
    )
    e_spl2_corr = modules["spl2"](
        e_x_hf,
        e_mp2_corr,
        grid["rho_4_3"],
        grid["grad_square_over_rho_4_3"],
        (0.117, 10.68, 1.1472, -0.7397),
    )
    e_os_spl2_corr = modules["spl2"](
        e_x_hf,
        1.8 * mp2.e_corr_os,
        grid["rho_4_3"],
        grid["grad_square_over_rho_4_3"],
        (0.527, 58.850, 1.278, -1.059),
    )
    e_os_mpac25_corr = modules["mpac25"](
        e_x_hf,
        1.7 * mp2.e_corr_os,
        grid["rho_4_3"],
        grid["grad_square_over_rho_4_3"],
    )
    strong = modules["strong"](mf, grids_level=4, rho_trunc=1.0e-14)
    e_ueg = modules["ueg"](strong.w_inf, strong.w_half, strong.w_three_quarter)
    e_hfac24_corr = modules["hfac24"](
        e_mp2_corr, strong.w_inf, strong.w_half, strong.w_three_quarter, nquad=160
    )

    return {
        "e_hf": float(mf.e_tot),
        "e_mp2_os": float(mp2.e_corr_os),
        "e_mp2_ss": float(mp2.e_corr_ss),
        "e_mp2_corr": float(e_mp2_corr),
        "e_x_hf": float(e_x_hf),
        "rho_4_3": float(grid["rho_4_3"]),
        "grad_square_over_rho_4_3": float(grid["grad_square_over_rho_4_3"]),
        "pc": float(pc),
        "rho_3_2": float(grid["rho_3_2"]),
        "grad_square_over_rho_7_6": float(grid["grad_square_over_rho_7_6"]),
        "w_inf": float(strong.w_inf),
        "w_half": float(strong.w_half),
        "w_three_quarter": float(strong.w_three_quarter),
        "e_spl2_corr": float(e_spl2_corr),
        "e_spl2_total": float(mf.e_tot + e_spl2_corr),
        "e_os_spl2_corr": float(e_os_spl2_corr),
        "e_os_spl2_total": float(mf.e_tot + e_os_spl2_corr),
        "e_mpac25_corr": float(e_mpac25_corr),
        "e_mpac25_total": float(mf.e_tot + e_mpac25_corr),
        "e_os_mpac25_corr": float(e_os_mpac25_corr),
        "e_os_mpac25_total": float(mf.e_tot + e_os_mpac25_corr),
        "e_ueg_ihf": float(e_ueg),
        "e_hfac24_corr": float(e_hfac24_corr),
        "e_hfac24_total": float(mf.e_tot + e_hfac24_corr),
    }


def write_results(rows: list[dict[str, object]], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "quick_vs_pyscf_mpac_s22_01_04.csv"
    json_path = outdir / "quick_vs_pyscf_mpac_s22_01_04.json"
    md_path = outdir / "quick_vs_pyscf_mpac_s22_01_04.md"

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["entry", "quantity", "quick", "pyscf", "diff"]
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(rows, indent=2) + "\n")

    max_abs = {}
    for key in COMPARE_KEYS:
        diffs = [abs(float(row["diff"])) for row in rows if row["quantity"] == key]
        max_abs[key] = max(diffs) if diffs else 0.0

    lines = [
        "# QUICK MPAC/HFAC S22 01-04 Validation",
        "",
        "Reference: PySCF conventional RHF + canonical MP2, STO-3G, "
        "no frozen core, no density fitting.",
        "",
        "| quantity | max abs diff / Eh |",
        "|---|---:|",
    ]
    for key in COMPARE_KEYS:
        lines.append(f"| `{key}` | {max_abs[key]:.6e} |")
    lines.append("")
    lines.append("Full per-system values are in the CSV and JSON files.")
    md_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-quick", action="store_true")
    parser.add_argument("--quick-exe", type=Path, default=REPO_ROOT / "build-serial/src/quick")
    parser.add_argument("--quick-basis", type=Path, default=REPO_ROOT / "basis")
    parser.add_argument("--outdir", type=Path, default=Path(__file__).with_name("results"))
    args = parser.parse_args()

    if args.run_quick:
        for entry in ENTRIES:
            run_quick(entry, args.quick_exe, args.quick_basis)

    modules = load_pyscf_modules()
    rows: list[dict[str, object]] = []
    for entry in ENTRIES:
        quick = parse_quick_output(entry)
        pyscf = pyscf_reference(entry, modules)
        for key in COMPARE_KEYS:
            rows.append(
                {
                    "entry": entry,
                    "quantity": key,
                    "quick": quick[key],
                    "pyscf": pyscf[key],
                    "diff": quick[key] - pyscf[key],
                }
            )

    write_results(rows, args.outdir)
    print(f"Wrote validation results to {args.outdir}")


if __name__ == "__main__":
    main()
