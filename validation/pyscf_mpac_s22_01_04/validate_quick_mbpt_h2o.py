#!/usr/bin/env python3
"""Validate QUICK H2O/STO-3G MBPT variants against conventional PySCF."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from pyscf import ao2mo, gto, scf


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUTS = {
    "mp2": "ene_H2O_mp2_sto3g",
    "sos_mp2": "ene_H2O_sos_mp2_sto3g",
    "lt_mp2": "ene_H2O_lt_mp2_sto3g",
    "lt_sos_mp2": "ene_H2O_lt_sos_mp2_sto3g",
    "k_mp2": "ene_H2O_k_mp2_sto3g",
    "kos_mp2": "ene_H2O_kos_mp2_sto3g",
}

QUICK_LABELS = {
    "e_hf": "TOTAL ENERGY",
    "mp2_os": "MP2 OPPOSITE-SPIN CORRELATION",
    "mp2_ss": "MP2 SAME-SPIN CORRELATION",
    "mp2_corr": "SECOND ORDER ENERGY",
    "mp2_total": "EMP2",
    "sos_corr": "SOS-MP2 CORRELATION",
    "sos_total": "ESOS-MP2",
    "lt_corr": "LT-MP2 CORRELATION",
    "lt_total": "ELT-MP2",
    "lt_sos_corr": "LT-SOS-MP2 CORRELATION",
    "lt_sos_total": "ELT-SOS-MP2",
    "k_os": "K-MP2 OPPOSITE-SPIN",
    "k_ss": "K-MP2 SAME-SPIN",
    "k_corr": "K-MP2 CORRELATION",
    "k_total": "EK-MP2",
    "kos_corr": "KOS-MP2 CORRELATION",
    "kos_total": "EKOS-MP2",
}

COMPARE_KEYS = (
    "e_hf",
    "mp2_os",
    "mp2_ss",
    "mp2_corr",
    "mp2_total",
    "sos_corr",
    "sos_total",
    "lt_corr",
    "lt_total",
    "lt_sos_corr",
    "lt_sos_total",
    "k_os",
    "k_ss",
    "k_corr",
    "k_total",
    "kos_corr",
    "kos_total",
)


def input_path(name: str) -> Path:
    return REPO_ROOT / "test" / f"{INPUTS[name]}.in"


def output_path(name: str) -> Path:
    return REPO_ROOT / "test" / f"{INPUTS[name]}.out"


def atom_block() -> str:
    atoms = []
    for line in input_path("mp2").read_text().splitlines()[2:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) >= 4 and fields[0][0].isalpha():
            atoms.append(" ".join(fields[:4]))
    return "\n".join(atoms)


def run_quick(name: str, quick_exe: Path, quick_basis: Path) -> None:
    env = os.environ.copy()
    env["QUICK_BASIS"] = str(quick_basis)
    subprocess.run(
        [str(quick_exe), str(input_path(name).relative_to(REPO_ROOT))],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )


def parse_quick_outputs() -> dict[str, float]:
    values: dict[str, float] = {}
    for name in INPUTS:
        for line in output_path(name).read_text().splitlines():
            stripped = line.strip()
            for key, label in QUICK_LABELS.items():
                if stripped.startswith(label):
                    values[key] = float(stripped.split("=")[-1])
    missing = [key for key in COMPARE_KEYS if key not in values]
    if missing:
        raise RuntimeError(f"Missing QUICK H2O values: {missing}")
    return values


def kappa_damping(delta: float, kappa: float) -> float:
    attenuation = 1.0 - np.exp(-kappa * delta)
    return float(attenuation * attenuation)


def pyscf_reference() -> dict[str, float]:
    mol = gto.M(atom=atom_block(), basis="sto-3g", unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1.0e-10
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF RHF did not converge for H2O/STO-3G.")

    nocc = mol.nelectron // 2
    nmo = mf.mo_coeff.shape[1]
    nvir = nmo - nocc
    eri = ao2mo.kernel(mol, mf.mo_coeff, compact=False).reshape(nmo, nmo, nmo, nmo)
    eps = mf.mo_energy

    e_os = 0.0
    e_ss = 0.0
    e_k_os = 0.0
    e_k_ss = 0.0
    e_kos_raw = 0.0

    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvir):
                aa = nocc + a
                for b in range(nvir):
                    bb = nocc + b
                    coulomb = eri[i, aa, j, bb]
                    exchange = eri[i, bb, j, aa]
                    denom = eps[i] + eps[j] - eps[aa] - eps[bb]
                    delta = -denom
                    inv_denom = 1.0 / denom
                    damp_k = kappa_damping(delta, 1.10)
                    damp_kos = kappa_damping(delta, 0.90)

                    e_os += coulomb * coulomb * inv_denom
                    e_ss += coulomb * (coulomb - exchange) * inv_denom
                    e_k_os += coulomb * coulomb * inv_denom * damp_k
                    e_k_ss += coulomb * (coulomb - exchange) * inv_denom * damp_k
                    e_kos_raw += coulomb * coulomb * inv_denom * damp_kos

    e_mp2 = e_os + e_ss
    e_sos = 1.30 * e_os
    e_k = e_k_os + e_k_ss
    e_kos = 2.10 * e_kos_raw
    return {
        "e_hf": float(mf.e_tot),
        "mp2_os": float(e_os),
        "mp2_ss": float(e_ss),
        "mp2_corr": float(e_mp2),
        "mp2_total": float(mf.e_tot + e_mp2),
        "sos_corr": float(e_sos),
        "sos_total": float(mf.e_tot + e_sos),
        "lt_corr": float(e_mp2),
        "lt_total": float(mf.e_tot + e_mp2),
        "lt_sos_corr": float(e_sos),
        "lt_sos_total": float(mf.e_tot + e_sos),
        "k_os": float(e_k_os),
        "k_ss": float(e_k_ss),
        "k_corr": float(e_k),
        "k_total": float(mf.e_tot + e_k),
        "kos_corr": float(e_kos),
        "kos_total": float(mf.e_tot + e_kos),
    }


def write_results(rows: list[dict[str, object]], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "quick_vs_pyscf_mbpt_h2o.csv"
    json_path = outdir / "quick_vs_pyscf_mbpt_h2o.json"
    md_path = outdir / "quick_vs_pyscf_mbpt_h2o.md"

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["quantity", "quick", "pyscf", "diff"]
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(rows, indent=2) + "\n")

    lines = [
        "# QUICK MBPT H2O/STO-3G Validation",
        "",
        "Reference: PySCF conventional RHF plus explicit AO-to-MO MP2 denominators, "
        "STO-3G, no frozen core, no density fitting.",
        "",
        "| quantity | QUICK / Eh | PySCF / Eh | diff / Eh |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['quantity']}` | {row['quick']:.12f} | "
            f"{row['pyscf']:.12f} | {row['diff']:.6e} |"
        )
    md_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-quick", action="store_true")
    parser.add_argument("--quick-exe", type=Path, default=REPO_ROOT / "build-serial/src/quick")
    parser.add_argument("--quick-basis", type=Path, default=REPO_ROOT / "basis")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).with_name("results_mbpt_h2o"),
    )
    args = parser.parse_args()

    if args.run_quick:
        for name in INPUTS:
            run_quick(name, args.quick_exe, args.quick_basis)

    quick = parse_quick_outputs()
    pyscf = pyscf_reference()
    rows = [
        {
            "quantity": key,
            "quick": quick[key],
            "pyscf": pyscf[key],
            "diff": quick[key] - pyscf[key],
        }
        for key in COMPARE_KEYS
    ]
    write_results(rows, args.outdir)
    print(f"Wrote validation results to {args.outdir}")


if __name__ == "__main__":
    main()
