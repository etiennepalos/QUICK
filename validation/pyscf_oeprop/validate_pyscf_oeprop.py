#!/usr/bin/env python3
"""Cross-code validation of QUICK OEPROP CPU results against PySCF.

The reference values are generated from PySCF electrostatic potentials.  EFIELD
and EFG are finite differences of the PySCF ESP, using QUICK's convention

    E_i(C) = -dV(C)/dC_i
    G_ij(C) = dE_i(C)/dC_j = -d^2 V(C)/(dC_i dC_j).

This script intentionally runs QUICK in a private work directory, then compares
the generated property files to PySCF values on the same points.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from pyscf import dft, gto, scf

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional for the numeric run
    plt = None


BOHR_PER_ANGSTROM = 1.8897259886
FD_STEP_BOHR = 5.0e-4

ATOM_SYMBOLS = {
    "H", "HE",
    "LI", "BE", "B", "C", "N", "O", "F", "NE",
    "NA", "MG", "AL", "SI", "P", "S", "CL", "AR",
    "K", "CA", "SC", "TI", "V", "CR", "MN", "FE", "CO", "NI", "CU", "ZN",
    "GA", "GE", "AS", "SE", "BR", "KR",
    "RB", "SR", "Y", "ZR", "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD",
    "IN", "SN", "SB", "TE", "I", "XE",
}


@dataclass
class QuickInput:
    path: Path
    keywords: str
    atoms: list[str]
    grid_angstrom: np.ndarray
    method: str
    basis: str
    charge: int
    dispersion: str
    external_field: np.ndarray
    external_origin_bohr: np.ndarray


@dataclass
class PySCFState:
    mol: gto.Mole
    mf: object
    dm: np.ndarray
    energy: float


def parse_quick_input(path: Path) -> QuickInput:
    lines = path.read_text().splitlines()
    keyword_lines: list[str] = []
    atoms: list[str] = []
    grid: list[list[float]] = []

    mode = "keywords"
    for raw in lines:
        line = raw.strip()
        if not line:
            if mode == "atoms" and atoms:
                mode = "grid"
            continue

        toks = line.split()
        first = toks[0].upper()
        if mode == "keywords":
            if len(toks) >= 4 and first in ATOM_SYMBOLS:
                mode = "atoms"
                atoms.append(raw)
            else:
                keyword_lines.append(raw)
        elif mode == "atoms":
            if len(toks) >= 4 and first in ATOM_SYMBOLS:
                atoms.append(raw)
            else:
                mode = "grid"
                grid.append([float(toks[0]), float(toks[1]), float(toks[2])])
        else:
            grid.append([float(toks[0]), float(toks[1]), float(toks[2])])

    keywords = " ".join(keyword_lines)
    key_upper = keywords.upper()

    if "PBE0" in key_upper:
        method = "PBE0"
    elif "B3LYP" in key_upper:
        method = "B3LYP"
    elif "HF" in key_upper or "RHF" in key_upper:
        method = "HF"
    else:
        raise ValueError(f"Unsupported method in {path}: {keywords}")

    basis_match = re.search(r"\bBASIS\s*=\s*([A-Za-z0-9+\-*_.]+)", keywords, re.I)
    if not basis_match:
        raise ValueError(f"No BASIS keyword found in {path}")
    basis = basis_match.group(1)

    charge_match = re.search(r"\bCHARGE\s*=\s*([-+]?\d+)", keywords, re.I)
    charge = int(charge_match.group(1)) if charge_match else 0
    dispersion = ""
    if "D3BJ" in key_upper:
        dispersion = "D3BJ"
    elif re.search(r"\bD3\b", key_upper):
        dispersion = "D3"

    field = np.zeros(3)
    origin = np.zeros(3)
    axes = {"X": 0, "Y": 1, "Z": 2}
    for label, idx in axes.items():
        for prefix in ("EXTERNAL_EFIELD", "FINITE_FIELD"):
            m = re.search(rf"\b{prefix}_{label}\s*=\s*([-+0-9.EeDd]+)", keywords, re.I)
            if m:
                field[idx] = float(m.group(1).replace("D", "E").replace("d", "e"))
            mo = re.search(rf"\b{prefix}_ORIGIN_{label}\s*=\s*([-+0-9.EeDd]+)", keywords, re.I)
            if mo:
                origin[idx] = float(mo.group(1).replace("D", "E").replace("d", "e"))

    return QuickInput(
        path=path,
        keywords=keywords,
        atoms=atoms,
        grid_angstrom=np.asarray(grid, dtype=float),
        method=method,
        basis=basis,
        charge=charge,
        dispersion=dispersion,
        external_field=field,
        external_origin_bohr=origin,
    )


def run_quick(repo_root: Path, quick_input: Path, run_dir: Path, quick_exe: Path,
              text_override: str | None = None) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / quick_input.name
    if text_override is None:
        shutil.copy2(quick_input, target)
    else:
        target.write_text(text_override)

    env = os.environ.copy()
    env["QUICK_BASIS"] = str(repo_root / "basis")
    subprocess.run(
        [str(quick_exe), target.name],
        cwd=run_dir,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return run_dir / f"{quick_input.stem}.out"


def read_property_rows(path: Path, ncols: int) -> np.ndarray:
    rows: list[list[float]] = []
    for line in path.read_text().splitlines():
        toks = line.split()
        if len(toks) != ncols:
            continue
        try:
            rows.append([float(tok) for tok in toks])
        except ValueError:
            pass
    if not rows:
        raise ValueError(f"No numeric rows with {ncols} columns found in {path}")
    return np.asarray(rows, dtype=float)


def read_total_energy(path: Path) -> float:
    for line in path.read_text().splitlines():
        if "TOTAL ENERGY" in line and "=" in line:
            return float(line.split("=")[1].split()[0])
    raise ValueError(f"No TOTAL ENERGY line found in {path}")


def read_energy_terms(path: Path) -> dict[str, float]:
    terms = {"total": math.nan, "dispersion": 0.0}
    for line in path.read_text().splitlines():
        stripped = line.lstrip()
        if stripped.startswith("TOTAL ENERGY") and "=" in stripped:
            terms["total"] = float(line.split("=")[1].split()[0])
        elif stripped.startswith("DISPERSION CORRECTION") and "=" in stripped:
            terms["dispersion"] = float(line.split("=")[1].split()[0])
    if math.isnan(terms["total"]):
        raise ValueError(f"No TOTAL ENERGY line found in {path}")
    return terms


def make_pyscf_state(qin: QuickInput, field: np.ndarray | None = None,
                     conv_tol: float | None = None) -> PySCFState:
    mol = gto.M(
        atom="\n".join(qin.atoms),
        basis=qin.basis.lower(),
        unit="Angstrom",
        charge=qin.charge,
        spin=0,
        cart=True,
        verbose=0,
    )

    if qin.method in {"B3LYP", "PBE0"}:
        mf = dft.RKS(mol)
        mf.xc = qin.method.lower()
        mf.grids.level = 3
        mf.conv_tol = conv_tol if conv_tol is not None else 4.0e-8
    elif qin.method == "HF":
        mf = scf.RHF(mol)
        mf.conv_tol = conv_tol if conv_tol is not None else 1.0e-9
    else:
        raise ValueError(qin.method)

    if field is not None and np.linalg.norm(field) > 0.0:
        h0 = mf.get_hcore()
        overlap = mol.intor("int1e_ovlp")
        moment = mol.intor("int1e_r")
        origin = qin.external_origin_bohr
        h_field = np.zeros_like(h0)
        for idir in range(3):
            h_field += field[idir] * (moment[idir] - origin[idir] * overlap)

        charges = mol.atom_charges()
        coords = mol.atom_coords()
        ecore_field = -sum(
            charges[iatom] * float(np.dot(field, coords[iatom] - origin))
            for iatom in range(mol.natm)
        )

        mf.get_hcore = lambda mol_arg=None, h0=h0, h_field=h_field: h0 + h_field
        mf.energy_nuc = (
            lambda *args, ecore_field=ecore_field, mol=mol:
            mol.energy_nuc() + ecore_field
        )

    energy = float(mf.kernel())
    if not mf.converged:
        raise RuntimeError(f"PySCF SCF did not converge for {qin.path.name}")
    return PySCFState(mol=mol, mf=mf, dm=mf.make_rdm1(), energy=energy)


def esp_single_point(state: PySCFState, point_bohr: np.ndarray) -> float:
    mol = state.mol
    mol.set_rinv_origin(point_bohr)
    rinv = mol.intor("int1e_rinv")
    electronic = -float(np.einsum("ij,ji", state.dm, rinv))
    diff = point_bohr[None, :] - mol.atom_coords()
    nuclear = float(np.sum(mol.atom_charges() / np.linalg.norm(diff, axis=1)))
    return nuclear + electronic


def pyscf_properties_on_points(state: PySCFState, points_bohr: np.ndarray,
                               progress_label: str) -> dict[str, np.ndarray]:
    npoints = points_bohr.shape[0]
    esp = np.zeros(npoints)
    efield = np.zeros((npoints, 3))
    efg = np.zeros((npoints, 3, 3))
    h = FD_STEP_BOHR

    eye = np.eye(3)
    for ipt, point in enumerate(points_bohr):
        if ipt and ipt % 100 == 0:
            print(f"  {progress_label}: {ipt}/{npoints} points", flush=True)

        v0 = esp_single_point(state, point)
        esp[ipt] = v0
        vp = np.zeros(3)
        vm = np.zeros(3)
        for idir in range(3):
            vp[idir] = esp_single_point(state, point + h * eye[idir])
            vm[idir] = esp_single_point(state, point - h * eye[idir])
            efield[ipt, idir] = -(vp[idir] - vm[idir]) / (2.0 * h)
            efg[ipt, idir, idir] = -(vp[idir] - 2.0 * v0 + vm[idir]) / (h * h)

        for idir in range(3):
            for jdir in range(idir + 1, 3):
                vpp = esp_single_point(state, point + h * eye[idir] + h * eye[jdir])
                vpm = esp_single_point(state, point + h * eye[idir] - h * eye[jdir])
                vmp = esp_single_point(state, point - h * eye[idir] + h * eye[jdir])
                vmm = esp_single_point(state, point - h * eye[idir] - h * eye[jdir])
                val = -(vpp - vpm - vmp + vmm) / (4.0 * h * h)
                efg[ipt, idir, jdir] = val
                efg[ipt, jdir, idir] = val

    return {"esp": esp, "efield": efield, "efg": efg}


def metrics(name: str, quick: np.ndarray, pyscf: np.ndarray, tolerance: float) -> dict:
    diff = pyscf.reshape(-1) - quick.reshape(-1)
    absdiff = np.abs(diff)
    denom = np.maximum(np.abs(quick.reshape(-1)), 1.0e-12)
    return {
        "name": name,
        "nvalues": int(diff.size),
        "max_abs": float(absdiff.max()),
        "rms": float(math.sqrt(float(np.mean(diff * diff)))),
        "mean_abs": float(absdiff.mean()),
        "max_rel": float(np.max(absdiff / denom)),
        "tolerance": tolerance,
        "pass": bool(absdiff.max() <= tolerance),
    }


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: dict, metric_rows: list[dict]) -> None:
    lines = [
        "# PySCF OEPROP Validation",
        "",
        f"Python: `{summary['python']}`",
        f"PySCF: `{summary['pyscf_version']}`",
        f"Finite-difference step for PySCF EFIELD/EFG: `{FD_STEP_BOHR:.1e}` bohr",
        "",
        "## SCF Energies",
        "",
        "| Case | QUICK / Eh | PySCF / Eh | PySCF - QUICK / Eh |",
        "|---|---:|---:|---:|",
    ]
    for row in summary["energies"]:
        lines.append(
            f"| {row['case']} | {row['quick']:.12f} | {row['pyscf']:.12f} | "
            f"{row['delta']:.3e} |"
        )

    lines += [
        "",
        "## Property Metrics",
        "",
        "| Comparison | Values | Max abs. | RMS | Tolerance | Pass |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for row in metric_rows:
        lines.append(
            f"| {row['name']} | {row['nvalues']} | {row['max_abs']:.3e} | "
            f"{row['rms']:.3e} | {row['tolerance']:.1e} | "
            f"{'yes' if row['pass'] else 'no'} |"
        )

    lines += [
        "",
        "## Interpretation Notes",
        "",
        "The acetone case is a cross-code PBE0-D3BJ/cc-pVDZ comparison. The "
        "local PySCF installation does not include a D3 backend, so the PySCF "
        "reference uses the PBE0 SCF density and the QUICK energy comparison "
        "uses QUICK's printed total energy after subtracting the printed D3BJ "
        "dispersion correction. This is appropriate for ESP, EFIELD, and EFG "
        "because the D3BJ correction is a geometry-dependent post-SCF energy "
        "term and does not alter the one-particle density. The remaining "
        "differences are interpreted as cross-code DFT quadrature and "
        "SCF-density differences. The H2O RHF/STO-3G surface and finite-field "
        "checks are much stricter because the underlying SCF densities are "
        "nearly identical.",
        "",
        "## Finite External Field",
        "",
        "| Quantity | QUICK | PySCF | PySCF - QUICK |",
        "|---|---:|---:|---:|",
    ]
    for row in summary["finite_field"]:
        lines.append(
            f"| {row['quantity']} | {row['quick']:.12f} | {row['pyscf']:.12f} | "
            f"{row['delta']:.3e} |"
        )

    path.write_text("\n".join(lines) + "\n")


def make_scatter_figure(path: Path, comparisons: list[tuple[str, np.ndarray, np.ndarray]]) -> None:
    if plt is None:
        return

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4), constrained_layout=True)
    for ax, (title, quick, pyscf) in zip(axes.ravel(), comparisons):
        x = quick.reshape(-1)
        y = pyscf.reshape(-1)
        ax.scatter(x, y, s=5, alpha=0.45, linewidths=0)
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        pad = 0.04 * (hi - lo if hi > lo else 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k-", lw=0.8)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("QUICK / a.u.")
        ax.set_ylabel("PySCF / a.u.")
    fig.savefig(path)
    plt.close(fig)


def replace_external_field(text: str, value: float) -> str:
    def repl(match: re.Match) -> str:
        return f"{match.group(1)}{value:.12E}"

    pattern = r"\b((?:EXTERNAL_EFIELD|FINITE_FIELD)_X\s*=\s*)([-+0-9.EeDd]+)"
    if not re.search(pattern, text, re.I):
        raise ValueError("No X external-field keyword found")
    return re.sub(pattern, repl, text, flags=re.I)


def main() -> int:
    here = Path(__file__).resolve().parent
    repo = here.parents[1]
    results_dir = here / "results"
    work_dir = here / "work"
    figures_dir = here / "figures"
    results_dir.mkdir(exist_ok=True)
    work_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    quick_exe = repo / "build-serial" / "src" / "quick"
    if not quick_exe.exists():
        raise FileNotFoundError(f"Serial QUICK executable not found: {quick_exe}")

    cases = {
        "external": {
            "esp": repo / "test" / "esp_grid_acetone_pbe0_d3bj_ccpvdz.in",
            "efield": repo / "test" / "efield_grid_acetone_pbe0_d3bj_ccpvdz.in",
            "efg": repo / "test" / "efg_grid_acetone_pbe0_d3bj_ccpvdz.in",
        },
        "surface": {
            "esp": repo / "test" / "esp_grid_density_surface_H2O_rhf_sto3g.in",
            "efield": repo / "test" / "efield_density_surface_H2O_rhf_sto3g.in",
            "efg": repo / "test" / "efg_density_surface_H2O_rhf_sto3g.in",
        },
    }

    metric_rows: list[dict] = []
    energy_rows: list[dict] = []
    scatter: list[tuple[str, np.ndarray, np.ndarray]] = []

    tolerances = {
        "external ESP_GRID acetone PBE0-D3BJ/cc-pVDZ": 1.0e-5,
        "external EFIELD_GRID acetone PBE0-D3BJ/cc-pVDZ": 5.0e-6,
        "external EFG_GRID acetone PBE0-D3BJ/cc-pVDZ": 1.0e-5,
        "surface ESP_SURFACE H2O RHF/STO-3G": 5.0e-6,
        "surface EFIELD_SURFACE H2O RHF/STO-3G": 5.0e-6,
        "surface EFG_SURFACE H2O RHF/STO-3G": 1.0e-5,
        "finite field H2O RHF/STO-3G +Fx energy": 1.0e-6,
    }

    for family, paths in cases.items():
        print(f"Running QUICK {family} cases", flush=True)
        run_base = work_dir / "quick_runs" / family
        out_paths = {}
        for prop, inpath in paths.items():
            out_paths[prop] = run_quick(repo, inpath, run_base / prop, quick_exe)

        qin = parse_quick_input(paths["efg"])
        print(f"Running PySCF SCF for {family}: {qin.method}/{qin.basis}", flush=True)
        state = make_pyscf_state(qin)
        energy_terms = read_energy_terms(out_paths["efg"])
        quick_energy = energy_terms["total"] - energy_terms["dispersion"]
        case_label = f"{family} {qin.method}/{qin.basis}"
        if qin.dispersion:
            case_label = f"{family} {qin.method}-{qin.dispersion}/{qin.basis} without dispersion energy"
        energy_rows.append({
            "case": case_label,
            "quick": quick_energy,
            "pyscf": state.energy,
            "delta": state.energy - quick_energy,
        })

        if family == "external":
            points_bohr = qin.grid_angstrom * BOHR_PER_ANGSTROM
        else:
            esp_rows = read_property_rows(run_base / "esp" / f"{paths['esp'].stem}.esp", 4)
            points_bohr = esp_rows[:, 0:3] * BOHR_PER_ANGSTROM

        props = pyscf_properties_on_points(state, points_bohr, family)

        q_esp = read_property_rows(run_base / "esp" / f"{paths['esp'].stem}.esp", 4)[:, 3]
        q_efield = read_property_rows(run_base / "efield" / f"{paths['efield'].stem}.efield", 6)[:, 3:6]
        q_efg = read_property_rows(run_base / "efg" / f"{paths['efg'].stem}.efg", 12)[:, 3:12].reshape(-1, 3, 3)

        labels = {
            "external": [
                "external ESP_GRID acetone PBE0-D3BJ/cc-pVDZ",
                "external EFIELD_GRID acetone PBE0-D3BJ/cc-pVDZ",
                "external EFG_GRID acetone PBE0-D3BJ/cc-pVDZ",
            ],
            "surface": [
                "surface ESP_SURFACE H2O RHF/STO-3G",
                "surface EFIELD_SURFACE H2O RHF/STO-3G",
                "surface EFG_SURFACE H2O RHF/STO-3G",
            ],
        }[family]

        for label, quick_arr, pyscf_arr in [
            (labels[0], q_esp, props["esp"]),
            (labels[1], q_efield, props["efield"]),
            (labels[2], q_efg, props["efg"]),
        ]:
            metric_rows.append(metrics(label, quick_arr, pyscf_arr, tolerances[label]))
            scatter.append((label.replace(" acetone ", "\nacetone ").replace(" H2O ", "\nH2O "),
                            quick_arr, pyscf_arr))

    print("Running finite external-field validation", flush=True)
    ff_input = repo / "test" / "ene_H2O_external_efield_rhf_sto3g.in"
    ff_qin = parse_quick_input(ff_input)
    ff_text = ff_input.read_text()
    ff_base = work_dir / "quick_runs" / "finite_field"

    ff_rows = []
    for label, field_value in [("+Fx", 1.0e-4), ("-Fx", -1.0e-4), ("0", 0.0)]:
        out = run_quick(
            repo,
            ff_input,
            ff_base / label.replace("+", "plus").replace("-", "minus"),
            quick_exe,
            text_override=replace_external_field(ff_text, field_value),
        )
        q_energy = read_total_energy(out)
        state = make_pyscf_state(ff_qin, field=np.array([field_value, 0.0, 0.0]))
        ff_rows.append({"label": label, "field": field_value, "quick": q_energy, "pyscf": state.energy})

    ff_summary = []
    for row in ff_rows:
        ff_summary.append({
            "quantity": f"E({row['label']}) / Eh",
            "quick": row["quick"],
            "pyscf": row["pyscf"],
            "delta": row["pyscf"] - row["quick"],
        })

    q_plus = next(row for row in ff_rows if row["label"] == "+Fx")
    q_minus = next(row for row in ff_rows if row["label"] == "-Fx")
    derivative_quick = (q_plus["quick"] - q_minus["quick"]) / (2.0e-4)
    derivative_pyscf = (q_plus["pyscf"] - q_minus["pyscf"]) / (2.0e-4)
    ff_summary.append({
        "quantity": "central dE/dFx / a.u.",
        "quick": derivative_quick,
        "pyscf": derivative_pyscf,
        "delta": derivative_pyscf - derivative_quick,
    })

    metric_rows.append(metrics(
        "finite field H2O RHF/STO-3G +Fx energy",
        np.array([q_plus["quick"]]),
        np.array([q_plus["pyscf"]]),
        tolerances["finite field H2O RHF/STO-3G +Fx energy"],
    ))

    summary = {
        "python": sys.executable,
        "pyscf_version": __import__("pyscf").__version__,
        "fd_step_bohr": FD_STEP_BOHR,
        "energies": energy_rows,
        "finite_field": ff_summary,
        "metrics": metric_rows,
    }

    (results_dir / "pyscf_validation_summary.json").write_text(json.dumps(summary, indent=2))
    write_csv(results_dir / "pyscf_validation_metrics.csv", metric_rows)
    write_markdown(results_dir / "pyscf_validation_summary.md", summary, metric_rows)
    figure_path = figures_dir / "pyscf_validation_scatter.pdf"
    make_scatter_figure(figure_path, scatter)
    if figure_path.exists():
        paper_figures = repo / "paper" / "figures"
        paper_figures.mkdir(exist_ok=True)
        shutil.copy2(figure_path, paper_figures / figure_path.name)

    all_pass = all(row["pass"] for row in metric_rows)
    print(f"Wrote {results_dir / 'pyscf_validation_summary.md'}")
    print(f"All toleranced comparisons pass: {all_pass}")
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
