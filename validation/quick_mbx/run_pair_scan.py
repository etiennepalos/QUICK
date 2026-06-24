#!/usr/bin/env python3
"""Run mixed QUICK-MBX pair scans and produce publication-style plots."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from run_lambros_dimer_scan import HARTREE_TO_KCAL, NUMBER_RE, draw_dashed_line, load_fonts, run


Atom = tuple[str, float, float, float]

H2O_FROM_CH4_EXAMPLE: list[Atom] = [
    ("O", 2.0786004022, -0.8324584110, 3.0706206061),
    ("H", 2.2502730350, -1.0317483584, 3.9948528330),
    ("H", 2.0608921579, 0.1418457568, 3.0121656318),
]
CH4_FROM_H2O_EXAMPLE: list[Atom] = [
    ("C", 0.1780116536, 0.1439591651, 0.0767959473),
    ("H", -0.4732221193, -0.4346691605, -0.5725415936),
    ("H", 1.1935894004, 0.1140036913, -0.3086389627),
    ("H", -0.1786432286, 1.1713010682, 0.1063843766),
    ("H", 0.1644153918, -0.2800416213, 1.0771061590),
]
CL_FROM_H2O_EXAMPLE: list[Atom] = [
    ("Cl", -2.2371172880, -0.0867486952, 1.6637199587),
]
H2O_FROM_CL_EXAMPLE: list[Atom] = [
    ("O", -0.0749612883, -1.9439663623, -0.5660661146),
    ("H", -0.4061648717, -1.2773088447, -1.1890851631),
    ("H", -0.6788183774, -1.7987964639, 0.1834998125),
]
O2_TRIPLET: list[Atom] = [
    ("O", 0.0, 0.0, -0.60375),
    ("O", 0.0, 0.0, 0.60375),
]
H2O_FOR_O2: list[Atom] = [
    ("O", 3.20, 0.0, 0.0),
    ("H", 3.958602, 0.0, 0.587079),
    ("H", 2.441398, 0.0, 0.587079),
]


@dataclass(frozen=True)
class CaseSpec:
    driver_case: str
    title: str
    panel_label: str
    x_label: str
    quantum_label: str
    classical_label: str | None
    hybrid_label: str
    qm_atoms: list[Atom]
    qm_mbx_name: str | None
    mbx_atoms: list[Atom]
    mbx_name: str
    mbx_json: Path
    residual_pair: tuple[str, str] | None
    npoints: int


H2O_CH4_JSON = Path("/private/tmp/MBX-quick-plan/examples/PEFs/042_ch4-h2o_mb-nrg_2bnb/C++/mbx.json")
CL_H2O_JSON = Path("/private/tmp/MBX-quick-plan/examples/PEFs/022_h2o-cl_mb-nrg_2bnb/C++/mbx.json")
H2O_JSON = Path("/private/tmp/MBX-quick-plan/examples/PEFs/001_mbpol/C++/mbx.json")

CASES: dict[str, CaseSpec] = {
    "h2o_ch4_h2o_qm": CaseSpec(
        driver_case="h2o_ch4_h2o_qm",
        title="H2O-CH4 PES: panel (a), H2O is QM",
        panel_label="(a) H2O(QM) / CH4(MBX)",
        x_label="O-C distance (Angstrom)",
        quantum_label="Quantum: PBE0-D3 full pair",
        classical_label="Classical: MB-nrg H2O-CH4",
        hybrid_label="Hybrid: H2O(QM)/CH4(MBX) + MBX 2B/disp",
        qm_atoms=H2O_FROM_CH4_EXAMPLE,
        qm_mbx_name="h2o",
        mbx_atoms=CH4_FROM_H2O_EXAMPLE,
        mbx_name="ch4",
        mbx_json=H2O_CH4_JSON,
        residual_pair=("ch4", "h2o"),
        npoints=10,
    ),
    "h2o_ch4_ch4_qm": CaseSpec(
        driver_case="h2o_ch4_ch4_qm",
        title="H2O-CH4 PES: panel (b), CH4 is QM",
        panel_label="(b) CH4(QM) / H2O(MBX)",
        x_label="C-O distance (Angstrom)",
        quantum_label="Quantum: PBE0-D3 full pair",
        classical_label="Classical: MB-nrg H2O-CH4",
        hybrid_label="Hybrid: CH4(QM)/H2O(MBX) + MBX 2B/disp",
        qm_atoms=CH4_FROM_H2O_EXAMPLE,
        qm_mbx_name="ch4",
        mbx_atoms=H2O_FROM_CH4_EXAMPLE,
        mbx_name="h2o",
        mbx_json=H2O_CH4_JSON,
        residual_pair=("ch4", "h2o"),
        npoints=10,
    ),
    "cl_h2o_cl_qm": CaseSpec(
        driver_case="cl_h2o_cl_qm",
        title="Cl-/H2O PES: Cl- is QM",
        panel_label="Cl-(QM) / H2O(MBX)",
        x_label="Cl-O distance (Angstrom)",
        quantum_label="Quantum: PBE0-D3 full pair",
        classical_label="Classical: MB-nrg Cl-/H2O",
        hybrid_label="Hybrid: Cl-(QM)/H2O(MBX) + MBX 2B/disp",
        qm_atoms=CL_FROM_H2O_EXAMPLE,
        qm_mbx_name="cl-",
        mbx_atoms=H2O_FROM_CL_EXAMPLE,
        mbx_name="h2o",
        mbx_json=CL_H2O_JSON,
        residual_pair=("cl-", "h2o"),
        npoints=10,
    ),
    "o2_h2o_o2_qm": CaseSpec(
        driver_case="o2_h2o_o2_qm",
        title="Triplet O2/H2O open-shell test",
        panel_label="Triplet O2(QM) / H2O(MBX)",
        x_label="O2(first O)-water O distance (Angstrom)",
        quantum_label="Quantum: triplet PBE0-D3 full pair",
        classical_label=None,
        hybrid_label="Hybrid: O2(QM)/H2O(MBX), elec+pol only",
        qm_atoms=O2_TRIPLET,
        qm_mbx_name=None,
        mbx_atoms=H2O_FOR_O2,
        mbx_name="h2o",
        mbx_json=H2O_JSON,
        residual_pair=None,
        npoints=10,
    ),
}


def translate_second_to_distance(first: list[Atom], second_ref: list[Atom], distance: float) -> list[Atom]:
    ax, ay, az = first[0][1:]
    bx, by, bz = second_ref[0][1:]
    vx, vy, vz = bx - ax, by - ay, bz - az
    norm = (vx * vx + vy * vy + vz * vz) ** 0.5
    ux, uy, uz = vx / norm, vy / norm, vz / norm
    tx, ty, tz = ax + distance * ux, ay + distance * uy, az + distance * uz
    sx, sy, sz = tx - bx, ty - by, tz - bz
    return [(sym, x + sx, y + sy, z + sz) for sym, x, y, z in second_ref]


def write_mbx_nrg(path: Path, monomers: list[tuple[str, list[Atom]]]) -> None:
    lines = ["SYSTEM NRG"]
    for monomer_name, atoms in monomers:
        lines += ["MOLECULE", f"MONOMER {monomer_name}"]
        for sym, x, y, z in atoms:
            lines.append(f" {sym:2s} {x:20.10f} {y:20.10f} {z:20.10f}")
        lines += ["ENDMON", "ENDMOL"]
    lines += ["ENDSYS", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def mbx_energy_kcal(
    single_point: Path,
    mbx_json: Path,
    monomers: list[tuple[str, list[Atom]]],
    workdir: Path,
    env: dict[str, str],
    stem: str,
) -> float:
    inp = workdir / f"{stem}.nrg"
    write_mbx_nrg(inp, monomers)
    result = run([str(single_point), str(inp), str(mbx_json)], workdir, env)
    match = re.search(r"Energy=\s*(" + NUMBER_RE.pattern + r")", result.stdout)
    if not match:
        raise RuntimeError(f"Could not parse MBX energy from:\n{result.stdout}")
    return float(match.group(1))


def write_elec_pol_baseline_json(source: Path, target: Path, residual_pair: tuple[str, str]) -> None:
    with source.open(encoding="utf-8") as handle:
        data = json.load(handle)
    mbx = data.setdefault("MBX", {})
    mbx["ignore_2b_poly"] = [list(residual_pair)]
    mbx["ignore_dispersion"] = [list(residual_pair)]
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=3)
        handle.write("\n")


def parse_scan_stdout(stdout: str) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for line in stdout.splitlines():
        if not line or line.startswith("#") or line.startswith("distance_ang,"):
            continue
        if not re.match(r"\s*[-+]?\d", line):
            continue
        distance, qm_dimer, qm_a, qm_b, hybrid = next(csv.reader([line]))
        rows.append(
            {
                "distance_ang": float(distance),
                "qm_dimer_au": float(qm_dimer),
                "qm_a_iso_au": float(qm_a),
                "qm_b_iso_au": float(qm_b),
                "hybrid_ab_au": float(hybrid),
            }
        )
    return rows


def process_case(
    spec: CaseSpec,
    scan_exe: Path,
    mbx_home: Path,
    basis: Path,
    workdir: Path,
    csv_path: Path,
    npoints: int | None,
) -> list[dict[str, float | str]]:
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    shutil.copy2(spec.mbx_json, workdir / "mbx.json")

    env = os.environ.copy()
    env["QUICK_MBX_PAIR_CASE"] = spec.driver_case
    if npoints is not None:
        env["QUICK_MBX_PAIR_SCAN_NPOINTS"] = str(npoints)
    libdir = mbx_home / "lib"
    env["DYLD_LIBRARY_PATH"] = str(libdir) + (":" + env["DYLD_LIBRARY_PATH"] if env.get("DYLD_LIBRARY_PATH") else "")
    env["QUICK_BASIS"] = str(basis.resolve())

    single_point = mbx_home / "bin" / "single_point"
    mbx_b_iso = mbx_energy_kcal(
        single_point, spec.mbx_json, [(spec.mbx_name, spec.mbx_atoms)], workdir, env, f"{spec.driver_case}_mbx_b"
    )
    mbx_a_iso: float | None = None
    baseline_json: Path | None = None
    if spec.qm_mbx_name is not None:
        mbx_a_iso = mbx_energy_kcal(
            single_point,
            spec.mbx_json,
            [(spec.qm_mbx_name, spec.qm_atoms)],
            workdir,
            env,
            f"{spec.driver_case}_mbx_a",
        )
    if spec.residual_pair is not None:
        baseline_json = workdir / "mbx_elec_pol_baseline.json"
        write_elec_pol_baseline_json(spec.mbx_json, baseline_json, spec.residual_pair)

    result = run([str(scan_exe.resolve())], workdir, env)
    rows = parse_scan_stdout(result.stdout)
    if not rows:
        raise RuntimeError(f"No scan rows parsed from:\n{result.stdout}")

    out_rows: list[dict[str, float | str]] = []
    for row in rows:
        translated_b = translate_second_to_distance(spec.qm_atoms, spec.mbx_atoms, row["distance_ang"])
        out: dict[str, float | str] = dict(row)
        out["case"] = spec.driver_case
        out["pbe0_d3_kcal"] = (row["qm_dimer_au"] - row["qm_a_iso_au"] - row["qm_b_iso_au"]) * HARTREE_TO_KCAL
        out["hybrid_ab_raw_kcal"] = (row["hybrid_ab_au"] - row["qm_a_iso_au"]) * HARTREE_TO_KCAL - mbx_b_iso

        if mbx_a_iso is not None:
            mbx_dimer = mbx_energy_kcal(
                single_point,
                spec.mbx_json,
                [(spec.qm_mbx_name or "", spec.qm_atoms), (spec.mbx_name, translated_b)],
                workdir,
                env,
                f"{spec.driver_case}_mbx_dimer_{row['distance_ang']:.2f}",
            )
            out["mbx_model_kcal"] = mbx_dimer - mbx_a_iso - mbx_b_iso
            out["mbx_total_kcal"] = mbx_dimer
            if baseline_json is not None:
                mbx_baseline = mbx_energy_kcal(
                    single_point,
                    baseline_json,
                    [(spec.qm_mbx_name or "", spec.qm_atoms), (spec.mbx_name, translated_b)],
                    workdir,
                    env,
                    f"{spec.driver_case}_mbx_baseline_{row['distance_ang']:.2f}",
                )
                out["mbx_sr_disp_cross_kcal"] = mbx_dimer - mbx_baseline
                out["hybrid_ab_corrected_kcal"] = float(out["hybrid_ab_raw_kcal"]) + float(out["mbx_sr_disp_cross_kcal"])
        out_rows.append(out)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
        "distance_ang",
        "pbe0_d3_kcal",
        "mbx_model_kcal",
        "hybrid_ab_raw_kcal",
        "mbx_sr_disp_cross_kcal",
        "hybrid_ab_corrected_kcal",
        "qm_dimer_au",
        "qm_a_iso_au",
        "qm_b_iso_au",
        "hybrid_ab_au",
        "mbx_total_kcal",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)
    return out_rows


def draw_case_plot(spec: CaseSpec, rows: list[dict[str, float | str]], png: Path, pdf: Path | None = None) -> None:
    draw_multi_panel([(spec, rows)], png, pdf)


def draw_multi_panel(
    panels: list[tuple[CaseSpec, list[dict[str, float | str]]]],
    png: Path,
    pdf: Path | None = None,
) -> None:
    panel_w, panel_h = 900, 680
    width = panel_w * len(panels) + 140
    height = 720
    margin_l, margin_r, margin_t, margin_b = 112, 46, 150, 110
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    _, small_font, label_font, title_font = load_fonts()

    title = "H2O-CH4 PES: panel comparison" if len(panels) > 1 else panels[0][0].title
    draw.text((80, 34), title, fill=(10, 10, 10), font=title_font)
    if all(spec.residual_pair is not None for spec, _ in panels):
        subtitle = "Quantum full pair vs MBX full pair vs corrected QUICK-MBX hybrid"
    else:
        subtitle = "Open-shell QUICK-MBX electrostatic/polarization-only coupling; no MBX O2 residual"
    draw.text((80, 74), subtitle, fill=(65, 65, 65), font=label_font)

    all_y: list[float] = []
    for spec, rows in panels:
        all_y += [float(row["pbe0_d3_kcal"]) for row in rows]
        if spec.classical_label is not None:
            all_y += [float(row["mbx_model_kcal"]) for row in rows if row.get("mbx_model_kcal") not in ("", None)]
        ykey = "hybrid_ab_corrected_kcal" if rows[0].get("hybrid_ab_corrected_kcal") not in ("", None) else "hybrid_ab_raw_kcal"
        all_y += [float(row[ykey]) for row in rows]
    ymin, ymax = min(all_y), max(all_y)
    pad_y = max(0.08, 0.16 * (ymax - ymin))
    ymin -= pad_y
    ymax += pad_y

    for ipanel, (spec, rows) in enumerate(panels):
        origin_x = 70 + ipanel * panel_w
        plot_l = origin_x + margin_l
        plot_t = margin_t
        plot_w = panel_w - margin_l - margin_r
        plot_h = panel_h - margin_t - margin_b
        xs = [float(row["distance_ang"]) for row in rows]
        xmin, xmax = min(xs), max(xs)
        if xmax == xmin:
            xmin -= 0.5
            xmax += 0.5

        def px(x: float) -> int:
            return int(plot_l + (x - xmin) / (xmax - xmin) * plot_w)

        def py(y: float) -> int:
            return int(plot_t + (ymax - y) / (ymax - ymin) * plot_h)

        draw.text((plot_l, plot_t - 34), spec.panel_label, fill=(10, 10, 10), font=label_font)
        draw.rectangle((plot_l, plot_t, plot_l + plot_w, plot_t + plot_h), outline=(30, 30, 30), width=2)
        for i in range(6):
            y = ymin + i * (ymax - ymin) / 5
            yy = py(y)
            draw.line((plot_l, yy, plot_l + plot_w, yy), fill=(226, 226, 226))
            draw.text((origin_x + 20, yy - 10), f"{y:7.3f}", fill=(40, 40, 40), font=small_font)
        for i in range(5):
            x = xmin + i * (xmax - xmin) / 4
            xx = px(x)
            draw.line((xx, plot_t, xx, plot_t + plot_h), fill=(238, 238, 238))
            draw.text((xx - 22, plot_t + plot_h + 18), f"{x:.2f}", fill=(40, 40, 40), font=small_font)

        series: list[tuple[str, str, tuple[int, int, int], str]] = [
            ("pbe0_d3_kcal", spec.quantum_label, (31, 119, 180), "solid"),
        ]
        if spec.classical_label is not None:
            series.append(("mbx_model_kcal", spec.classical_label, (44, 160, 44), "solid"))
        ykey = "hybrid_ab_corrected_kcal" if rows[0].get("hybrid_ab_corrected_kcal") not in ("", None) else "hybrid_ab_raw_kcal"
        series.append((ykey, spec.hybrid_label, (214, 39, 40), "solid"))

        for iseries, (key, label, color, style) in enumerate(series):
            pts = [(px(float(row["distance_ang"])), py(float(row[key]))) for row in rows if row.get(key) not in ("", None)]
            if style == "dash":
                draw_dashed_line(draw, pts, color, 4)
            else:
                draw.line(pts, fill=color, width=4)
            for x, y in pts:
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline=(255, 255, 255), width=2)
            if key == "pbe0_d3_kcal":
                legend_label = "Quantum"
            elif key == "mbx_model_kcal":
                legend_label = "Classical MBX"
            elif key == "hybrid_ab_raw_kcal" and spec.residual_pair is None:
                legend_label = "Hybrid elec+pol only"
            else:
                legend_label = "Hybrid"
            lx = plot_l + plot_w - 210
            ly = plot_t + 20 + 28 * iseries
            draw.line((lx, ly + 8, lx + 36, ly + 8), fill=color, width=4)
            draw.ellipse((lx + 13, ly + 3, lx + 23, ly + 13), fill=color, outline=(255, 255, 255), width=1)
            draw.text((lx + 42, ly - 1), legend_label, fill=(20, 20, 20), font=small_font)

        draw.text((plot_l + plot_w // 2 - 110, plot_t + plot_h + 44), spec.x_label, fill=(20, 20, 20), font=label_font)
        if ipanel == 0:
            y_label = "Interaction energy (kcal/mol)"
            bbox = draw.textbbox((0, 0), y_label, font=label_font)
            label_img = Image.new("RGBA", (bbox[2] - bbox[0] + 8, bbox[3] - bbox[1] + 8), (255, 255, 255, 0))
            label_draw = ImageDraw.Draw(label_img)
            label_draw.text((4, 4), y_label, fill=(20, 20, 20), font=label_font)
            label_img = label_img.rotate(90, expand=True)
            image.paste(label_img, (8, plot_t + plot_h // 2 - label_img.height // 2), label_img)

    png.parent.mkdir(parents=True, exist_ok=True)
    image.save(png)
    if pdf is not None:
        pdf.parent.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(pdf, "PDF", resolution=300.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(list(CASES) + ["h2o_ch4_two_panel"]), required=True)
    parser.add_argument("--scan-exe", type=Path, required=True)
    parser.add_argument("--mbx-home", type=Path, default=Path("/private/tmp/mbx-quick-install"))
    parser.add_argument("--basis", type=Path, default=Path("basis"))
    parser.add_argument("--workdir", type=Path, default=Path("/private/tmp/quick-mbx-pair-scan"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot-pdf", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--npoints", type=int)
    args = parser.parse_args()

    if args.case == "h2o_ch4_two_panel":
        panel_specs = [CASES["h2o_ch4_h2o_qm"], CASES["h2o_ch4_ch4_qm"]]
        panel_rows: list[tuple[CaseSpec, list[dict[str, float | str]]]] = []
        all_rows: list[dict[str, float | str]] = []
        for spec in panel_specs:
            rows = process_case(
                spec,
                args.scan_exe,
                args.mbx_home,
                args.basis,
                args.workdir / spec.driver_case,
                args.csv.with_name(args.csv.stem + f"_{spec.driver_case}.csv"),
                args.npoints,
            )
            panel_rows.append((spec, rows))
            all_rows += rows
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "case",
                "distance_ang",
                "pbe0_d3_kcal",
                "mbx_model_kcal",
                "hybrid_ab_raw_kcal",
                "mbx_sr_disp_cross_kcal",
                "hybrid_ab_corrected_kcal",
                "qm_dimer_au",
                "qm_a_iso_au",
                "qm_b_iso_au",
                "hybrid_ab_au",
                "mbx_total_kcal",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in all_rows:
                writer.writerow(row)
        draw_multi_panel(panel_rows, args.output, args.plot_pdf or args.output.with_suffix(".pdf"))
        print(f"Wrote {args.csv}")
        print(f"Wrote {args.output}")
        print(f"Wrote {args.plot_pdf or args.output.with_suffix('.pdf')}")
    else:
        spec = CASES[args.case]
        rows = process_case(spec, args.scan_exe, args.mbx_home, args.basis, args.workdir, args.csv, args.npoints)
        draw_case_plot(spec, rows, args.output, args.plot_pdf or args.output.with_suffix(".pdf"))
        print(f"Wrote {args.csv}")
        print(f"Wrote {args.output}")
        print(f"Wrote {args.plot_pdf or args.output.with_suffix('.pdf')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
