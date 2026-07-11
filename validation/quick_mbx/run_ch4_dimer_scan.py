#!/usr/bin/env python3
"""Run and plot a small PBE0-D3 / MB-nrg methane-dimer scan.

This is a lightweight second-system validation for the QUICK-MBX adapter.
It uses the first two methane monomers from the MBX
040_ch4-ch4_mb-nrg_2bnb example, translates monomer B along the C--C axis,
and compares:

* QUICK PBE0-D3/aug-cc-pVTZ dimer interaction energy,
* MBX MB-nrg methane dimer interaction energy,
* QUICK-MBX hybrid interaction energy with monomer A as PBE0-D3 and monomer B
  as MBX ch4,
* the same hybrid corrected by the MBX ch4/ch4 2B-plus-dispersion residual.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

from run_lambros_dimer_scan import HARTREE_TO_KCAL, NUMBER_RE, draw_dashed_line, load_fonts, run


MONOMER_A = [
    ("C", 0.1713338779, 0.0402580564, 0.2343860323),
    ("H", -0.0070796542, -0.8125863746, 0.8833225480),
    ("H", 0.5295486445, 0.8773662734, 0.8265597026),
    ("H", 0.9203628978, -0.2207373935, -0.5083610174),
    ("H", -0.7529439533, 0.3158362319, -0.2656777218),
]

MONOMER_B_REF = [
    ("C", 3.0280437961, 2.6077617055, 0.8685578966),
    ("H", 2.6352959969, 2.4318876714, 1.8662202179),
    ("H", 3.2755779869, 1.6557409443, 0.4078436157),
    ("H", 2.2795425973, 3.1170415698, 0.2678406518),
    ("H", 3.9204494427, 3.2239426955, 0.9341346223),
]


def monomer_b_at_distance(rcc: float) -> list[tuple[str, float, float, float]]:
    ax, ay, az = MONOMER_A[0][1:]
    bx, by, bz = MONOMER_B_REF[0][1:]
    vx, vy, vz = bx - ax, by - ay, bz - az
    norm = (vx * vx + vy * vy + vz * vz) ** 0.5
    ux, uy, uz = vx / norm, vy / norm, vz / norm
    tx, ty, tz = ax + rcc * ux, ay + rcc * uy, az + rcc * uz
    sx, sy, sz = tx - bx, ty - by, tz - bz
    return [(sym, x + sx, y + sy, z + sz) for sym, x, y, z in MONOMER_B_REF]


def write_mbx_nrg(path: Path, monomers: list[list[tuple[str, float, float, float]]]) -> None:
    lines = ["SYSTEM NRG"]
    for atoms in monomers:
        lines += ["MOLECULE", "MONOMER ch4"]
        for sym, x, y, z in atoms:
            lines.append(f" {sym:2s} {x:20.10f} {y:20.10f} {z:20.10f}")
        lines += ["ENDMON", "ENDMOL"]
    lines += ["ENDSYS", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def mbx_energy_kcal(
    single_point: Path,
    mbx_json: Path,
    monomers: list[list[tuple[str, float, float, float]]],
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


def write_elec_pol_baseline_json(source: Path, target: Path) -> None:
    """Write an MBX JSON file retaining the ch4/ch4 electrostatic/polar baseline."""

    with source.open(encoding="utf-8") as handle:
        data = json.load(handle)
    mbx = data.setdefault("MBX", {})
    mbx["ignore_2b_poly"] = [["ch4", "ch4"]]
    mbx["ignore_dispersion"] = [["ch4", "ch4"]]
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


def draw_plot(
    rows: list[dict[str, float]],
    png: Path,
    pdf: Path | None = None,
    include_raw_diagnostic: bool = False,
) -> None:
    width, height = 1600, 980
    margin_l, margin_r, margin_t, margin_b = 225, 60, 108, 132
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    _, small_font, label_font, title_font = load_fonts()

    series = [
        ("pbe0_d3_kcal", "Quantum: PBE0-D3 full dimer", (31, 119, 180), "solid"),
        ("mbnrg_kcal", "Classical: MB-nrg ch4 full dimer", (44, 160, 44), "solid"),
    ]
    if include_raw_diagnostic:
        series.append(("hybrid_ab_raw_kcal", "Diagnostic: hybrid elec+pol only", (135, 135, 135), "dash"))
    series.append(("hybrid_ab_sr_disp_corrected_kcal", "Hybrid: QM/MBX ch4 + MBX 2B/disp", (214, 39, 40), "solid"))

    xs = [row["distance_ang"] for row in rows]
    ys = [row[key] for row in rows for key, _, _, _ in series]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad_y = max(0.08, 0.16 * (ymax - ymin))
    ymin -= pad_y
    ymax += pad_y

    def px(x: float) -> int:
        return int(margin_l + (x - xmin) / (xmax - xmin) * plot_w)

    def py(y: float) -> int:
        return int(margin_t + (ymax - y) / (ymax - ymin) * plot_h)

    draw.rectangle((margin_l, margin_t, margin_l + plot_w, margin_t + plot_h), outline=(30, 30, 30), width=2)
    for i in range(6):
        y = ymin + i * (ymax - ymin) / 5
        yy = py(y)
        draw.line((margin_l, yy, margin_l + plot_w, yy), fill=(226, 226, 226))
        draw.text((44, yy - 10), f"{y:7.3f}", fill=(40, 40, 40), font=small_font)
    for i in range(6):
        x = xmin + i * (xmax - xmin) / 5
        xx = px(x)
        draw.line((xx, margin_t, xx, margin_t + plot_h), fill=(238, 238, 238))
        draw.text((xx - 24, margin_t + plot_h + 18), f"{x:.2f}", fill=(40, 40, 40), font=small_font)

    draw.text((margin_l, 28), "Methane dimer PES: quantum vs classical vs hybrid", fill=(10, 10, 10), font=title_font)
    draw.text(
        (margin_l, 68),
        "QUICK PBE0-D3/aug-cc-pVTZ vs MBX MB-nrg ch4 vs QUICK-MBX with MBX 2B/dispersion residual",
        fill=(65, 65, 65),
        font=label_font,
    )
    draw.text((margin_l + plot_w // 2 - 120, height - 54), "C-C distance (Angstrom)", fill=(20, 20, 20), font=label_font)

    y_label = "Interaction energy (kcal/mol)"
    bbox = draw.textbbox((0, 0), y_label, font=label_font)
    label_img = Image.new("RGBA", (bbox[2] - bbox[0] + 8, bbox[3] - bbox[1] + 8), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_img)
    label_draw.text((4, 4), y_label, fill=(20, 20, 20), font=label_font)
    label_img = label_img.rotate(90, expand=True)
    image.paste(label_img, (4, margin_t + plot_h // 2 - label_img.height // 2), label_img)

    for iseries, (key, label, color, style) in enumerate(series):
        pts = [(px(row["distance_ang"]), py(row[key])) for row in rows]
        if style == "dash":
            draw_dashed_line(draw, pts, color, 4)
        else:
            draw.line(pts, fill=color, width=4)
        for x, y in pts:
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color, outline=(255, 255, 255), width=2)
        lx = margin_l + plot_w - 560
        ly = margin_t + 20 + 30 * iseries
        if style == "dash":
            draw_dashed_line(draw, [(lx, ly + 8), (lx + 40, ly + 8)], color, 4)
        else:
            draw.line((lx, ly + 8, lx + 40, ly + 8), fill=color, width=4)
        draw.ellipse((lx + 14, ly + 2, lx + 26, ly + 14), fill=color, outline=(255, 255, 255), width=2)
        draw.text((lx + 46, ly - 1), label, fill=(20, 20, 20), font=small_font)

    png.parent.mkdir(parents=True, exist_ok=True)
    image.save(png)
    if pdf is not None:
        pdf.parent.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(pdf, "PDF", resolution=300.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-exe", type=Path, required=True)
    parser.add_argument("--mbx-home", type=Path, default=Path("/private/tmp/mbx-quick-install"))
    parser.add_argument(
        "--mbx-json",
        type=Path,
        default=Path("/private/tmp/MBX-quick-plan/examples/PEFs/040_ch4-ch4_mb-nrg_2bnb/C++/mbx.json"),
    )
    parser.add_argument("--basis", type=Path, default=Path("basis"))
    parser.add_argument("--workdir", type=Path, default=Path("/private/tmp/quick-mbx-ch4-dimer-scan"))
    parser.add_argument("--output", type=Path, default=Path("validation/quick_mbx/ch4_dimer_three_way_augccpvtz.png"))
    parser.add_argument("--plot-pdf", type=Path)
    parser.add_argument("--csv", type=Path, default=Path("validation/quick_mbx/ch4_dimer_three_way_augccpvtz.csv"))
    parser.add_argument("--include-raw-diagnostic", action="store_true")
    args = parser.parse_args()

    workdir = args.workdir.resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    shutil.copy2(args.mbx_json, workdir / "mbx.json")
    elec_pol_baseline_json = workdir / "mbx_elec_pol_baseline.json"
    write_elec_pol_baseline_json(args.mbx_json, elec_pol_baseline_json)

    env = os.environ.copy()
    libdir = args.mbx_home / "lib"
    env["DYLD_LIBRARY_PATH"] = str(libdir) + (":" + env["DYLD_LIBRARY_PATH"] if env.get("DYLD_LIBRARY_PATH") else "")
    env["QUICK_BASIS"] = str(args.basis.resolve())

    single_point = args.mbx_home / "bin" / "single_point"
    mbx_a_iso = mbx_energy_kcal(single_point, args.mbx_json, [MONOMER_A], workdir, env, "mbx_ch4_monomer_a")
    mbx_b_iso = mbx_energy_kcal(single_point, args.mbx_json, [MONOMER_B_REF], workdir, env, "mbx_ch4_monomer_b")

    result = run([str(args.scan_exe.resolve())], workdir, env)
    rows = parse_scan_stdout(result.stdout)
    if not rows:
        raise RuntimeError(f"No scan rows parsed from:\n{result.stdout}")

    for row in rows:
        monomer_b = monomer_b_at_distance(row["distance_ang"])
        mbx_dimer = mbx_energy_kcal(
            single_point,
            args.mbx_json,
            [MONOMER_A, monomer_b],
            workdir,
            env,
            f"mbx_ch4_dimer_{row['distance_ang']:.2f}",
        )
        mbx_dimer_elec_pol_baseline = mbx_energy_kcal(
            single_point,
            elec_pol_baseline_json,
            [MONOMER_A, monomer_b],
            workdir,
            env,
            f"mbx_ch4_dimer_elec_pol_baseline_{row['distance_ang']:.2f}",
        )
        row["mbnrg_sr_disp_cross_kcal"] = mbx_dimer - mbx_dimer_elec_pol_baseline
        row["mbnrg_total_kcal"] = mbx_dimer
        row["mbnrg_kcal"] = mbx_dimer - mbx_a_iso - mbx_b_iso
        row["pbe0_d3_kcal"] = (row["qm_dimer_au"] - row["qm_a_iso_au"] - row["qm_b_iso_au"]) * HARTREE_TO_KCAL
        row["hybrid_ab_raw_kcal"] = (row["hybrid_ab_au"] - row["qm_a_iso_au"]) * HARTREE_TO_KCAL - mbx_b_iso
        row["hybrid_ab_sr_disp_corrected_kcal"] = row["hybrid_ab_raw_kcal"] + row["mbnrg_sr_disp_cross_kcal"]

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "distance_ang",
        "pbe0_d3_kcal",
        "mbnrg_kcal",
        "hybrid_ab_raw_kcal",
        "mbnrg_sr_disp_cross_kcal",
        "hybrid_ab_sr_disp_corrected_kcal",
        "qm_dimer_au",
        "qm_a_iso_au",
        "qm_b_iso_au",
        "hybrid_ab_au",
        "mbnrg_total_kcal",
    ]
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})

    plot_pdf = args.plot_pdf if args.plot_pdf is not None else args.output.with_suffix(".pdf")
    draw_plot(rows, args.output, plot_pdf, include_raw_diagnostic=args.include_raw_diagnostic)
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.output}")
    print(f"Wrote {plot_pdf}")
    print("Baseline JSON: ignore_2b_poly=[[ch4,ch4]], ignore_dispersion=[[ch4,ch4]]")
    for row in rows:
        print(
            f"R={row['distance_ang']:.2f} "
            f"PBE0-D3={row['pbe0_d3_kcal']: .6f} "
            f"MB-nrg={row['mbnrg_kcal']: .6f} "
            f"Hybrid raw={row['hybrid_ab_raw_kcal']: .6f} "
            f"2B/disp={row['mbnrg_sr_disp_cross_kcal']: .6f} "
            f"Hybrid+2B/disp={row['hybrid_ab_sr_disp_corrected_kcal']: .6f} kcal/mol"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
