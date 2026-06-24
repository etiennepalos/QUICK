#!/usr/bin/env python3
"""Run and plot a three-way PBE0-D3 / MB-pol water-dimer scan.

The scan is deliberately lightweight and reproducible on a laptop. It keeps
the monomer orientations from the MBX 001_mbpol example, translates monomer B
along the A--B O--O axis, and compares the same XYZ points with:

* QUICK PBE0-D3/aug-cc-pVTZ dimer interaction energy,
* MBX MB-pol dimer interaction energy,
* QUICK-MBX hybrid interaction energy with monomer A as PBE0-D3 and monomer B
  as MB-pol,
* QUICK-MBX hybrid interaction energy corrected by the MB-pol cross
  short-range/dispersion residual on the same dimer geometry.

This is a first reproduction-style calculation inspired by Lambros et al.
Figure 1, not the exact MP2/aug-cc-pVQZ optimized scan from the paper.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HARTREE_TO_KCAL = 627.5094740631
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?")

MONOMER_A = [
    ("O", -1.58972425, 1.04337922, -0.08780840),
    ("H", -0.63591971, 0.97898520, 0.00000000),
    ("H", -1.90066280, 1.74501050, -0.66454990),
]
MONOMER_B_REF = [
    ("O", 1.64924507, 1.08594656, 0.00000000),
    ("H", 2.60878026, 1.09587704, -0.02817115),
    ("H", 1.33830653, 1.78757784, 0.57674150),
]


def run(cmd: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True, check=True)


def monomer_b_at_distance(roo: float) -> list[tuple[str, float, float, float]]:
    ax, ay, az = MONOMER_A[0][1:]
    bx, by, bz = MONOMER_B_REF[0][1:]
    vx, vy, vz = bx - ax, by - ay, bz - az
    norm = (vx * vx + vy * vy + vz * vz) ** 0.5
    ux, uy, uz = vx / norm, vy / norm, vz / norm
    tx, ty, tz = ax + roo * ux, ay + roo * uy, az + roo * uz
    sx, sy, sz = tx - bx, ty - by, tz - bz
    return [(sym, x + sx, y + sy, z + sz) for sym, x, y, z in MONOMER_B_REF]


def write_mbx_nrg(path: Path, monomers: list[list[tuple[str, float, float, float]]]) -> None:
    lines = [f"SYSTEM {len(monomers)}H2O"]
    for atoms in monomers:
        lines += ["MOLECULE", "MONOMER h2o"]
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
    """Write an MBX JSON file retaining the h2o/h2o electrostatic/polar baseline."""

    with source.open(encoding="utf-8") as handle:
        data = json.load(handle)
    mbx = data.setdefault("MBX", {})
    mbx["ignore_2b_poly"] = [["h2o", "h2o"]]
    mbx["ignore_dispersion"] = [["h2o", "h2o"]]
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


def load_fonts() -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, ...]:
    font_path = Path(
        "/Users/etiennepalos/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/poppler/fonts/DejaVuSans.ttf"
    )
    bold_path = Path(
        "/Users/etiennepalos/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/libreoffice-headless/libreoffice/LibreOfficeDev.app/Contents/Resources/fonts/truetype/DejaVuSans-Bold.ttf"
    )
    if font_path.exists():
        font = ImageFont.truetype(str(font_path), 18)
        small_font = ImageFont.truetype(str(font_path), 16)
        label_font = ImageFont.truetype(str(font_path), 20)
    else:
        font = small_font = label_font = ImageFont.load_default()
    title_font = ImageFont.truetype(str(bold_path), 26) if bold_path.exists() else font
    return font, small_font, label_font, title_font


def draw_dashed_line(draw: ImageDraw.ImageDraw, pts: list[tuple[int, int]], fill: tuple[int, int, int], width: int) -> None:
    dash, gap = 18, 10
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        dx = x1 - x0
        dy = y1 - y0
        length = (dx * dx + dy * dy) ** 0.5
        if length <= 0:
            continue
        ux = dx / length
        uy = dy / length
        t = 0.0
        while t < length:
            t2 = min(t + dash, length)
            draw.line((x0 + ux * t, y0 + uy * t, x0 + ux * t2, y0 + uy * t2), fill=fill, width=width)
            t += dash + gap


def draw_plot(
    rows: list[dict[str, float]],
    png: Path,
    pdf: Path | None = None,
    include_raw_diagnostic: bool = False,
) -> None:
    width, height = 1680, 1040
    margin_l, margin_r, margin_t, margin_b = 225, 60, 110, 135
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font, small_font, label_font, title_font = load_fonts()

    series = [
        ("pbe0_d3_kcal", "Quantum: PBE0-D3 full dimer", (31, 119, 180), "solid"),
        ("mbpol_kcal", "Classical: MB-pol full dimer", (44, 160, 44), "solid"),
    ]
    if "hybrid_ab_sr_disp_corrected_kcal" in rows[0]:
        if include_raw_diagnostic:
            series.append(("hybrid_ab_raw_kcal", "Diagnostic: hybrid elec+pol only", (135, 135, 135), "dash"))
        series.append(("hybrid_ab_sr_disp_corrected_kcal", "Hybrid: QM/MB-pol + MBX sr/disp", (214, 39, 40), "solid"))
    else:
        series.append(("hybrid_ab_raw_kcal", "Hybrid: QM/MB-pol elec+pol only", (214, 39, 40), "solid"))

    xs = [row["distance_ang"] for row in rows]
    ys = [row[key] for row in rows for key, _, _, _ in series]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad_y = max(0.35, 0.14 * (ymax - ymin))
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
        draw.text((48, yy - 10), f"{y:6.2f}", fill=(40, 40, 40), font=small_font)
    for i in range(6):
        x = xmin + i * (xmax - xmin) / 5
        xx = px(x)
        draw.line((xx, margin_t, xx, margin_t + plot_h), fill=(238, 238, 238))
        draw.text((xx - 22, margin_t + plot_h + 18), f"{x:.2f}", fill=(40, 40, 40), font=small_font)

    draw.text((margin_l, 28), "Water dimer PES: quantum vs classical vs hybrid", fill=(10, 10, 10), font=title_font)
    draw.text(
        (margin_l, 68),
        "QUICK PBE0-D3/aug-cc-pVTZ vs MBX MB-pol vs QUICK-MBX with MBX short-range/dispersion residual",
        fill=(65, 65, 65),
        font=label_font,
    )
    draw.text((margin_l + plot_w // 2 - 115, height - 54), "O-O distance (Angstrom)", fill=(20, 20, 20), font=label_font)

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
        lx = margin_l + plot_w - 520
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
    parser.add_argument("--mbx-json", type=Path, default=Path("/private/tmp/MBX-quick-plan/examples/PEFs/001_mbpol/C++/mbx.json"))
    parser.add_argument("--basis", type=Path, default=Path("basis"))
    parser.add_argument("--workdir", type=Path, default=Path("/private/tmp/quick-mbx-lambros-dimer-scan"))
    parser.add_argument("--output", type=Path, default=Path("validation/quick_mbx/lambros_dimer_three_way_augccpvtz.png"))
    parser.add_argument("--plot-pdf", type=Path)
    parser.add_argument("--csv", type=Path, default=Path("validation/quick_mbx/lambros_dimer_three_way_augccpvtz.csv"))
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
    mbx_a_iso = mbx_energy_kcal(single_point, args.mbx_json, [MONOMER_A], workdir, env, "mbx_monomer_a")
    mbx_b_iso = mbx_energy_kcal(single_point, args.mbx_json, [MONOMER_B_REF], workdir, env, "mbx_monomer_b")

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
            f"mbx_dimer_{row['distance_ang']:.2f}",
        )
        mbx_dimer_elec_pol_baseline = mbx_energy_kcal(
            single_point,
            elec_pol_baseline_json,
            [MONOMER_A, monomer_b],
            workdir,
            env,
            f"mbx_dimer_elec_pol_baseline_{row['distance_ang']:.2f}",
        )
        row["mbpol_sr_disp_cross_kcal"] = mbx_dimer - mbx_dimer_elec_pol_baseline
        row["mbpol_total_kcal"] = mbx_dimer
        row["mbpol_kcal"] = mbx_dimer - mbx_a_iso - mbx_b_iso
        row["pbe0_d3_kcal"] = (row["qm_dimer_au"] - row["qm_a_iso_au"] - row["qm_b_iso_au"]) * HARTREE_TO_KCAL
        row["hybrid_ab_raw_kcal"] = (row["hybrid_ab_au"] - row["qm_a_iso_au"]) * HARTREE_TO_KCAL - mbx_b_iso
        row["hybrid_ab_sr_disp_corrected_kcal"] = row["hybrid_ab_raw_kcal"] + row["mbpol_sr_disp_cross_kcal"]

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "distance_ang",
        "pbe0_d3_kcal",
        "mbpol_kcal",
        "hybrid_ab_raw_kcal",
        "mbpol_sr_disp_cross_kcal",
        "hybrid_ab_sr_disp_corrected_kcal",
        "qm_dimer_au",
        "qm_a_iso_au",
        "qm_b_iso_au",
        "hybrid_ab_au",
        "mbpol_total_kcal",
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
    for row in rows:
        print(
            f"R={row['distance_ang']:.2f} "
            f"PBE0-D3={row['pbe0_d3_kcal']: .6f} "
            f"MB-pol={row['mbpol_kcal']: .6f} "
            f"Hybrid raw={row['hybrid_ab_raw_kcal']: .6f} "
            f"sr/disp={row['mbpol_sr_disp_cross_kcal']: .6f} "
            f"Hybrid+sr/disp={row['hybrid_ab_sr_disp_corrected_kcal']: .6f} kcal/mol"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
