#!/usr/bin/env python3
"""Analyze PBE-D3/MB-PBE water-dimer hybrid bookkeeping."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw

from run_lambros_dimer_scan import draw_dashed_line, load_fonts


def read_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out = {key: float(value) for key, value in row.items() if key != "point"}
            out["point"] = float(row["point"])
            rows.append(out)
    return rows


def analyze_rows(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    analyzed: list[dict[str, float]] = []
    for row in rows:
        out = dict(row)
        out["mbpbe_baseline_kcal"] = row["mbpbe_kcal"] - row["mbpbe_sr_disp_cross_kcal"]
        out["qmmb_baseline_gap_kcal"] = row["hybrid_mbpbe_raw_kcal"] - out["mbpbe_baseline_kcal"]
        out["hybrid_mbpbe_baseline_aligned_kcal"] = (
            row["hybrid_mbpbe_corrected_kcal"] - out["qmmb_baseline_gap_kcal"]
        )
        out["hybrid_mbpbe_current_error_kcal"] = row["hybrid_mbpbe_corrected_kcal"] - row["pbe_d3_kcal"]
        out["hybrid_mbpbe_aligned_error_kcal"] = (
            out["hybrid_mbpbe_baseline_aligned_kcal"] - row["pbe_d3_kcal"]
        )
        analyzed.append(out)
    return analyzed


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = [
        "point",
        "actual_distance_ang",
        "pbe_d3_kcal",
        "hybrid_mbpbe_raw_kcal",
        "mbpbe_baseline_kcal",
        "qmmb_baseline_gap_kcal",
        "mbpbe_sr_disp_cross_kcal",
        "hybrid_mbpbe_corrected_kcal",
        "hybrid_mbpbe_baseline_aligned_kcal",
        "hybrid_mbpbe_current_error_kcal",
        "hybrid_mbpbe_aligned_error_kcal",
        "mbpbe_kcal",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def draw_plot(rows: list[dict[str, float]], png: Path, pdf: Path | None) -> None:
    width, height = 1780, 1180
    margin_l, margin_r, margin_t, margin_b = 230, 80, 125, 145
    gap = 70
    top_h = 610
    bottom_h = height - margin_t - margin_b - top_h - gap
    plot_w = width - margin_l - margin_r

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    _, small_font, label_font, title_font = load_fonts()

    xs = [row["actual_distance_ang"] for row in rows]
    xmin, xmax = min(xs), max(xs)

    top_series = [
        ("pbe_d3_kcal", "PBE-D3 full QM", (31, 119, 180), "solid"),
        ("hybrid_mbpbe_corrected_kcal", "PBE-D3/MB-PBE current", (255, 127, 14), "solid"),
        ("hybrid_mbpbe_baseline_aligned_kcal", "PBE-D3/MB-PBE baseline-aligned diagnostic", (44, 160, 44), "solid"),
    ]
    bottom_series = [
        ("hybrid_mbpbe_current_error_kcal", "current - PBE-D3", (255, 127, 14), "solid"),
        ("hybrid_mbpbe_aligned_error_kcal", "aligned - PBE-D3", (44, 160, 44), "solid"),
        ("qmmb_baseline_gap_kcal", "raw QM/MB baseline - MB-PBE baseline", (127, 127, 127), "dash"),
    ]

    def px(x: float) -> int:
        return int(margin_l + (x - xmin) / (xmax - xmin) * plot_w)

    def draw_panel(
        y0: int,
        h: int,
        series: list[tuple[str, str, tuple[int, int, int], str]],
        ylabel: str,
        force_zero: bool = False,
    ) -> None:
        vals = [row[key] for row in rows for key, _, _, _ in series]
        ymin, ymax = min(vals), max(vals)
        if force_zero:
            ymin = min(ymin, 0.0)
            ymax = max(ymax, 0.0)
        pad = max(0.15, 0.14 * (ymax - ymin))
        ymin -= pad
        ymax += pad

        def py(y: float) -> int:
            return int(y0 + (ymax - y) / (ymax - ymin) * h)

        draw.rectangle((margin_l, y0, margin_l + plot_w, y0 + h), outline=(30, 30, 30), width=2)
        for i in range(6):
            y = ymin + i * (ymax - ymin) / 5
            yy = py(y)
            draw.line((margin_l, yy, margin_l + plot_w, yy), fill=(226, 226, 226))
            draw.text((45, yy - 10), f"{y:7.2f}", fill=(40, 40, 40), font=small_font)
        for i in range(6):
            x = xmin + i * (xmax - xmin) / 5
            xx = px(x)
            draw.line((xx, y0, xx, y0 + h), fill=(238, 238, 238))
            draw.text((xx - 24, y0 + h + 14), f"{x:.2f}", fill=(40, 40, 40), font=small_font)
        if force_zero:
            zy = py(0.0)
            draw.line((margin_l, zy, margin_l + plot_w, zy), fill=(80, 80, 80), width=2)

        bbox = draw.textbbox((0, 0), ylabel, font=label_font)
        label_img = Image.new("RGBA", (bbox[2] - bbox[0] + 8, bbox[3] - bbox[1] + 8), (255, 255, 255, 0))
        label_draw = ImageDraw.Draw(label_img)
        label_draw.text((4, 4), ylabel, fill=(20, 20, 20), font=label_font)
        label_img = label_img.rotate(90, expand=True)
        image.paste(label_img, (6, y0 + h // 2 - label_img.height // 2), label_img)

        lx = margin_l + plot_w - 650
        ly = y0 + 20
        for iseries, (key, label, color, style) in enumerate(series):
            pts = [(px(row["actual_distance_ang"]), py(row[key])) for row in rows]
            if style == "dash":
                draw_dashed_line(draw, pts, color, 4)
            else:
                draw.line(pts, fill=color, width=4)
            for x, y in pts:
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline=(255, 255, 255), width=2)
            yy = ly + 28 * iseries
            if style == "dash":
                draw_dashed_line(draw, [(lx, yy + 8), (lx + 40, yy + 8)], color, 4)
            else:
                draw.line((lx, yy + 8, lx + 40, yy + 8), fill=color, width=4)
            draw.ellipse((lx + 15, yy + 3, lx + 25, yy + 13), fill=color, outline=(255, 255, 255), width=2)
            draw.text((lx + 50, yy - 1), label, fill=(20, 20, 20), font=small_font)

    draw.text((margin_l, 30), "PBE-D3/MB-PBE water-dimer bookkeeping diagnostic", fill=(10, 10, 10), font=title_font)
    draw.text(
        (margin_l, 72),
        "The aligned diagnostic subtracts the raw QM/MB electrostatic-polarization baseline mismatch.",
        fill=(65, 65, 65),
        font=label_font,
    )
    draw_panel(margin_t, top_h, top_series, "Interaction energy (kcal/mol)")
    draw_panel(margin_t + top_h + gap, bottom_h, bottom_series, "Difference / gap (kcal/mol)", force_zero=True)
    draw.text((margin_l + plot_w // 2 - 95, height - 54), "O-O distance (Angstrom)", fill=(20, 20, 20), font=label_font)

    png.parent.mkdir(parents=True, exist_ok=True)
    image.save(png)
    if pdf is not None:
        pdf.parent.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(pdf, "PDF", resolution=300.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("validation/quick_mbx/water_dimer_mbpolopt_pbe_d3_mbpbe_20pt.csv"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("validation/quick_mbx/water_dimer_mbpbe_bookkeeping_diagnostic.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/quick_mbx/water_dimer_mbpbe_bookkeeping_diagnostic.png"),
    )
    parser.add_argument("--plot-pdf", type=Path)
    args = parser.parse_args()

    rows = analyze_rows(read_rows(args.input))
    write_csv(args.csv, rows)
    draw_plot(rows, args.output, args.plot_pdf or args.output.with_suffix(".pdf"))

    max_current = max(abs(row["hybrid_mbpbe_current_error_kcal"]) for row in rows)
    max_aligned = max(abs(row["hybrid_mbpbe_aligned_error_kcal"]) for row in rows)
    rms_current = (sum(row["hybrid_mbpbe_current_error_kcal"] ** 2 for row in rows) / len(rows)) ** 0.5
    rms_aligned = (sum(row["hybrid_mbpbe_aligned_error_kcal"] ** 2 for row in rows) / len(rows)) ** 0.5
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.plot_pdf or args.output.with_suffix('.pdf')}")
    print(f"current max_abs_error={max_current:.6f} rms_error={rms_current:.6f} kcal/mol")
    print(f"aligned max_abs_error={max_aligned:.6f} rms_error={rms_aligned:.6f} kcal/mol")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
