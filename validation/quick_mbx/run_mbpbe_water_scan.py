#!/usr/bin/env python3
"""Run a PBE-D3 / MB-pol / MB-PBE water-dimer scan on optimized geometries."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

from run_lambros_dimer_scan import HARTREE_TO_KCAL, NUMBER_RE, draw_dashed_line, load_fonts, run


Atom = tuple[str, float, float, float]


def read_xyz_stack(path: Path) -> list[tuple[int, float | None, list[Atom]]]:
    frames: list[tuple[int, float | None, list[Atom]]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    point = 1
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        natom = int(lines[i].strip())
        comment = lines[i + 1] if i + 1 < len(lines) else ""
        atoms: list[Atom] = []
        for atom_line in lines[i + 2 : i + 2 + natom]:
            parts = atom_line.split()
            atoms.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
        target_match = re.search(r"target_R_OO_ang=(" + NUMBER_RE.pattern + r")", comment)
        target_r = float(target_match.group(1)) if target_match else None
        frames.append((point, target_r, atoms))
        point += 1
        i += natom + 2
    return frames


def oo_distance(atoms: list[Atom]) -> float:
    _, ax, ay, az = atoms[0]
    _, bx, by, bz = atoms[3]
    return ((bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2) ** 0.5


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


def write_elec_pol_baseline_json(source: Path, target: Path, monomer_name: str) -> None:
    with source.open(encoding="utf-8") as handle:
        data = json.load(handle)
    mbx = data.setdefault("MBX", {})
    mbx["ignore_2b_poly"] = [[monomer_name, monomer_name]]
    mbx["ignore_dispersion"] = [[monomer_name, monomer_name]]
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=3)
        handle.write("\n")


def parse_scan_stdout(stdout: str) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for line in stdout.splitlines():
        if not line or line.startswith("#") or line.startswith("point,"):
            continue
        if not re.match(r"\s*\d", line):
            continue
        point, distance, qm_dimer, qm_a, qm_b, hybrid_mbpol, hybrid_mbpbe = next(csv.reader([line]))
        rows.append(
            {
                "point": int(point),
                "distance_ang": float(distance),
                "qm_dimer_au": float(qm_dimer),
                "qm_a_iso_au": float(qm_a),
                "qm_b_iso_au": float(qm_b),
                "hybrid_mbpol_au": float(hybrid_mbpol),
                "hybrid_mbpbe_au": float(hybrid_mbpbe),
            }
        )
    return rows


def compute_rows(
    frames: list[tuple[int, float | None, list[Atom]]],
    scan_rows: list[dict[str, float]],
    single_point: Path,
    mbx_json: Path,
    baseline_jsons: dict[str, Path],
    workdir: Path,
    env: dict[str, str],
) -> list[dict[str, float]]:
    if len(frames) != len(scan_rows):
        raise RuntimeError(f"XYZ frame count ({len(frames)}) does not match QUICK scan rows ({len(scan_rows)}).")

    rows: list[dict[str, float]] = []
    for (point, target_r, atoms), row in zip(frames, scan_rows):
        mon_a = atoms[:3]
        mon_b = atoms[3:]
        out = dict(row)
        out["target_distance_ang"] = target_r if target_r is not None else oo_distance(atoms)
        out["actual_distance_ang"] = oo_distance(atoms)
        out["pbe_d3_kcal"] = (row["qm_dimer_au"] - row["qm_a_iso_au"] - row["qm_b_iso_au"]) * HARTREE_TO_KCAL

        for model, hkey in [("h2o", "mbpol"), ("mbpbe", "mbpbe")]:
            mbx_a = mbx_energy_kcal(single_point, mbx_json, [(model, mon_a)], workdir, env, f"{hkey}_a_{point:02d}")
            mbx_b = mbx_energy_kcal(single_point, mbx_json, [(model, mon_b)], workdir, env, f"{hkey}_b_{point:02d}")
            mbx_dimer = mbx_energy_kcal(
                single_point,
                mbx_json,
                [(model, mon_a), (model, mon_b)],
                workdir,
                env,
                f"{hkey}_dimer_{point:02d}",
            )
            mbx_baseline = mbx_energy_kcal(
                single_point,
                baseline_jsons[model],
                [(model, mon_a), (model, mon_b)],
                workdir,
                env,
                f"{hkey}_baseline_{point:02d}",
            )
            out[f"{hkey}_kcal"] = mbx_dimer - mbx_a - mbx_b
            out[f"{hkey}_sr_disp_cross_kcal"] = mbx_dimer - mbx_baseline
            out[f"{hkey}_total_kcal"] = mbx_dimer
            hybrid_au_key = "hybrid_mbpol_au" if model == "h2o" else "hybrid_mbpbe_au"
            out[f"hybrid_{hkey}_raw_kcal"] = (row[hybrid_au_key] - row["qm_a_iso_au"]) * HARTREE_TO_KCAL - mbx_b
            out[f"hybrid_{hkey}_corrected_kcal"] = (
                out[f"hybrid_{hkey}_raw_kcal"] + out[f"{hkey}_sr_disp_cross_kcal"]
            )
        rows.append(out)
    return rows


def draw_plot(
    rows: list[dict[str, float]],
    png: Path,
    pdf: Path | None,
    include_raw: bool,
    title: str,
    subtitle: str,
) -> None:
    width, height = 1780, 1060
    margin_l, margin_r, margin_t, margin_b = 230, 70, 120, 140
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    _, small_font, label_font, title_font = load_fonts()

    series: list[tuple[str, str, tuple[int, int, int], str]] = [
        ("pbe_d3_kcal", "Quantum: PBE-D3 full dimer", (31, 119, 180), "solid"),
        ("mbpol_kcal", "Classical: MB-pol", (44, 160, 44), "solid"),
        ("mbpbe_kcal", "Classical: MB-PBE", (23, 156, 174), "solid"),
        ("hybrid_mbpol_corrected_kcal", "Hybrid: PBE-D3/MB-pol", (214, 39, 40), "solid"),
        ("hybrid_mbpbe_corrected_kcal", "Hybrid: PBE-D3/MB-PBE", (255, 127, 14), "solid"),
    ]
    if include_raw:
        series += [
            ("hybrid_mbpol_raw_kcal", "Raw PBE-D3/MB-pol elec+pol", (145, 145, 145), "dash"),
            ("hybrid_mbpbe_raw_kcal", "Raw PBE-D3/MB-PBE elec+pol", (95, 95, 95), "dash"),
        ]

    xs = [row["actual_distance_ang"] for row in rows]
    ys = [row[key] for row in rows for key, _, _, _ in series]
    xmin, xmax = min(xs), max(xs)
    if xmax == xmin:
        xmin -= 0.5
        xmax += 0.5
    ymin, ymax = min(ys), max(ys)
    pad_y = max(0.25, 0.14 * (ymax - ymin))
    ymin -= pad_y
    ymax += pad_y

    def px(x: float) -> int:
        return int(margin_l + (x - xmin) / (xmax - xmin) * plot_w)

    def py(y: float) -> int:
        return int(margin_t + (ymax - y) / (ymax - ymin) * plot_h)

    draw.text((margin_l, 30), title, fill=(10, 10, 10), font=title_font)
    draw.text((margin_l, 72), subtitle, fill=(65, 65, 65), font=label_font)
    draw.rectangle((margin_l, margin_t, margin_l + plot_w, margin_t + plot_h), outline=(30, 30, 30), width=2)

    for i in range(6):
        y = ymin + i * (ymax - ymin) / 5
        yy = py(y)
        draw.line((margin_l, yy, margin_l + plot_w, yy), fill=(226, 226, 226))
        draw.text((48, yy - 10), f"{y:7.2f}", fill=(40, 40, 40), font=small_font)
    for i in range(6):
        x = xmin + i * (xmax - xmin) / 5
        xx = px(x)
        draw.line((xx, margin_t, xx, margin_t + plot_h), fill=(238, 238, 238))
        draw.text((xx - 24, margin_t + plot_h + 18), f"{x:.2f}", fill=(40, 40, 40), font=small_font)

    y_label = "Interaction energy (kcal/mol)"
    bbox = draw.textbbox((0, 0), y_label, font=label_font)
    label_img = Image.new("RGBA", (bbox[2] - bbox[0] + 8, bbox[3] - bbox[1] + 8), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_img)
    label_draw.text((4, 4), y_label, fill=(20, 20, 20), font=label_font)
    label_img = label_img.rotate(90, expand=True)
    image.paste(label_img, (6, margin_t + plot_h // 2 - label_img.height // 2), label_img)
    draw.text((margin_l + plot_w // 2 - 88, height - 54), "O-O distance (Angstrom)", fill=(20, 20, 20), font=label_font)

    legend_x = margin_l + plot_w - 500
    legend_y = margin_t + 20
    for iseries, (key, label, color, style) in enumerate(series):
        pts = [(px(row["actual_distance_ang"]), py(row[key])) for row in rows]
        if style == "dash":
            draw_dashed_line(draw, pts, color, 4)
        else:
            draw.line(pts, fill=color, width=4)
        for x, y in pts:
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color, outline=(255, 255, 255), width=2)
        ly = legend_y + 29 * iseries
        if style == "dash":
            draw_dashed_line(draw, [(legend_x, ly + 8), (legend_x + 40, ly + 8)], color, 4)
        else:
            draw.line((legend_x, ly + 8, legend_x + 40, ly + 8), fill=color, width=4)
        draw.ellipse((legend_x + 14, ly + 2, legend_x + 26, ly + 14), fill=color, outline=(255, 255, 255), width=2)
        draw.text((legend_x + 48, ly - 1), label, fill=(20, 20, 20), font=small_font)

    png.parent.mkdir(parents=True, exist_ok=True)
    image.save(png)
    if pdf is not None:
        pdf.parent.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(pdf, "PDF", resolution=300.0)


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = [
        "point",
        "target_distance_ang",
        "actual_distance_ang",
        "pbe_d3_kcal",
        "mbpol_kcal",
        "mbpbe_kcal",
        "hybrid_mbpol_raw_kcal",
        "mbpol_sr_disp_cross_kcal",
        "hybrid_mbpol_corrected_kcal",
        "hybrid_mbpbe_raw_kcal",
        "mbpbe_sr_disp_cross_kcal",
        "hybrid_mbpbe_corrected_kcal",
        "qm_dimer_au",
        "qm_a_iso_au",
        "qm_b_iso_au",
        "hybrid_mbpol_au",
        "hybrid_mbpbe_au",
        "mbpol_total_kcal",
        "mbpbe_total_kcal",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def representative_table(rows: list[dict[str, float]]) -> list[dict[str, float | str]]:
    short = min(rows, key=lambda row: row["actual_distance_ang"])
    minimum = min(rows, key=lambda row: row["pbe_d3_kcal"])
    long = max(rows, key=lambda row: row["actual_distance_ang"])
    return [
        {"region": "short", **short},
        {"region": "minimum", **minimum},
        {"region": "long", **long},
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-exe", type=Path, required=True)
    parser.add_argument(
        "--xyz",
        type=Path,
        default=Path("validation/quick_mbx/water_dimer_b3lyp_d3_augccpvtz_constrained_10pt.xyz"),
    )
    parser.add_argument("--mbx-home", type=Path, default=Path("/private/tmp/mbx-quick-install"))
    parser.add_argument("--mbx-json", type=Path, default=Path("/private/tmp/MBX-quick-plan/examples/PEFs/001_mbpol/C++/mbx.json"))
    parser.add_argument("--basis", type=Path, default=Path("basis"))
    parser.add_argument("--workdir", type=Path, default=Path("/private/tmp/quick-mbx-mbpbe-water-scan"))
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("validation/quick_mbx/water_dimer_b3lypopt_pbe_d3_mbpbe_10pt.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/quick_mbx/water_dimer_b3lypopt_pbe_d3_mbpbe_10pt.png"),
    )
    parser.add_argument("--plot-pdf", type=Path)
    parser.add_argument("--include-raw-diagnostic", action="store_true")
    parser.add_argument("--plot-title", default="Water dimer fixed-geometry MB-PBE scan")
    parser.add_argument(
        "--plot-subtitle",
        default="PBE-D3/aug-cc-pVTZ quantum reference vs MB-pol, MB-PBE, and corrected QUICK-MBX hybrids",
    )
    args = parser.parse_args()

    frames = read_xyz_stack(args.xyz)
    workdir = args.workdir.resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    shutil.copy2(args.mbx_json, workdir / "mbx.json")
    shutil.copy2(args.xyz, workdir / "water_dimer_optimized.xyz")

    baseline_jsons = {
        "h2o": workdir / "mbpol_elec_pol_baseline.json",
        "mbpbe": workdir / "mbpbe_elec_pol_baseline.json",
    }
    write_elec_pol_baseline_json(args.mbx_json, baseline_jsons["h2o"], "h2o")
    write_elec_pol_baseline_json(args.mbx_json, baseline_jsons["mbpbe"], "mbpbe")

    env = os.environ.copy()
    libdir = args.mbx_home / "lib"
    env["DYLD_LIBRARY_PATH"] = str(libdir) + (":" + env["DYLD_LIBRARY_PATH"] if env.get("DYLD_LIBRARY_PATH") else "")
    env["QUICK_BASIS"] = str(args.basis.resolve())
    env["QUICK_MBX_WATER_XYZ"] = str((workdir / "water_dimer_optimized.xyz").resolve())

    result = subprocess.run([str(args.scan_exe.resolve())], cwd=workdir, env=env, text=True, capture_output=True, check=False)
    (workdir / "quick_mbx_water_xyz_scan.stdout").write_text(result.stdout, encoding="utf-8")
    (workdir / "quick_mbx_water_xyz_scan.stderr").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"QUICK-MBX water XYZ scan failed with exit code {result.returncode}. "
            f"See {workdir / 'quick_mbx_water_xyz_scan.stdout'}."
        )
    scan_rows = parse_scan_stdout(result.stdout)
    if not scan_rows:
        raise RuntimeError(f"No scan rows parsed from:\n{result.stdout}")

    rows = compute_rows(
        frames,
        scan_rows,
        args.mbx_home / "bin" / "single_point",
        args.mbx_json,
        baseline_jsons,
        workdir,
        env,
    )
    write_csv(args.csv, rows)
    plot_pdf = args.plot_pdf or args.output.with_suffix(".pdf")
    draw_plot(
        rows,
        args.output,
        plot_pdf,
        include_raw=args.include_raw_diagnostic,
        title=args.plot_title,
        subtitle=args.plot_subtitle,
    )

    print(f"Wrote {args.csv}")
    print(f"Wrote {args.output}")
    print(f"Wrote {plot_pdf}")
    for row in representative_table(rows):
        print(
            f"{row['region']:>7s} R={row['actual_distance_ang']:.3f} "
            f"PBE-D3={row['pbe_d3_kcal']: .6f} "
            f"MB-pol={row['mbpol_kcal']: .6f} "
            f"MB-PBE={row['mbpbe_kcal']: .6f} "
            f"PBE-D3/MB-pol={row['hybrid_mbpol_corrected_kcal']: .6f} "
            f"PBE-D3/MB-PBE={row['hybrid_mbpbe_corrected_kcal']: .6f} kcal/mol"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
