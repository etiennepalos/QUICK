#!/usr/bin/env python3
"""Finite-behavior regression using QUICK's existing OEPROP acceptance CLI."""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

SCOPE = (
    "Finite CPU/GPU behavior at extreme observation coordinates only. "
    "These cancellation/underflow stress points do not establish relative "
    "far-field accuracy. No timings or acceleration are measured."
)
OMITTED_E = re.compile(r"^([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))([+-][0-9]{3})$")
FIELDS = {
    "esp": ["esp"],
    "efield": ["efield_" + axis for axis in "xyz"],
    "efg": ["efg_" + a + b for a in "xyz" for b in "xyz"],
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def fortran_float(token):
    """Read ordinary E/D numbers and Fortran's omitted E for ±ddd exponents."""
    token = token.strip().replace("D", "E").replace("d", "e")
    match = OMITTED_E.fullmatch(token)
    if match:
        token = match[1] + "E" + match[2]
    return float(token)


def make_grid(path):
    scales = [1e+62, 1e+70, 1e+80]
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, 0, -1), (1, 1, 1), (-1, 1, -1)]
    points = [tuple((scales[i % 3] * v for v in directions[i // 3 % 6])) for i in range(129)]
    path.write_text(
        "129\n" + "".join(
            " ".join(f"{value:.17e}" for value in point) + "\n" for point in points
        )
    )
    return points


def csv_rows(path):
    with path.open() as stream:
        reader = csv.DictReader(stream)
        require(
            reader.fieldnames is not None
            and len(set(reader.fieldnames)) == len(reader.fieldnames),
            "Missing/duplicate CSV header " + str(path),
        )
        rows = list(reader)
    require(
        rows and all(
            None not in row and all(value is not None for value in row.values())
            for row in rows
        ),
        "Empty/malformed CSV " + str(path),
    )
    return rows


def environment(driver, basis):
    result = dict(os.environ)
    result.pop("LD_PRELOAD", None)
    result.pop("QUICK_DERIVATIVE_PHASE_LOG", None)
    result.pop("QUICK_EFG_TRACE_FILE", None)
    result.update(
        QUICK_BASIS=str(basis), OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
    )
    result["LD_LIBRARY_PATH"] = str(driver.parent) + ":" + result.get("LD_LIBRARY_PATH", "")
    return result


def runtime(driver, env, backend):
    loaded = subprocess.run(
        ["ldd", str(driver)], env=env, text=True, capture_output=True, check=True
    ).stdout
    resolved = {}
    for line in loaded.splitlines():
        match = re.search(r"=>\s+(/\S+)\s+\(", line)
        if match:
            path = Path(match[1]).resolve()
            resolved[str(path)] = sha(path)
    quick = [Path(path) for path in resolved if Path(path).name.startswith("libquick")]
    require(len(quick) == 1, "Expected one resolved QUICK library for " + backend)
    if backend == "cpu":
        require(quick[0].name == "libquick.so", "CPU driver resolved an accelerator library")
    else:
        require(
            quick[0].name in ["libquick_cuda.so", "libquick_hip.so"],
            "GPU driver did not resolve an accelerator library",
        )
    cache = driver.parent.parent / "CMakeCache.txt"
    return {
        "driver": str(driver),
        "driver_sha256": sha(driver),
        "resolved_libraries_sha256": resolved,
        "ldd": re.sub(r"0x[0-9a-fA-F]+", "address omitted", loaded),
        "build_cache": {"path": str(cache), "sha256": sha(cache)}
        if cache.is_file() else None,
    }


def source_files(root):
    if root is None:
        return {}
    require((root / "src").is_dir(), "Source root has no src directory")
    paths = [p for p in (root / "src").rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    paths += (
        [p for p in (root / "tools").rglob("generate_oeprop*.py") if p.is_file()]
        if (root / "tools").is_dir() else []
    )
    if (root / "CMakeLists.txt").is_file():
        paths.append(root / "CMakeLists.txt")
    return {str(path): sha(path) for path in sorted(paths)}


def inspect_outputs(work, backend, points):
    rows = csv_rows(work / "range.api.csv")
    require(
        len(rows) == 129 and [int(row["probe"]) for row in rows] == list(range(1, 130)),
        "Wrong point count/order",
    )
    for row, point in zip(rows, points):
        require(int(row["frame"]) == 1, "Unexpected property frame")
        require(
            tuple(fortran_float(row[axis + "_ang"]) for axis in "xyz") == point,
            "Probe coordinate changed",
        )
    counts = {
        name: sum(
            not math.isfinite(fortran_float(row[key])) for row in rows for key in keys
        )
        for name, keys in FIELDS.items()
    }
    data = {
        "nonfinite_components": counts,
        "nonfinite_csv_values": sum(
            not math.isfinite(fortran_float(value)) for row in rows for value in row.values()
        ),
        "property_points": 129,
        "property_components": 129 * 13,
    }
    checks = csv_rows(work / "range.checks.csv")
    data["nonfinite_acceptance_checks"] = sum(
        not math.isfinite(fortran_float(row["max_abs"])) for row in checks
    )
    lifecycle = csv_rows(work / "range.lifecycle.csv")
    require(
        len(lifecycle) == 1 and lifecycle[0] == {
            "scf_calls": "1", "property_calls_preserve_scf": "1", "post_delete_ierr": "47"
        },
        "Property calls changed the SCF counter or lifecycle",
    )
    data["lifecycle"] = lifecycle[0]
    kernels = csv_rows(work / "range.kernels.csv")
    require(len(kernels) == 1, "Expected serial rank evidence")
    launches = {
        key: int(kernels[0][key])
        for key in ["esp_launches", "efield_launches", "efg_launches"]
    }
    require(
        all(value == 0 for value in launches.values()) if backend == "cpu"
        else all(value > 0 for value in launches.values()),
        "Unexpected kernel launch evidence",
    )
    data["kernel_launches"] = launches
    output = work / "oeprop_acceptance.out"
    require(
        output.is_file() and output.read_text().count("REACH CONVERGENCE AFTER") == 1,
        "Expected exactly one initialization SCF",
    )
    data["initialization_scf_calls"] = 1
    data["output_sha256"] = {path.name: sha(path) for path in sorted(work.iterdir()) if path.is_file()}
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-driver", required=True, type=Path)
    parser.add_argument("--gpu-driver", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path, help="Repository providing the basis directory")
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument(
        "--keyword", default=(
            "PBE0 D3BJ BASIS=cc-pVDZ CHARGE=0 XCCUTOFF=1.0D-12 "
            "BASISCUTOFF=1.0D-12 CUTOFF=1.0D-12 DENSERMS=4.0D-8 ENERGY"
        ),
    )
    parser.add_argument("--cpu-source-root", type=Path)
    parser.add_argument("--gpu-source-root", type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    require(args.timeout > 0, "Timeout must be positive")
    drivers = {k: getattr(args, k + "_driver").resolve() for k in ["cpu", "gpu"]}
    basis = (args.repo_root / "basis").resolve()
    geometry = args.geometry.resolve()
    work = args.workdir.resolve()
    require(
        all(path.is_file() and os.access(path, os.X_OK) for path in drivers.values()),
        "Driver unavailable/non-executable",
    )
    require(basis.is_dir() and geometry.is_file(), "Missing basis or geometry")
    require(
        len(args.keyword) <= 256 and "ENERGY" in args.keyword
        and "GRADIENT" not in args.keyword,
        "Use one initialization energy SCF keyword of at most 256 characters",
    )
    require(not work.exists(), "Use a fresh workdir")
    work.mkdir(parents=True)
    grid = work / "large.probe"
    points = make_grid(grid)
    local_geometry = work / "molecule.xyz"
    shutil.copyfile(geometry, local_geometry)
    report = {
        "passed": False, "prepared_only": args.prepare_only, "scope": SCOPE, "points": 129,
        "coordinate_units": "Angstrom; acceptance driver converts to bohr",
        "input_coordinate_scales_ang": [1e+62, 1e+70, 1e+80], "keyword": args.keyword,
        "processes": {}, "harness_sha256": sha(Path(__file__)),
        "geometry_original": str(geometry), "geometry_sha256": sha(geometry),
        "grid_sha256": sha(grid),
    }

    def save():
        (work / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save()
    envs = {k: environment(driver, basis) for k, driver in drivers.items()}

    def snapshot():
        return {
            "runtime": {k: runtime(driver, envs[k], k) for k, driver in drivers.items()},
            "cpu_source_sha256": source_files(
                args.cpu_source_root.resolve() if args.cpu_source_root else None
            ),
            "gpu_source_sha256": source_files(
                args.gpu_source_root.resolve() if args.gpu_source_root else None
            ),
            "basis_sha256": {
                str(p): sha(p) for p in sorted(basis.rglob("*")) if p.is_file()
            },
            "geometry_sha256": sha(geometry),
            "local_geometry_sha256": sha(local_geometry), "grid_sha256": sha(grid),
        }
    frozen = snapshot()
    report["provenance_before"] = frozen
    save()
    if args.prepare_only:
        report["note"] = "Inputs and provenance prepared; no scientific process ran and passed remains false."
        save()
        return 0
    try:
        common_density = None
        for backend in ["cpu", "gpu"]:
            directory = work / backend
            directory.mkdir()
            density_arg = "-" if backend == "cpu" else str(common_density)
            argv = [
                str(drivers[backend]), "acceptance", str(local_geometry), str(grid),
                "range", args.keyword, density_arg,
            ]
            require(
                all(len(x) <= 256 for x in argv[2:]),
                "Acceptance CLI argument exceeds its 256-character buffer",
            )
            require(snapshot() == frozen, "Provenance changed before " + backend)
            if common_density is not None:
                require(
                    sha(common_density) == report["common_cpu_density_sha256"],
                    "Common CPU density changed",
                )
            with (directory / "run.log").open("w") as stream:
                completed = subprocess.run(
                    argv, cwd=directory, env=envs[backend], stdout=stream,
                    stderr=subprocess.STDOUT, timeout=args.timeout,
                )
            process = {
                "argv": argv, "cwd": str(directory), "returncode": completed.returncode,
                "environment": {
                    key: envs[backend].get(key) for key in [
                        "QUICK_BASIS", "LD_LIBRARY_PATH", "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES",
                    ]
                },
            }
            report["processes"][backend] = process
            save()
            require(completed.returncode == 0, "Acceptance CLI failed: " + str(directory / "run.log"))
            require(snapshot() == frozen, "Provenance changed during " + backend)
            process.update(inspect_outputs(directory, backend, points))
            save()
            require(
                not any(process["nonfinite_components"].values())
                and process["nonfinite_csv_values"] == 0
                and process["nonfinite_acceptance_checks"] == 0,
                "Nonfinite " + backend + " property components: "
                + str(process["nonfinite_components"]),
            )
            if backend == "cpu":
                common_density = directory / "range.density.dat"
                report["common_cpu_density_sha256"] = sha(common_density)
                save()
            else:
                require(
                    sha(directory / "range.density.dat") == report["common_cpu_density_sha256"]
                    and sha(common_density) == report["common_cpu_density_sha256"],
                    "GPU did not preserve the identical exported CPU AO density",
                )
        report["provenance_after"] = snapshot()
        require(report["provenance_after"] == frozen, "Final provenance changed")
        report["passed"] = True
        report["scientific_property_components"] = 2 * 129 * 13
        save()
        print(json.dumps({
            "passed": True, "scope": SCOPE, "points": 129,
            "property_components": 2 * 129 * 13,
            "common_cpu_density_sha256": report["common_cpu_density_sha256"],
        }))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        report["error"] = str(error)
        save()
        print(str(error), file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
