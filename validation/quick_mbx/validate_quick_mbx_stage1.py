#!/usr/bin/env python3
"""Stage-1 validation driver for QUICK-MBX preparation work.

The checks here intentionally avoid requiring MBX unless explicit MBX paths are
provided.  They validate the QUICK-side ingredients used by the first
energy-only QM/MB-pol target:

* ESP/EFIELD/EFG property generation on serial and MPI builds.
* Analytic EFG against the retained finite-difference EFG reference.
* Optional MBX standalone example parity when an MBX install and MBX example
  tree are supplied.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


NUMBER_RE = re.compile(r"[-+]?\d*\.\d+(?:[Ee][-+]?\d+)?|[-+]?\d+\.?(?:[Ee][-+]?\d+)?")


PROPERTY_CASES = [
    ("esp_grid_density_surface_H2O_rhf_sto3g", ".esp", 2.0e-7),
    ("efield_density_surface_H2O_rhf_sto3g", ".efield", 2.0e-7),
    ("efg_density_surface_H2O_rhf_sto3g", ".efg", 2.0e-7),
]

EFG_ANALYTIC = "efg_grid_acetone_b3lyp_def2svp"
EFG_NUMERICAL = "efg_grid_numerical_acetone_b3lyp_def2svp"


def run(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    result = subprocess.run(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(cmd)}\n{result.stdout}")


def numbers(path: Path) -> list[float]:
    vals: list[float] = []
    for line in path.read_text(errors="ignore").splitlines():
        vals.extend(float(x) for x in NUMBER_RE.findall(line))
    return vals


def max_abs_diff(a: Path, b: Path) -> float:
    av = numbers(a)
    bv = numbers(b)
    if len(av) != len(bv):
        raise RuntimeError(f"numeric length mismatch: {a} has {len(av)}, {b} has {len(bv)}")
    if not av:
        raise RuntimeError(f"no numeric data parsed from {a}")
    return max(abs(x - y) for x, y in zip(av, bv))


def first_energy(path: Path) -> float:
    energy_re = re.compile(r"Energy(?:\s*\[[^\]]+\])?\s*=\s*(" + NUMBER_RE.pattern + r")")
    for line in path.read_text(errors="ignore").splitlines():
        match = energy_re.search(line)
        if match:
            return float(match.group(1))
    raise RuntimeError(f"no energy line parsed from {path}")


def copy_input(repo: Path, stem: str, work: Path) -> Path:
    src = repo / "test" / f"{stem}.in"
    dst = work / f"{stem}.in"
    shutil.copy2(src, dst)
    return dst


def run_quick(exe: Path, inp: Path, basis: Path, work: Path, mpi: list[str] | None = None) -> None:
    env = os.environ.copy()
    env["QUICK_BASIS"] = str(basis)
    cmd = [str(exe), inp.name]
    if mpi:
        cmd = mpi + cmd
    run(cmd, cwd=work, env=env)


def check_saved(repo: Path, work: Path, stem: str, ext: str, tol: float, label: str) -> dict[str, object]:
    saved = repo / "test" / "saved" / f"{stem}{ext}"
    produced = work / f"{stem}{ext}"
    diff = max_abs_diff(saved, produced)
    return {"check": label, "max_abs_diff": diff, "tolerance": tol, "passed": diff <= tol}


def mbx_runtime_env(mbx_home: Path, mbx_source: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["MBX_HOME"] = str(mbx_home)
    for var in ("DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH"):
        entries = [str(p) for p in (mbx_home / "lib", mbx_home / "lib64") if p.exists()]
        if entries:
            env[var] = os.pathsep.join(entries + ([env[var]] if env.get(var) else []))
    plugin_roots = []
    for root in (mbx_home, mbx_source):
        if root:
            plugin = root / "plugins" / "python" / "mbx"
            if plugin.exists():
                plugin_roots.append(str(plugin))
    if plugin_roots:
        env["PYTHONPATH"] = os.pathsep.join(plugin_roots + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


def run_mbx_example(
    mbx_home: Path,
    mbx_source: Path | None,
    example_dir: Path | None,
    single_point: Path | None,
    work: Path,
    tol: float,
) -> dict[str, object]:
    source_root = mbx_source or mbx_home
    example = example_dir or source_root / "examples" / "PEFs" / "001_mbpol" / "C++"
    exe = single_point or mbx_home / "bin" / "single_point"
    required = [example / "input.nrg", example / "mbx.json", example / "expected_output", exe]
    missing = [str(p) for p in required if not p.exists()]
    env = mbx_runtime_env(mbx_home, mbx_source)
    if not missing:
        mbx_work = work / "mbx_example"
        mbx_work.mkdir(parents=True, exist_ok=True)
        shutil.copy2(example / "input.nrg", mbx_work / "input.nrg")
        shutil.copy2(example / "mbx.json", mbx_work / "mbx.json")
        shutil.copy2(example / "expected_output", mbx_work / "expected_output")

        result = subprocess.run(
            [str(exe), "input.nrg", "mbx.json"],
            cwd=mbx_work,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        output = mbx_work / "output"
        output.write_text(result.stdout)
        if result.returncode != 0:
            return {
                "check": "MBX standalone parity",
                "passed": False,
                "mode": "single_point",
                "returncode": result.returncode,
                "output": result.stdout,
            }

        expected_energy = first_energy(mbx_work / "expected_output")
        produced_energy = first_energy(output)
        diff = abs(expected_energy - produced_energy)
        return {
            "check": "MBX standalone parity",
            "mode": "single_point",
            "expected_energy_kcal_mol": expected_energy,
            "produced_energy_kcal_mol": produced_energy,
            "max_abs_diff": diff,
            "tolerance": tol,
            "passed": diff <= tol,
        }

    python_example = example_dir or source_root / "examples" / "PEFs" / "001_mbpol" / "python"
    script = source_root / "examples" / "PEFs" / "src" / "python" / "example.py"
    python_required = [python_example / "input.xyz", python_example / "mbx.json", python_example / "expected_output", script]
    python_missing = [str(p) for p in python_required if not p.exists()]
    if python_missing:
        return {
            "check": "MBX standalone parity",
            "skipped": True,
            "reason": f"missing single_point files: {missing}; missing python files: {python_missing}",
        }

    mbx_work = work / "mbx_example"
    mbx_work.mkdir(parents=True, exist_ok=True)
    shutil.copy2(python_example / "input.xyz", mbx_work / "input.xyz")
    shutil.copy2(python_example / "mbx.json", mbx_work / "mbx.json")
    shutil.copy2(python_example / "expected_output", mbx_work / "expected_output")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=mbx_work,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = mbx_work / "output"
    output.write_text(result.stdout)
    if result.returncode != 0:
        return {
            "check": "MBX standalone parity",
            "passed": False,
            "mode": "python",
            "returncode": result.returncode,
            "output": result.stdout,
        }

    diff = max_abs_diff(mbx_work / "expected_output", output)
    return {"check": "MBX standalone parity", "mode": "python", "max_abs_diff": diff, "tolerance": tol, "passed": diff <= tol}


def run_quick_mbx_smoke(
    exe: Path,
    mbx_home: Path,
    mbx_source: Path | None,
    example_dir: Path | None,
    basis: Path,
    work: Path,
    mpi: list[str] | None = None,
    ext_perm_check_tol: float = 1.0e-6,
) -> dict[str, object]:
    source_root = mbx_source or mbx_home
    example = example_dir or source_root / "examples" / "PEFs" / "001_mbpol" / "C++"
    json_file = example / "mbx.json"
    missing = [str(p) for p in (exe, json_file) if not p.exists()]
    label = "QUICK-MBX API smoke" if mpi is None else "QUICK-MBX API smoke MPI"
    if missing:
        return {"check": label, "skipped": True, "reason": f"missing files: {missing}"}

    smoke_work = work / ("quick_mbx_smoke" if mpi is None else "quick_mbx_smoke_mpi")
    smoke_work.mkdir(parents=True, exist_ok=True)
    shutil.copy2(json_file, smoke_work / "mbx.json")

    env = mbx_runtime_env(mbx_home, mbx_source)
    env["QUICK_BASIS"] = str(basis)
    cmd = [str(exe)]
    if mpi:
        cmd = mpi + cmd
    result = subprocess.run(
        cmd,
        cwd=smoke_work,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = smoke_work / "output"
    output.write_text(result.stdout)
    if result.returncode != 0:
        return {"check": label, "passed": False, "returncode": result.returncode, "output": result.stdout}

    energy_matches = re.findall(r"(QUICK_MBX_SMOKE_[A-Z_]*ENERGY_AU)\s+(" + NUMBER_RE.pattern + r")", result.stdout)
    if not energy_matches:
        return {"check": label, "passed": False, "reason": "energy marker not found", "output": result.stdout}
    energies = {name: float(value) for name, value in energy_matches}
    quick_outputs = parse_quick_mbx_outputs(smoke_work)
    max_ext_perm_check = 0.0
    if quick_outputs:
        max_ext_perm_check = max(abs(case["terms"].get("QUICK-MBX EXT PERM CHECK", 0.0)) for case in quick_outputs.values())
    converged = all(bool(case["converged"]) for case in quick_outputs.values()) if quick_outputs else False
    return {
        "check": label,
        "energy_au": next(iter(energies.values())),
        "energies_au": energies,
        "max_ext_perm_half_residual_au": max_ext_perm_check,
        "quick_outputs": quick_outputs,
        "scf_converged": converged,
        "passed": all(math.isfinite(value) for value in energies.values())
        and converged
        and max_ext_perm_check <= ext_perm_check_tol,
    }


def parse_quick_mbx_outputs(smoke_work: Path) -> dict[str, dict[str, object]]:
    parsed: dict[str, dict[str, object]] = {}
    term_re = re.compile(r"^\s*(QUICK-MBX [A-Z0-9 /_-]+?)\s*=\s*(" + NUMBER_RE.pattern + r")")
    for out_file in sorted(smoke_work.glob("quick_mbx_api_smoke_*.out")):
        terms: dict[str, float] = {}
        text = out_file.read_text(errors="ignore")
        for line in text.splitlines():
            match = term_re.search(line)
            if match:
                terms[match.group(1).strip()] = float(match.group(2))
        parsed[out_file.stem] = {"converged": "REACH CONVERGENCE" in text, "terms": terms}
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--quick-serial", type=Path, default=None)
    parser.add_argument("--quick-mpi", type=Path, default=None)
    parser.add_argument("--quick-mbx-smoke", type=Path, default=None)
    parser.add_argument("--quick-mbx-smoke-mpi", type=Path, default=None)
    parser.add_argument("--mpirun", nargs="+", default=["mpirun", "-np", "2"])
    parser.add_argument("--basis", type=Path, default=None)
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/quick_mbx_stage1_validation"))
    parser.add_argument("--skip-mpi", action="store_true")
    parser.add_argument("--efg-tol", type=float, default=2.0e-7)
    parser.add_argument("--mbx-library", type=Path, default=None)
    parser.add_argument("--mbx-home", type=Path, default=Path(os.environ["MBX_HOME"]) if os.environ.get("MBX_HOME") else None)
    parser.add_argument("--mbx-source", type=Path, default=None)
    parser.add_argument("--mbx-example-dir", type=Path, default=None)
    parser.add_argument("--mbx-single-point", type=Path, default=None)
    parser.add_argument("--mbx-parity-tol", type=float, default=1.0e-8)
    parser.add_argument("--quick-mbx-smoke-tol", type=float, default=1.0e-8)
    parser.add_argument("--quick-mbx-ext-perm-check-tol", type=float, default=1.0e-6)
    args = parser.parse_args()

    repo = args.repo.resolve()
    quick_serial = args.quick_serial or repo / "build-serial" / "src" / "quick"
    quick_mpi = args.quick_mpi or repo / "build-mpi" / "src" / "quick.MPI"
    quick_mbx_smoke = args.quick_mbx_smoke or repo / "build-mbx-serial" / "src" / "test-api-mbx"
    quick_mbx_smoke_mpi = args.quick_mbx_smoke_mpi or repo / "build-mbx-mpi" / "src" / "test-api-mbx.MPI"
    basis = args.basis or repo / "basis"
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []

    serial_work = work / "serial"
    mpi_work = work / "mpi"
    analytic_work = work / "efg_analytic"
    numerical_work = work / "efg_numerical"
    for d in (serial_work, mpi_work, analytic_work, numerical_work):
        d.mkdir(parents=True, exist_ok=True)

    for stem, ext, tol in PROPERTY_CASES:
        inp = copy_input(repo, stem, serial_work)
        run_quick(quick_serial, inp, basis, serial_work)
        results.append(check_saved(repo, serial_work, stem, ext, tol, f"serial saved {stem}{ext}"))

        if not args.skip_mpi:
            inp = copy_input(repo, stem, mpi_work)
            run_quick(quick_mpi, inp, basis, mpi_work, args.mpirun)
            diff = max_abs_diff(serial_work / f"{stem}{ext}", mpi_work / f"{stem}{ext}")
            results.append(
                {"check": f"serial vs mpi {stem}{ext}", "max_abs_diff": diff, "tolerance": tol, "passed": diff <= tol}
            )

    inp = copy_input(repo, EFG_ANALYTIC, analytic_work)
    run_quick(quick_serial, inp, basis, analytic_work)
    inp = copy_input(repo, EFG_NUMERICAL, numerical_work)
    run_quick(quick_serial, inp, basis, numerical_work)
    diff = max_abs_diff(analytic_work / f"{EFG_ANALYTIC}.efg", numerical_work / f"{EFG_NUMERICAL}.efg")
    results.append(
        {"check": "analytic EFG vs numerical EFG", "max_abs_diff": diff, "tolerance": args.efg_tol, "passed": diff <= args.efg_tol}
    )

    if args.mbx_library:
        results.append({"check": "MBX library present", "path": str(args.mbx_library), "passed": args.mbx_library.exists()})
    if args.mbx_home:
        mbx_home = args.mbx_home.resolve()
        mbx_source = args.mbx_source.resolve() if args.mbx_source else None
        mbx_example_dir = args.mbx_example_dir.resolve() if args.mbx_example_dir else None
        results.append(
            run_mbx_example(
                mbx_home,
                mbx_source,
                mbx_example_dir,
                args.mbx_single_point.resolve() if args.mbx_single_point else None,
                work,
                args.mbx_parity_tol,
            )
        )
        quick_mbx_smoke_result: dict[str, object] | None = None
        quick_mbx_smoke_mpi_result: dict[str, object] | None = None
        if quick_mbx_smoke.exists():
            quick_mbx_smoke_result = run_quick_mbx_smoke(
                quick_mbx_smoke.resolve(),
                mbx_home,
                mbx_source,
                mbx_example_dir,
                basis,
                work,
                ext_perm_check_tol=args.quick_mbx_ext_perm_check_tol,
            )
            results.append(quick_mbx_smoke_result)
        else:
            results.append(
                {
                    "check": "QUICK-MBX API smoke",
                    "skipped": True,
                    "reason": f"build an MBX-enabled QUICK target or pass --quick-mbx-smoke; missing {quick_mbx_smoke}",
                }
            )
        if not args.skip_mpi:
            if quick_mbx_smoke_mpi.exists():
                quick_mbx_smoke_mpi_result = run_quick_mbx_smoke(
                    quick_mbx_smoke_mpi.resolve(),
                    mbx_home,
                    mbx_source,
                    mbx_example_dir,
                    basis,
                    work,
                    args.mpirun,
                    args.quick_mbx_ext_perm_check_tol,
                )
                results.append(quick_mbx_smoke_mpi_result)
            else:
                results.append(
                    {
                        "check": "QUICK-MBX API smoke MPI",
                        "skipped": True,
                        "reason": f"build an MBX-enabled MPI QUICK target or pass --quick-mbx-smoke-mpi; missing {quick_mbx_smoke_mpi}",
                    }
                )
        if (
            quick_mbx_smoke_result
            and quick_mbx_smoke_mpi_result
            and quick_mbx_smoke_result.get("passed")
            and quick_mbx_smoke_mpi_result.get("passed")
        ):
            serial_energies = quick_mbx_smoke_result.get("energies_au")
            mpi_energies = quick_mbx_smoke_mpi_result.get("energies_au")
            if isinstance(serial_energies, dict) and isinstance(mpi_energies, dict):
                common = sorted(set(serial_energies) & set(mpi_energies))
                if not common:
                    diff = math.inf
                    per_label_diff = {}
                else:
                    per_label_diff = {
                        label: abs(float(serial_energies[label]) - float(mpi_energies[label])) for label in common
                    }
                    diff = max(per_label_diff.values())
            else:
                per_label_diff = {}
                diff = abs(float(quick_mbx_smoke_result["energy_au"]) - float(quick_mbx_smoke_mpi_result["energy_au"]))
            results.append(
                {
                    "check": "QUICK-MBX API smoke serial vs MPI",
                    "max_abs_diff": diff,
                    "per_label_diff": per_label_diff,
                    "tolerance": args.quick_mbx_smoke_tol,
                    "passed": diff <= args.quick_mbx_smoke_tol,
                }
            )
    else:
        results.append({"check": "MBX standalone parity", "skipped": True, "reason": "provide --mbx-home after MBX install"})
        results.append({"check": "QUICK-MBX API smoke", "skipped": True, "reason": "provide --mbx-home after MBX install"})

    summary = {"passed": all(bool(x.get("passed", False)) or bool(x.get("skipped", False)) for x in results), "results": results}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
