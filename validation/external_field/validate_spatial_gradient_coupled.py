#!/usr/bin/env python3
"""Check spatial gradients together with ddX, PBE0 and sampled short BOMD.

Requires NumPy and separately compiled spatial-gradient API and instrumented
BOMD clients. Evidence directories must be new. BOMD instrumentation only
exports the live computed gradient; it must not change forces or integrator.
"""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import validate_spatial_gradient as affine
sys.path.insert(0, str(ROOT / "validation/solvation"))
sys.path.insert(0, str(ROOT / "validation/oeprop_bomd"))
import validate_solvation_bomd as solvent_bomd
from validate_oeprop_bomd import max_difference, property_difference, write_control

BASE = "HF BASIS=STO-3G GRADIENT CUTOFF=1e-12 GRADCUTOFF=1e-12 DENSERMS=1e-11"
APPLIED = " EFIELD_FX=.0002 EFIELD_GXX=2e-5 EFIELD_GYY=-2e-5 EFIELD_GXY=3e-5"
SOLVENT = " DDX={} DDX_FIELD=SOLUTE DDX_LMAX=6 DDX_NLEBEDEV=302 DDX_NRAD=75 DDX_NANG=302"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for backend in ("cpu", "cuda"):
        parser.add_argument(f"--{backend}-api-driver", type=Path, required=True)
        parser.add_argument(f"--{backend}-bomd-driver", type=Path, required=True)
        parser.add_argument(f"--{backend}-basis", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800.)
    args = parser.parse_args()
    args.workdir = args.workdir.resolve()
    args.workdir.mkdir(parents=True, exist_ok=False)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be positive and finite")
    drivers, envs, bases = {}, {}, {}
    for backend in ("cpu", "cuda"):
        drivers[backend] = {kind: getattr(args, f"{backend}_{kind}_driver").resolve(strict=True)
                            for kind in ("api", "bomd")}
        bases[backend] = getattr(args, f"{backend}_basis").resolve(strict=True)
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", QUICK_BASIS=str(bases[backend]))
        envs[backend] = env
    report = {"passed": False, "started_utc": datetime.now(timezone.utc).isoformat(), "runs": [], "checks": [],
              "scope": "Static symmetric affine field + equilibrium ddX PCM/COSMO and PBE0 central CPU/CUDA parity; four-step sampled BOMD software checks",
              "fixed_gates": {"CPU_CUDA_energy_hartree": 1e-7, "HF_CPU_CUDA_gradient_hartree_per_bohr": 1e-7,
                              "PBE0_CPU_CUDA_gradient_hartree_per_bohr": 1e-6, "PCM_GPU_FD_gradient_hartree_per_bohr": 1e-5,
                              "properties_on_off_full_gradient_hartree_per_bohr": 1e-8, "short_BOMD_drift_hartree": 1e-5},
              "limitations": ["Tiny finite water, four .025 fs steps = .1 fs, 50 K deterministic initial velocity scaling; not production dynamics or statistics",
                              "Finite ddX discretization at LMAX=6, cavity Lebedev=302, volume 75x302; checks discrete-model consistency, not continuum convergence",
                              "PBE0 check establishes backend parity at one geometry; no new independent XC-grid accuracy claim",
                              "DDX_FIELD=SOLUTE: imposed potential couples directly to solute; the solvent reaction-field convention is unchanged",
                              "ESP/E/EFG property outputs keep their existing source-only convention and omit the direct imposed and ddX reaction fields",
                              "No PBC, thermostat, time dependence, electrode or constant-potential model is tested"]}

    def check(name, error, tolerance, **details):
        report["checks"].append({"name": name, "error": float(error), "tolerance": tolerance,
                                 "passed": bool(math.isfinite(error) and error <= tolerance), **details})

    def progress():
        (args.workdir / "progress.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    def api_run(backend, label, keyword, atoms=None, gradient=True):
        atoms = copy.deepcopy(affine.WATER) if atoms is None else atoms
        if not gradient:
            keyword = re.sub(r"(?<!\S)GRADIENT(?!\S)", "", keyword)
        if len(keyword) > 256:
            raise ValueError("API keyword exceeds 256 characters")
        folder = args.workdir / "api" / f"{len(report['runs']):03d}-{backend}-{label}"
        folder.mkdir(parents=True, exist_ok=False)
        text = keyword + "\n3 0\n" + "".join(str(z) + " " + " ".join(f"{v:.17g}" for v in point) + "\n" for z, point in atoms)
        path = folder / "field.dat"; path.write_text(text)
        command = [str(drivers[backend]["api"]), "gradient" if gradient else "energy", path.name]
        row = {"backend": backend, "label": label, "kind": "API", "keyword": keyword,
               "command": command, "folder": str(folder), "input_sha256": affine.sha(path)}
        report["runs"].append(row)
        print("RUN", backend, label, flush=True)
        start = time.monotonic()
        with (folder / "execution.log").open("w") as log:
            process = subprocess.run(command, cwd=folder, env=envs[backend], stdout=log,
                                     stderr=subprocess.STDOUT, timeout=args.timeout)
        row.update(returncode=process.returncode, wall_seconds=time.monotonic() - start)
        if process.returncode:
            raise ValueError(f"API run failed: {folder}")
        result = affine.parse(folder / "result.dat")
        if gradient and (not result.get("analytic_gradient_active", False)
                         or max(abs(value) for value in result["qm_gradient"]) < 1e-5):
            raise ValueError("Nonstationary gradient fixture did not return an active nonzero gradient")
        output = (folder / "spatial_gradient_api.out").read_text(errors="replace")
        cycles = re.findall(r"REACH CONVERGENCE AFTER\s+(\d+)\s+CYCLES", output)
        check(f"{backend}:{label}:explicit_converged_SCF", int(len(cycles) != 1 or not result["scf_converged"]), 0.)
        check(f"{backend}:{label}:all_nuclear_gradient_components", abs(len(result["qm_gradient"]) - (9 if gradient else 0)), 0.)
        expected_calls = [1, int(gradient)] if backend == "cuda" else [0, 0]
        check(f"{backend}:{label}:native_field_GPU_dispatch", int(result["native_gpu_field_counts"] != expected_calls), 0.)
        check(f"{backend}:{label}:parsed_affine_tensor", affine.difference(sum(result["external_tensor"], []),
               [2e-5, 3e-5, 0., 3e-5, -2e-5, 0., 0., 0., 0.]), 1e-18)
        if "DDX=" in keyword:
            expected = "DDX model=2" if "DDX=PCM" in keyword else "DDX model=1"
            check(f"{backend}:{label}:active_ddX_model", int(expected not in output), 0.)
            check(f"{backend}:{label}:complete_default_volume_grid", int("volume points=67950" not in output), 0.)
        row.update(result=result, output_sha256={p.name: affine.sha(p) for p in folder.iterdir() if p.is_file()})
        progress()
        return result

    try:
        sources = [Path(__file__).resolve(), HERE / "validate_spatial_gradient.py", HERE / "quick_spatial_gradient_api_driver.f90",
                   ROOT / "validation/oeprop_bomd/quick_oeprop_bomd_driver.f90", Path(solvent_bomd.__file__).resolve(),
                   ROOT / "validation/oeprop_bomd/validate_oeprop_bomd.py", ROOT / "validation/oeprop_bomd/water.xyz",
                   ROOT / "validation/oeprop_bomd/water_fixed.probe"]
        sources.extend(basis / name for basis in bases.values() for name in ("STO-3G.BAS", "6-31GDP.BAS") if (basis / name).exists())
        report["inputs_before"] = {str(p): affine.sha(p) for p in set(sources)}
        report["runtime_before"] = {backend: {kind: affine.runtime(path, envs[backend]) for kind, path in kinds.items()}
                                    for backend, kinds in drivers.items()}
        for backend in drivers:
            expected = (bases[backend].parent / "lib" / ("libquick_cuda.so" if backend == "cuda" else "libquick.so")).resolve(strict=True)
            for kind in drivers[backend]:
                check(f"{backend}:{kind}:expected_installed_runtime",
                      int(Path(report["runtime_before"][backend][kind]["quick_library"]) != expected), 0.)

        central = {}
        for model in ("PCM", "COSMO"):
            keyword = BASE + APPLIED + SOLVENT.format(model)
            central[model] = {backend: api_run(backend, model.lower() + "-central", keyword) for backend in drivers}
            check(model + ":CPU_CUDA_energy", abs(central[model]["cpu"]["energy"] - central[model]["cuda"]["energy"]), 1e-7)
            check(model + ":CPU_CUDA_full_nuclear_gradient",
                  affine.difference(central[model]["cpu"]["qm_gradient"], central[model]["cuda"]["qm_gradient"]), 1e-7)
        step = .0005
        keyword = BASE + APPLIED + SOLVENT.format("PCM")
        gradient = []
        for index in range(9):
            atom, axis = divmod(index, 3)
            energies = []
            for sign in (1, -1):
                atoms = copy.deepcopy(affine.WATER)
                atoms[atom][1][axis] += sign * step * affine.BOHR_ANGSTROM
                energies.append(api_run("cuda", f"PCM-FD-{index}-{sign:+d}", keyword, atoms, False)["energy"])
            gradient.append((energies[0] - energies[1]) / (2 * step))
        check("PCM:CUDA_all_nuclear_gradient_vs_energy_FD", affine.difference(gradient, central["PCM"]["cuda"]["qm_gradient"]),
              1e-5, step_bohr=step, finite_difference=gradient)

        keyword = BASE.replace("HF ", "DFT PBE0 ").replace("STO-3G", "6-31G**") + " XCCUTOFF=1e-12 BASISCUTOFF=1e-12" + APPLIED
        dft = {backend: api_run(backend, "PBE0-631Gss-central", keyword) for backend in drivers}
        check("PBE0:CPU_CUDA_energy", abs(dft["cpu"]["energy"] - dft["cuda"]["energy"]), 1e-7)
        check("PBE0:CPU_CUDA_full_nuclear_gradient", affine.difference(dft["cpu"]["qm_gradient"], dft["cuda"]["qm_gradient"]), 1e-6)
        for backend in drivers:
            check(backend + ":PBE0_Cartesian_25AO", abs(dft[backend]["basis_functions"] - 25), 0.)

        keyword = (BASE + APPLIED + SOLVENT.format("PCM")).upper()
        trajectories = {}
        report["bomd"] = {}
        for backend in drivers:
            workdir = args.workdir / "bomd" / backend
            workdir.mkdir(parents=True, exist_ok=False)
            case_report = {"runs": [], "checks": [], "runtime_before": report["runtime_before"][backend]["bomd"]}
            case_args = SimpleNamespace(workdir=workdir, driver=drivers[backend]["bomd"], launcher="", ranks=1,
                                        timeout=args.timeout, drift_tolerance=1e-5)
            campaign = solvent_bomd.Campaign(case_args, case_report, envs[backend])
            on = campaign.run("properties-on", keyword, 4, stride=1, dt=.025)
            off = campaign.run("properties-off", keyword, 4, stride=0, dt=.025)
            for label, entry in (("on", on), ("off", off)):
                summary = entry["record"]["summary"]
                expected = [5, 5] if backend == "cuda" else [0, 0]
                check(f"{backend}:BOMD-{label}:native_field_GPU_dispatch", int(summary["applied_field_master_gpu_calls"] != expected), 0.)
                forces = np.loadtxt(entry["folder"] / "trajectory_forces.dat")
                if forces.shape != (15, 5) or not np.isfinite(forces).all():
                    raise ValueError("Expected complete finite instrumented BOMD gradients (5 calls x 3 atoms)")
                check(f"{backend}:BOMD-{label}:force_indices", max_difference(forces[:, :2],
                       np.asarray([[call, atom] for call in range(1, 6) for atom in range(1, 4)])), 0.)
                entry["forces"] = forces[:, 2:]
            check(backend + ":BOMD_sampling_preserves_all_forces", max_difference(on["forces"], off["forces"]), 1e-8)
            check(backend + ":BOMD_sampling_preserves_energy", max_difference(on["energies"][:, :4], off["energies"][:, :4]), 1e-8)
            check(backend + ":BOMD_sampling_preserves_geometry", max_difference(on["trajectory"], off["trajectory"]), 1e-8)
            check(backend + ":BOMD_sampling_preserves_final_velocity", max_difference(on["checkpoint"]["velocities"], off["checkpoint"]["velocities"]), 1e-8)
            resumed = campaign.run("same-Hamiltonian-restart", keyword, 0, stride=1, dt=.025, restart=on["folder"] / "trajectory.chk")
            check(backend + ":BOMD_restart_geometry", max_difference(on["trajectory"][-1:], resumed["trajectory"]), 1e-10)
            check(backend + ":BOMD_restart_energy", max_difference(on["energies"][-1:, :4], resumed["energies"][:, :4]), 1e-7)
            for column, error in property_difference([row for row in on["properties"] if int(row["step"]) == 4], resumed["properties"]).items():
                check(backend + ":BOMD_restart_property:" + column, error, 2e-7)
            folder = workdir / "reject-changed-tensor"; folder.mkdir(exist_ok=False)
            changed = keyword.replace("EFIELD_GXY=3E-5", "EFIELD_GXY=4E-5")
            if changed == keyword:
                raise ValueError("Restart mutation did not change the field gradient")
            write_control(folder / "control.nml", geometry=ROOT / "validation/oeprop_bomd/water.xyz",
                          probes=ROOT / "validation/oeprop_bomd/water_fixed.probe", prefix=Path("trajectory"), mode="fixed",
                          steps=0, stride=0, dt_fs=.025, keyword=changed, restart=on["folder"] / "trajectory.chk")
            process = subprocess.run([str(drivers[backend]["bomd"]), "control.nml"], cwd=folder, env=envs[backend],
                                     capture_output=True, text=True, timeout=args.timeout)
            (folder / "execution.log").write_text(process.stdout + process.stderr)
            check(backend + ":BOMD_restart_rejects_changed_G", int(process.returncode == 0 or "read checkpoint" not in process.stdout + process.stderr), 0.)
            report["bomd"][backend] = case_report
            report["checks"].extend({**item, "name": backend + ":" + item["name"]} for item in case_report["checks"])
            trajectories[backend] = on
            progress()
        check("BOMD:CPU_CUDA_energies", max_difference(trajectories["cpu"]["energies"][:, :4], trajectories["cuda"]["energies"][:, :4]), 1e-7)
        check("BOMD:CPU_CUDA_geometry", max_difference(trajectories["cpu"]["trajectory"], trajectories["cuda"]["trajectory"]), 1e-7)
        check("BOMD:CPU_CUDA_full_force_history", max_difference(trajectories["cpu"]["forces"], trajectories["cuda"]["forces"]), 1e-7)
        for column, error in property_difference(trajectories["cpu"]["properties"], trajectories["cuda"]["properties"]).items():
            check("BOMD:CPU_CUDA_property:" + column, error, 2e-7)
        report["runtime_after"] = {backend: {kind: affine.runtime(path, envs[backend]) for kind, path in kinds.items()}
                                   for backend, kinds in drivers.items()}
        report["inputs_after"] = {str(p): affine.sha(p) for p in set(sources)}
        check("runtime_unchanged", int(report["runtime_before"] != report["runtime_after"]), 0.)
        check("harness_fixture_basis_sources_unchanged", int(report["inputs_before"] != report["inputs_after"]), 0.)
        report["passed"] = all(item["passed"] for item in report["checks"])
    except Exception as error:
        import traceback
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["failed_checks"] = sum(not item["passed"] for item in report["checks"])
    path = args.workdir / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": report["passed"], "checks": len(report["checks"]), "failed_checks": report["failed_checks"],
                      "error": report.get("error"), "report": str(path)}), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
