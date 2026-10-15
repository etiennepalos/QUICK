#!/usr/bin/env python3
"""Validate applied-field and continuous-density ddX Born--Oppenheimer dynamics.

This is a separate feature campaign; the established ddX-disabled OEPROP
acceptance suite and its numerical gates are not changed. Requires NumPy.
"""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "validation/oeprop_bomd"))
from validate_oeprop_bomd import (ENERGY_COLUMNS, INVARIANT_COLUMNS, PROPERTY_COLUMNS,
    invariant_errors, max_difference, property_difference, read_checkpoint, read_csv,
    read_trajectory, write_control)
from validate_solvation import identities, runtime, sha256

BASE = "HF BASIS=STO-3G CUTOFF=1D-12 GRADCUTOFF=1D-12 DENSERMS=1D-10 GRADIENT CHARGE=0"
FIELD = " EXTERNAL_EFIELD_X=0.0002 EXTERNAL_EFIELD_Y=-0.0001"
SOLVENT = " DDX={} DDX_LMAX=6 DDX_NLEBEDEV=302 DDX_NRAD=75 DDX_NANG=302"
CASES = {"field": BASE + FIELD,
         "cosmo": BASE + SOLVENT.format("COSMO"),
         "pcm": BASE + SOLVENT.format("PCM"),
         "cosmo_field": BASE + SOLVENT.format("COSMO") + FIELD + " DDX_FIELD=SOLUTE",
         "pcm_field": BASE + SOLVENT.format("PCM") + FIELD + " DDX_FIELD=SOLUTE",
         "pbe0_pcm": BASE.replace("HF ", "DFT PBE0 ")
             + " XCCUTOFF=1D-12 BASISCUTOFF=1D-12" + SOLVENT.format("PCM"),
         "pbe0_pcm_spd": BASE.replace("HF ", "DFT PBE0 D3 ").replace("STO-3G", "DEF2-SVP")
             + " XCCUTOFF=1D-12 BASISCUTOFF=1D-12" + SOLVENT.format("PCM")}


class Campaign:
    def __init__(self, args, report, env):
        self.args, self.report, self.env = args, report, env
        self.gpu = "_cuda" in report["runtime_before"]["quick_library"]

    def check(self, name, error, tolerance, **details):
        self.report["checks"].append({"name": name, "error": float(error), "tolerance": tolerance,
            "passed": math.isfinite(error) and error <= tolerance, **details})

    def run(self, label, keyword, steps, stride=1, dt=0.05, restart=None):
        if len(keyword) > 256:
            raise ValueError("Keyword exceeds the established QUICK API's 256 characters")
        folder = self.args.workdir / label
        folder.mkdir(exist_ok=False)
        start_step = read_checkpoint(restart)["step"] if restart else 0
        absolute_steps = list(range(start_step, start_step + steps + 1))
        sampled = [s for s in absolute_steps if stride > 0 and s % stride == 0]
        write_control(folder / "control.nml", geometry=ROOT / "validation/oeprop_bomd/water.xyz",
            probes=ROOT / "validation/oeprop_bomd/water_fixed.probe", prefix=Path("trajectory"),
            mode="fixed", steps=steps, stride=stride, dt_fs=dt, keyword=keyword, restart=restart)
        command = shlex.split(self.args.launcher) + [str(self.args.driver), "control.nml"]
        record = {"label": label, "keyword": keyword, "command": command, "cwd": str(folder),
                  "input_sha256": sha256(folder / "control.nml")}
        self.report["runs"].append(record)
        print(f"RUN {label}", flush=True)
        start = time.monotonic()
        with (folder / "execution.log").open("w") as log:
            process = subprocess.Popen(command, cwd=folder, env=self.env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                record["returncode"] = process.wait(timeout=self.args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError(f"BOMD timeout: {folder}")
        record["wall_seconds"] = time.monotonic()-start
        if record["returncode"]:
            raise RuntimeError(f"BOMD exited {record['returncode']}: {folder}")
        summary = json.loads((folder / "trajectory_summary.json").read_text())
        energies = read_csv(folder / "trajectory_energy.csv", ["step"] + ENERGY_COLUMNS)
        properties = read_csv(folder / "trajectory_oeprop.csv", ["step", "probe", "time_fs", "probe_mode",
            "x_ang", "y_ang", "z_ang"] + PROPERTY_COLUMNS + INVARIANT_COLUMNS, allow_empty=stride == 0)
        ranks = read_csv(folder / "trajectory_kernels.csv", ["rank", "force_calls", "api_energy_gradient_calls",
            "property_calls", "properties_extra_api_calls", "esp_launches", "efield_launches", "efg_launches"])
        trajectory = read_trajectory(folder / "trajectory_trajectory.xyz", absolute_steps, dt)
        checkpoint = read_checkpoint(folder / "trajectory.chk")
        text = (folder / "quick_oeprop_bomd.out").read_text(errors="replace")
        cycles = [int(c) for c in re.findall(r"REACH CONVERGENCE AFTER\s+(\d+)\s+CYCLES", text)]
        force_calls = steps+1
        self.check(label+":explicit_converged_scfs", abs(len(cycles)-force_calls), 0)
        self.check(label+":no_failed_scf", int(bool(re.search(
            r"NO CONVERGENCE|RAN OUT OF CYCLES|FAILED TO CONVERGE", text, re.I))), 0)
        self.check(label+":force_calls", max(abs(summary[k]-force_calls) for k in
            ("force_calls", "api_energy_gradient_calls")), 0)
        if "EXTERNAL_EFIELD_X=" in keyword:
            field_calls = summary["applied_field_master_gpu_calls"]
            expected_field = [force_calls]*2 if self.gpu else [0]*2
            self.check(label+":native_field_gpu_operations", int(field_calls != expected_field), 0,
                       observed=field_calls,expected=expected_field)
        self.check(label+":properties_no_extra_scf", max(abs(summary["property_calls"]-len(sampled)),
            abs(summary["probe_generation_calls"]-len(sampled)), abs(summary["properties_extra_api_calls"])), 0)
        self.check(label+":step_sequence", int([int(r["step"]) for r in energies] != absolute_steps), 0)
        self.check(label+":energy_timestamps", max(abs(float(r["time_fs"])-int(r["step"])*dt) for r in energies), 1e-12)
        self.check(label+":checkpoint", max(abs(checkpoint["step"]-absolute_steps[-1]),
            abs(checkpoint["dt_fs"]-dt), max_difference(checkpoint["coordinates"], trajectory[-1])), 1e-11)
        self.check(label+":restart_hamiltonian", int(checkpoint["keyword"] != keyword.upper()
            or not summary["restart_hamiltonian_verified"]), 0)
        nprobe = int((ROOT / "validation/oeprop_bomd/water_fixed.probe").read_text().splitlines()[0])
        self.check(label+":property_rows", int([(int(r["step"]), int(r["probe"])) for r in properties] !=
            [(s,p) for s in sampled for p in range(1,nprobe+1)]), 0)
        if properties:
            for name, error in invariant_errors(properties).items():
                self.check(label+":tensor_"+name, error, 2e-10)
            self.check(label+":property_timestamps", max(abs(float(r["time_fs"])-int(r["step"])*dt)
                for r in properties), 1e-12)
            fixed = np.asarray([[float(v) for v in row.split()] for row in
                (ROOT / "validation/oeprop_bomd/water_fixed.probe").read_text().splitlines()[1:]])
            self.check(label+":fixed_probe_coordinates", max(max_difference(
                [[float(r[f"{axis}_ang"]) for axis in "xyz"] for r in properties if int(r["step"]) == s], fixed)
                for s in sampled), 1e-10)
            self.check(label+":fixed_probe_mode", int(any(r["probe_mode"] != "fixed" for r in properties)), 0)
        expected_ranks = self.args.ranks
        self.check(label+":ranks", int([int(r["rank"]) for r in ranks] != list(range(expected_ranks))
            or summary["mpi_ranks"] != expected_ranks), 0)
        active = "DDX=" in keyword and "DDX_EPS=1 " not in keyword+" "
        expected_launches = [len(sampled)]*3
        if active:
            expected_launches[0] += sum(cycles)+force_calls
            expected_launches[1] += force_calls
            self.check(label+":solvent_updates", abs(summary["solvation_updates"]-(sum(cycles)+force_calls)), 0)
            expected_volume = [sum(cycles)+force_calls]*2+[force_calls] if self.gpu else [0]*3
            if len(summary["solvation_gpu_volume_calls"]) != 3:
                raise ValueError("Solvent volume counter shape must be exactly three")
            self.check(label+":solvent_gpu_operations", max(abs(a-b) for a,b in zip(
                summary["solvation_gpu_volume_calls"], expected_volume)), 0)
            cache = summary["solvation_gpu_coulomb_cache"]
            if len(cache) != 7:
                raise ValueError("Solvent Coulomb cache counter shape must be exactly seven")
            if self.gpu:
                hits, builds, tbytes, gemvs, direct, fallbacks, allocated = cache
                self.check(label+":gpu_coulomb_geometry_lifecycle", max(abs(builds-force_calls),
                    abs(gemvs-(sum(cycles)+force_calls)),abs(hits-(gemvs-builds)),abs(direct),abs(fallbacks),
                    int(tbytes <= 0 or allocated < tbytes)), 0, observed=cache,
                    columns=["hits","builds","T_bytes","gemvs","direct_calls","fallbacks","allocated_bytes"])
            else:
                self.check(label+":cpu_no_gpu_coulomb_cache", max(abs(v) for v in cache), 0)
            phase_times = summary["solvation_master_phase_seconds"]
            phase_labels = {"cavity_model", "volume_grid", "solid_harmonics", "cavity_esp", "volume_density",
                "density_projection", "ddx_solve", "local_matrix", "coulomb_matrix", "cavity_efield",
                "ddx_gradient", "coulomb_gradient", "volume_gradient", "partition_gradient"}
            self.check(label+":solvent_timings", int(set(phase_times) != phase_labels or any(
                not math.isfinite(v) or v < 0 for v in phase_times.values())), 0)
        if not self.gpu:
            expected_launches = [0]*3
        for row in ranks:
            self.check(label+f":rank{row['rank']}_launches", max(abs(int(row[k])-v) for k,v in zip(
                ("esp_launches", "efield_launches", "efg_launches"), expected_launches)), 0,
                expected=expected_launches, observed=[int(row[k]) for k in ("esp_launches", "efield_launches", "efg_launches")])
            self.check(label+f":rank{row['rank']}_calls", max(abs(int(row[k])-v) for k,v in
                (("force_calls", force_calls), ("api_energy_gradient_calls", force_calls),
                 ("property_calls", len(sampled)), ("properties_extra_api_calls", 0))), 0)
        reference = checkpoint["reference_total_au"]
        self.check(label+":energy_identity", max(abs(float(r["total_au"])-float(r["potential_au"])
            -float(r["kinetic_au"])) for r in energies), 1e-12)
        self.check(label+":drift_identity", max(abs(float(r["delta_total_au"])
            -(float(r["total_au"])-reference)) for r in energies), 1e-12)
        drift = max(abs(float(r["total_au"])-reference) for r in energies)
        self.check(label+":summary_drift", abs(summary["max_abs_drift_au"]-drift), 1e-12)
        self.check(label+":summary_initial_energy", abs(summary["initial_total_au"]-reference), 1e-12)
        self.check(label+":summary_final_energy", abs(summary["final_total_au"]-float(energies[-1]["total_au"])), 1e-12)
        self.check(label+":short_nve_drift", drift, self.args.drift_tolerance)
        numeric_energies = np.asarray([[float(row[col]) for col in ENERGY_COLUMNS] for row in energies])
        record.update(summary=summary, recomputed_max_abs_drift_au=drift, scf_cycles=cycles, energies=numeric_energies.tolist(),
            trajectory_angstrom=trajectory.tolist(), checkpoint=checkpoint,
            output_sha256={p.name: sha256(p) for p in folder.iterdir() if p.is_file()})
        return {"record": record, "properties": properties, "trajectory": trajectory,
                "energies": numeric_energies, "checkpoint": checkpoint, "folder": folder}

    def case(self, name):
        keyword = CASES[name]
        enabled = self.run(name+"-stride1", keyword, self.args.steps)
        disabled = self.run(name+"-stride0", keyword, self.args.steps, stride=0)
        self.check(name+":property_sampling_preserves_energy", max_difference(
            enabled["energies"][:, :4], disabled["energies"][:, :4]), 1e-7)
        self.check(name+":property_sampling_preserves_geometry", max_difference(
            enabled["trajectory"], disabled["trajectory"]), 1e-7)
        self.check(name+":property_sampling_preserves_velocities", max_difference(
            enabled["checkpoint"]["velocities"], disabled["checkpoint"]["velocities"]), 1e-7)
        split = self.args.steps//2
        first = self.run(name+"-split", keyword, split, stride=2)
        resumed = self.run(name+"-restart", keyword, self.args.steps-split, stride=2,
                           restart=first["folder"] / "trajectory.chk")
        self.check(name+":restart_geometry", max_difference(enabled["trajectory"][split:], resumed["trajectory"]), 1e-7)
        self.check(name+":restart_energies", max_difference(enabled["energies"][split:,:4], resumed["energies"][:,:4]), 1e-7)
        self.check(name+":restart_velocities", max_difference(enabled["checkpoint"]["velocities"],
            resumed["checkpoint"]["velocities"]), 1e-7)
        subset = [r for r in enabled["properties"] if int(r["step"]) >= split and int(r["step"])%2 == 0]
        for column, error in property_difference(subset, resumed["properties"]).items():
            self.check(name+":restart_"+column, error, 2e-7)
        if self.args.energy_steps:
            coarse_dt = self.args.energy_dt
            fine_dt = 0.5*coarse_dt
            coarse = self.run(name+f"-nve-dt{coarse_dt:g}", keyword, self.args.energy_steps, dt=coarse_dt)
            fine = self.run(name+f"-nve-dt{fine_dt:g}", keyword, 2*self.args.energy_steps, dt=fine_dt)
            c = coarse["record"]["recomputed_max_abs_drift_au"]
            f = fine["record"]["recomputed_max_abs_drift_au"]
            self.check(name+":timestep_refinement", max(0, f-0.4*c), 5e-8,
                coarse_drift=c, fine_drift=f, coarse_dt_fs=coarse_dt, fine_dt_fs=fine_dt,
                equal_physical_time_fs=coarse_dt*self.args.energy_steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--basis", type=Path, default=ROOT/"basis")
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--launcher", default="")
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--energy-steps", type=int, default=40,
                        help="Coarse NVE steps; the refined trajectory uses twice this count")
    parser.add_argument("--energy-dt", type=float, default=0.05,
                        help="Coarse NVE timestep in fs; refined timestep is half this value")
    parser.add_argument("--drift-tolerance", type=float, default=1e-5)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    if (args.steps < 4 or args.energy_steps < 0 or args.ranks < 1
            or not math.isfinite(args.energy_dt) or args.energy_dt <= 0 or 0.5*args.energy_dt <= 0
            or not math.isfinite(args.timeout) or args.timeout <= 0
            or not math.isfinite(args.drift_tolerance) or args.drift_tolerance < 0):
        parser.error("Require steps >=4, energy-steps >=0, ranks>=1, finite positive energy-dt/timeout, "
                     "and finite nonnegative drift tolerance")
    args.driver, args.basis = args.driver.resolve(strict=True), args.basis.resolve(strict=True)
    args.workdir = args.workdir.resolve()
    args.workdir.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(QUICK_BASIS=str(args.basis), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    sources = [Path(__file__), ROOT/"CMakeLists.txt", ROOT/"QUICKConfig.cmake.in",
        *[p for d in (ROOT/"src", ROOT/"quick-cmake", ROOT/"tools", ROOT/"validation", args.basis)
          for p in d.rglob("*") if p.is_file() and "__pycache__" not in p.parts]]
    report = {"passed": False, "started_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}, "runs": [], "checks": [],
        "scope": "Finite water; electrostatic equilibrium solvent; defined solute-only applied-field convention; short NVE software/force validation, not production ensembles or spectra"}
    try:
        report["source_before"], report["runtime_before"] = identities(sources), runtime(args.driver, env)
        campaign = Campaign(args, report, env)
        for name in args.cases:
            campaign.case(name)
        report["source_after"], report["runtime_after"] = identities(sources), runtime(args.driver, env)
        campaign.check("source_unchanged", int(report["source_before"] != report["source_after"]), 0)
        campaign.check("runtime_unchanged", int(report["runtime_before"] != report["runtime_after"]), 0)
        report["passed"] = bool(report["checks"]) and all(c["passed"] for c in report["checks"])
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["failed_checks"] = sum(not c["passed"] for c in report["checks"])
    (args.workdir/"report.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"passed": report["passed"], "runs": len(report["runs"]),
        "failed_checks": report["failed_checks"], "error": report.get("error")}), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
