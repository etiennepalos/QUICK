#!/usr/bin/env python3
"""Validate a short, fully coupled CUDA solvent/field trajectory on supplied XYZ.

Software and force-transfer checks only; no ensemble or long-time MD claim.
The input geometry, probes, source tree, loaded libraries and raw outputs are
fingerprinted. Acceptance gates are fixed before executing the calculation.
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
import signal
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--driver', type=Path, required=True)
    p.add_argument('--geometry', type=Path, default=Path(__file__).with_name('fixtures')/'water27_h2o20es.xyz')
    p.add_argument('--probes', type=Path, default=Path(__file__).with_name('fixtures')/'water27_h2o20es.probe')
    p.add_argument('--basis', type=Path, help='Defaults to the selected repository basis directory')
    p.add_argument('--workdir', type=Path, required=True)
    p.add_argument('--model', choices=('COSMO', 'PCM'), default='PCM')
    p.add_argument('--coordinate-atoms',type=int,nargs=2,default=(1,4))
    p.add_argument('--steps', type=int, default=2)
    p.add_argument('--timeout', type=float, default=7200.)
    a = p.parse_args()
    if a.steps < 2 or not math.isfinite(a.timeout) or a.timeout <= 0:
        p.error('Require at least two steps and a finite positive timeout')
    if a.basis is None:
        a.basis = a.repo/'basis'
    for key in ('repo', 'driver', 'geometry', 'probes', 'basis'):
        setattr(a, key, getattr(a, key).resolve(strict=True))
    input_natom=int(a.geometry.read_text().splitlines()[0])
    if (len(set(a.coordinate_atoms))!=2 or any(i<1 or i>input_natom for i in a.coordinate_atoms)):
        p.error('Coordinate atom indices must be distinct valid one-based atom indices')
    a.workdir = a.workdir.resolve()
    a.workdir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(a.repo / 'validation/oeprop_bomd'))
    sys.path.insert(0, str(a.repo / 'validation/solvation'))
    from validate_oeprop_bomd import (write_control, read_csv, read_checkpoint,
        read_trajectory, invariant_errors, max_difference, ENERGY_COLUMNS,
        PROPERTY_COLUMNS, INVARIANT_COLUMNS)
    from validate_solvation import identities, runtime, sha256
    env = os.environ.copy()
    env.update(QUICK_BASIS=str(a.basis), OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    keyword = ('DFT PBE0 D3 BASIS=DEF2-SVP SCF=120 DENSERMS=1D-9 CUTOFF=1D-12 '
        'GRADCUTOFF=1D-12 XCCUTOFF=1D-12 BASISCUTOFF=1D-12 GRADIENT '
        f'DDX={a.model} EXTERNAL_EFIELD_X=0.0002 DDX_FIELD=SOLUTE')
    if len(keyword) > 256:
        raise ValueError('Keyword exceeds QUICK API256-character limit')
    sources = [Path(__file__).resolve(), a.geometry, a.probes,
        a.repo/'CMakeLists.txt', a.repo/'QUICKConfig.cmake.in',
        *[f for d in ('src', 'quick-cmake', 'tools', 'validation', 'basis')
          for f in (a.repo/d).rglob('*') if f.is_file() and '__pycache__' not in f.parts],
        *[f for f in a.basis.rglob('*') if f.is_file()]]
    report = {'passed': False, 'started_utc': datetime.now(timezone.utc).isoformat(),
        'arguments': {k: str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        'keyword': keyword, 'checks': [], 'dt_fs': .025,
        'scope': 'CUDA coupled PBE0-D3/def2-SVP with default ddX cavity/volume grids, solute-only field, ESP/EFIELD/EFG every accepted step. Short software/force-transfer check; no ensemble or long-time drift claim.',
        'gates': {'short_nve_drift_hartree': 1e-5, 'energy_identity_hartree': 2e-12,
                  'net_gradient_au': 3e-7, 'tensor_invariants': 2e-9}}
    def check(name, error, tolerance, **details):
        report['checks'].append(dict(name=name, error=float(error), tolerance=tolerance,
            passed=math.isfinite(error) and error <= tolerance, **details))
    try:
        report['source_before'] = identities(sources)
        report['runtime_before'] = runtime(a.driver, env)
        if '_cuda' not in report['runtime_before']['quick_library']:
            raise ValueError('This transfer campaign requires native CUDA')
        write_control(a.workdir/'control.nml', geometry=a.geometry, probes=a.probes,
            prefix=Path('trajectory'), mode='fixed', steps=a.steps, stride=1,
            dt_fs=.025, keyword=keyword, coordinate_atoms=tuple(a.coordinate_atoms))
        report['input_sha256'] = sha256(a.workdir/'control.nml')
        command = [str(a.driver), 'control.nml']
        report['command'] = command
        start = time.monotonic()
        with (a.workdir/'execution.log').open('w') as log:
            proc = subprocess.Popen(command, cwd=a.workdir, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
            try:
                report['returncode'] = proc.wait(timeout=a.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                raise RuntimeError('Coupled trajectory timed out')
        report['wall_seconds'] = time.monotonic()-start
        if report['returncode']:
            raise RuntimeError('Coupled trajectory exited '+str(report['returncode']))
        summary = json.loads((a.workdir/'trajectory_summary.json').read_text())
        report['summary'] = summary
        energies = read_csv(a.workdir/'trajectory_energy.csv', ['step']+ENERGY_COLUMNS)
        properties = read_csv(a.workdir/'trajectory_oeprop.csv',
            ['step','probe','time_fs','probe_mode','x_ang','y_ang','z_ang']
            +PROPERTY_COLUMNS+INVARIANT_COLUMNS)
        kernels = read_csv(a.workdir/'trajectory_kernels.csv',
            ['rank','force_calls','api_energy_gradient_calls','property_calls','properties_extra_api_calls',
             'esp_launches','efield_launches','efg_launches'])
        checkpoint = read_checkpoint(a.workdir/'trajectory.chk')
        expected_steps = list(range(a.steps+1))
        trajectory = read_trajectory(a.workdir/'trajectory_trajectory.xyz', expected_steps, .025)
        natom = int(a.geometry.read_text().splitlines()[0])
        check('atom_count', abs(trajectory.shape[1]-natom), 0)
        report['natom'] = natom
        atom_records=[line.split() for line in a.geometry.read_text().splitlines()[2:]]
        if len(atom_records)!=natom or any(len(row)!=4 for row in atom_records):
            raise ValueError('Malformed input XYZ geometry')
        input_coordinates=[[float(value) for value in row[1:]] for row in atom_records]
        check('initial_geometry',max_difference(trajectory[0],input_coordinates),1e-11)
        text = (a.workdir/'quick_oeprop_bomd.out').read_text(errors='replace')
        cycles = [int(v) for v in re.findall(r'REACH CONVERGENCE AFTER\s+(\d+)\s+CYCLES',text)]
        report['scf_cycles'] = cycles
        force_count = a.steps+1
        update_count = sum(cycles)+force_count
        check('explicit_converged_scfs', abs(len(cycles)-force_count), 0)
        check('no_unconverged_scf', int(bool(re.search(
            r'NO CONVERGENCE|RAN OUT OF CYCLES|FAILED TO CONVERGE', text, re.I))),0)
        for key in ('force_calls','api_energy_gradient_calls','property_calls','probe_generation_calls'):
            check(key, abs(summary[key]-force_count),0)
        check('no_extra_property_scf', abs(summary['properties_extra_api_calls']),0)
        check('native_field_operations', int(summary['applied_field_master_gpu_calls'] != [force_count]*2),0)
        check('volume_gpu_operations', int(summary['solvation_gpu_volume_calls'] != [update_count]*2+[force_count]),0)
        check('solvent_updates', abs(summary['solvation_updates']-update_count),0)
        cache = summary['solvation_gpu_coulomb_cache']
        if len(cache)!=7:
            raise ValueError('Coulomb cache counter shape must be exactly seven')
        check('coulomb_cache_storage',int(cache[2]<=0 or cache[6]<cache[2]),0)
        check('coulomb_cache_builds',abs(cache[1]-force_count),0)
        check('coulomb_cache_gemvs',abs(cache[3]-update_count),0)
        check('coulomb_cache_hits',abs(cache[0]-sum(cycles)),0)
        check('no_direct_or_fallback',abs(cache[4])+abs(cache[5]),0)
        expected_kernels = [force_count+update_count, 2*force_count, force_count]
        check('one_gpu_rank',int(len(kernels)!=1 or int(kernels[0]['rank'])!=0 or summary['mpi_ranks']!=1),0)
        if len(kernels)!=1:
            raise ValueError('Require exactly one kernel rank row')
        for key in ('force_calls','api_energy_gradient_calls','property_calls'):
            check('kernel_'+key,abs(int(kernels[0][key])-force_count),0)
        check('kernel_properties_no_extra_scf',abs(int(kernels[0]['properties_extra_api_calls'])),0)
        for key, expected in zip(('esp_launches','efield_launches','efg_launches'),expected_kernels):
            check(key,abs(int(kernels[0][key])-expected),0)
        check('energy_step_sequence',int([int(r['step']) for r in energies]!=expected_steps),0)
        check('energy_timestamps',max(abs(float(r['time_fs'])-int(r['step'])*.025) for r in energies),1e-12)
        reference=checkpoint['reference_total_au']
        drift=max(abs(float(r['total_au'])-reference) for r in energies)
        report['recomputed_max_abs_drift_au']=drift
        check('energy_identity',max(abs(float(r['total_au'])-float(r['potential_au'])-float(r['kinetic_au'])) for r in energies),2e-12)
        check('raw_drift_identity',max(abs(float(r['delta_total_au'])-(float(r['total_au'])-reference)) for r in energies),2e-12)
        check('summary_drift',abs(summary['max_abs_drift_au']-drift),2e-12)
        check('short_nve_drift',drift,1e-5)
        check('summary_initial_energy',abs(summary['initial_total_au']-reference),2e-12)
        check('summary_final_energy',abs(summary['final_total_au']-float(energies[-1]['total_au'])),2e-12)
        net_gradient=max(float(r['net_gradient_au_per_bohr']) for r in energies)
        check('summary_net_gradient',abs(summary['max_net_gradient_au_per_bohr']-net_gradient),2e-12)
        check('neutral_net_gradient',net_gradient,3e-7)
        nprobe=int(a.probes.read_text().splitlines()[0])
        check('property_step_probe_sequence',int([(int(r['step']),int(r['probe'])) for r in properties]
            !=[(s,i) for s in expected_steps for i in range(1,nprobe+1)]),0)
        probe_rows=[line.split() for line in a.probes.read_text().splitlines()[1:]]
        if len(probe_rows)!=nprobe or any(len(row)!=3 for row in probe_rows):
            raise ValueError('Malformed fixed probe input')
        probe_coordinates=[[float(value) for value in row] for row in probe_rows]
        check('property_timestamps',max(abs(float(r['time_fs'])-int(r['step'])*.025) for r in properties),1e-12)
        check('property_fixed_coordinates',max(max_difference(
            [[float(r[axis+'_ang']) for axis in 'xyz'] for r in properties if int(r['step'])==step],
            probe_coordinates) for step in expected_steps),1e-10)
        check('property_fixed_mode',int(any(r['probe_mode']!='fixed' for r in properties)),0)
        clearance=min(math.sqrt(sum((float(probe[axis])-float(atom[axis]))**2 for axis in range(3)))
            for frame in trajectory for atom in frame for probe in probe_coordinates)
        report['minimum_probe_nuclear_distance_angstrom']=clearance
        check('probe_nuclear_clearance',max(0.0,.8-clearance),0)
        for name,error in invariant_errors(properties).items():
            check('tensor_'+name,error,2e-9)
        check('checkpoint_step',abs(checkpoint['step']-a.steps),0)
        check('checkpoint_dt',abs(checkpoint['dt_fs']-.025),1e-12)
        check('restart_hamiltonian_verified',int(not summary['restart_hamiltonian_verified']),0)
        check('checkpoint_geometry',max_difference(checkpoint['coordinates'],trajectory[-1]),1e-11)
        check('checkpoint_hamiltonian',int(checkpoint['keyword']!=keyword.upper()),0)
        report['source_after']=identities(sources)
        report['runtime_after']=runtime(a.driver,env)
        check('source_unchanged',int(report['source_before']!=report['source_after']),0)
        check('runtime_unchanged',int(report['runtime_before']!=report['runtime_after']),0)
        report['passed']=bool(report['checks']) and all(c['passed'] for c in report['checks'])
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
    report['raw_output_sha256']={f.name:sha256(f) for f in a.workdir.iterdir() if f.is_file()}
    report['finished_utc']=datetime.now(timezone.utc).isoformat()
    report['failed_checks']=[c for c in report['checks'] if not c['passed']]
    (a.workdir/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:report.get(k) for k in ('passed','natom','wall_seconds','failed_checks','error')}),flush=True)
    return 0 if report['passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())
