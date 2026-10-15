#!/usr/bin/env python3
"""Separate matched-SG1 force consistency from independent XC convergence.

This bounded CPU diagnostic recomputes every molecular gradient component
from SCF energies at geometry-specific quadratures. It never changes QUICK
source, replaces historical failed reports, or treats matched-grid agreement
as convergence of SG1. The supported fixture is neutral RKS water/STO-3G.
"""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import pyscf
from pyscf import dft, gto, lib
from pyscf.gto.basis import parse_gaussian

BOHR = 0.52917720859


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--driver', type=Path, required=True)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--quick-report', type=Path, required=True)
    parser.add_argument('--higher-grid-report', type=Path, required=True)
    parser.add_argument('--steps', type=float, nargs='+', default=[.001, .0005])
    parser.add_argument('--grid-levels', type=int, nargs='*', default=[5, 7, 9])
    args = parser.parse_args()
    if (len(set(args.steps)) != len(args.steps)
            or any(not math.isfinite(h) or h <= 0 for h in args.steps)
            or args.grid_levels != sorted(set(args.grid_levels))
            or any(level < 0 or level > 9 for level in args.grid_levels)):
        parser.error('Require positive unique steps and increasing PySCF grid levels in [0,9]')
    for name in ('repo', 'driver', 'quick_report', 'higher_grid_report'):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.workdir = args.workdir.resolve()
    args.workdir.mkdir(parents=True, exist_ok=False)
    lib.num_threads(1)
    sys.path.insert(0, str(args.repo / 'validation/solvation'))
    sys.path.insert(0, str(args.repo / 'validation/external_field'))
    from validate_solvation import runtime
    from validate_pyscf_external_field import parse_model

    native = json.loads(args.quick_report.read_text())
    if not native['passed']:
        raise ValueError('Require an accepted native field report')
    target = next(row for row in native['runs']
                  if row['label'] == 'water_pbe0_sto3g-mixed_plus' and not row['reference'])
    frozen_input = Path(target['workdir']) / 'field.in'
    if sha(frozen_input) != target['input_sha256']:
        raise ValueError('Historical native input changed')
    model = parse_model(frozen_input)
    if (model['method'] != 'PBE0' or model['basis'].upper() != 'STO-3G'
            or model['charge'] or model['spin'] or model['unrestricted']):
        raise ValueError('Only neutral restricted PBE0/STO-3G is supported')
    higher = json.loads(args.higher_grid_report.read_text())
    higher_output = args.higher_grid_report.parent / 'oracle-output.json'
    if sha(higher_output) != higher['oracle_output_sha256'] or not higher['exact_quick_basis']:
        raise ValueError('Historical independent exact-basis reference changed')

    symbols = [row[0] for row in model['atoms_bohr']]
    base = np.array([row[1] for row in model['atoms_bohr']])
    basisfile = args.repo / 'basis/STO-3G.BAS'
    native_basis = (Path(native['arguments']['basis']) / 'STO-3G.BAS').resolve()
    if any(native[when].get(str(native_basis)) != sha(basisfile)
           for when in ('source_inputs_before', 'source_inputs_after')):
        raise ValueError('Current exact basis differs from the accepted native fixture')
    basis = {}
    for symbol in set(symbols):
        blocks = [block for block in basisfile.read_text().split('****')
                  if re.search(r'^\s*' + symbol + r'\s+0\s*$', block, re.M | re.I)]
        if len(blocks) != 1:
            raise ValueError('Missing exact QUICK basis block')
        basis[symbol] = parse_gaussian.parse(blocks[0], optimize=False)

    env = os.environ.copy()
    env.update(QUICK_BASIS=str(args.repo / 'basis'), OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    native_runtime = runtime(args.driver, env)
    if any(marker in native_runtime['quick_library'] for marker in ('_cuda', '_hip', '_mpi')):
        raise ValueError('This diagnostic requires a serial CPU exporter')
    source_files = [Path(__file__).resolve(), args.driver, basisfile,
                    args.repo / 'validation/external_field/quick_sg1_export_driver.f90',
                    args.repo / 'src/modules/quick_gridpoints_module.f90',
                    args.repo / 'src/modules/quick_scf_operator_module.f90',
                    args.repo / 'src/modules/quick_method_module.f90',
                    args.quick_report, frozen_input, args.higher_grid_report, higher_output]
    package = Path(pyscf.__file__).resolve().parent
    independent_files = [Path(sys.executable).resolve()]
    independent_files += [package / rel for rel in (
        'gto/mole.py', 'gto/basis/parse_gaussian.py', 'scf/hf.py', 'dft/rks.py',
        'dft/gen_grid.py', 'dft/numint.py', 'dft/libxc.py')]
    independent_files += sorted({p.resolve() for p in (package / 'lib').rglob('*.so*') if p.is_file()})
    before = {str(p): sha(p) for p in source_files + independent_files}
    report = {
        'passed': False, 'complete': False, 'started_utc': datetime.now(timezone.utc).isoformat(),
        'scope': __doc__, 'arguments': {k: str(v) if isinstance(v, Path) else v
                                     for k, v in vars(args).items()},
        'pyscf_version': pyscf.__version__, 'numpy_version': np.__version__,
        'settings': {'cartesian': True, 'basis': 'exact QUICK STO-3G.BAS contractions',
                     'threads': 1, 'scf_energy_tolerance': 1e-12,
                     'orbital_gradient_tolerance': 1e-9, 'direct_scf_tolerance': 1e-14,
                     'additional_density_pruning': False,
                     'native_weight_skip_reproduced': True,
                     'native_bin_primitive_low_density_screening_reproduced': False,
                     'field_origin_bohr': [0., 0., 0.],
                     'origin_note': 'Neutral molecule; fixed-origin total energy and forces are invariant.'},
        'tolerances': {'energy_hartree': 5e-5, 'gradient_hartree_per_bohr': 5e-5},
        'historical_absolute_report': {'path': str(args.higher_grid_report),
            'sha256': sha(args.higher_grid_report), 'passed': higher['passed'],
            'failed_checks': [c for c in higher['checks'] if not c['passed']]},
        'source_runtime_before': before, 'quick_runtime_before': native_runtime,
        'runs': [], 'comparisons': [], 'checks': [],
        'limitations': ['Matched-SG1 consistency does not establish SG1 grid convergence.',
                        'Independent level grids use PySCF radial/pruning/partition conventions.',
                        'These water/STO-3G checks are not convergence of a WATER20 production model.',
                        'No GPU execution; no ddX, D3, affine tensor, or unrestricted reference here.',
                        'Original higher-grid failures remain unchanged.']}
    started = time.monotonic()

    def save():
        (args.workdir / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    def check(name, error, tolerance, **details):
        report['checks'].append({'name': name, 'max_abs_error': float(error),
                                 'tolerance': tolerance,
                                 'passed': math.isfinite(error) and error <= tolerance, **details})

    def independent(folder, coordinates, field, grid=None, level=None):
        mol = gto.M(atom=list(zip(symbols, coordinates)), basis=basis,
                    unit='Bohr', cart=True, verbose=0)
        mf = dft.RKS(mol)
        mf.xc = 'PBE0'
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-9
        mf.direct_scf_tol = 1e-14
        mf.max_cycle = 250
        mf.small_rho_cutoff = 0
        mf.verbose = 4
        mf.stdout = (folder / 'pyscf.log').open('w')
        if grid is not None:
            mf.grids.coords = np.ascontiguousarray(grid[:, :3])
            mf.grids.weights = np.ascontiguousarray(grid[:, 3])
            mf.grids.non0tab = mf.grids.make_mask(mol, mf.grids.coords)
        else:
            mf.grids.level = level
        moments = mol.intor('int1e_r', comp=3, hermi=1)
        core = mf.get_hcore() + np.einsum('x,xij->ij', field, moments)
        nuclear = mol.energy_nuc() - np.einsum('a,ax,x->', mol.atom_charges(), coordinates, field)
        mf.get_hcore = lambda *unused: core
        mf.energy_nuc = lambda *unused: nuclear
        energy = float(mf.kernel())
        if not mf.converged or not math.isfinite(energy):
            raise RuntimeError('Independent SCF did not converge: ' + str(folder))
        dm = mf.make_rdm1()
        rho = dft.numint.eval_rho(mol, dft.numint.eval_ao(mol, mf.grids.coords), dm)
        result = {'energy_hartree': energy, 'converged': bool(mf.converged),
                  'grid_points': len(mf.grids.weights), 'basis_functions': mol.nao_nr(),
                  'integrated_electrons': float(np.dot(mf.grids.weights, rho)),
                  'density_ao': dm.tolist()}
        mf.stdout.close()
        (folder / 'pyscf-result.json').write_text(json.dumps(result, indent=2) + '\n')
        return result

    geometries = [('central', None, None, 0., 0)]
    geometries += [(f'{atom + 1}{"xyz"[axis]}-h{step:g}-s{sign:+d}', atom, axis, step, sign)
                   for atom in range(len(symbols)) for axis in range(3)
                   for step in args.steps for sign in (1, -1)]
    try:
        for field_name, field in [('zero', np.zeros(3)), ('field', np.array(model['field_au']))]:
            for grid_name in ['matched-SG1'] + [f'level-{level}' for level in args.grid_levels]:
                for label, atom, axis, step, sign in geometries:
                    run_name = f'{field_name}/{grid_name}/{label}'
                    folder = args.workdir / run_name
                    folder.mkdir(parents=True)
                    coordinates = base.copy()
                    if atom is not None:
                        coordinates[atom, axis] += step * sign
                    request = {'symbols': symbols, 'coordinates_bohr': coordinates.tolist(),
                               'field_au': field.tolist(), 'grid': grid_name}
                    (folder / 'request.json').write_text(json.dumps(request, indent=2) + '\n')
                    row = {'label': run_name, 'field': field_name, 'grid': grid_name,
                           'atom_index': atom, 'axis_index': axis, 'step_bohr': step, 'sign': sign}
                    tick = time.monotonic()
                    grid = None
                    if grid_name == 'matched-SG1':
                        card = ('DFT PBE0 BASIS=STO-3G CHARGE=0 MULT=1 GRADIENT SCF=250 '
                                'DENSERMS=1D-10 CUTOFF=1D-12 GRADCUTOFF=1D-12 '
                                'XCCUTOFF=1D-12 BASISCUTOFF=1D-12')
                        card += ''.join(f' EXTERNAL_EFIELD_{ax.upper()}={value:.12g}'
                                        for ax, value in zip('xyz', field) if value)
                        if len(card) > 256:
                            raise ValueError('Native API keyword exceeds its length')
                        (folder / 'input.dat').write_text(card + f'\n{len(symbols)}\n' + ''.join(
                            f'{gto.charge(sym)} ' + ' '.join(f'{v:.17g}' for v in xyz * BOHR) + '\n'
                            for sym, xyz in zip(symbols, coordinates)))
                        with (folder / 'execution.log').open('w') as log:
                            proc = subprocess.run([str(args.driver), 'input.dat'], cwd=folder, env=env,
                                                  stdout=log, stderr=subprocess.STDOUT, timeout=120)
                        if proc.returncode:
                            raise RuntimeError('Native exporter failed: ' + run_name)
                        raw = (folder / 'result.dat').read_text().splitlines()
                        values = {line.split()[0]: float(line.split()[1]) for line in raw
                                  if line.split()[0] != 'gradient'}
                        gradient = np.array([[float(x) for x in line.split()[2:]]
                                             for line in raw if line.startswith('gradient')])
                        grid = np.loadtxt(folder / 'grid.dat', skiprows=1)
                        if gradient.shape != (len(symbols), 3) or grid.ndim != 2 or grid.shape[1] != 6:
                            raise ValueError('Malformed native output')
                        if not np.all(np.isfinite(grid)) or not np.all(np.isfinite(gradient)):
                            raise ValueError('Nonfinite native output')
                        skipped = grid[:, 3] < values['matrix_cutoff']
                        grid[skipped, 3] = 0.
                        row.update(quick_energy=values['energy'], quick_gradient=gradient.tolist(),
                                   quick_settings=values, native_weight_skips=int(skipped.sum()))
                    oracle = independent(folder, coordinates, field, grid,
                                         int(grid_name.split('-')[1]) if grid is None else None)
                    row.update(pyscf_energy=oracle['energy_hartree'], grid_points=oracle['grid_points'],
                               integrated_electrons=oracle['integrated_electrons'],
                               wall_seconds=time.monotonic() - tick,
                               raw_hashes={p.name: sha(p) for p in folder.iterdir() if p.is_file()})
                    report['runs'].append(row)
                    save()
                    print(run_name, 'seconds', round(row['wall_seconds'], 3), flush=True)
        for field_name in ('zero', 'field'):
            center = next(r for r in report['runs'] if r['field'] == field_name
                          and r['grid'] == 'matched-SG1' and r['atom_index'] is None)
            analytic = np.array(center['quick_gradient']).reshape(-1)
            last = {}
            for grid_name in ['matched-SG1'] + [f'level-{level}' for level in args.grid_levels]:
                runs = [r for r in report['runs'] if r['field'] == field_name and r['grid'] == grid_name]
                central = next(r for r in runs if r['atom_index'] is None)
                for step in args.steps:
                    pgradient, qgradient = [], []
                    for atom in range(len(symbols)):
                        for axis in range(3):
                            plus, minus = [next(r for r in runs if r['atom_index'] == atom
                                                and r['axis_index'] == axis and r['step_bohr'] == step
                                                and r['sign'] == sign) for sign in (1, -1)]
                            pgradient.append((plus['pyscf_energy'] - minus['pyscf_energy']) / (2 * step))
                            if grid_name == 'matched-SG1':
                                qgradient.append((plus['quick_energy'] - minus['quick_energy']) / (2 * step))
                    error = float(np.max(np.abs(analytic - pgradient)))
                    item = {'field': field_name, 'grid': grid_name, 'step_bohr': step,
                            'quick_analytic': analytic.tolist(), 'pyscf_FD_gradient': pgradient,
                            'quick_analytic_minus_pyscf_FD': (analytic - pgradient).tolist(),
                            'max_abs_quick_analytic_minus_pyscf_FD': error,
                            'central_energy_hartree': central['pyscf_energy']}
                    label = f'{field_name}:{grid_name}:h={step:g}'
                    check(label + ':native_analytic_vs_independent_FD', error, 5e-5)
                    if qgradient:
                        item['quick_FD_gradient'] = qgradient
                        check(label + ':native_analytic_vs_native_FD',
                              float(np.max(np.abs(analytic - qgradient))), 5e-5)
                    if step in last:
                        previous = last[step]
                        dg = float(np.max(np.abs(np.array(pgradient) - previous['pyscf_FD_gradient'])))
                        de = abs(central['pyscf_energy'] - previous['central_energy_hartree'])
                        item['previous_grid'] = previous['grid']
                        item['max_abs_gradient_change_from_previous_grid'] = dg
                        item['abs_energy_change_from_previous_grid'] = de
                        check(label + ':grid_refinement_gradient', dg, 5e-5)
                        check(label + ':grid_refinement_energy', de, 5e-5)
                    last[step] = item
                    report['comparisons'].append(item)
                if grid_name == 'matched-SG1':
                    check(field_name + ':matched-SG1:max_abs_independent_energy',
                          max(abs(r['quick_energy'] - r['pyscf_energy']) for r in runs), 5e-5)
        report['complete'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        report['source_runtime_after'] = {str(p): sha(p) for p in source_files + independent_files}
        report['quick_runtime_after'] = runtime(args.driver, env)
        check('source_runtime_unchanged', float(before != report['source_runtime_after']), 0.)
        check('quick_runtime_unchanged', float(native_runtime != report['quick_runtime_after']), 0.)
        report['failed_checks'] = sum(not c['passed'] for c in report['checks'])
        report['passed'] = report['complete'] and not report['failed_checks']
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        report['wall_seconds'] = time.monotonic() - started
        save()
    print(json.dumps({k: report[k] for k in ('passed', 'complete', 'failed_checks', 'wall_seconds')}, indent=2))
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
