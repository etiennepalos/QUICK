#!/usr/bin/env python3
"""Compare opt-in native XC grids with a preserved converged-grid reference.

The input reference is validate_pbe0_xc_convergence.py's complete water
report, including its retained SG1 failures. No historical check is changed.
Native central energies/full gradients are compared for every selected grid;
the final grid also gets full native energy finite differences at both h.
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
import subprocess
import sys
import time

BOHR = 0.52917720859


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--driver', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--grids', nargs='+', default=['50x194', '75x302', '100x434', '150x590'])
    parser.add_argument('--steps', type=float, nargs='+', default=[.001, .0005])
    parser.add_argument('--matrixzero', type=float, default=1e-10,
                        help='Existing QUICK density/weight threshold; refine jointly with the grid')
    parser.add_argument('--central-only', action='store_true')
    parser.add_argument('--allow-gpu', action='store_true', help='Requires an independently authorized GPU window')
    args = parser.parse_args()
    for name in ('repo', 'driver', 'reference'):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.workdir = args.workdir.resolve()
    args.workdir.mkdir(parents=True, exist_ok=False)
    grids = [tuple(int(x) for x in text.split('x')) for text in args.grids]
    if any(len(pair) != 2 or min(pair) < 1 for pair in grids):
        parser.error('Use positive radialxangular grids')
    if any(not math.isfinite(h) or h <= 0 for h in args.steps):
        parser.error('Use finite positive steps')
    if not math.isfinite(args.matrixzero) or args.matrixzero <= 0:
        parser.error('Use a finite positive MATRIXZERO threshold')
    reference = json.loads(args.reference.read_text())
    if not reference['complete']:
        raise ValueError('Independent reference is incomplete')
    independent = [row for row in reference['comparisons'] if row['grid'].startswith('level-')]
    highest = max(int(row['grid'].split('-')[1]) for row in independent)
    if highest < 7:
        raise ValueError('Require an independently refined grid, at least level7')
    oracle = [row for row in independent if row['grid'] == f'level-{highest}']
    for row in oracle:
        if (row['max_abs_gradient_change_from_previous_grid'] > 5e-5
                or row['abs_energy_change_from_previous_grid'] > 5e-5):
            raise ValueError('Independent highest-grid refinement has not met the unchanged gates')
    if sha(args.repo / 'basis/STO-3G.BAS') != reference['source_runtime_before'][str(Path(reference['arguments']['repo']) / 'basis/STO-3G.BAS')]:
        raise ValueError('Current native basis differs from independent exact QUICK basis')
    sys.path.insert(0, str(args.repo / 'validation/solvation'))
    from validate_solvation import runtime
    env = os.environ.copy()
    env.update(QUICK_BASIS=str(args.repo / 'basis'), OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    runtime_before = runtime(args.driver, env)
    if not args.allow_gpu and any(s in runtime_before['quick_library'] for s in ('_cuda', '_hip')):
        raise ValueError('GPU execution requires --allow-gpu and an authorized window')
    paths = [Path(__file__).resolve(), args.driver, args.reference,
             args.repo / 'src/modules/quick_gridpoints_module.f90',
             args.repo / 'src/modules/quick_method_module.f90', args.repo / 'basis/STO-3G.BAS']
    before = {str(path): sha(path) for path in paths}
    report = {'passed': False, 'complete': False, 'started_utc': datetime.now(timezone.utc).isoformat(),
              'scope': __doc__, 'arguments': {k: str(v) if isinstance(v, Path) else v
                                           for k, v in vars(args).items()},
              'source_runtime_before': before, 'runtime_before': runtime_before,
              'tolerances': {'energy_hartree': 5e-5, 'gradient_hartree_per_bohr': 5e-5},
              'reference_grid_level': highest,
              'reference_historical_failed_checks': reference['historical_absolute_report']['failed_checks'],
              'reference_current_failed_checks': [c for c in reference['checks'] if not c['passed']],
              'runs': [], 'comparisons': [], 'checks': [],
              'limitations': ['Water/STO-3G only; production basis/cluster convergence is separate.',
                              'Finite-difference truncation is examined at both specified steps.',
                              'No original failure is waived; all native grid candidates retain their checks.']}
    start = time.monotonic()

    def save():
        (args.workdir / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    def check(name, error, tolerance):
        report['checks'].append({'name': name, 'max_abs_error': error, 'tolerance': tolerance,
                                 'passed': math.isfinite(error) and error <= tolerance})

    def native(field_name, grid, atom=None, axis=None, step=0., sign=0):
        geometry_label = 'central' if atom is None else f'{atom + 1}{"xyz"[axis]}-h{step:g}-s{sign:+d}'
        folder = args.workdir / field_name / f'{grid[0]}x{grid[1]}' / geometry_label
        folder.mkdir(parents=True)
        req = json.loads((args.reference.parent / field_name / 'matched-SG1/central/request.json').read_text())
        coords = [row.copy() for row in req['coordinates_bohr']]
        if atom is not None:
            coords[atom][axis] += sign * step
        if req['symbols'] != ['O', 'H', 'H']:
            raise ValueError('Expected the preserved water fixture in O,H,H order')
        card = ('DFT PBE0 BASIS=STO-3G GRADIENT SCF=250 DENSERMS=1e-10 CUTOFF=1e-12 '
                'GRADCUTOFF=1e-12 XCCUTOFF=1e-12 BASISCUTOFF=1e-12 '
                f'XC_RADIAL={grid[0]} XC_ANGULAR={grid[1]} MATRIXZERO={args.matrixzero:.12g}')
        card += ''.join(f' EXTERNAL_EFIELD_{ax.upper()}={value:.12g}'
                        for ax, value in zip('xyz', req['field_au']) if value)
        if len(card) > 256:
            raise ValueError('Native keyword exceeds API capacity')
        text = card + '\n3\n' + ''.join(str(number) + ' ' + ' '.join(f'{v * BOHR:.17g}' for v in xyz) + '\n'
                                           for number, xyz in zip((8, 1, 1), coords))
        (folder / 'input.dat').write_text(text)
        tick = time.monotonic()
        with (folder / 'execution.log').open('w') as log:
            proc = subprocess.run([str(args.driver), 'input.dat'], cwd=folder, env=env,
                                  stdout=log, stderr=subprocess.STDOUT, timeout=300)
        if proc.returncode:
            raise RuntimeError('Native grid run failed: ' + str(folder))
        lines = (folder / 'result.dat').read_text().splitlines()
        scalar = {line.split()[0]: float(line.split()[1]) for line in lines
                  if line.split()[0] != 'gradient'}
        gradient = [float(v) for line in lines if line.startswith('gradient') for v in line.split()[2:]]
        if len(gradient) != 9 or not all(math.isfinite(v) for v in gradient + [scalar['energy']]):
            raise ValueError('Invalid native gradient or energy')
        row = {'field': field_name, 'grid': list(grid), 'geometry': geometry_label,
               'energy_hartree': scalar['energy'], 'gradient_hartree_per_bohr': gradient,
               'grid_points': int((folder / 'grid.dat').read_text().splitlines()[0]),
               'wall_seconds': time.monotonic() - tick,
               'raw_hashes': {p.name: sha(p) for p in folder.iterdir() if p.is_file()}}
        report['runs'].append(row)
        save()
        print(field_name, grid, geometry_label, row['energy_hartree'], flush=True)
        return row

    try:
        for field_name in ('zero', 'field'):
            previous = None
            for grid in grids:
                central = native(field_name, grid)
                for ref in [r for r in oracle if r['field'] == field_name]:
                    error = max(abs(a - b) for a, b in zip(central['gradient_hartree_per_bohr'], ref['pyscf_FD_gradient']))
                    energy_error = abs(central['energy_hartree'] - ref['central_energy_hartree'])
                    label = f'{field_name}:{grid[0]}x{grid[1]}:independent:h={ref["step_bohr"]:g}'
                    check(label + ':gradient', error, 5e-5)
                    check(label + ':energy', energy_error, 5e-5)
                    report['comparisons'].append({'name': label, 'max_abs_gradient_error': error,
                                                 'abs_energy_error': energy_error,
                                                 'component_errors': [a - b for a, b in zip(
                                                     central['gradient_hartree_per_bohr'], ref['pyscf_FD_gradient'])]})
                if previous is not None:
                    error = max(abs(a - b) for a, b in zip(central['gradient_hartree_per_bohr'], previous['gradient_hartree_per_bohr']))
                    check(f'{field_name}:{grid}:previous-grid:gradient', error, 5e-5)
                    check(f'{field_name}:{grid}:previous-grid:energy', abs(central['energy_hartree'] - previous['energy_hartree']), 5e-5)
                previous = central
            if not args.central_only:
                for step in args.steps:
                    fd = []
                    for atom in range(3):
                        for axis in range(3):
                            plus, minus = [native(field_name, grids[-1], atom, axis, step, sign) for sign in (1, -1)]
                            fd.append((plus['energy_hartree'] - minus['energy_hartree']) / (2 * step))
                    error = max(abs(a - b) for a, b in zip(previous['gradient_hartree_per_bohr'], fd))
                    check(f'{field_name}:selected-native-FD:h={step:g}', error, 5e-5)
                    report['comparisons'].append({'name': f'{field_name}:selected-native-FD:h={step:g}',
                                                 'gradient_FD': fd, 'max_abs_gradient_error': error})
        report['complete'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        report['source_runtime_after'] = {str(path): sha(path) for path in paths}
        report['runtime_after'] = runtime(args.driver, env)
        check('source_runtime_unchanged', float(before != report['source_runtime_after']), 0.)
        check('runtime_unchanged', float(runtime_before != report['runtime_after']), 0.)
        report['failed_checks'] = sum(not c['passed'] for c in report['checks'])
        report['passed'] = report['complete'] and not report['failed_checks']
        report['wall_seconds'] = time.monotonic() - start
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
    print(json.dumps({k: report[k] for k in ('passed', 'complete', 'failed_checks', 'wall_seconds')}, indent=2))
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
