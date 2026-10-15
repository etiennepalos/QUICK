#!/usr/bin/env python3
"""Bounded PBE0/Cartesian-def2-SVP grid transfer to WATER27 water and dimer.

Six native central calculations use three explicit grids and one mixed
uniform field. Independent PySCF level7/9 full analytic forces include grid
response and explicit field-moment derivatives. Two-step energy differences
on one nonzero component per molecule independently check that extension.
This isolates XC convergence; neither D3 nor solvent is included.
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

import numpy as np
import pyscf
from pyscf import dft, gto, lib
from pyscf.gto.basis import parse_gaussian

BOHR = .52917720859
FIELD = np.array([.0002, -.00015, .0001])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--driver', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--workdir', type=Path, required=True)
    args = parser.parse_args()
    for name in ('repo', 'driver', 'dataset'):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.workdir = args.workdir.resolve()
    args.workdir.mkdir(parents=True, exist_ok=False)
    lib.num_threads(1)
    sys.path.insert(0, str(args.repo / 'validation/solvation'))
    from validate_solvation import runtime
    env = os.environ.copy()
    env.update(QUICK_BASIS=str(args.repo / 'basis'), OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    before_runtime = runtime(args.driver, env)
    if any(s in before_runtime['quick_library'] for s in ('_cuda', '_hip', '_mpi')):
        raise ValueError('This bounded transfer requires the serial CPU exporter')
    basisfile = args.repo / 'basis/DEF2-SVP.BAS'
    text = basisfile.read_text()
    basis = {}
    import re
    for sym in ('H', 'O'):
        blocks = [b for b in text.split('****') if re.search(r'^\s*' + sym + r'\s+0\s*$', b, re.M | re.I)]
        if len(blocks) != 1:
            raise ValueError('Incomplete exact QUICK def2-SVP basis')
        basis[sym] = parse_gaussian.parse(blocks[0], optimize=False)
    sources = [Path(__file__).resolve(), args.driver, basisfile,
               args.repo / 'src/modules/quick_gridpoints_module.f90',
               args.repo / 'src/modules/quick_method_module.f90']
    sources += [args.dataset / name for name in ('H2O.xyz', 'H2O2.xyz')]
    package = Path(pyscf.__file__).resolve().parent
    sources += [package / rel for rel in ('grad/rhf.py', 'grad/rks.py', 'dft/rks.py',
                 'dft/numint.py', 'dft/gen_grid.py', 'gto/moleintor.py', 'gto/basis/parse_gaussian.py')]
    sources += sorted({p.resolve() for p in (package / 'lib').rglob('*.so*') if p.is_file()})
    before = {str(p): sha(p) for p in sources}
    report = {'passed': False, 'complete': False, 'started_utc': datetime.now(timezone.utc).isoformat(),
              'scope': __doc__, 'arguments': {k: str(v) for k, v in vars(args).items()},
              'source_runtime_before': before, 'runtime_before': before_runtime,
              'pyscf_version': pyscf.__version__, 'numpy_version': np.__version__,
              'settings': {'basis': 'exact QUICK DEF2-SVP.BAS', 'cartesian': True,
                           'field_au': FIELD.tolist(), 'origin_bohr': [0., 0., 0.],
                           'matrixzero': 1e-14, 'threads': 1, 'pyscf_small_rho_cutoff': 0.,
                           'pyscf_grid_response': True, 'pyscf_energy_tolerance': 1e-12,
                           'pyscf_orbital_gradient_tolerance': 1e-9,
                           'pyscf_direct_scf_tolerance': 1e-14},
              'checks': [], 'runs': [], 'comparisons': [],
              'limitations': ['Two fixed WATER27 geometries with one static mixed field; no solvent, D3, trajectory or bulk claim.',
                              'AO normalization conventions may differ; energies/forces are basis-representation invariant. No raw AO density equivalence is asserted.',
                              'Reference field-gradient extension has selected-component energy-FD checks at two steps.']}
    start = time.monotonic()

    def save():
        (args.workdir / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    def check(name, error, tolerance):
        report['checks'].append({'name': name, 'max_abs_error': float(error), 'tolerance': tolerance,
                                 'passed': math.isfinite(error) and error <= tolerance})

    def oracle(name, folder, symbols, coordinates, level, need_gradient):
        folder.mkdir(parents=True)
        mol = gto.M(atom=list(zip(symbols, coordinates)), basis=basis, unit='Bohr', cart=True, verbose=0)
        mf = dft.RKS(mol)
        mf.xc = 'PBE0'
        mf.grids.level = level
        mf.small_rho_cutoff = 0.
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-9
        mf.direct_scf_tol = 1e-14
        mf.max_cycle = 250
        mf.verbose = 4
        mf.stdout = (folder / 'pyscf.log').open('w')
        moments = mol.intor('int1e_r', comp=3, hermi=1)
        core = mf.get_hcore() + np.einsum('i,iab->ab', FIELD, moments)
        nuclear = mol.energy_nuc() - np.einsum('a,ai,i->', mol.atom_charges(), coordinates, FIELD)
        mf.get_hcore = lambda *unused: core
        mf.energy_nuc = lambda *unused: nuclear
        energy = float(mf.kernel())
        if not mf.converged or not math.isfinite(energy):
            raise RuntimeError('Independent SCF did not converge: ' + str(folder))
        result = {'label': name, 'backend': 'PySCF', 'energy_hartree': energy,
                  'coordinates_bohr': coordinates.tolist(), 'basis_functions': mol.nao_nr(),
                  'grid_level': level, 'grid_points': len(mf.grids.weights)}
        if need_gradient:
            grad = mf.nuc_grad_method()
            grad.grid_response = True
            base_generator = grad.hcore_generator(mol)
            base_nuclear = grad.grad_nuc(mol)
            # libcint int1e_irp is <bra|r_i * d_j ket>. A center derivative
            # is minus the AO spatial derivative. Add both AO-center terms;
            # SCF orbital energies already include F.r, hence its Pulay term.
            ket = np.einsum('i,ijab->jab', FIELD,
                            mol.intor('int1e_irp', comp=9).reshape(3, 3, mol.nao_nr(), mol.nao_nr()))
            slices = mol.aoslice_by_atom()

            def field_generator(unused=None):
                def derivative(atom):
                    p0, p1 = slices[atom, 2:]
                    h1 = base_generator(atom).copy()
                    h1[:, :, p0:p1] -= ket[:, :, p0:p1]
                    h1[:, p0:p1, :] -= ket.transpose(0, 2, 1)[:, p0:p1, :]
                    return h1
                return derivative

            grad.hcore_generator = field_generator
            total_nuclear = base_nuclear - mol.atom_charges()[:, None] * FIELD
            grad.grad_nuc = lambda mol=None, atmlst=None: total_nuclear if atmlst is None else total_nuclear[atmlst]
            gradient = np.asarray(grad.kernel())
            if gradient.shape != coordinates.shape or not np.all(np.isfinite(gradient)):
                raise ValueError('Invalid independent full gradient')
            result['gradient_hartree_per_bohr'] = gradient.tolist()
        mf.stdout.close()
        (folder / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        result['raw_hashes'] = {p.name: sha(p) for p in folder.iterdir() if p.is_file()}
        report['runs'].append(result)
        save()
        print(name, energy, flush=True)
        return result

    try:
        for case, filename, fd_axis in [('water', 'H2O.xyz', 2), ('dimer', 'H2O2.xyz', 0)]:
            rows = (args.dataset / filename).read_text().splitlines()
            atoms = [line.split() for line in rows[2:] if line.strip()]
            symbols = [row[0] for row in atoms]
            coordinates = np.array([[float(v) / BOHR for v in row[1:]] for row in atoms])
            expected = 25 if case == 'water' else 50
            refs = {}
            for level in (7, 9):
                refs[level] = oracle(f'{case}:level{level}', args.workdir / case / f'level{level}',
                                     symbols, coordinates, level, True)
                check(f'{case}:level{level}:cartesian_basis_count', abs(refs[level]['basis_functions'] - expected), 0.)
                check(f'{case}:level{level}:neutral_translation_gradient',
                      float(np.max(np.abs(np.sum(refs[level]['gradient_hartree_per_bohr'], axis=0)))), 1e-6)
            check(case + ':level7_to9:energy', abs(refs[7]['energy_hartree'] - refs[9]['energy_hartree']), 5e-5)
            check(case + ':level7_to9:full_gradient', float(np.max(np.abs(
                np.array(refs[7]['gradient_hartree_per_bohr']) - refs[9]['gradient_hartree_per_bohr']))), 5e-5)
            for step in (.001, .0005):
                energies = []
                for sign in (1, -1):
                    displaced = coordinates.copy()
                    displaced[0, fd_axis] += sign * step
                    label = f'{case}:FD:h{step:g}:s{sign:+d}'
                    value = oracle(label, args.workdir / case / f'fd-h{step:g}-s{sign:+d}',
                                   symbols, displaced, 9, False)
                    energies.append(value['energy_hartree'])
                fd = (energies[0] - energies[1]) / (2 * step)
                error = abs(fd - refs[9]['gradient_hartree_per_bohr'][0][fd_axis])
                check(f'{case}:independent_field_gradient_FD:h={step:g}', error, 1e-6)
                report['comparisons'].append({'name': f'{case}:independent_FD:h={step:g}',
                                             'FD_gradient': fd, 'max_abs_error': error})
            previous = None
            for nrad, nang in ((100, 434), (150, 590), (200, 974)):
                folder = args.workdir / case / f'native-{nrad}x{nang}'
                folder.mkdir()
                card = ('DFT PBE0 BASIS=DEF2-SVP GRADIENT SCF=250 DENSERMS=1e-10 CUTOFF=1e-12 '
                        'GRADCUTOFF=1e-12 XCCUTOFF=1e-12 BASISCUTOFF=1e-12 MATRIXZERO=1e-14 '
                        f'XC_RADIAL={nrad} XC_ANGULAR={nang}'
                        ' EXTERNAL_EFIELD_X=0.0002 EXTERNAL_EFIELD_Y=-0.00015 EXTERNAL_EFIELD_Z=0.0001')
                if len(card) > 256:
                    raise ValueError('Native API keyword exceeds capacity')
                (folder / 'input.dat').write_text(card + f'\n{len(symbols)}\n' + ''.join(
                    str(gto.charge(sym)) + ' ' + ' '.join(f'{v * BOHR:.17g}' for v in point) + '\n'
                    for sym, point in zip(symbols, coordinates)))
                with (folder / 'execution.log').open('w') as log:
                    proc = subprocess.run([str(args.driver), 'input.dat'], cwd=folder, env=env,
                                          stdout=log, stderr=subprocess.STDOUT, timeout=600)
                if proc.returncode:
                    raise RuntimeError('Native basis-transfer SCF failed: ' + str(folder))
                raw = (folder / 'result.dat').read_text().splitlines()
                scalar = {line.split()[0]: float(line.split()[1]) for line in raw if not line.startswith('gradient')}
                gradient = np.array([[float(v) for v in line.split()[2:]] for line in raw if line.startswith('gradient')])
                nb = int((folder / 'density.dat').read_text().splitlines()[0])
                check(f'{case}:{nrad}x{nang}:cartesian_basis_count', abs(nb - expected), 0.)
                error = float(np.max(np.abs(gradient - refs[9]['gradient_hartree_per_bohr'])))
                energy_error = abs(scalar['energy'] - refs[9]['energy_hartree'])
                check(f'{case}:{nrad}x{nang}:independent_full_gradient', error, 5e-5)
                check(f'{case}:{nrad}x{nang}:independent_energy', energy_error, 5e-5)
                if previous is not None:
                    check(f'{case}:{nrad}x{nang}:refinement_full_gradient', float(np.max(np.abs(gradient - previous[1]))), 5e-5)
                    check(f'{case}:{nrad}x{nang}:refinement_energy', abs(scalar['energy'] - previous[0]), 5e-5)
                previous = scalar['energy'], gradient
                result = {'label': f'{case}:{nrad}x{nang}', 'backend': 'QUICK',
                          'energy_hartree': scalar['energy'], 'gradient_hartree_per_bohr': gradient.tolist(),
                          'max_abs_gradient_error': error, 'abs_energy_error': energy_error,
                          'basis_functions': nb, 'grid_points': int((folder / 'grid.dat').read_text().splitlines()[0]),
                          'raw_hashes': {p.name: sha(p) for p in folder.iterdir() if p.is_file()}}
                report['runs'].append(result)
                save()
                print(result['label'], energy_error, error, flush=True)
        report['complete'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        report['source_runtime_after'] = {str(p): sha(p) for p in sources}
        report['runtime_after'] = runtime(args.driver, env)
        check('source_runtime_unchanged', float(before != report['source_runtime_after']), 0.)
        check('runtime_unchanged', float(before_runtime != report['runtime_after']), 0.)
        report['failed_checks'] = sum(not c['passed'] for c in report['checks'])
        report['passed'] = report['complete'] and not report['failed_checks']
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        report['wall_seconds'] = time.monotonic() - start
        save()
    print(json.dumps({k: report[k] for k in ('passed', 'complete', 'failed_checks', 'wall_seconds')}, indent=2))
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
