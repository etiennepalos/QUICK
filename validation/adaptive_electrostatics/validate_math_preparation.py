#!/usr/bin/env python3
"""Independent bound/POU feasibility checks; no SCFs, QUICK calls or GPU work.

All inequalities called majorants here are real-arithmetic identities
evaluated in ordinary FP64. Unknown total numerical budgets remain None.
Timing is Python prototype timing and cannot establish native/GPU speedup.
"""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import scipy
from reference_adaptive import (components, boys, primitive_normalization,
    prepare_pairs, electronic_jet, cell_majorants, tensor, packed_norm,
    gaussian_fourth, FixedCover, slots, validate_error_mode)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--mpmath-site', type=Path,
                        help='Append an existing site containing pure-Python mpmath, without overriding this environment')
    args = parser.parse_args()
    args.repo = args.repo.resolve(strict=True)
    args.workdir = args.workdir.resolve()
    args.workdir.mkdir(parents=True, exist_ok=False)
    if args.mpmath_site:
        sys.path.append(str(args.mpmath_site.resolve(strict=True)))
    import mpmath as mp
    source = [Path(__file__).resolve(), Path(__file__).with_name('reference_adaptive.py'),
              args.repo / 'src/modules/quick_field_hessian_module.f90',
              args.repo / 'src/gpu/gpu_field_hessian.cu']
    before = {str(p): sha(p) for p in source}
    report = {'passed': False, 'complete': False, 'started_utc': datetime.now(timezone.utc).isoformat(),
              'scope': __doc__, 'python': sys.executable, 'numpy': np.__version__, 'scipy': scipy.__version__,
              'mpmath': mp.__version__, 'source_before': before, 'checks': [], 'cell_tests': [],
              'pou_trials': [], 'compact_F10_R_slots_per_layer': len(slots(10)[0]),
              'production_integration': False, 'certified': False,
              'total_numerical_error_budget_available': False,
              'limits': {'max_anchors': 512, 'max_active_patches': 8,
                         'query_counts': [1000, 10000], 'SCFs': 0, 'GPU_runs': 0},
              'limitations': ['Sampled component checks validate an implementation; they are not floating-point certification.',
                              'Boys values use independent SciPy hypergeometric evaluation, crosschecked against 80-digit quadrature.',
                              'Synthetic small Gaussian densities and fixed uniform covers; no production speed or adaptive-tree claim.',
                              'Only truncation majorants are evaluated. Unknown anchor/final arithmetic errors are never set to zero.']}
    start = time.monotonic()

    def save():
        (args.workdir / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    def check(name, error, tolerance):
        report['checks'].append({'name': name, 'error': float(error), 'tolerance': tolerance,
                                 'passed': bool(np.isfinite(error) and error <= tolerance)})

    def ao(angular, exponent, center=(0., 0., 0.)):
        return {'center': list(center), 'angular': angular,
                'primitives': [(exponent, primitive_normalization(exponent, angular))]}

    rng = np.random.default_rng(74019)
    try:
        mp.mp.dps = 80
        errors = []
        for t in (0., 1e-8, .3, 32., 100., 1000.):
            observed = boys(10, t)[:, 0]
            for m in range(11):
                exact = mp.quad(lambda x: x**(2*m)*mp.exp(-mp.mpf(t)*x*x), [0, 1])
                errors.append(abs(observed[m]-float(exact))/float(exact))
        check('F0_to_F10_independent_80digit_quadrature', max(errors), 2e-12)
        try:
            validate_error_mode('certified')
        except NotImplementedError:
            check('unavailable_certified_mode_rejected', 0., 0.)
        else:
            check('unavailable_certified_mode_rejected', 1., 0.)

        for alpha in (.001, .7, 1000.):
            points = np.array([[0., 0., 0.], [.2, -.1, .3], [.7, .3, -.2]])/np.sqrt(2*alpha)
            pairs = prepare_pairs([ao((0, 0, 0), alpha)], [[1.]])
            jet = electronic_jet(pairs, points, 4)
            observed = -tensor(jet[4], 4)
            exact = gaussian_fourth(points, alpha)
            check(f'sGaussian_a{alpha}:independent_closed_rank4',
                  np.max(abs(observed-exact))/max(np.max(abs(exact)), 1e-300), 5e-13)
            check(f'sGaussian_a{alpha}:center_H_zero', np.max(abs(jet[3][0])), 0.)

        aos20 = [ao(c, alpha) for l, alpha in enumerate((.7, .5, .4, .3)) for c in components(l)]
        c = rng.normal(size=(20, 5))*.15
        density20 = c @ c.T
        pairs20 = prepare_pairs(aos20, density20)
        hetero = [ao((0, 0, 0), .7), ao((3, 0, 0), .3, (.7, -.3, .2)),
                  ao((1, 2, 0), .4, (-.2, .4, .3)), ao((0, 1, 0), .5)]
        c4 = rng.normal(size=(4, 3))*.3
        pairs4 = prepare_pairs(hetero, c4 @ c4.T)
        sg = prepare_pairs([ao((0, 0, 0), .7)], [[1.]])
        boxes = [(np.array([-.2]*3), np.array([.25]*3)),
                 (np.array([4.75, -.25, -.25]), np.array([5.25, .25, .25]))]
        for name, pairs in [('sGaussian', sg), ('all_spdf_20AO', pairs20), ('heterocenter_4AO', pairs4)]:
            for boxid, (lower, upper) in enumerate(boxes):
                from itertools import product
                points = np.vstack((lower+(upper-lower)*rng.random((16, 3)),
                                    np.array(list(product(*zip(lower, upper)))), (lower+upper)/2))
                jet = electronic_jet(pairs, points, 4)
                b = cell_majorants(pairs, lower, upper)
                tighter = cell_majorants(pairs, lower, upper, coefficient_mode='abs_signed')
                item = {'model': name, 'box': boxid, 'primitive_pairs': len(pairs),
                        'lower': lower.tolist(), 'upper': upper.tolist(), 'ranks': {}}
                for rank in (3, 4):
                    actual = np.max(abs(jet[rank]), axis=0)
                    check(f'{name}:box{boxid}:rank{rank}:positive_component_majorant',
                          np.max(np.maximum(actual-b[rank], 0)), 1e-10*max(1, float(np.max(actual))))
                    check(f'{name}:box{boxid}:rank{rank}:absE_component_majorant',
                          np.max(np.maximum(actual-tighter[rank], 0)), 1e-10*max(1, float(np.max(actual))))
                    true_max = np.max(packed_norm(jet[rank], rank))
                    item['ranks'][str(rank)] = {'sampled_max_norm': float(true_max),
                        'positive_majorant_norm': float(packed_norm(b[rank], rank)),
                        'absolute_signedE_majorant_norm': float(packed_norm(tighter[rank], rank)),
                        'majorant_to_sampled_ratio': float(packed_norm(b[rank], rank)/true_max)}
                report['cell_tests'].append(item)
                save()

        # Independent libcint rank3 derivatives differentiated in C for rank4.
        # No SCF and no QUICK recurrence is involved in this reference.
        sys.path.insert(0, str(args.repo / 'validation/field_hessian'))
        from reference_field_hessian import attraction_third
        from pyscf import gto, lib
        lib.num_threads(1)
        mol = gto.M(atom='He 0 0 0', basis={'He': [[l, [a, 1.]] for l, a in enumerate((.7, .5, .4, .3))]},
                    cart=True, unit='Bohr', verbose=0)
        norm = np.sqrt(np.diag(mol.intor('int1e_ovlp')))
        point = np.array([.35, .27, -.18])
        direct = -tensor(electronic_jet(pairs20, [point], 4)[4], 4)[0]
        fderrors = []
        for step in (.001, .0005):
            observed = np.empty((3, 3, 3, 3))
            for axis in range(3):
                delta = np.eye(3)[axis]*step
                h = []
                for displaced in (point+delta, point-delta):
                    i3 = attraction_third(mol, displaced)/norm[:, None]/norm[None, :]
                    h.append(np.einsum('mn,ijkmn->ijk', density20, i3))
                observed[:, :, :, axis] = (h[0]-h[1])/(2*step)
            error = float(np.max(abs(observed-direct)))
            fderrors.append(error)
            check(f'all_spdf_rank4:libcint_H_FD:h={step}', error, 2e-4)
        check('all_spdf_rank4:second_order_FD_reduction', fderrors[1]/fderrors[0], .31)
        report['independent_spdf_rank4_FD_errors'] = fderrors

        for name, lower, upper in [('near', [-.4]*3, [.4]*3),
                                   ('far', [4.75, -.25, -.25], [5.25, .25, .25])]:
            points = np.array(lower)+(np.array(upper)-lower)*rng.random((10000, 3))
            tick = time.monotonic()
            true_jet = electronic_jet(sg, points, 2)
            direct_seconds = time.monotonic()-tick
            exact = (tensor(true_jet[0], 0), -tensor(true_jet[1], 1), -tensor(true_jet[2], 2))
            for nside in (2, 4, 8):
                tick = time.monotonic()
                cover = FixedCover(sg, lower, upper, nside)
                setup_seconds = time.monotonic()-tick
                for degree in (2, 3):
                    for npoints in (1000, 10000):
                        tick = time.monotonic()
                        value = cover.evaluate(points[:npoints], degree)
                        query_seconds = time.monotonic()-tick
                        errors = np.column_stack((abs(value['V']-exact[0][:npoints]),
                            np.linalg.norm(value['E']-exact[1][:npoints], axis=1),
                            np.linalg.norm(value['G']-exact[2][:npoints], axis=(1, 2))))
                        budget = value['truncation_majorant']
                        prefix = f'{name}:nside{nside}:degree{degree}:N{npoints}'
                        check(prefix+':real_arithmetic_truncation_majorant_sampled',
                              np.max(np.maximum(errors-budget, 0)), 1e-10)
                        check(prefix+':coverage_lower_bound', max((.75**12)-value['minimum_denominator'], 0), 1e-14)
                        check(prefix+':weight_sum', value['weight_sum_error'], 1e-13)
                        check(prefix+':weight_gradient_sum', value['weight_gradient_sum_error'], 1e-11)
                        check(prefix+':weight_hessian_sum', value['weight_hessian_sum_error'], 1e-8)
                        report['pou_trials'].append({'case': name, 'nside': nside, 'anchors': nside**3,
                            'degree': degree, 'npoints': npoints, 'setup_seconds': setup_seconds,
                            'query_seconds': query_seconds, 'direct_10000_seconds': direct_seconds,
                            'max_actual_V_E_G_errors': errors.max(axis=0).tolist(),
                            'max_truncation_majorants': budget.max(axis=0).tolist(),
                            'numeric_error_budget': None, 'total_error_bound': None,
                            'total_tolerance_acceptance_claimed': False, 'certified': False})
                        save()
                # Frozen-cover derivative and potential-gauge checks.
                probes = .9*points[:3]+.1*(np.array(lower)+upper)/2
                probes[0] = cover.centers[len(cover.centers)//2]
                base = cover.evaluate(probes, 3)
                for step in (1e-4, 5e-5):
                    efd, gfd = np.empty((3, 3)), np.empty((3, 3, 3))
                    for axis in range(3):
                        delta = np.eye(3)[axis]*step
                        plus, minus = [cover.evaluate(probes+sign*delta, 3) for sign in (1, -1)]
                        efd[:, axis] = -(plus['V']-minus['V'])/(2*step)
                        gfd[:, :, axis] = (plus['E']-minus['E'])/(2*step)
                    check(f'{name}:nside{nside}:frozen_cover_E_from_V:h{step}', np.max(abs(efd-base['E'])), 2e-5)
                    check(f'{name}:nside{nside}:frozen_cover_G_from_E:h{step}', np.max(abs(gfd-base['G'])), 2e-4)
                cover.v += 2.3
                shifted = cover.evaluate(probes, 3)
                check(f'{name}:nside{nside}:constant_potential_gauge_V', np.max(abs(shifted['V']-base['V']-2.3)), 1e-13)
                check(f'{name}:nside{nside}:constant_potential_gauge_E', np.max(abs(shifted['E']-base['E'])), 1e-12)
                check(f'{name}:nside{nside}:constant_potential_gauge_G', np.max(abs(shifted['G']-base['G'])), 1e-9)
                # Prescribed nonzero anchor perturbations have known norms.
                # This exercises their amplification by ALL POU derivatives.
                cover.v -= 2.3
                da = np.tile([1e-4, 2e-4, 3e-4, 4e-4], (len(cover.centers), 1))
                cover.v += da[:, 0]
                cover.e[:, 0] += da[:, 1]
                cover.g[:, 0, 0] += da[:, 2]
                cover.H[:, 0, 0, 0] += da[:, 3]
                value = cover.evaluate(points[:1000], 3, da)
                actual = np.column_stack((abs(value['V']-exact[0][:1000]),
                    np.linalg.norm(value['E']-exact[1][:1000], axis=1),
                    np.linalg.norm(value['G']-exact[2][:1000], axis=(1, 2))))
                check(f'{name}:nside{nside}:prescribed_nonzero_anchor_error_budget',
                      np.max(np.maximum(actual-value['conditional_truncation_plus_supplied_anchor_budget'], 0)), 1e-10)
                check(f'{name}:nside{nside}:unknown_total_numeric_budget_is_not_zero',
                      int(value['numeric_error_budget'] is not None or value['total_error_bound'] is not None), 0.)
        report['complete'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        report['source_after'] = {str(p): sha(p) for p in source}
        check('source_unchanged', int(before != report['source_after']), 0.)
        report['failed_checks'] = sum(not c['passed'] for c in report['checks'])
        report['passed'] = report['complete'] and not report['failed_checks']
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        report['wall_seconds'] = time.monotonic()-start
        save()
    print(json.dumps({k: report[k] for k in ('passed', 'complete', 'failed_checks', 'wall_seconds')}, indent=2))
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
