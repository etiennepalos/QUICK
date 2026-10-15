#!/usr/bin/env python3
"""Standalone production PoU validation; no QUICK SCFs, chemistry, or GPU jobs.

The independent vectorized FixedCover supplies V/E/G. A small second-order
automatic differentiation oracle enumerates every anchor and supplies weights
and the tighter query-radius budgets. Polynomial support majorants are analytic
coefficient bounds over whole support boxes, never sampled maxima.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from reference_adaptive import FixedCover, tensor


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Jet:
    """Independent two-derivative scalar algebra, used only by the oracle."""
    def __init__(self, value, d=None, dd=None):
        self.v = float(value)
        self.d = np.zeros(3) if d is None else np.asarray(d, float)
        self.dd = np.zeros((3, 3)) if dd is None else np.asarray(dd, float)

    def __add__(self, other):
        other = other if isinstance(other, Jet) else Jet(other)
        return Jet(self.v+other.v, self.d+other.d, self.dd+other.dd)

    __radd__ = __add__

    def __neg__(self):
        return Jet(-self.v, -self.d, -self.dd)

    def __sub__(self, other):
        return self+-other if isinstance(other, Jet) else self+(-other)

    def __mul__(self, other):
        other = other if isinstance(other, Jet) else Jet(other)
        return Jet(self.v*other.v, self.d*other.v+self.v*other.d,
                   self.dd*other.v+self.v*other.dd+np.outer(self.d, other.d)+np.outer(other.d, self.d))

    __rmul__ = __mul__

    def inverse(self):
        return Jet(1/self.v, -self.d/self.v**2,
                   2*np.outer(self.d, self.d)/self.v**3-self.dd/self.v**2)

    def __truediv__(self, other):
        return self*(other.inverse() if isinstance(other, Jet) else 1/other)


def weights(cover, point):
    """All-anchor enumeration, independent of native eight-candidate lookup."""
    active = []
    for k, center in enumerate(cover.centers):
        delta = point-center
        if np.any(abs(delta/cover.h) >= 1):
            continue
        q = Jet(1)
        for axis in range(3):
            derivative = np.eye(3)[axis]/cover.h[axis]
            t = Jet(delta[axis]/cover.h[axis], derivative)
            z = Jet(1)-t*t
            q = q*z*z*z*z
        active.append((k, q))
    denominator = sum((q for _, q in active), Jet(0))
    return [(k, q/denominator) for k, q in active], denominator.v


def budget_oracle(cover, points, degree, errors=None):
    result, overlap, partition = [], 0, []
    for point in points:
        active, denominator = weights(cover, point)
        overlap = max(overlap, len(active))
        partition.append([abs(sum(w.v for _, w in active)-1),
                          np.linalg.norm(sum((w.d for _, w in active), np.zeros(3))),
                          np.linalg.norm(sum((w.dd for _, w in active), np.zeros((3, 3))))])
        b = np.zeros(3)
        for k, w in active:
            r = np.linalg.norm(point-cover.centers[k])
            m = cover.majorants[k, degree-2]
            if degree == 2:
                local = m*np.array([r**3/6, r*r/2, r])
            else:
                local = m*np.array([r**4/24, r**3/6, r*r/2])
            if errors is not None:
                dv, de, dg, dh = errors[k]
                local += [dv+de*r+dg*r*r/2, de+dg*r, dg]
                if degree == 3:
                    local += dh*np.array([r**3/6, r*r/2, r])
            b += w.v*local+[0, np.linalg.norm(w.d)*local[0],
                           2*np.linalg.norm(w.d)*local[1]+np.linalg.norm(w.dd)*local[0]]
        result.append(b)
    return np.array(result), overlap, np.max(partition, axis=0)


def make_cover(nside, seed=17):
    cover = FixedCover.__new__(FixedCover)
    cover.lower = np.array([-.55, -.2, .1])
    cover.upper = np.array([.85, .7, 2.3])
    cover.nside = nside
    cover.h = (cover.upper-cover.lower)/nside
    cover.centers = np.array([cover.lower+(np.array(i)+.5)*cover.h
                             for i in itertools.product(range(nside), repeat=3)])
    n = len(cover.centers)
    rng = np.random.default_rng(seed+nside)
    cover.v = rng.normal(0, .2, n)
    cover.e = rng.normal(0, .4, (n, 3))
    cover.g = tensor(rng.normal(0, .1, (n, 6)), 2)
    cover.H = tensor(rng.normal(0, .1, (n, 10)), 3)
    cover.majorants = 2+abs(rng.normal(size=(n, 2)))
    return cover


def probe_points(cover):
    lo, hi = cover.lower, cover.upper
    mid = (lo+hi)/2
    points = [mid]
    points += [np.array(p) for p in itertools.product(*zip(lo, hi))]
    for axis in range(3):
        for side in (lo[axis], hi[axis]):
            p = mid.copy(); p[axis] = side; points.append(p)
        others = [i for i in range(3) if i != axis]
        for a, b in itertools.product((0, 1), repeat=2):
            p = mid.copy(); p[others] = [([lo, hi][a])[others[0]], ([lo, hi][b])[others[1]]]
            points.append(p)
        # Leaf faces and compact-support boundaries, both one-sided limits.
        for x in np.arange(.5, cover.nside, .5):
            value = lo[axis]+x*cover.h[axis]
            for eps in (-1e-12, 0., 1e-12, -1e-6, 1e-6):
                p = mid.copy(); p[axis] = value+eps*cover.h[axis]
                if np.all((p >= lo) & (p <= hi)):
                    points.append(p)
    rng = np.random.default_rng(933+cover.nside)
    points += list(rng.uniform(lo, hi, (129, 3)))
    return np.array(points)


POLYNOMIAL = {(4, 0, 0): .04, (0, 4, 0): .03, (0, 0, 4): .02,
              (2, 2, 0): .015, (1, 1, 2): -.012, (3, 0, 0): .08,
              (1, 1, 1): .09, (1, 0, 0): -.3, (0, 1, 0): .2,
              (0, 0, 0): .1, (2, 0, 0): .12, (0, 1, 1): -.07}


def polynomial(points, derivative=(0, 0, 0), absolute_box=None):
    points = np.atleast_2d(points)
    out = np.zeros(len(points))
    for powers, coefficient in POLYNOMIAL.items():
        if any(p < d for p, d in zip(powers, derivative)):
            continue
        factor = coefficient
        for p, d in zip(powers, derivative):
            factor *= math.factorial(p)/math.factorial(p-d)
        power = np.array(powers)-derivative
        if absolute_box is None:
            out += factor*np.prod(points**power, axis=1)
        else:
            out += abs(factor)*np.prod(np.asarray(absolute_box)**power)
    return out


def poly_tensor(points, rank, absolute_box=None):
    out = np.empty((len(np.atleast_2d(points)),)+(3,)*rank)
    for indices in itertools.product(range(3), repeat=rank):
        orders = tuple(indices.count(k) for k in range(3))
        out[(slice(None),)+indices] = polynomial(points, orders, absolute_box)
    return out


def polynomial_cover(nside):
    cover = make_cover(nside)
    cover.v = polynomial(cover.centers)
    cover.e, cover.g, cover.H = [-poly_tensor(cover.centers, r) for r in (1, 2, 3)]
    cover.majorants = np.array([[np.linalg.norm(poly_tensor([center], r,
        np.maximum(abs(center-cover.h), abs(center+cover.h)))) for r in (3, 4)]
        for center in cover.centers])
    return cover


class Campaign:
    def __init__(self, workdir, executable):
        self.root, self.exe = workdir, executable
        self.checks, self.runs = [], []

    def check(self, name, passed, **details):
        self.checks.append(dict(name=name, pass_=bool(passed), **details))
        if not passed:
            raise AssertionError(f'{name}: {details}')

    def close(self, name, value, expected, atol=3e-11, rtol=3e-12):
        error = float(np.max(abs(np.asarray(value)-expected)))
        scale = float(np.max(abs(expected)))
        self.check(name, error <= atol+rtol*scale, maximum_absolute_error=error,
                   absolute_gate=atol, relative_gate=rtol, expected_scale=scale)

    def run(self, name, cover, points, degree=3, errors=None, mode=0, nside=None):
        directory = self.root/'cases'/name
        directory.mkdir(parents=True)
        n, np_ = len(cover.v), len(points)
        blocks = [cover.lower, cover.upper, cover.v, cover.e.T,
                  cover.g.transpose(1, 2, 0), cover.H.transpose(1, 2, 3, 0), cover.majorants.T]
        if errors is not None:
            blocks.append(errors.T)
        blocks.append(np.asarray(points).T)
        text = f'{cover.nside if nside is None else nside} {degree} {n} {np_} {int(errors is not None)} {mode}\n'
        text += ''.join(' '.join(f'{v:.17e}' for v in np.asarray(b).ravel(order='F'))+'\n'
                        for b in blocks if np.asarray(b).size)
        inp, output = directory/'input.dat', directory/'output.dat'
        inp.write_text(text)
        argv = [str(self.exe), str(inp), str(output)]
        start = time.perf_counter()
        result = subprocess.run(argv, text=True, capture_output=True, timeout=30)
        (directory/'stdout.log').write_text(result.stdout)
        (directory/'stderr.log').write_text(result.stderr)
        record = dict(name=name, argv=argv, returncode=result.returncode,
                      elapsed_seconds=time.perf_counter()-start,
                      input_sha256=sha(inp), output_sha256=sha(output) if output.exists() else None)
        self.runs.append(record)
        self.check(name+'/process', result.returncode == 0, returncode=result.returncode)
        lines = output.read_text().splitlines()
        ierr, overlap = map(int, lines[0].split())
        count, preserved = lines[1].split()
        data = np.array([[float(s) for s in line.split()] for line in lines[2:]]) if np_ else np.empty((0, 17))
        self.check(name+'/records', int(count) == np_ and data.shape == (np_, 17))
        return dict(ierr=ierr, overlap=overlap, preserved=preserved == 'T',
                    V=data[:, 1], E=data[:, 2:5], G=data[:, 5:14].reshape(-1, 3, 3).transpose(0, 2, 1),
                    budgets=data[:, 14:17])

    def positive(self, name, cover, points, degree, errors=None):
        out = self.run(name, cover, points, degree, errors)
        self.check(name+'/accepted', out['ierr'] == 0, ierr=out['ierr'])
        ref = cover.evaluate(points, degree, errors)
        for key in ('V', 'E', 'G'):
            self.close(name+'/reference-'+key, out[key], ref[key])
        budget, overlap, partition = budget_oracle(cover, points, degree, errors)
        self.close(name+'/independent-query-radius-budget', out['budgets'], budget, atol=2e-10, rtol=5e-12)
        self.check(name+'/overlap', out['overlap'] == overlap <= 8,
                   native=out['overlap'], exhaustive=overlap)
        coarse = ref['truncation_majorant'] if errors is None else ref['conditional_truncation_plus_supplied_anchor_budget']
        self.check(name+'/coarse-support-enclosure', bool(np.all(out['budgets'] <= coarse+2e-10)))
        self.check(name+'/budget-nonnegative', bool(np.all(out['budgets'] >= 0)))
        self.close(name+'/tensor-symmetry', out['G'], out['G'].transpose(0, 2, 1), atol=2e-12, rtol=3e-13)
        self.check(name+'/AD-partition-identities', max(partition) < 2e-11, maximum_errors=partition.tolist())
        return out


def run_campaign(c):
    for nside in (1, 2, 4):
        cover = make_cover(nside)
        points = probe_points(cover)
        for degree in (2, 3):
            c.positive(f'random-n{nside}-d{degree}', cover, points, degree)
        for count in (1, 7, 127, 128, 129, 257):
            rng = np.random.default_rng(count)
            c.positive(f'tail-n{nside}-{count}', cover, rng.uniform(cover.lower, cover.upper, (count, 3)), 3)

    cover = make_cover(4)
    rng = np.random.default_rng(331)
    points = rng.uniform(cover.lower+.15*(cover.upper-cover.lower),
                         cover.upper-.15*(cover.upper-cover.lower), (19, 3))
    errors = rng.uniform(1e-5, 4e-4, (len(cover.v), 4))
    for degree in (2, 3):
        c.positive(f'anchor-budget-d{degree}', cover, points, degree, errors)
        center = c.run(f'fd-center-d{degree}', cover, points, degree)
        fd_errors = []
        for h in (2e-5, 1e-5):
            efd, gfd = np.empty_like(center['E']), np.empty_like(center['G'])
            for axis in range(3):
                delta = np.eye(3)[axis]*h
                plus = c.run(f'fd-d{degree}-{h}-{axis}-plus', cover, points+delta, degree)
                minus = c.run(f'fd-d{degree}-{h}-{axis}-minus', cover, points-delta, degree)
                c.check(f'fd-d{degree}-{h}-{axis}/accepted', plus['ierr'] == minus['ierr'] == 0)
                efd[:, axis] = -(plus['V']-minus['V'])/(2*h)
                gfd[:, :, axis] = (plus['E']-minus['E'])/(2*h)
            c.close(f'fd-d{degree}-{h}/E=-gradV', efd, center['E'], atol=2e-5, rtol=0)
            c.close(f'fd-d{degree}-{h}/G=gradE', gfd, center['G'], atol=3e-4, rtol=0)
            fd_errors.append([float(np.max(abs(efd-center['E']))), float(np.max(abs(gfd-center['G'])))])
        c.check(f'fd-d{degree}/two-step-stability', np.all(np.array(fd_errors[1]) <= np.array(fd_errors[0])*1.1+1e-8), errors=fd_errors)

    for nside in (1, 2, 4):
        cover = polynomial_cover(nside)
        points = probe_points(cover)
        truth = [polynomial(points), -poly_tensor(points, 1), -poly_tensor(points, 2)]
        for degree in (2, 3):
            for perturbed in (False, True):
                fixture = copy.deepcopy(cover)
                errors = None
                if perturbed:
                    errors = np.tile([1e-4, 2e-4, 3e-4, 4e-4], (len(cover.v), 1))
                    signs = (-1.)**np.arange(len(cover.v))
                    fixture.v += signs*errors[:, 0]
                    fixture.e[:, 0] += signs*errors[:, 1]
                    fixture.g[:, 0, 0] += signs*errors[:, 2]
                    fixture.H[:, 0, 0, 0] += signs*errors[:, 3]
                name = f'polynomial-n{nside}-d{degree}-perturbed{int(perturbed)}'
                out = c.positive(name, fixture, points, degree, errors)
                for axis, key in enumerate(('V', 'E', 'G')):
                    diff = out[key]-truth[axis]
                    norm = abs(diff) if axis == 0 else np.linalg.norm(diff.reshape(len(points), -1), axis=1)
                    c.check(name+'/whole-support-majorant-'+key,
                            bool(np.all(norm <= out['budgets'][:, axis]+2e-10)),
                            maximum_actual_error=float(np.max(norm)),
                            maximum_budget=float(np.max(out['budgets'][:, axis])),
                            maximum_excess=float(np.max(norm-out['budgets'][:, axis])))

    cover = make_cover(4)
    points = probe_points(cover)
    base = c.run('gauge-baseline', cover, points)
    shifted = copy.deepcopy(cover); shifted.v += 2.3
    out = c.run('gauge-shift-2.3', shifted, points)
    c.check('gauge/accepted', base['ierr'] == out['ierr'] == 0)
    c.close('gauge/V-constant', out['V']-base['V'], np.full(len(points), 2.3), atol=2e-12)
    for key in ('E', 'G', 'budgets'):
        c.close('gauge/invariant-'+key, out[key], base[key], atol=2e-11)
    constant = copy.deepcopy(cover)
    constant.v[:] = 2.**40
    constant.e[:] = 0; constant.g[:] = 0; constant.H[:] = 0
    out = c.run('gauge-exact-large-constant', constant, points)
    c.check('gauge-large/accepted', out['ierr'] == 0)
    c.close('gauge-large/V-exact', out['V'], constant.v[0]*np.ones(len(points)), atol=0, rtol=0)
    c.close('gauge-large/E-exact-zero', out['E'], np.zeros_like(out['E']), atol=0, rtol=0)
    c.close('gauge-large/G-exact-zero', out['G'], np.zeros_like(out['G']), atol=0, rtol=0)

    cover = make_cover(2)
    points = np.array([(cover.lower+cover.upper)/2])
    negative = []
    for mode in (1, 2, 4, 5, 6):
        negative.append((f'shape-{mode}', copy.deepcopy(cover), points.copy(), dict(mode=mode), 46))
    negative.append(('anchor-error-shape', copy.deepcopy(cover), points.copy(), dict(mode=3, errors=np.ones((8, 4))), 46))
    for degree in (1, 4):
        negative.append((f'degree-{degree}', copy.deepcopy(cover), points.copy(), dict(degree=degree), 46))
    for nside in (0, 1025):
        negative.append((f'nside-{nside}', copy.deepcopy(cover), points.copy(), dict(nside=nside), 46))
    negative.append(('empty-query', copy.deepcopy(cover), np.empty((0, 3)), {}, 46))
    negative.append(('anchor-count', copy.deepcopy(cover), points.copy(), dict(nside=1), 46))
    for field in ('v', 'e', 'g', 'H', 'majorants', 'lower', 'upper'):
        fixture = copy.deepcopy(cover); getattr(fixture, field).flat[0] = np.nan
        negative.append((f'nan-{field}', fixture, points.copy(), {}, 55))
    nanpoint = points.copy(); nanpoint[0, 1] = np.nan
    negative.append(('nan-point', copy.deepcopy(cover), nanpoint, {}, 55))
    outside = points.copy(); outside[0, 0] = np.nextafter(cover.lower[0], -np.inf)
    negative.append(('outside-nextafter', copy.deepcopy(cover), outside, {}, 55))
    flat = copy.deepcopy(cover); flat.upper[0] = flat.lower[0]
    negative.append(('flat-box', flat, points.copy(), {}, 55))
    negmajor = copy.deepcopy(cover); negmajor.majorants[0, 0] = -1
    negative.append(('negative-majorant', negmajor, points.copy(), {}, 55))
    for kind in ('negative', 'nan'):
        errors = np.ones((8, 4)); errors[0, 0] = -1 if kind == 'negative' else np.nan
        negative.append((f'anchor-error-{kind}', copy.deepcopy(cover), points.copy(), dict(errors=errors), 46))
    asym = copy.deepcopy(cover); asym.g[0, 0, 1] += .4
    negative.append(('asymmetric-G', asym, points.copy(), {}, 55))
    asym = copy.deepcopy(cover); asym.H[0, 0, 1, 2] += .6
    negative.append(('asymmetric-H', asym, points.copy(), {}, 55))
    overflow = make_cover(1); overflow.lower[:] = -10; overflow.upper[:] = 10
    overflow.majorants[:] = 1e308
    negative.append(('late-overflow', overflow, np.array([[0, 0, 0], [10, 10, 10]]), {}, 55))
    for name, fixture, probes, kwargs, expected in negative:
        out = c.run('invalid-'+name, fixture, probes, **kwargs)
        c.check('invalid-'+name+'/code', out['ierr'] == expected, actual=out['ierr'], expected=expected)
        c.check('invalid-'+name+'/transactional', out['preserved'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', default='gfortran')
    parser.add_argument('--repo', type=Path, default=HERE.parents[1])
    parser.add_argument('--workdir', type=Path, required=True)
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=False)
    source, build = args.workdir/'source', args.workdir/'build'
    source.mkdir(); build.mkdir()
    files = [args.repo/'src/modules/quick_partition_of_unity_module.f90',
             HERE/'quick_native_pou_driver.f90', args.repo/'src/util/util.fh',
             Path(__file__), HERE/'reference_adaptive.py']
    hashes = {}
    for path in files:
        shutil.copy2(path, source/path.name)
        hashes[str(path)] = sha(path)
    exe = args.workdir/'native-pou'
    argv = [args.compiler, '-cpp', '-O0', '-ffree-line-length-none', '-fcheck=all', '-fbacktrace',
            '-I'+str(source), '-J'+str(build), '-I'+str(build),
            str(source/'quick_partition_of_unity_module.f90'), str(source/'quick_native_pou_driver.f90'), '-o', str(exe)]
    result = subprocess.run(argv, cwd=build, text=True, capture_output=True, timeout=120)
    (args.workdir/'compile.json').write_text(json.dumps(dict(argv=argv, returncode=result.returncode,
        stdout=result.stdout, stderr=result.stderr, source_hashes=hashes), indent=2)+'\n')
    if result.returncode:
        raise RuntimeError(result.stderr)
    campaign = Campaign(args.workdir, exe)
    failure = None
    try:
        run_campaign(campaign)
    except Exception as error:
        failure = f'{type(error).__name__}: {error}'
    report = dict(status='PASS' if failure is None else 'FAIL', failure=failure,
                  scope='Standalone production PoU module, bounds checked; zero SCFs or GPU jobs',
                  checks=campaign.checks, number_checks=len(campaign.checks),
                  native_runs=campaign.runs, number_native_runs=len(campaign.runs),
                  source_hashes=hashes, executable_sha256=sha(exe),
                  current_source_matches_snapshot={str(p): sha(p) == hashes[str(p)] for p in files},
                  limitations=['Synthetic anchors and analytic polynomial sources; no molecular accuracy claim',
                    'Real-arithmetic Taylor/support budgets evaluated in FP64; total numerical budget unavailable',
                    'Standalone module validation does not establish full QUICK API, MPI, or CUDA integration',
                    'Finite differences are diagnostics, not continuous derivative certificates'],
                  gates=dict(reference_absolute=3e-11, reference_relative=3e-12,
                             budget_absolute=2e-10, budget_relative=5e-12,
                             finite_difference_E_absolute=2e-5, finite_difference_G_absolute=3e-4))
    (args.workdir/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('status', 'failure', 'number_checks', 'number_native_runs')}))
    return 0 if failure is None else 1


if __name__ == '__main__':
    raise SystemExit(main())
