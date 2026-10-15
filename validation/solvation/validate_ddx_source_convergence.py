#!/usr/bin/env python3
"""Check ddX source bases against independent Cartesian/Legendre expressions."""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess


def expected_m0(l: int, point: tuple[float, ...], mode: int) -> tuple[float, list[float]]:
    r = math.hypot(*point)
    c = math.sqrt(4 * math.pi / (2*l + 1))
    if r == 0:
        return (math.sqrt(4 * math.pi) if l == 0 else 0.0,
                [0.0, 0.0, math.sqrt(4*math.pi/3)/2.5 if l == 1 else 0.0])
    t = point[2] / r
    p, dp, prev, dprev = 1.0, 0.0, 0.0, 0.0
    for n in range(1, l+1):
        p, prev, dp, dprev = (((2*n-1)*t*p-(n-1)*prev)/n, p,
                              ((2*n-1)*(p+t*dp)-(n-1)*dprev)/n, dp)
    power = -(l+1) if mode == 2 and r > 2.5 else l
    radial = (2.5/r)**(l+1) if power < 0 else (r/2.5)**l
    value = c * radial * p
    gradient = [c*radial/r*(power*p*x/r + dp*((1.0 if k == 2 else 0.0)-t*x/r))
                for k, x in enumerate(point)]
    return value, gradient


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--driver', type=Path, required=True)
    ap.add_argument('--workdir', type=Path, required=True)
    ap.add_argument('--launcher', default='')
    args = ap.parse_args()
    args.driver = args.driver.resolve(strict=True)
    args.workdir = args.workdir.resolve()
    args.workdir.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    command = shlex.split(args.launcher)+[str(args.driver)]
    with (args.workdir/'execution.log').open('w') as log:
        run = subprocess.run(command, cwd=args.workdir, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=90)
    report = dict(command=command, returncode=run.returncode, checks=[], exceptions=[],
                  driver_sha256=hashlib.sha256(args.driver.read_bytes()).hexdigest())
    def check(name: str, error: float, tolerance: float) -> None:
        report['checks'].append(dict(name=name, error=error, tolerance=tolerance,
                                    passed=math.isfinite(error) and error <= tolerance))
    if run.returncode == 0:
        rows = list(csv.DictReader((args.workdir/'source_basis.csv').open()))
        lookup = {(int(r['mode']),int(r['point']),int(r['l']),int(r['m'])):r for r in rows}
        check('complete_rows', abs(len(rows)-19*361), 0.0)
        scalar_error = gradient_error = inside_error = 0.0
        for r in rows:
            mode,p,l,m = (int(r[k]) for k in ('mode','point','l','m'))
            if m == 0:
                v,g = expected_m0(l,tuple(float(r[k]) for k in ('x','y','z')),mode)
                scalar_error = max(scalar_error,abs(float(r['value'])-v)/max(1,abs(v)))
                gradient_error = max(gradient_error,max(abs(float(r[k])-x) for k,x in zip(('gx','gy','gz'),g))/max(1,*map(abs,g)))
            if mode == 2 and p in (1,2,3,4,8):
                old = lookup[1,p,l,m]
                inside_error = max(inside_error,max(abs(float(r[k])-float(old[k])) for k in ('value','gx','gy','gz')))
        check('independent_legendre_m0_value_all_orders',scalar_error,2e-13)
        check('independent_legendre_m0_gradient_all_orders',gradient_error,5e-12)
        check('inside_and_exact_boundary_regular_preserved',inside_error,0.0)
        for step in (1,2):
            error = 0.0
            for mode in (1,2):
                for p in range(1,8):
                    if p == 4: continue
                    for l in range(19):
                        shell = [lookup[mode,p,l,m] for m in range(-l,l+1)]
                        scale = max(1.0,*(abs(float(r[k])) for r in shell for k in ('gx','gy','gz')))
                        error = max(error,max(abs(float(r[f'fd{step}{axis}'])-float(r['g'+axis])) for r in shell for axis in 'xyz')/scale)
            check(f'all_component_gradient_fd_step{step}',error,2e-7)
        jump = 0.0
        for l in range(19):
            scale = max(1.0,*(abs(float(lookup[2,p,l,m]['value'])) for p in (8,9) for m in range(-l,l+1)))
            jump = max(jump,max(abs(float(lookup[2,8,l,m]['value'])-float(lookup[2,9,l,m]['value'])) for m in range(-l,l+1))/scale)
        check('boundary_value_continuity_delta1e-8',jump,5e-7)
        check('native_success_and_reset',float('default_reset_and_invalid_mode_pass' not in (args.workdir/'execution.log').read_text()),0.0)
    report['passed'] = run.returncode == 0 and bool(report['checks']) and all(c['passed'] for c in report['checks'])
    (args.workdir/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
