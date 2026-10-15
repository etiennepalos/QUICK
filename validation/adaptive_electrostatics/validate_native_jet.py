#!/usr/bin/env python3
"""Native unscreened electronic jet/F10 majorant correctness; no SCF/GPU.

Run with one CPU thread, preferably taskset on the assigned validation core.
This verifies real-arithmetic formulas evaluated in FP64, not certification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from reference_adaptive import (cell_majorants, components, electronic_jet,
    gaussian_fourth, packed_norm, prepare_pairs, primitive_normalization, tensor)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flatten(a):
    a = np.asarray(a)
    fmt = str if np.issubdtype(a.dtype, np.integer) else lambda x: format(x, '.17e')
    return ' '.join(fmt(x) for x in a.flatten(order='F'))+'\n'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--workdir', type=Path, required=True)
    p.add_argument('--compiler', type=Path, required=True)
    args = p.parse_args()
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    build = work/'build'
    build.mkdir()
    source = args.repo/'src/modules/quick_electrostatic_jet_module.f90'
    driver = Path(__file__).with_name('native_jet_driver.f90')
    files = [source, driver, Path(__file__), Path(__file__).with_name('reference_adaptive.py')]
    before = {str(x): sha(x) for x in files}
    commands = []
    checks = []
    cases = []
    start = time.perf_counter()
    env = os.environ.copy()
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[variable] = '1'

    def check(name, passed, **details):
        checks.append({'name': name, 'passed': bool(passed), **details})

    def run(command, name):
        commands.append(command)
        r = subprocess.run(command, text=True, capture_output=True, env=env, cwd=build)
        (work/(name+'.stdout')).write_text(r.stdout)
        (work/(name+'.stderr')).write_text(r.stderr)
        if r.returncode:
            raise RuntimeError(f'{name} failed: {r.returncode}\n{r.stderr}')
        return r

    exe = build/'native-jet'
    run([str(args.compiler), '-cpp', '-O2', '-Wall', '-Wextra', '-fcheck=all',
        '-ffree-line-length-none', '-I', str(args.repo/'src/util'), '-J', str(build),
        str(source), str(driver), '-o', str(exe)], 'compile')

    def normalized_ao(center, angular, exponent=.7, second=False):
        primitives = [(exponent, primitive_normalization(exponent, angular))]
        if second:
            primitives.append((.35*exponent,
                -.17*primitive_normalization(.35*exponent, angular)))
        return {'center': np.asarray(center), 'angular': angular, 'primitives': primitives}

    def native(name, aos, density, points, boxes, owned=None, mode=0):
        point_array = np.atleast_2d(points)
        nb = len(aos)
        na = nb
        mp = max(len(a['primitives']) for a in aos)
        npnt = len(point_array)
        ncell = len(boxes)
        nc = np.array([len(a['primitives']) for a in aos])
        angular = np.array([a['angular'] for a in aos]).T
        expo, coeff = np.ones((mp,nb)), np.zeros((mp,nb))
        for mu,a in enumerate(aos):
            for ip,(x,c) in enumerate(a['primitives']):
                expo[ip,mu], coeff[ip,mu] = x,c
        owned = np.ones(nb,dtype=int) if owned is None else np.asarray(owned,dtype=int)
        arrays = [nc,angular,np.arange(1,nb+1),owned,expo,coeff,
            np.array([a['center'] for a in aos]).T,density,point_array.T,
            np.array([b[0] for b in boxes]).T,np.array([b[1] for b in boxes]).T]
        inp,out = work/(name+'.in'),work/(name+'.out')
        inp.write_text(f'{nb} {na} {mp} {npnt} {ncell} {mode}\n'+''.join(map(flatten,arrays)))
        elapsed = time.perf_counter()
        run([str(exe),str(inp),str(out)],name)
        elapsed = time.perf_counter()-elapsed
        lines = out.read_text().splitlines()
        errors = [int(x) for x in lines[0].split()]
        statuses = []
        for line in lines[1:3]:
            x = line.split()
            statuses.append({'no_intentional_screen': x[0]=='T', 'numerical_budget_known': x[1]=='T',
                'certified': x[2]=='T', 'underflow_detected': x[3]=='T',
                'primitive_pairs': int(x[4]), 'prefactor_underflow_count': int(x[5]),
                'boys_underflow_count': int(x[6])})
        jet = np.array([[float(x) for x in line.split()] for line in lines[3:3+npnt]])
        bounds = np.array([[float(x) for x in line.split()] for line in lines[3+npnt:]])
        result = {'errors': errors,'statuses': statuses, 'jet': jet,'bounds': bounds}
        cases.append({'name': name,'errors': errors,'statuses': statuses,'elapsed_seconds': elapsed,
            'input_sha256': sha(inp),'output_sha256': sha(out)})
        for i,status in enumerate(statuses):
            check(f'{name}:status{i}', status['no_intentional_screen'] and
                not status['numerical_budget_known'] and not status['certified'])
        return result

    def expected(pairs, points, boxes):
        vderiv = electronic_jet(pairs,points,3)
        jets = np.column_stack([tensor(vderiv[0],0),-tensor(vderiv[1],1),
            -tensor(vderiv[2],2).reshape(len(points),9,order='F'),
            -tensor(vderiv[3],3).reshape(len(points),27,order='F')])
        jets += np.array([.125]+[-.25]*3+[.5]*9+[-.75]*27)
        bounds = []
        for lo,hi in boxes:
            b = cell_majorants(pairs,lo,hi)
            bounds.append(np.r_[np.concatenate(b)+.0625,packed_norm(b[3],3),packed_norm(b[4],4)])
        return jets,np.array(bounds)

    points = np.array([[0.,0.,0.],[.1,-.2,.3],[.7,.4,-.3],[5.,.1,-.1],[-.4,.7,1.2]])
    boxes = [(np.array([-.25,-.3,-.35]),np.array([.3,.4,.2])),
        (np.array([4.75,-.25,-.25]),np.array([5.25,.25,.25])),
        (np.array([.123,-.456,.789]),np.array([.123,-.456,.789]))]
    models = []
    for exponent in (.001,.7,1000.):
        aos = [normalized_ao([0.,0.,0.],(0,0,0),exponent)]
        models.append((f's-{exponent}',aos,np.ones((1,1))))
    aos20 = [normalized_ao([0.,0.,0.],c,.7/(1+.2*rank),True)
        for rank in range(4) for c in components(rank)]
    rng = np.random.default_rng(702612)
    c = rng.normal(size=(20,4))*.15
    models.append(('all-spdf',aos20,c@c.T))
    aos4 = [normalized_ao([.1,-.2,.3],(0,0,0),.8,True),
        normalized_ao([-.4,.3,.2],(1,0,0),.6,True),
        normalized_ao([.2,.1,-.5],(1,1,0),.4,True),
        normalized_ao([-.2,-.1,.4],(1,1,1),.3,True)]
    c = rng.normal(size=(4,2))*.2
    models.append(('heterocenter',aos4,c@c.T))
    for name,aos,d in models:
        got = native(name,aos,d,points,boxes)
        ej,eb = expected(prepare_pairs(aos,d),points,boxes)
        check(name+':success',got['errors']==[0,0])
        jet_error = float(np.max(abs(got['jet']-ej)))
        bound_rel = float(np.max(abs(got['bounds']-eb)/(1+abs(eb))))
        check(name+':jet',jet_error < 2e-10, max_absolute_error=jet_error)
        check(name+':majorants',bound_rel < 2e-12, max_scaled_error=bound_rel)
        # All symmetric tensor permutations must be bitwise identical after
        # removing identical additive initial values.
        h = got['jet'][:,13:].reshape(len(points),3,3,3,order='F')
        check(name+':full-H-symmetry',np.array_equal(h,h.swapaxes(1,2)) and
            np.array_equal(h,h.swapaxes(1,3)))

    # Higher-index triangle ownership must reproduce the serial raw result.
    aos,d = aos4,models[-1][2]
    both = native('owned-all',aos,d,points,boxes)
    left = native('owned-even',aos,d,points,boxes,[1,0,1,0])
    right = native('owned-odd',aos,d,points,boxes,[0,1,0,1])
    seed = np.array([.125]+[-.25]*3+[.5]*9+[-.75]*27)
    err = np.max(abs(left['jet']+right['jet']-seed-both['jet']))
    check('ownership:jet-sum',err < 2e-13,max_absolute_error=float(err))
    err = np.max(abs(left['bounds'][:,:35]+right['bounds'][:,:35]-.0625-both['bounds'][:,:35]))
    check('ownership:majorant-sum',err < 2e-12,max_absolute_error=float(err))
    none = native('owned-none',aos,d,points,boxes,[0]*4)
    check('ownership:none',np.array_equal(none['jet'],np.broadcast_to(seed,none['jet'].shape)) and
        np.all(none['bounds'][:,:35]==.0625) and all(x['primitive_pairs']==0 for x in none['statuses']))

    # Error outputs are unchanged even after a late point-range failure.
    for mode in range(1,6):
        got = native(f'error-{mode}',aos,d,points,boxes,mode=mode)
        expected_errors = [0,55] if mode==3 else ([55,0] if mode in (2,5) else [55,55])
        check(f'error-{mode}:codes',got['errors']==expected_errors,actual=got['errors'])
        if got['errors'][0]:
            check(f'error-{mode}:jet-transaction',np.array_equal(got['jet'],np.broadcast_to(seed,got['jet'].shape)))
        if got['errors'][1]:
            check(f'error-{mode}:majorant-transaction',np.all(got['bounds'][:,:35]==.0625))

    # Detect Gaussian underflow but do not turn its unknown FP error into a
    # zero numerical budget. Cross terms vanish in hardware; no cutoff used.
    far_aos = [normalized_ao([0.,0.,0.],(0,0,0)),normalized_ao([100.,0.,0.],(0,0,0))]
    underflow = native('range-underflow',far_aos,np.ones((2,2)),points,boxes)
    check('underflow:success',underflow['errors']==[0,0])
    check('underflow:reported',all(s['underflow_detected'] and s['prefactor_underflow_count'] > 0
        and not s['numerical_budget_known'] for s in underflow['statuses']))

    after = {str(x):sha(x) for x in files}
    check('source-stability',before==after)
    report = {'schema':1,'status':'PASS' if all(x['passed'] for x in checks) else 'FAIL',
        'scope':'raw electronic V/E/G/H and continuous component majorants through rank4, ordinary FP64',
        'no_scf':True,'no_gpu':True,'certified':False,'numerical_error_budget_available':False,
        'elapsed_seconds':time.perf_counter()-start,'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'source_sha256':before,'commands':commands,'checks':checks,'cases':cases,
        'counts':{'passed':sum(x['passed'] for x in checks),'total':len(checks)}}
    (work/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'counts':report['counts'],'report':str(work/'report.json')}))
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    sys.exit(main())
