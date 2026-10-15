#!/usr/bin/env python3
"""Validate native FP64 CUDA PoU queries against CPU and independent AD fixtures.

This driver launches GPU work. Run only with an exclusive device allocation.
It performs no SCFs and does not modify QUICK or an installed runtime.
"""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
from __future__ import annotations

import argparse
import copy
import ctypes as ct
import json
from pathlib import Path
import time
import traceback

import numpy as np

import validate_native_pou as reference


class Campaign(reference.Campaign):
    def __init__(self, workdir, library, cpu_driver):
        super().__init__(workdir, None)
        self.library = ct.CDLL(str(library))
        self.function = self.library.gpu_adaptive_electrostatics_
        pi, pd = ct.POINTER(ct.c_int), ct.POINTER(ct.c_double)
        self.function.argtypes = [pi, pi]+[pd]*7+[pi, pd, pi]+[pd]*5+[pi, pi]
        self.function.restype = None
        self.cpu = reference.Campaign(workdir/'cpu', cpu_driver)

    def diagnostics(self):
        count, live, last = ct.c_int(), ct.c_longlong(), ct.c_longlong()
        self.library.gpu_adaptive_electrostatics_launch_count_(ct.byref(count))
        self.library.gpu_adaptive_electrostatics_allocation_bytes_(ct.byref(live), ct.byref(last))
        return dict(launches=count.value,live_bytes=live.value,last_bytes=last.value)

    def run(self, name, cover, points, degree=3, errors=None, mode=0, nside=None,
            optional_flag=None, compare_cpu=True):
        if mode:
            raise ValueError('Assumed-shape Fortran checks are outside the raw C interface')
        directory=self.root/'cases'/name
        directory.mkdir(parents=True, exist_ok=False)
        arrays=[cover.lower, cover.upper, cover.v, cover.e.T, cover.g.transpose(1,2,0),
                cover.H.transpose(1,2,3,0), cover.majorants.T,
                np.zeros((4,1)) if errors is None else errors.T, np.asarray(points).T]
        arrays=[np.asfortranarray(a,dtype=np.float64) for a in arrays]
        npoints=len(points)
        values=np.full(npoints,13.25)
        fields=np.full((3,npoints),-24.5,order='F')
        gradients=np.full((3,3,npoints),37.75,order='F')
        budgets=np.full((3,npoints),-48.125,order='F')
        ns=ct.c_int(cover.nside if nside is None else nside)
        deg=ct.c_int(degree)
        flag=ct.c_int(int(errors is not None) if optional_flag is None else optional_flag)
        np_=ct.c_int(npoints);overlap=ct.c_int(-1);ierr=ct.c_int(-1)
        pointer=lambda a:a.ctypes.data_as(ct.POINTER(ct.c_double))
        before=self.diagnostics()
        start=time.perf_counter()
        self.function(ct.byref(ns),ct.byref(deg),*[pointer(a) for a in arrays[:7]],
                      ct.byref(flag),pointer(arrays[7]),ct.byref(np_),pointer(arrays[8]),
                      *[pointer(a) for a in (values,fields,gradients,budgets)],
                      ct.byref(overlap),ct.byref(ierr))
        elapsed=time.perf_counter()-start
        after=self.diagnostics()
        np.savez(directory/'arrays.npz',lower=arrays[0],upper=arrays[1],va=arrays[2],ea=arrays[3],
                 ga=arrays[4],ha=arrays[5],majorants=arrays[6],anchor_errors=arrays[7],
                 points=arrays[8],v=values,e=fields,g=gradients,budgets=budgets)
        record=dict(name=name,nside=ns.value,degree=degree,npoints=npoints,
                    has_anchor_errors=flag.value,ierr=ierr.value,overlap=overlap.value,
                    wall_seconds=elapsed,diagnostics_before=before,diagnostics_after=after,
                    arrays_sha256=reference.sha(directory/'arrays.npz'))
        self.runs.append(record)
        self.check(name+'/all-device-storage-released',after['live_bytes']==0,diagnostics=after)
        out=dict(ierr=ierr.value,overlap=overlap.value,
                 preserved=bool(np.all(values==13.25) and np.all(fields==-24.5)
                     and np.all(gradients==37.75) and np.all(budgets==-48.125)),
                 V=values,E=fields.T,G=gradients.transpose(2,0,1),budgets=budgets.T)
        if ierr.value==0:
            expected_bytes=8*(19*npoints+(42+4*int(bool(flag.value)))*ns.value**3)+8
            self.check(name+'/one-native-query-launch',after['launches']==before['launches']+1)
            self.check(name+'/allocation-accounting',after['last_bytes']==expected_bytes,
                       expected_bytes=expected_bytes,actual_bytes=after['last_bytes'])
            if compare_cpu:
                cpu=self.cpu.run(name,cover,points,degree,errors,nside=nside)
                self.check(name+'/cpu-accepted',cpu['ierr']==0)
                for key in ('V','E','G','budgets'):
                    self.close(name+'/cpu-equivalence-'+key,out[key],cpu[key],atol=3e-10,rtol=8e-12)
                self.check(name+'/cpu-overlap',out['overlap']==cpu['overlap'])
        elif after['launches']==before['launches']:
            self.check(name+'/rejected-before-device-allocation',after['last_bytes']==0)
        return out


def run_campaign(c):
    # Block tails include both sides of every 128-thread launch boundary.
    for nside in (1,2,4):
        cover=reference.make_cover(nside)
        for degree in (2,3):
            c.positive(f'boundary-n{nside}-d{degree}',cover,reference.probe_points(cover),degree)
        for count in (1,7,127,128,129,255,256,257):
            rng=np.random.default_rng(1100+count+nside)
            probes=rng.uniform(cover.lower,cover.upper,(count,3))
            c.positive(f'tail-n{nside}-{count}',cover,probes,3)

    cover=reference.make_cover(4)
    rng=np.random.default_rng(331)
    points=rng.uniform(cover.lower+.15*(cover.upper-cover.lower),
                       cover.upper-.15*(cover.upper-cover.lower),(11,3))
    errors=rng.uniform(1e-5,4e-4,(len(cover.v),4))
    for degree in (2,3):
        c.positive(f'anchor-errors-d{degree}',cover,points,degree,errors)
        center=c.run(f'fd-center-d{degree}',cover,points,degree)
        residuals=[]
        for step in (2e-5,1e-5):
            efd=np.empty_like(center['E']);gfd=np.empty_like(center['G'])
            for axis in range(3):
                delta=np.eye(3)[axis]*step
                plus=c.run(f'fd-d{degree}-{step}-{axis}-plus',cover,points+delta,degree)
                minus=c.run(f'fd-d{degree}-{step}-{axis}-minus',cover,points-delta,degree)
                efd[:,axis]=-(plus['V']-minus['V'])/(2*step)
                gfd[:,:,axis]=(plus['E']-minus['E'])/(2*step)
            c.close(f'fd-d{degree}-{step}/E=-gradV',efd,center['E'],atol=2e-5,rtol=0)
            c.close(f'fd-d{degree}-{step}/G=gradE',gfd,center['G'],atol=3e-4,rtol=0)
            residuals.append([float(np.max(abs(efd-center['E']))),float(np.max(abs(gfd-center['G'])))])
        c.check(f'fd-d{degree}/two-step-stability',np.all(np.array(residuals[1])<=np.array(residuals[0])*1.1+1e-8),errors=residuals)

    for nside in (1,2,4):
        cover=reference.polynomial_cover(nside);points=reference.probe_points(cover)
        truth=[reference.polynomial(points),-reference.poly_tensor(points,1),-reference.poly_tensor(points,2)]
        for degree in (2,3):
            for perturb in (False,True):
                fixture=copy.deepcopy(cover);errors=None
                if perturb:
                    errors=np.tile([1e-4,2e-4,3e-4,4e-4],(len(cover.v),1))
                    signs=(-1.)**np.arange(len(cover.v))
                    fixture.v+=signs*errors[:,0];fixture.e[:,0]+=signs*errors[:,1]
                    fixture.g[:,0,0]+=signs*errors[:,2];fixture.H[:,0,0,0]+=signs*errors[:,3]
                name=f'polynomial-n{nside}-d{degree}-perturbed{int(perturb)}'
                out=c.positive(name,fixture,points,degree,errors)
                for i,key in enumerate(('V','E','G')):
                    diff=out[key]-truth[i]
                    norm=abs(diff) if i==0 else np.linalg.norm(diff.reshape(len(points),-1),axis=1)
                    c.check(name+'/whole-support-enclosure-'+key,np.all(norm<=out['budgets'][:,i]+2e-10),
                            maximum_excess=float(np.max(norm-out['budgets'][:,i])))

    cover=reference.make_cover(4);points=reference.probe_points(cover)
    base=c.run('gauge-baseline',cover,points)
    shifted=copy.deepcopy(cover);shifted.v+=2.3
    out=c.run('gauge-shift',shifted,points)
    c.close('gauge/V-offset',out['V']-base['V'],np.full(len(points),2.3),atol=2e-12)
    for key in ('E','G','budgets'):c.close('gauge/invariant-'+key,out[key],base[key],atol=2e-11)
    cover.v[:]=2.**40;cover.e[:]=0;cover.g[:]=0;cover.H[:]=0
    out=c.run('gauge-exact-large-constant',cover,points)
    c.close('gauge-large/V-exact',out['V'],cover.v[0]*np.ones(len(points)),atol=0,rtol=0)
    for key in ('E','G'):c.close('gauge-large/'+key+'-exact-zero',out[key],np.zeros_like(out[key]),atol=0,rtol=0)

    # Raw numeric rejection occurs after several successful launches, proving
    # new-call allocation diagnostics cannot retain a previous positive value.
    cover=reference.make_cover(2);points=np.array([(cover.lower+cover.upper)/2])
    invalid=[]
    for field in ('v','e','g','H','majorants','lower','upper'):
        fixture=copy.deepcopy(cover);getattr(fixture,field).flat[0]=np.nan
        invalid.append(('nan-'+field,fixture,points,{},55))
    for tensor in ('g','H'):
        fixture=copy.deepcopy(cover)
        if tensor=='g':fixture.g[0,0,1]+=.4
        else:fixture.H[0,0,1,2]+=.6
        invalid.append(('asymmetric-'+tensor,fixture,points,{},55))
    outside=points.copy();outside[0,0]=np.nextafter(cover.lower[0],-np.inf)
    invalid.append(('outside-nextafter',cover,outside,{},55))
    invalid.append(('empty-query',cover,np.empty((0,3)),{},46))
    invalid.append(('bad-optional-flag',cover,points,dict(optional_flag=2),55))
    invalid.append(('invalid-degree',cover,points,dict(degree=4),46))
    negative=np.ones((len(cover.v),4));negative[0,0]=-1
    invalid.append(('negative-anchor-errors',cover,points,dict(errors=negative),55))
    overflow=reference.make_cover(1);overflow.lower[:]=-10;overflow.upper[:]=10
    overflow.majorants[:]=1e308
    invalid.append(('late-overflow',overflow,np.array([[0,0,0],[10,10,10]]),{},55))
    for name,fixture,probes,kwargs,expected in invalid:
        out=c.run('invalid-'+name,fixture,probes,compare_cpu=False,**kwargs)
        c.check('invalid-'+name+'/code',out['ierr']==expected,actual=out['ierr'],expected=expected)
        c.check('invalid-'+name+'/transactional',out['preserved'])
    # Recovery after a late device-side numeric failure uses the same process.
    c.positive('valid-after-invalid',cover,points,3)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-library',type=Path,required=True)
    parser.add_argument('--cpu-driver',type=Path,required=True)
    parser.add_argument('--workdir',type=Path,required=True)
    args=parser.parse_args()
    args.gpu_library=args.gpu_library.resolve(strict=True);args.cpu_driver=args.cpu_driver.resolve(strict=True)
    args.workdir=args.workdir.resolve();args.workdir.mkdir(parents=True,exist_ok=False)
    c=Campaign(args.workdir,args.gpu_library,args.cpu_driver)
    report=dict(scope=__doc__,gpu_library=str(args.gpu_library),gpu_library_sha256=reference.sha(args.gpu_library),
                cpu_driver=str(args.cpu_driver),cpu_driver_sha256=reference.sha(args.cpu_driver),
                validator_sha256=reference.sha(__file__),oracle_sha256=reference.sha(reference.__file__),
                exceptions=[],passed=False)
    try:run_campaign(c)
    except Exception:report['exceptions'].append(traceback.format_exc())
    report.update(checks=c.checks,runs=c.runs,cpu_checks=c.cpu.checks,cpu_runs=c.cpu.runs,
                  final_diagnostics=c.diagnostics())
    report['passed']=not report['exceptions'] and bool(c.checks) and all(x['pass_'] for x in c.checks)
    (args.workdir/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=report['passed'],checks=len(c.checks),runs=len(c.runs),exceptions=report['exceptions']),indent=2))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
