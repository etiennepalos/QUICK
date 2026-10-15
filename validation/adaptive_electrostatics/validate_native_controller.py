#!/usr/bin/env python3
"""Standalone, bounded experimental CPU controller/POU acceptance; no SCF/GPU."""
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

from reference_adaptive import FixedCover, electronic_jet, prepare_pairs, primitive_normalization, tensor


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def numbers(a):
    return ' '.join(format(x,'.17e') for x in np.asarray(a).flatten(order='F'))+'\n'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--workdir',type=Path,required=True)
    p.add_argument('--compiler',type=Path,required=True)
    a=p.parse_args()
    a.workdir.mkdir(parents=True,exist_ok=False)
    build=a.workdir/'build';build.mkdir()
    source=[a.repo/'src/modules'/f'quick_{s}_module.f90' for s in
        ('electrostatic_jet','partition_of_unity','adaptive_electrostatics')]
    driver=Path(__file__).with_name('native_controller_driver.f90')
    files=source+[driver,Path(__file__),Path(__file__).with_name('reference_adaptive.py')]
    before={str(x):sha(x) for x in files}
    checks=[];cases=[];commands=[];start=time.perf_counter()
    env=os.environ.copy()
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):env[k]='1'
    # Compile the immutable per-campaign copies, not files being reviewed by
    # another worker. A later source change is reported explicitly below.
    snapshots=[]
    for s in source:
        copy=build/s.name;copy.write_bytes(s.read_bytes());snapshots.append(copy)
    exe=build/'native-controller'

    def run(command,name):
        commands.append(command)
        r=subprocess.run(command,text=True,capture_output=True,cwd=build,env=env)
        (a.workdir/(name+'.stdout')).write_text(r.stdout)
        (a.workdir/(name+'.stderr')).write_text(r.stderr)
        if r.returncode:raise RuntimeError(f'{name}: {r.returncode}\n{r.stderr}')

    run([str(a.compiler),'-cpp','-O2','-Wall','-Wextra','-fcheck=all','-ffree-line-length-none',
        '-I',str(a.repo/'src/util'),'-J',str(build)]+list(map(str,snapshots))+[str(driver),'-o',str(exe)],'compile')

    def check(name,ok,**details):checks.append({'name':name,'passed':bool(ok),**details})

    lo=np.array([4.75,-.25,-.25]);hi=np.array([5.25,.25,.25])
    rng=np.random.default_rng(152701)
    points=rng.uniform(lo,hi,(32,3))
    seed=np.array([.125]+[-.25]*3+[.5]*9)

    def native(name,points=points,degree=3,error_mode=2,max_depth=3,max_anchors=600,max_overlap=8,
               seconds=30.,memory=256.,tol=(1.,1.,1.),factor=2.,lower=lo,upper=hi,
               mode=0,method=0,amplitude=1.):
        inp=a.workdir/(name+'.in');out=a.workdir/(name+'.out')
        inp.write_text(f'{len(points)} {mode} {method} {amplitude:.17e}\n'+
            f'{degree} {error_mode} {max_depth} {max_anchors} {max_overlap}\n'+
            numbers([seconds,memory,*tol,factor])+numbers([*lower,*upper])+numbers(np.asarray(points).T))
        elapsed=time.perf_counter();run([str(exe),str(inp),str(out)],name);elapsed=time.perf_counter()-elapsed
        lines=out.read_text().splitlines();x=lines[0].split();y=lines[1].split()
        report={'ierr':int(x[0]),'fallback':x[1]=='T','accepted':x[2]=='T','certified':x[3]=='T',
            'numerical_budget_known':x[4]=='T','continuous':x[5]=='T','underflow':x[6]=='T',
            'depth':int(y[0]),'anchors':int(y[1]),'bounds':int(y[2]),'overlap':int(y[3]),
            'budget':[float(v) for v in lines[2].split()[3:6]],
            'total_budget':[float(v) for v in lines[2].split()[6:]],
            'times_and_bytes':[float(v) for v in lines[3].split()],'reason':lines[4].strip(),
            'elapsed_seconds':elapsed,'input_sha256':sha(inp),'output_sha256':sha(out)}
        jet=np.array([[float(v) for v in line.split()] for line in lines[5:]])
        cases.append({'name':name,**report})
        check(name+':unknown-FP-budget',not report['certified'] and not report['numerical_budget_known'] and
            all(np.isnan(report['total_budget'])))
        if not report['accepted']:
            check(name+':unchanged-output',np.array_equal(jet,np.broadcast_to(seed,jet.shape)))
        return report,jet

    ao={'center':np.zeros(3),'angular':(0,0,0),'primitives':[(.7,primitive_normalization(.7,(0,0,0)))]}
    pairs=prepare_pairs([ao],np.ones((1,1)))

    def compare(name,report,jet,points,degree,continuous,amplitude=1.):
        cover=FixedCover(pairs,lo,hi,2**report['depth'])
        result=cover.evaluate(points,degree)
        expected=np.column_stack([result['V'],result['E'],result['G'].reshape(len(points),9,order='F')])*amplitude
        difference=float(np.max(abs(jet-expected)))
        check(name+':independent-POU',difference<2e-12,max_absolute_error=difference)
        direct=electronic_jet(pairs,points,2)
        truth=[tensor(direct[0],0),-tensor(direct[1],1),-tensor(direct[2],2)]
        errors=[float(np.max(abs(jet[:,0]-amplitude*truth[0]))),
            float(np.max(np.linalg.norm(jet[:,1:4]-amplitude*truth[1],axis=1))),
            float(np.max(np.linalg.norm(jet[:,4:].reshape(len(points),3,3,order='F')-amplitude*truth[2],axis=(1,2))))]
        check(name+':finite-observables',np.all(np.isfinite(jet)),max_actual_errors=errors,
            real_arithmetic_truncation_budget=report['budget'])
        if continuous:
            check(name+':observed-error-within-truncation-budget',all(e<=b+2e-12 for e,b in zip(errors,report['budget'])))

    for degree in (2,3):
        for error_mode in (1,2):
            name=f'degree{degree}-mode{error_mode}'
            r,j=native(name,degree=degree,error_mode=error_mode)
            check(name+':accepted',r['ierr']==0 and r['accepted'] and not r['fallback'])
            check(name+':mode-label',r['continuous']==(error_mode==2))
            compare(name,r,j,points,degree,error_mode==2)

    # Budgets at successive levels need not be monotone. Test explicit caps
    # with a demanding target and preserve every failed attempt count.
    for name,kw,reason in [
        ('depth-cap',{'max_depth':0},'depth cap'),
        ('anchor-cap',{'max_anchors':8},'anchor cap'),
        ('memory-cap',{'memory':1e-4},'memory cap'),
        ('time-cap',{'seconds':1e-12},'time cap'),
    ]:
        r,j=native(name,tol=(1e-20,1e-20,1e-20),**kw)
        check(name+':fallback',r['ierr']==0 and r['fallback'] and reason in r['reason'],reason=r['reason'],anchors=r['anchors'])
    r,j=native('overlap-cap',method=0,max_overlap=1,tol=(1e-20,1e-20,1e-20),max_depth=1)
    check('overlap-cap:fallback',r['ierr']==0 and r['fallback'] and r['reason']=='overlap cap')
    check('cumulative-anchors',r['anchors']==9,actual=r['anchors'])

    for name,kwargs,code in [
        ('certified',{'error_mode':3},59),('bad-degree',{'degree':4},58),
        ('bad-tolerance',{'tol':(0.,1.,1.)},58),('bad-mode',{'error_mode':0},58),
        ('bad-depth',{'max_depth':7},58),('bad-shape',{'mode':1},46),
        ('nonfinite-query',{'mode':2},55),('outside-box',{'points':np.array([[6.,0.,0.]])},0),
    ]:
        r,j=native(name,**kwargs)
        check(name+':error',r['ierr']==code,actual=r['ierr'],expected=code)
        if name=='outside-box':
            check(name+':whole-batch-fallback',r['fallback'] and not r['accepted'] and
                r['reason']=='queries outside fixed domain')

    # Explicit fixed level1, eight anchors, nontrivial normalized weight
    # derivatives. Freeze all anchors/boxes for both finite-difference sides.
    for degree in (2,3):
        r,j=native(f'fixed-{degree}',method=1,max_depth=1,degree=degree)
        compare(f'fixed-{degree}',r,j,points,degree,True)
        r_g,j_g=native(f'gauge-{degree}',method=1,max_depth=1,degree=degree,mode=3)
        check(f'gauge-{degree}:constant-offset',np.max(abs(j_g[:,0]-j[:,0]-2.3))<2e-14 and
            np.max(abs(j_g[:,1:]-j[:,1:]))<2e-12)
        for axis in range(3):
            step=2e-5;plus=points.copy();minus=points.copy()
            plus[:,axis]+=step;minus[:,axis]-=step
            rp,jp=native(f'fixed-{degree}-plus{axis}',method=1,max_depth=1,degree=degree,points=plus)
            rm,jm=native(f'fixed-{degree}-minus{axis}',method=1,max_depth=1,degree=degree,points=minus)
            efd=-(jp[:,0]-jm[:,0])/(2*step)
            gfd=(jp[:,1:4]-jm[:,1:4])/(2*step)
            g=j[:,4:].reshape(len(points),3,3,order='F')
            eerr=float(np.max(abs(efd-j[:,1+axis])))
            gerr=float(np.max(abs(gfd-g[:,:,axis])))
            check(f'fixed-{degree}:FD-axis{axis}',eerr<2e-8 and gerr<2e-7,max_E_error=eerr,max_G_error=gerr)
    for name,mode in [('asymmetric-G',4),('asymmetric-H',5)]:
        r,j=native(name,method=1,max_depth=1,mode=mode)
        check(name+':rejected',r['ierr']==55,actual=r['ierr'])

    for n in (1000,10000):
        queries=rng.uniform(lo,hi,(n,3))
        r,j=native(f'queries-{n}',points=queries)
        check(f'queries-{n}:accepted',r['ierr']==0 and r['accepted'])
        compare(f'queries-{n}',r,j,queries,3,True)
    r,j=native('density-scaled',amplitude=1.37)
    compare('density-scaled',r,j,points,3,True,amplitude=1.37)

    after={str(x):sha(x) for x in files}
    check('source-stability',before==after,before=before,after=after)
    report={'schema':1,'status':'PASS' if all(x['passed'] for x in checks) else 'FAIL',
        'scope':'experimental CPU coherent POU/controller correctness and bounded fallback; no certification/speedup claim',
        'no_scf':True,'no_gpu':True,'elapsed_seconds':time.perf_counter()-start,
        'cpu_affinity':sorted(os.sched_getaffinity(0)),'source_sha256':before,'commands':commands,
        'counts':{'passed':sum(c['passed'] for c in checks),'total':len(checks)},'checks':checks,'cases':cases}
    # NaN is kept in raw native outputs; JSON uses null for an unavailable
    # budget instead of non-standard JSON NaN or a fabricated numeric zero.
    for case in report['cases']:
        for key in ('budget','total_budget'):
            case[key]=[None if not np.isfinite(v) else v for v in case[key]]
    (a.workdir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'counts':report['counts'],'report':str(a.workdir/'report.json')}))
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':sys.exit(main())
