#!/usr/bin/env python3
"""Linked CPU/default-off/MPI adaptive API acceptance on small molecules."""
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

from reference_adaptive import FixedCover, prepare_pairs


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def table(p):
    return {line.split()[0]:line.split()[1:] for line in p.read_text().splitlines()}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--runtime-root',type=Path,required=True)
    p.add_argument('--workdir',type=Path,required=True)
    p.add_argument('--compiler',type=Path,required=True)
    p.add_argument('--mpifort',type=Path,required=True)
    p.add_argument('--mpirun',type=Path,required=True)
    p.add_argument('--ddx-lib',type=Path,required=True)
    a=p.parse_args()
    a.workdir.mkdir(parents=True,exist_ok=False)
    client_source=Path(__file__).with_name('quick_adaptive_api_driver.f90')
    native_files=[a.repo/'src/modules'/f'quick_{s}_module.f90' for s in
        ('electrostatic_jet','partition_of_unity','adaptive_electrostatics','api')]
    source_files=native_files+[client_source,Path(__file__),Path(__file__).with_name('reference_adaptive.py')]
    source_hashes={str(s):sha(s) for s in source_files}
    frozen_hashes={s.name:sha(a.runtime_root/'source/src/modules'/s.name) for s in native_files}
    supports_gpu_queries='logical :: gpu_queries=' in (a.runtime_root/'source/src/modules/quick_adaptive_electrostatics_module.f90').read_text()
    nrequests=16 if supports_gpu_queries else 15
    checks=[];commands=[];cases=[];start=time.perf_counter()
    env=os.environ.copy()
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):env[k]='1'
    env['OMPI_ALLOW_RUN_AS_ROOT']='1';env['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM']='1'
    env['PATH']=str(a.compiler.parent)+':'+env.get('PATH','')

    def check(name,ok,**details):checks.append({'name':name,'passed':bool(ok),**details})

    def run(command,cwd,name,timeout=90):
        commands.append({'argv':list(map(str,command)),'cwd':str(cwd),'timeout':timeout})
        localenv=env.copy()
        if 'install' in str(command[0]):raise RuntimeError('Unexpected executable selection')
        backend='cpu-mpi' if 'mpi' in name else ('default-off' if 'default-off' in name else 'cpu')
        localenv['QUICK_BASIS']=str(a.runtime_root/backend/'install/basis')
        try:r=subprocess.run(list(map(str,command)),cwd=cwd,text=True,capture_output=True,env=localenv,timeout=timeout)
        except subprocess.TimeoutExpired as ex:
            (cwd/(name+'.stdout')).write_text(ex.stdout.decode() if isinstance(ex.stdout,bytes) else (ex.stdout or ''))
            (cwd/(name+'.stderr')).write_text(ex.stderr.decode() if isinstance(ex.stderr,bytes) else (ex.stderr or ''))
            raise RuntimeError(f'{name}: timed out; raw logs preserved') from ex
        (cwd/(name+'.stdout')).write_text(r.stdout);(cwd/(name+'.stderr')).write_text(r.stderr)
        if r.returncode:raise RuntimeError(f'{name}: exit {r.returncode}\n{r.stderr}\n{r.stdout[-3000:]}')

    clients={}
    for backend in ('cpu','cpu-mpi','default-off'):
        install=a.runtime_root/backend/'install'
        mod='mpi' if backend=='cpu-mpi' else 'serial'
        exe=a.workdir/('api-'+backend)
        command=[a.mpifort if backend=='cpu-mpi' else a.compiler,'-cpp','-ffree-line-length-none',
            '-I'+str(a.repo/'src/util'),'-I'+str(install/'include/amber-modules/quick'/mod)]
        if backend=='cpu-mpi':command+=['-DMPIV']
        if supports_gpu_queries:command+=['-DADAPTIVE_GPU_QUERIES']
        command += [client_source,'-L'+str(install/'lib'),'-Wl,-rpath,'+str(install/'lib'),
            '-Wl,-rpath,'+str(a.ddx_lib),'-Wl,-rpath-link,'+str(a.ddx_lib),
            '-lquick_mpi' if backend=='cpu-mpi' else '-lquick','-o',exe]
        run(command,a.workdir,'compile-'+backend)
        clients[backend]=exe
    for s in native_files:check('runtime-source:'+s.name,source_hashes[str(s)]==frozen_hashes[s.name])

    water=[(8,[0.,0.,0.]),(1,[.757,0.,.587]),(1,[-.757,0.,.587])]
    common='ENERGY SCF=150 DENSERMS=1e-10 CUTOFF=1e-12'
    fixtures={
        'rhf-water':('HF BASIS=STO-3G '+common,water,[]),
        'pbe0-water':('PBE0 BASIS=STO-3G '+common+' XC_RADIAL=150 XC_ANGULAR=590 MATRIXZERO=1e-14',water,[]),
        'uhf-oh':('UHF BASIS=STO-3G MULT=2 '+common,[(8,[0.,0.,0.]),(1,[0.,0.,.97])],[]),
        'charged-mmm':('HF BASIS=STO-3G CHARGE=1 '+common,
            water+[(1,[0.,.85,-.4])],[([3.,1.,0.],.05),([-3.,-.5,.2],-.07)]),
    }
    rng=np.random.default_rng(631290)
    points=rng.uniform([4.76,-.24,-.24],[5.24,.24,.24],(32,3))
    fixture_paths={}
    for name,(keyword,atoms,charges) in fixtures.items():
        inp=a.workdir/(name+'.in')
        lines=[keyword,f'{len(atoms)} {len(charges)} {len(points)}']
        lines += [str(z)+' '+' '.join(format(x,'.17e') for x in c) for z,c in atoms]
        lines += [' '.join(format(x,'.17e') for x in [*c,q]) for c,q in charges]
        lines += [' '.join(format(x,'.17e') for x in pt) for pt in points]
        inp.write_text('\n'.join(lines)+'\n');fixture_paths[name]=inp

    def inspect(label,out,reference=None):
        state=table(out/'result.dat')
        check(label+':converged',state['converged']==['T'])
        check(label+':readiness',state['pre_ready_status']==['47'] and state['post_delete_status']==['47'])
        state_errors=[float(v) for v in state['state_errors']]
        check(label+':unchanged-state-H-no-extra-SCF',max(state_errors)==0.,state_errors=state_errors)
        request_records=[]
        for request in range(1,nrequests+1):
            meta=table(out/f'request-{request:02}.meta')
            data=np.loadtxt(out/f'request-{request:02}.dat')
            code=int(meta['ierr'][0]);flags=[x=='T' for x in meta['flags']]
            budget=np.array([float(x) for x in meta['budget']])
            total=np.array([float(x) for x in meta['total_budget']])
            check(label+f':request{request}:unknown-budget',not flags[2] and not flags[3] and np.all(np.isnan(total)))
            if request in (10,11,12,13,16):
                expected={10:59,11:46,12:55,13:55,16:56}[request]
                check(label+f':request{request}:error',code==expected,actual=code)
                check(label+f':request{request}:transaction',float(meta['output_error'][0])==0.)
            else:
                check(label+f':request{request}:success',code==0 and flags[0]!=flags[1])
                errors=[float(np.max(abs(data[:,3]-data[:,16]))),
                    float(np.max(np.linalg.norm(data[:,4:7]-data[:,17:20],axis=1))),
                    float(np.max(np.linalg.norm((data[:,7:16]-data[:,20:29]).reshape(-1,3,3),axis=(1,2))))]
                if flags[0]:check(label+f':request{request}:direct-fallback',max(errors)==0.,errors=errors)
                elif flags[4]:check(label+f':request{request}:actual-within-truncation',
                    all(e<=b+2e-8 for e,b in zip(errors,budget)),errors=errors,budget=budget.tolist())
            if request in (6,7,8,9):check(label+f':request{request}:fallback-flag',flags[0])
            if reference:
                ref=np.loadtxt(reference/f'request-{request:02}.dat')
                difference=float(np.max(abs(data-ref)))
                check(label+f':request{request}:serial-parity',difference<1e-10,max_absolute_error=difference)
            request_records.append({'request':request,'name':' '.join(meta['name']),'ierr':code,
                'flags':flags,'budget':[None if not np.isfinite(x) else float(x) for x in budget],
                'reason':' '.join(meta['reason'])})
        before=np.loadtxt(out/'request-02.dat');scaled=np.loadtxt(out/'request-14.dat');restored=np.loadtxt(out/'request-15.dat')
        check(label+':density-refresh-observed',np.max(abs(scaled[:,3:16]-before[:,3:16]))>1e-5)
        check(label+':density-refresh-restored',np.array_equal(before,restored))
        cases.append({'name':label,'energy':float(state['energy'][0]),'basis':int(state['basis'][0]),
            'unrestricted':state['unrestricted']==['T'],'requests':request_records})

    serial_dirs={}
    for name in fixtures:
        out=a.workdir/(name+'-cpu');out.mkdir();serial_dirs[name]=out
        run([clients['cpu'],fixture_paths[name],out],out,name+'-cpu')
        inspect(name+'-cpu',out)

    # Source-only independent coherent reference from exported *native*
    # normalized Cartesian coefficients and exact retained alpha+beta D.
    for name,out in serial_dirs.items():
        lines=(out/'ao_basis.dat').read_text().splitlines();nb=int(lines[0]);line=1;aos=[]
        geometry=np.loadtxt(out/'geometry_bohr.dat').reshape(-1,3)
        for mu in range(nb):
            header=list(map(int,lines[line].split()));line+=1
            primitives=[tuple(map(float,lines[line+j].split())) for j in range(header[-1])];line+=header[-1]
            aos.append({'center':geometry[header[1]-1],'angular':tuple(header[2:5]),'primitives':primitives})
        dlines=(out/'density.dat').read_text().splitlines();unrestricted=dlines[0].split()[1]=='T'
        raw=np.array([float(x) for x in dlines[1:]])
        density=raw[:nb*nb].reshape(nb,nb,order='F')
        if unrestricted:density=density+raw[nb*nb:].reshape(nb,nb,order='F')
        pairs=prepare_pairs(aos,density)
        sources=np.loadtxt(out/'classical_sources.dat').reshape(-1,4)
        u=points[:,None,:]-sources[None,:,:3];r=np.linalg.norm(u,axis=2);q=sources[:,3]
        vc=np.sum(q/r,axis=1);ec=np.einsum('s,psj,ps->pj',q,u,r**-3)
        gc=np.einsum('s,ij,ps->pij',q,np.eye(3),r**-3)-3*np.einsum('s,psi,psj,ps->pij',q,u,u,r**-5)
        for request,degree,factor in [(1,2,1.),(2,3,1.),(14,3,1.013)]:
            meta=table(out/f'request-{request:02}.meta');depth=int(meta['counts'][0])
            cover=FixedCover(pairs,np.array([4.75,-.25,-.25]),np.array([5.25,.25,.25]),2**depth)
            result=cover.evaluate(points,degree)
            expected=np.column_stack([factor*result['V']+vc,factor*result['E']+ec,
                (factor*result['G']+gc).reshape(len(points),9,order='F')])
            actual=np.loadtxt(out/f'request-{request:02}.dat')[:,3:16]
            difference=float(np.max(abs(expected-actual)))
            check(name+f':request{request}:independent-full-D',difference<1e-11,max_absolute_error=difference)

    for backend in ('default-off','cpu-mpi'):
        out=a.workdir/('rhf-water-'+backend);out.mkdir()
        cmd=[clients[backend],fixture_paths['rhf-water'],out]
        if backend=='cpu-mpi':cmd=[a.mpirun,'--bind-to','none','-np','2',*cmd]
        run(cmd,out,'rhf-water-'+backend)
        inspect('rhf-water-'+backend,out,serial_dirs['rhf-water'])
    split=a.workdir/'mpi-split';split.mkdir()
    out1=split/'water';out2=split/'charged';out1.mkdir();out2.mkdir()
    run([a.mpirun,'--bind-to','none','-np','4',clients['cpu-mpi'],fixture_paths['rhf-water'],out1,
        fixture_paths['charged-mmm'],out2],split,'mpi-split')
    inspect('mpi-split-water',out1,serial_dirs['rhf-water'])
    inspect('mpi-split-charged',out2,serial_dirs['charged-mmm'])

    after={str(s):sha(s) for s in source_files}
    check('source-stability',after==source_hashes)
    report={'schema':1,'status':'PASS' if all(x['passed'] for x in checks) else 'FAIL',
        'scope':'linked CPU/default-off/MPI experimental adaptive electrostatics; no certification/GPU/speedup claim',
        'source_sha256':source_hashes,'runtime_source_sha256':frozen_hashes,
        'runtime_libraries':{k:sha(a.runtime_root/k/'install/lib'/('libquick_mpi.so' if k=='cpu-mpi' else 'libquick.so'))
            for k in clients},'client_sha256':{k:sha(v) for k,v in clients.items()},
        'counts':{'passed':sum(x['passed'] for x in checks),'total':len(checks)},'checks':checks,'cases':cases,
        'commands':commands,'elapsed_seconds':time.perf_counter()-start,'cpu_affinity':sorted(os.sched_getaffinity(0))}
    (a.workdir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'counts':report['counts'],'report':str(a.workdir/'report.json')}))
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':sys.exit(main())
