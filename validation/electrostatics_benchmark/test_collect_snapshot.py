#!/usr/bin/env python3
"""One-pass collector I/O contract tests. Synthetic records, no native jobs."""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
import argparse
import copy
import csv
import itertools
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

import collect_snapshot as collector


def write_rows(path,rows):
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def fixture(folder,backend='cuda'):
    folder.mkdir()
    result={'basis_functions':['2'],'points':['3'],'collector_passes':['1'],'V_E_G_API_calls':['1'],
            'H_API_calls':['1'],'native_gpu_launches':['1']*4 if backend=='cuda' else ['0']*4,
            'extra_API_calls':['0'],'scf_converged':['T'],'hessian_requested':['T'],
            'initial_scf_energy':['-1.23'],'density_error':['0'],'fock_error':['0']}
    (folder/'result.dat').write_text(''.join(k+' '+' '.join(v)+'\n' for k,v in result.items()))
    (folder/'electrostatics_snapshot.out').write_text('======== BASIS INFOS ========\n BASIS FUNCTIONS = 2\n REACH CONVERGENCE AFTER 10 CYCLES\n')
    timings=[dict(route='VEG',iteration='0',seconds='.01',esp_launches='1',efield_launches='1',efg_launches='1',hessian_launches='0'),
             dict(route='H',iteration='0',seconds='.02',esp_launches='0',efield_launches='0',efg_launches='0',hessian_launches='1')]
    if backend=='cpu':
        for row in timings:
            for key in ('esp_launches','efield_launches','efg_launches','hessian_launches'):row[key]='0'
    write_rows(folder/'timings.csv',timings)
    data=np.zeros((3,44));data[:,0]=[1,2,3];data[:,1:4]=[[.2,.3,.4],[.5,.6,.7],[.8,.9,1.]]
    data[:,4:8]=np.arange(12).reshape(3,4)/10
    np.savetxt(folder/'properties.dat',data,header='3',comments='')
    (folder/'density.dat').write_text('2 F\n0\n0\n0\n0\n')
    (folder/'overlap.dat').write_text('2\n1\n0\n0\n1\n')
    (folder/'ao_basis.dat').write_text('2\n')
    plan=[dict(query_index=str(i+1),kind='center' if i==0 else 'sample',center_id='unit-fixture',
               **{x+'_bohr':str(data[i,j+1]) for j,x in enumerate('xyz')}) for i in range(3)]
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    checks=[]
    def execute(name,mutator=None,backend='cuda',keyword='HF BASIS=STO-3G ENERGY',expected=2,positive=False):
        folder=args.output/name;plan=fixture(folder,backend)
        if mutator:mutator(folder)
        try:
            value=collector.validate_exports(folder,keyword,plan,backend,expected)
            if not positive:raise AssertionError('Invalid export accepted: '+name)
            checks.append(dict(name=name,passed=True,observed_basis=value['basis_functions']))
        except ValueError as error:
            if positive:raise
            checks.append(dict(name=name,passed=True,rejection=str(error)))
    def result(key,value):
        def mutate(folder):
            rows=collector.read_result(folder/'result.dat');rows[key]=value
            (folder/'result.dat').write_text(''.join(k+' '+' '.join(v)+'\n' for k,v in rows.items()))
        return mutate
    def timings(change):
        def mutate(folder):
            rows=list(csv.DictReader((folder/'timings.csv').open()));change(rows);write_rows(folder/'timings.csv',rows)
        return mutate
    def data(change):
        def mutate(folder):
            rows=np.loadtxt(folder/'properties.dat',skiprows=1);change(rows);np.savetxt(folder/'properties.dat',rows,header='3',comments='')
        return mutate
    try:
        execute('valid-cuda',positive=True)
        execute('valid-cpu',backend='cpu',positive=True)
        for key,value in [('collector_passes',['2']),('H_API_calls',['2']),('V_E_G_API_calls',['0']),
                          ('extra_API_calls',['1']),('density_error',['1e-20']),('fock_error',['NaN']),
                          ('scf_converged',['F']),('hessian_requested',['F']),('native_gpu_launches',['0']*4),('points',['4'])]:
            execute('invalid-'+key,result(key,value))
        execute('invalid-basis',expected=3)
        execute('invalid-phase-order',timings(lambda r:r.reverse()))
        execute('invalid-repeat',timings(lambda r:r[0].update(iteration='1')))
        execute('invalid-phase-counter',timings(lambda r:r[1].update(hessian_launches='0')))
        execute('invalid-phase-time',timings(lambda r:r[0].update(seconds='NaN')))
        execute('invalid-probe-order',data(lambda r:r.__setitem__((0,0),2)))
        execute('invalid-probe-position',data(lambda r:r.__setitem__((0,1),1.9)))
        execute('invalid-nonfinite-H',data(lambda r:r.__setitem__((0,17),np.nan)))
        execute('invalid-G-permutation',data(lambda r:r.__setitem__((0,9),.1)))
        execute('invalid-H-permutation',data(lambda r:r.__setitem__((0,18),.1)))
        execute('invalid-density-header',lambda p:(p/'density.dat').write_text('3 F\n0\n0\n0\n0\n'))
        execute('invalid-density-finite',lambda p:(p/'density.dat').write_text('2 F\nNaN\n0\n0\n0\n'))
        execute('invalid-overlap-shape',lambda p:(p/'overlap.dat').write_text('2\n1\n0\n1\n'))
        execute('invalid-AO-count',lambda p:(p/'ao_basis.dat').write_text('3\n'))
        execute('invalid-extra-SCF',lambda p:(p/'electrostatics_snapshot.out').write_text((p/'electrostatics_snapshot.out').read_text()+' REACH CONVERGENCE AFTER 5 CYCLES\n'))
        execute('invalid-no-SCF',lambda p:(p/'electrostatics_snapshot.out').write_text('======== BASIS INFOS ========\n BASIS FUNCTIONS = 2\n'))
        execute('invalid-silent-D3',keyword='PBE0 D3 BASIS=STO-3G ENERGY')
        report=dict(passed=True,checks=checks,number_checks=len(checks),native_jobs=0,SCFs=0,GPU_kernels=0,
                    scope='Synthetic export-schema unit fixtures only; no native or scientific accuracy claim')
    except Exception as error:
        report=dict(passed=False,checks=checks,error=f'{type(error).__name__}: {error}')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report.get(key) for key in ('passed','number_checks','error')}))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
