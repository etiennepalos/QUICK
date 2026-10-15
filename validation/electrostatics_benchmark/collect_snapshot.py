#!/usr/bin/env python3
"""Collect one native frozen-density V/E/G/H snapshot and analyze it offline.

One process performs one initial SCF, one V/E/G API call and one H API call.
No CPU reference SCF is launched. AO n/grad(n), Poisson/STF and finite-sample
Taylor analysis use the exact retained density afterward, without an SCF.
"""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import benchmark_workflow as workflow
sys.path.insert(0,str(ROOT/'validation/field_uniformity'))
from field_uniformity import sha256,read_csv,plot_errors,SOURCE_CONVENTION


def require(condition,message):
    if not condition:raise ValueError(message)


def read_result(path):
    out={}
    for line in path.read_text().splitlines():
        words=line.split()
        if not words:continue
        require(words[0] not in out,'Duplicate native result key '+words[0])
        out[words[0]]=words[1:]
    return out


def validate_exports(native,keyword,plan,backend,expected_basis):
    result=read_result(native/'result.dat')
    for key in ('collector_passes','V_E_G_API_calls','H_API_calls'):
        require(result.get(key)==['1'],'Expected one collector call: '+key)
    for key in ('scf_converged','hessian_requested'):
        require(result.get(key)==['T'],'Native boolean gate failed: '+key)
    for key in ('extra_API_calls','density_error','fock_error'):
        values=result.get(key,[])
        require(len(values)==1 and math.isfinite(float(values[0])) and float(values[0])==0,
                'Native frozen-state gate failed: '+key)
    nbasis=int(result['basis_functions'][0]);np_=len(plan)
    require(int(result['points'][0])==np_,'Native point count differs from plan')
    require(expected_basis is None or nbasis==expected_basis,'Native basis count differs from explicit expectation')
    require(nbasis>0 and math.isfinite(float(result['initial_scf_energy'][0])),'Finite native energy/basis required')
    expected_total=[1,1,1,1] if backend=='cuda' else [0,0,0,0]
    require(list(map(int,result.get('native_gpu_launches',[])))==expected_total,'Wrong total native GPU launch counters')
    text=(native/'electrostatics_snapshot.out').read_text()
    scfs=len(re.findall(r'REACH CONVERGENCE AFTER\s+\d+\s+CYCLES',text))
    require(scfs==1,'Expected exactly one native converged SCF event')
    method=workflow.validate_method_output(text,keyword,expected_basis)
    require(method['basis_functions']==nbasis,'Printed/result native basis counts differ')
    timings=list(csv.DictReader((native/'timings.csv').open()))
    require([r['route'] for r in timings]==['VEG','H'],'Expected exactly the one-pass VEG/H timing rows')
    counters=('esp_launches','efield_launches','efg_launches','hessian_launches')
    for row,expected in zip(timings,([1,1,1,0],[0,0,0,1])):
        require(row['iteration']=='0','Snapshot collector must not repeat API calls')
        require(math.isfinite(float(row['seconds'])) and float(row['seconds'])>=0,'Finite nonnegative API phase time required')
        require([int(row[x]) for x in counters]==(expected if backend=='cuda' else [0]*4),'Wrong one-pass phase launch counters')
    properties=native/'properties.dat'
    require(int(properties.read_text().splitlines()[0])==np_,'Native property header count differs')
    data=np.atleast_2d(np.loadtxt(properties,skiprows=1))
    require(data.shape==(np_,44) and np.isfinite(data).all(),'Expected finite full 44-column V/E/G/H properties')
    require(np.array_equal(data[:,0],np.arange(1,np_+1)),'Native probe order differs')
    expected_points=np.array([[float(row[x+'_bohr']) for x in 'xyz'] for row in plan])
    require(np.allclose(data[:,1:4],expected_points,atol=1e-12,rtol=1e-12),'Native probes differ from query plan')
    g=data[:,8:17].reshape(np_,3,3)
    h=data[:,17:44].reshape(np_,3,3,3)
    discrepancies={}
    for name,tensor in (('G',g),('H',h)):
        worst=0.
        for permutation in itertools.permutations(range(1,tensor.ndim)):
            transposed=tensor.transpose((0,*permutation))
            worst=max(worst,float(np.max(abs(tensor-transposed))))
            require(np.allclose(tensor,transposed,atol=1e-14,rtol=1e-12),'Full native tensor symmetry gate failed: '+name)
        discrepancies[name]=worst
    density_lines=(native/'density.dat').read_text().splitlines()
    header=density_lines[0].split()
    require(len(header)==2 and int(header[0])==nbasis and header[1] in ('T','F'),'Malformed density header')
    flat=np.array([float(x) for line in density_lines[1:] for x in line.split()])
    require(len(flat)==nbasis**2*(2 if header[1]=='T' else 1) and np.isfinite(flat).all(),'Malformed finite density export')
    overlap=np.asarray(np.loadtxt(native/'overlap.dat',skiprows=1))
    require(int((native/'overlap.dat').read_text().splitlines()[0])==nbasis and overlap.size==nbasis**2
            and np.isfinite(overlap).all(),'Malformed finite overlap export')
    require(int((native/'ao_basis.dat').read_text().splitlines()[0])==nbasis,'AO descriptor basis dimension differs')
    return {'basis_functions':nbasis,'points':np_,'native_SCF_convergence_events':scfs,
            'source_convention':SOURCE_CONVENTION,'phase_timings':timings,'method_print':method,
            'full_tensor_max_permutation_discrepancy':discrepancies,'result':result}


def offline_gates(output,case_id,backend,model,plot):
    folder=output/(case_id+'-'+backend)
    reference=json.loads((folder/'ao-reference/reference.json').read_text())
    centers=read_csv(folder/'analysis/centers_analysis.csv',['center_id','poisson_g_residual','poisson_h_residual_norm'])
    nelectrons=sum(z for z,_ in model['atoms'])-model['charge']
    electron_error=abs(reference['electron_count_trace_DS']-nelectrons)
    require(electron_error<=1e-8,'Independent electron-count trace gate failed')
    poisson={key:max(abs(float(row[key])) for row in centers) for key in ('poisson_g_residual','poisson_h_residual_norm')}
    require(all(math.isfinite(v) and v<=2e-8 for v in poisson.values()),'Independent AO Poisson gate failed')
    require(reference['overlap_error']<=2e-10,'Independent Cartesian AO mapping/normalization overlap gate failed')
    values=read_csv(folder/'analysis/radius_summary.csv',['center_id','radius_bohr'])
    numeric=[{key:(v if key=='center_id' else float(v)) for key,v in row.items()} for row in values]
    orders={}
    for center_id in sorted({row['center_id'] for row in values}):
        selected=sorted([row for row in numeric if row['center_id']==center_id],key=lambda row:row['radius_bohr'])
        require(len(selected)>=2,'At least two tested radii required for reported observed order')
        orders[center_id]={}
        for approximation in ('constant','linear','quadratic'):
            errors=[row[approximation+'_max_error'] for row in selected]
            require(all(math.isfinite(v) and v>=0 for v in errors),'Malformed finite Taylor reconstruction error')
            orders[center_id][approximation]=(math.log(errors[1]/errors[0])/math.log(selected[1]['radius_bohr']/selected[0]['radius_bohr'])
                                              if min(errors[:2])>1e-14 else None)
        if plot:plot_errors(numeric,folder/'analysis'/('field_error_'+center_id+'.pdf'),center_id=center_id)
    analysis=json.loads((folder/'analysis/analysis.json').read_text())
    require(analysis['continuous_bound_claim'] is False,'Finite sample analysis must not claim a continuous bound')
    return {'independent_overlap_max_error':reference['overlap_error'],'electron_count_error':electron_error,
            'Poisson_max_absolute_residuals':poisson,'observed_first_two_radius_orders':orders,
            'finite_sample_analysis':str(folder/'analysis/analysis.json'),'continuous_bound_claim':False}


def run(args):
    work=args.workdir.resolve();work.mkdir(parents=True,exist_ok=False)
    report={'schema':'quick-one-pass-snapshot-1','passed':False,'prepared_only':args.prepare_only,
            'native_jobs_launched':0,'scope':__doc__,'source_convention':SOURCE_CONVENTION,
            'continuous_bound_claim':False,'gates':{'state_extra_SCF':0,'overlap':2e-10,'electron_count':1e-8,'Poisson_G_H':2e-8},
            'limitations':['One retained density and explicit finite directional samples, not an ensemble or relaxed-response scan',
                           'Phase times are one-shot diagnostics, not repeated performance estimates',
                           'Source-only observed fields exclude direct imposed/reaction backgrounds'], 'runs':[]}
    baseline=None;snapshot=None
    def save():
        (work/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        prepared=args.prepared.resolve(strict=True);index_path=prepared/'application-index.json';index=json.loads(index_path.read_text())
        require(index.get('schema')=='quick-field-uniformity-prepared-1','Unsupported prepared schema')
        cases=[case for case in index['cases'] if case['id']==args.case_id]
        require(len(cases)==1,'Select exactly one explicit prepared case')
        case=cases[0];folder=Path(case['directory']).resolve(strict=True)
        require(sha256(folder/'case-provenance.json')==case['case_provenance_sha256'],'Prepared case provenance changed')
        provenance=json.loads((folder/'case-provenance.json').read_text())
        require(all(sha256(folder/name)==digest for name,digest in provenance['prepared_files_sha256'].items()),'Prepared files changed')
        model=json.loads((folder/'model.json').read_text());keyword=(folder/'keyword.txt').read_text().strip().upper()
        require('DFTD3' not in keyword.split(),'Use public D3 input keyword')
        workflow.validate_expected_controls(keyword,index['protocol']['expected_controls'])
        plan=read_csv(folder/'query_plan.csv',['query_index','kind','center_id','x_bohr','y_bohr','z_bohr'])
        require(len(plan)==case['queries'],'Prepared case/query count differs')
        driver=args.driver.resolve(strict=True);prefix=args.install_prefix.resolve(strict=True);source=args.source_snapshot.resolve(strict=True)
        client_source=args.client_source.resolve(strict=True);compile_record=args.compile_record.resolve(strict=True)
        compiled=json.loads(compile_record.read_text())
        require(compiled.get('returncode')==0 and compiled['source_sha256']==sha256(client_source)
                and compiled['binary_sha256']==sha256(driver),'Collector binary/source do not match compile record')
        require(str(client_source) in compiled['argv'],'Compile record must name the exact archived client source')
        basis=(prefix/'basis'/Path(case['basis_file']).name).resolve(strict=True)
        require(sha256(basis)==sha256(case['basis_file'])==provenance['source_files_sha256'][case['basis_file']],
                'Installed basis differs from exact prepared basis')
        env=os.environ.copy();env.update(QUICK_BASIS=str(prefix/'basis'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        env['LD_LIBRARY_PATH']=':'.join([str(prefix/'lib'),*map(str,args.library_path),env.get('LD_LIBRARY_PATH','')])
        if args.backend=='cuda':env['CUDA_VISIBLE_DEVICES']=str(args.cuda_device)
        inputs=[index_path,*[p for p in folder.iterdir() if p.is_file()],basis,client_source,compile_record,
                args.oracle_python.resolve(strict=True),Path(sys.executable).resolve(strict=True)]
        harness=[Path(__file__),HERE/'benchmark_workflow.py',ROOT/'validation/oeprop_bomd/benchmark_overhead.py',
                 ROOT/'validation/oeprop_bomd/validate_oeprop_bomd.py',*sorted((ROOT/'validation/field_uniformity').glob('*.py')),
                 ROOT/'validation/field_hessian/reference_field_hessian.py']
        def snapshot():
            return {'runtime':workflow.old.driver_snapshot(driver,env),'production_source':workflow.production_snapshot(source),
                    'inputs_basis_clients_harness':{str(p):sha256(p) for p in [*inputs,*harness]}}
        baseline=snapshot();report['provenance_before']=baseline
        quick=[path for path in baseline['runtime']['resolved_libraries_sha256'] if re.fullmatch(r'libquick(?:_cuda)?\.so(?:\.[\d.]+)?',Path(path).name)]
        require(len(quick)==1 and Path(quick[0]).parent==(prefix/'lib').resolve(),'Loaded QUICK library is not the selected install')
        require(Path(quick[0]).name.startswith('libquick_cuda' if args.backend=='cuda' else 'libquick.so'),'Loaded QUICK backend differs')
        if args.build_report:
            bp=args.build_report.resolve(strict=True);build=json.loads(bp.read_text());mp=bp.parent/'source-manifest.json'
            require(build.get('passed') and sha256(mp)==build['source_manifest_sha256'],'Invalid immutable build report')
            manifest=json.loads(mp.read_text())
            require(all(manifest.get(name)==digest for name,digest in baseline['production_source'].items()),'Production source differs from compiled manifest')
            libraries={str(Path(path).resolve()):digest for entry in build['backends'].values() for path,digest in entry['libraries_sha256'].items()}
            require(libraries.get(quick[0])==baseline['runtime']['resolved_libraries_sha256'][quick[0]],'Loaded library absent from build report')
            report['build_record']={'path':str(bp),'sha256':sha256(bp),'source_manifest_sha256':sha256(mp)}
            # Include immutable build records in subsequent provenance checks.
            inputs.extend([bp,mp]);baseline=snapshot();report['provenance_before']=baseline
        native=work/'native';native.mkdir();shutil.copy2(folder/'input.dat',native/'input.dat')
        inputs.append(native/'input.dat');baseline=snapshot();report['provenance_before']=baseline
        argv=[str(driver),str(native/'input.dat'),'-']
        report.update(configuration={'case_id':args.case_id,'backend':args.backend,'keyword':keyword,'queries':case['queries'],
                'expected_basis_functions':args.expected_basis_functions,'source_snapshot':str(source),'prepared':str(prepared)},
            native_command=argv,native_directory=str(native),environment={k:env.get(k) for k in
            ('QUICK_BASIS','LD_LIBRARY_PATH','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES')})
        save()
        if args.prepare_only:
            report['preparation_passed']=True
        else:
            report['native_jobs_launched']=1;save();start=time.perf_counter()
            with (native/'execution.log').open('w') as log:
                process=subprocess.run(argv,cwd=native,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
            report['runs'].append({'argv':argv,'returncode':process.returncode,'fresh_process_wall_seconds':time.perf_counter()-start})
            require(process.returncode==0,'Native one-pass collector failed')
            require(sha256(native/'input.dat')==sha256(folder/'input.dat'),'Actual native input differs from prepared input')
            report['native_validation']=validate_exports(native,keyword,plan,args.backend,args.expected_basis_functions);save()
            runs_json=work/'native-exports.json';runs_json.write_text(json.dumps({args.case_id:{args.backend:str(native)}},indent=2)+'\n')
            cmd=[sys.executable,str(ROOT/'validation/field_uniformity/application.py'),'analyze','--prepared',str(prepared),
                 '--runs-json',str(runs_json),'--oracle-python',str(args.oracle_python.resolve(strict=True)),
                 '--output',str(work/'analysis'),'--tolerance-au',*map(str,args.tolerance_au)]
            if args.plot:cmd.append('--plot')
            start=time.perf_counter()
            with (work/'offline.log').open('w') as log:
                process=subprocess.run(cmd,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
            report['runs'].append({'argv':cmd,'returncode':process.returncode,'offline_seconds':time.perf_counter()-start,'native_SCFs':0})
            require(process.returncode==0 and json.loads((work/'analysis/report.json').read_text())['passed'],'Offline AO/conversion/analysis failed')
            report['independent_analysis']=offline_gates(work/'analysis',args.case_id,args.backend,model,args.plot)
            report['passed']=True
        report['provenance_after']=snapshot();require(report['provenance_after']==baseline,'Runtime/source/input/harness changed during collection')
        report['provenance_unchanged']=True
    except Exception as error:
        import traceback
        report.update(passed=False,error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
        if snapshot is not None and baseline is not None:
            try:
                report['provenance_after']=snapshot();report['provenance_unchanged']=report['provenance_after']==baseline
            except Exception as after_error:report['after_provenance_error']=str(after_error)
    report['retained_artifacts_sha256']={str(p.relative_to(work)):sha256(p) for p in work.rglob('*') if p.is_file() and p!=work/'report.json'}
    save();print(json.dumps({key:report.get(key) for key in ('passed','prepared_only','preparation_passed','native_jobs_launched','error')}))
    return 0 if report['passed'] or (args.prepare_only and report.get('preparation_passed') and not report.get('error')) else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('driver','install-prefix','source-snapshot','client-source','compile-record','prepared','oracle-python','workdir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--case-id',required=True);parser.add_argument('--backend',choices=('cpu','cuda'),required=True)
    parser.add_argument('--build-report',type=Path);parser.add_argument('--library-path',type=Path,action='append',default=[])
    parser.add_argument('--expected-basis-functions',type=int);parser.add_argument('--cuda-device',type=int,default=0)
    parser.add_argument('--timeout',type=float,default=7200);parser.add_argument('--tolerance-au',type=float,nargs='+',default=[1e-4,1e-3])
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--plot',action='store_true')
    args=parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout<=0:parser.error('Positive finite timeout required')
    if any(not math.isfinite(v) or v<=0 for v in args.tolerance_au):parser.error('Positive finite field-error tolerances required')
    if args.expected_basis_functions is not None and args.expected_basis_functions<1:parser.error('Positive expected AO count required')
    if args.cuda_device<0:parser.error('Nonnegative CUDA device required')
    return run(args)


if __name__=='__main__':raise SystemExit(main())
