#!/usr/bin/env python3
"""Read-only matched direct/adaptive BOMD acceptance; launches no native jobs.

Retrospective hashes identify retained evidence and check that this analysis
does not change it. They do not invent an execution-time runtime fingerprint.
"""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'validation/electrostatics_benchmark'))
import benchmark_workflow as workflow
old=workflow.old
legacy=sys.modules['validate_oeprop_bomd']


def require(condition,message):
    if not condition:raise ValueError(message)


def control_values(path):
    """Read scalar/array assignments outside quoted strings/comments.

    This reader is for the retained driver controls, not a general Fortran
    namelist replacement. Indexed/repeated assignments are rejected.
    """
    source=path.read_text();text=[];mask=[];quote=None;i=0
    while i<len(source):
        c=source[i]
        if quote:
            text.append(c);mask.append(' ')
            if c==quote:
                if i+1<len(source) and source[i+1]==quote:
                    text.append(source[i+1]);mask.append(' ');i+=1
                else:quote=None
        elif c in "'\"":
            quote=c;text.append(c);mask.append(' ')
        elif c=='!':
            while i<len(source) and source[i]!='\n':i+=1
            text.append('\n');mask.append('\n')
        else:text.append(c);mask.append(c)
        i+=1
    require(quote is None,'Unterminated quoted namelist value')
    text=''.join(text);mask=''.join(mask)
    matches=list(re.finditer(r'\b([A-Za-z][A-Za-z0-9_]*)(\s*\([^)]*\))?\s*=',mask));out={}
    require(bool(matches),'No retained namelist assignments')
    for k,match in enumerate(matches):
        key=match[1].lower();require(key not in out and not match[2],'Duplicate/indexed control requires an explicit normalized input: '+key)
        value=text[match.end():matches[k+1].start() if k+1<len(matches) else len(text)].strip().rstrip(',/ \t\r\n')
        if value.startswith(("'",'"')):
            q=value[0];require(len(value)>=2 and value[-1]==q,'Malformed quoted control '+key)
            out[key]=value[1:-1].replace(q+q,q)
        else:
            entries=[entry.strip() for entry in value.split(',') if entry.strip()]
            require(bool(entries),'Missing control value '+key)
            parsed=[]
            for entry in entries:
                if entry.lower() in ('.true.','.false.'):parsed.append(entry.lower()=='.true.')
                else:
                    number=float(entry.upper().replace('D','E'));require(math.isfinite(number),'Nonfinite control '+key);parsed.append(number)
            out[key]=parsed[0] if len(parsed)==1 else parsed
    return out


def csv_rows(path,required):
    with path.open(newline='') as stream:
        reader=csv.DictReader(stream);fields=reader.fieldnames or []
        require(len(fields)==len(set(fields)) and set(required)<=set(fields),'Malformed diagnostic CSV header')
        rows=list(reader)
    require(bool(rows) and all(None not in row and all(v is not None for v in row.values()) for row in rows),'Malformed/empty diagnostic CSV')
    return rows


def resolved(directory,path):
    return (directory/str(path)).resolve(strict=True)


def gpu_telemetry(rows,summary,requested,gpu):
    columns={'anchor_backend','query_backend','gpu_query_launches'}
    keys={'adaptive_anchor_backend','adaptive_bound_backend','adaptive_cuda_queries_requested',
          'adaptive_gpu_query_launches','adaptive_device_bytes_live_after','adaptive_device_bytes_last_call'}
    present_columns=columns & set(rows[0])
    present_keys=keys & set(summary)
    if not present_columns and not present_keys:
        require(not requested,'Requested CUDA queries require actual backend/counter telemetry')
        return {'available':False,'checked':False,'scope':'Legacy CSV/summary lacks GPU telemetry; legacy direct-route counters still checked'}
    require(present_columns==columns and present_keys==keys,'Incomplete optional GPU telemetry')
    require(summary['adaptive_cuda_queries_requested'] is requested,'Requested GPU control differs from summary')
    require(summary['adaptive_anchor_backend']=='cpu' and summary['adaptive_bound_backend']=='cpu',
            'Unexpected anchor/bound backend claim')
    require(not requested or gpu,'CUDA queries requested on a non-GPU BOMD backend')
    total=0;frames=[]
    for row in rows:
        anchor=row['anchor_backend'].strip().lower();query=row['query_backend'].strip().lower()
        launches=int(row['gpu_query_launches'])
        require(str(launches)==row['gpu_query_launches'].strip() and launches>=0,'Invalid frame GPU launch count')
        require(anchor=='cpu','Unexpected frame anchor backend')
        fallback=row['fallback']=='T'
        if fallback:
            require(query=='native direct','Whole-frame fallback must report the actual native direct query backend')
            # Rejected candidate levels can have already launched CUDA PoU.
            require(requested or launches==0,'Unrequested CUDA query launched before fallback')
        elif requested:
            require(query=='cuda' and launches==int(row['depth'])+1,
                    'Accepted requested-GPU frame lacks CUDA launches for each attempted cover level')
        else:
            require(query=='cpu' and launches==0,'Unrequested accepted GPU query/backend mismatch')
        total+=launches
        frames.append({'step':int(row['step']),'anchor_backend':anchor,'query_backend':query,'gpu_query_launches':launches})
    for key in ('adaptive_gpu_query_launches','adaptive_device_bytes_live_after','adaptive_device_bytes_last_call'):
        require(type(summary[key]) is int and summary[key]>=0,'Invalid summary GPU counter/byte value '+key)
    require(summary['adaptive_gpu_query_launches']==total,'GPU query summary differs from actual frame launch sum')
    require(summary['adaptive_device_bytes_live_after']==0,'Adaptive CUDA allocations remain live after run')
    require(total==0 or summary['adaptive_device_bytes_last_call']>0,'CUDA launches lack recorded device work allocation')
    return {'available':True,'checked':True,'requested_cuda_queries':requested,'actual_gpu_query_launches':total,
            'device_bytes_live_after':summary['adaptive_device_bytes_live_after'],
            'device_bytes_last_call':summary['adaptive_device_bytes_last_call'],'framewise':frames,
            'scope':'CPU anchors/bounds; actual per-frame query dispatch and counter sums; allocation cleanup. Candidate CUDA attempts may precede direct fallback.'}


def load(directory):
    directory=directory.resolve(strict=True);control=control_values(directory/'control.nml')
    require(control.get('probe_mode','fixed').lower()=='fixed','This matched validator requires fixed probes')
    require(control.get('field_hessian',False) is True,'Both runs must explicitly enable direct H')
    require(not control.get('restart_file',''),'Use complete step-zero paths for this matched validator')
    prefix=directory/str(control.get('output_prefix','quick_oeprop_bomd'))
    summary=json.loads(Path(str(prefix)+'_summary.json').read_text())
    steps=int(control.get('nsteps',10));stride=int(control.get('oeprop_stride',1));dt=float(control.get('dt_fs',.25))
    require(steps>=0 and stride>0,'Positive property stride and nonnegative MD step count required')
    expected_steps=list(range(steps+1));sampled=[s for s in expected_steps if s%stride==0]
    for key,value in {'start_step':0,'final_step':steps,'steps_run':steps,'oeprop_stride':stride,'force_calls':steps+1,
                      'api_energy_gradient_calls':steps+1,'property_calls':len(sampled),
                      'probe_generation_calls':len(sampled),'properties_extra_api_calls':0}.items():
        require(summary.get(key)==value,'Wrong summary counter '+key)
    require(abs(float(summary['dt_fs'])-dt)<1e-14,'Wrong summary time step')
    for key,value in summary.items():
        if isinstance(value,(int,float)):require(math.isfinite(value),'Nonfinite summary '+key)
    energy=legacy.read_csv(Path(str(prefix)+'_energy.csv'),['step',*legacy.ENERGY_COLUMNS])
    require([int(row['step']) for row in energy]==expected_steps,'Wrong energy frame sequence')
    for row in energy:
        require(abs(float(row['time_fs'])-int(row['step'])*dt)<=1e-12,'Wrong energy time')
        require(abs(float(row['total_au'])-float(row['potential_au'])-float(row['kinetic_au']))<=1e-12,'Energy sum inconsistent')
    trajectory=legacy.read_trajectory(Path(str(prefix)+'_trajectory.xyz'),expected_steps,dt)
    checkpoint=legacy.read_checkpoint(resolved(directory,control.get('checkpoint_file','bomd.chk')))
    require(checkpoint['step']==steps and checkpoint['keyword']==control['keywd'].upper()
            and abs(checkpoint['dt_fs']-dt)<1e-14,'Wrong checkpoint metadata')
    require(legacy.max_difference(checkpoint['coordinates'],trajectory[-1])<=1e-11,'Checkpoint/XYZ endpoint mismatch')
    geometry_path=resolved(directory,control.get('geometry_file','geometry.xyz'))
    lines=geometry_path.read_text().splitlines();natom=int(lines[0]);initial=np.array([row.split()[1:] for row in lines[2:2+natom]],float)
    require(trajectory.shape==(steps+1,natom,3) and legacy.max_difference(trajectory[0],initial)<=1e-10,'Initial trajectory differs from supplied geometry')
    probes_path=resolved(directory,control.get('probe_file','probes.dat'));nprobes=int(probes_path.read_text().splitlines()[0])
    probes=np.atleast_2d(np.loadtxt(probes_path,skiprows=1))
    require(probes.shape==(nprobes,3) and np.isfinite(probes).all(),'Malformed retained finite probes')
    properties=legacy.read_csv(Path(str(prefix)+'_oeprop.csv'),['step','time_fs','probe','probe_mode','x_ang','y_ang','z_ang',
                              *legacy.PROPERTY_COLUMNS,*legacy.INVARIANT_COLUMNS])
    identities=[(s,p) for s in sampled for p in range(1,nprobes+1)]
    require([(int(r['step']),int(r['probe'])) for r in properties]==identities,'Property sample identities differ')
    for row in properties:
        require(row['probe_mode']=='fixed' and abs(float(row['time_fs'])-int(row['step'])*dt)<=1e-12,'Wrong property mode/time')
        require(legacy.max_difference([float(row[x+'_ang']) for x in 'xyz'],probes[int(row['probe'])-1])<=1e-11,'Fixed probe changed')
    invariant=legacy.invariant_errors(properties)
    require(all(v<=old.TOLERANCES['property_au'] for v in invariant.values()),'Printed invariant consistency failed')
    output=directory/'quick_oeprop_bomd.out';text=output.read_text()
    require(len(re.findall(r'REACH CONVERGENCE AFTER\s+\d+\s+CYCLES',text))==steps+1,'Extra/missing native SCF convergence events')
    gradients=old.read_gradients(output,natom,steps+1)
    method=workflow.validate_method_output(text,control['keywd'])
    args=SimpleNamespace(nsteps=steps,dt_fs=dt,probe_count=nprobes)
    data=dict(summary=summary,energy=energy,trajectory=trajectory,checkpoint=checkpoint,properties=properties,gradients=gradients)
    # Reuse the H checks without renaming or writing any retained native file.
    hpath=Path(str(prefix)+'_field_hessian.csv')
    hrows=legacy.read_csv(hpath,['step','time_fs','probe','x_ang','y_ang','z_ang',*workflow.H_COLUMNS])
    require([(int(r['step']),int(r['probe'])) for r in hrows]==identities,'H sample identities differ')
    require(summary.get('field_hessian_calls')==len(sampled),'Wrong direct H call count')
    gpu=summary['backend'].lower() in ('cuda','gpu','hip')
    require(summary.get('field_hessian_master_gpu_launches')==(len(sampled) if gpu else 0),'Wrong direct H GPU launch count')
    require(summary.get('field_hessian_convention')=='H_ijk=d2 E_i/dC_j dC_k=-d3 V/dC_i dC_j dC_k','Wrong H sign convention')
    require(summary.get('field_hessian_units')=='hartree/(e bohr^3); laboratory Cartesian axes','Wrong H units')
    for h,p in zip(hrows,properties):
        require(abs(float(h['time_fs'])-int(h['step'])*dt)<=1e-12,'Wrong H time')
        require(legacy.max_difference([float(h[x+'_ang']) for x in 'xyz'],[float(p[x+'_ang']) for x in 'xyz'])<=1e-11,'H/V/E/G probe positions differ')
    diagnostics=[];telemetry={'available':False,'checked':False,'scope':'Direct reference has no adaptive work'}
    if control.get('adaptive_electrostatics',False):
        require(summary.get('adaptive_electrostatics') is True and summary.get('adaptive_calls')==len(sampled),'Wrong adaptive call count')
        require(summary.get('adaptive_values_rebuilt_each_frame') is True and summary.get('adaptive_total_numeric_budget_known') is False,
                'Adaptive density lifecycle/unknown numerical budget not declared')
        require(summary.get('adaptive_hessian')=='direct','Adaptive path must retain direct H')
        rows=csv_rows(Path(str(prefix)+'_adaptive.csv'),['step','degree','error_mode','fallback','depth','anchors','bounds','max_overlap',
                      'truncation_V','truncation_E','truncation_G','numeric_budget_known','preparation_s','query_s','reason'])
        require([int(row['step']) for row in rows]==sampled,'Adaptive diagnostics miss/reorder frames')
        telemetry=gpu_telemetry(rows,summary,control.get('adaptive_gpu_queries',False),gpu)
        tolerances=np.asarray(control.get('adaptive_tolerances',[1e-6,1e-5,1e-4]),float)
        require(tolerances.shape==(3,) and np.isfinite(tolerances).all() and np.all(tolerances>0),'Invalid adaptive requested tolerances')
        for row in rows:
            require(row['fallback'] in ('T','F') and row['numeric_budget_known']=='F','Wrong fallback/unknown-budget flags')
            fallback=row['fallback']=='T';budget=[]
            for key in ('truncation_V','truncation_E','truncation_G'):
                value=float(row[key]);require(not math.isinf(value) and (math.isnan(value) or value>=0),'Invalid truncation budget')
                require(fallback or math.isfinite(value),'Accepted approximation lacks a finite truncation budget')
                budget.append(value if math.isfinite(value) else None)
            if not fallback:require(np.all(np.array(budget)<=tolerances),'Accepted truncation budget exceeds requested tolerances')
            for key in ('preparation_s','query_s'):
                require(math.isfinite(float(row[key])) and float(row[key])>=0,'Invalid adaptive time')
            for key in ('depth','anchors','bounds','max_overlap'):
                require(int(row[key])>=0,'Invalid adaptive count')
            require(int(row['max_overlap'])<=8 and row['reason'].strip(),'Missing reason/overlap guard')
            diagnostics.append(dict(step=int(row['step']),fallback=fallback,degree=int(row['degree']),error_mode=int(row['error_mode']),
                depth=int(row['depth']),anchors=int(row['anchors']),bounds=int(row['bounds']),max_overlap=int(row['max_overlap']),
                truncation_budget=budget,numerical_budget_known=False,total_numerical_budget=None,
                interpretation='Unavailable numerical/anchor budget; not assumed zero',reason=row['reason']))
        require(summary.get('adaptive_whole_frame_fallbacks')==sum(r['fallback'] for r in diagnostics),'Fallback summary/frames differ')
    else:
        require('adaptive_calls' not in summary and not Path(str(prefix)+'_adaptive.csv').exists(),'Direct run performed adaptive work')
    ranks=legacy.read_csv(Path(str(prefix)+'_kernels.csv'),['rank',*old.COUNT_COLUMNS,*old.LAUNCH_COLUMNS,*old.PHASE_COLUMNS])
    require([int(r['rank']) for r in ranks]==list(range(summary['mpi_ranks'])),'Missing/reordered rank evidence')
    expected_counts=[steps+1,steps+1,len(sampled),len(sampled),0]
    expected_launches=sum(row['fallback'] for row in diagnostics) if diagnostics else len(sampled)
    for row in ranks:
        require(all(int(row[k])==v for k,v in zip(old.COUNT_COLUMNS,expected_counts)),'Rank counter mismatch')
        require(all(int(row[k])==(expected_launches if gpu else 0) for k in old.LAUNCH_COLUMNS),'Direct/fallback native kernel counts differ')
    files={str(path.resolve()):legacy.sha256(path) for path in directory.iterdir() if path.is_file()}
    files.update({str(path):legacy.sha256(path) for path in (geometry_path,probes_path)})
    return dict(directory=str(directory),control=control,data=data,hessian=hrows,adaptive_frames=diagnostics,
                sampled_steps=sampled,probes=probes,source_files=files,method=method,invariant_max_abs=invariant,
                gpu_telemetry=telemetry)


def compare(reference,candidate):
    left,right=reference['control'],candidate['control']
    for key,default in [('keywd',None),('nsteps',10),('dt_fs',.25),('temperature_k',50),('oeprop_stride',1),
                        ('probe_mode','fixed'),('nfrozen',0),('frozen_atoms',None),('anchor_atoms',None),('coordinate_atoms',None)]:
        require(left.get(key,default)==right.get(key,default),'Matched Hamiltonian/dynamics control differs: '+key)
    require(reference['sampled_steps']==candidate['sampled_steps'],'Matched sampled frames differ')
    require(legacy.max_difference(reference['probes'],candidate['probes'])==0,'Matched input probes differ')
    physics=old.compare_physics(reference['data'],candidate['data'])
    hessian=workflow.hessian_difference(reference['hessian'],candidate['hessian'])
    require(max(hessian.values())<=old.TOLERANCES['property_au'],'Direct H changed with adaptive observation')
    diffs=legacy.property_difference(reference['data']['properties'],candidate['data']['properties'])
    frames=[]
    requested=np.asarray(right.get('adaptive_tolerances',[1e-6,1e-5,1e-4]),float)
    for diagnostic in candidate['adaptive_frames']:
        step=diagnostic['step']
        a=[r for r in reference['data']['properties'] if int(r['step'])==step]
        b=[r for r in candidate['data']['properties'] if int(r['step'])==step]
        delta=np.array([[float(y[k])-float(x[k]) for k in legacy.PROPERTY_COLUMNS] for x,y in zip(a,b)])
        norms=np.array([max(abs(delta[:,0])),max(np.linalg.norm(delta[:,1:4],axis=1)),max(np.linalg.norm(delta[:,4:],axis=1))])
        gate=np.full(3,old.TOLERANCES['property_au']) if diagnostic['fallback'] else requested+old.TOLERANCES['property_au']
        require(np.all(norms<=gate),'Framewise direct/adaptive empirical differences exceed diagnostic gates')
        frames.append({**diagnostic,'max_actual_direct_difference_V_E_G_Frobenius':norms.tolist(),'empirical_absolute_gates':gate.tolist(),
                       'gate_scope':'Observed requested probes only; total arithmetic/anchor budget unavailable'})
    accepted=sum(not r['fallback'] for r in frames);fallback=len(frames)-accepted
    return dict(physics=physics,direct_H_max_abs_by_component=hessian,property_max_abs_by_component=diffs,
                framewise=frames,accepted_frames=accepted,fallback_frames=fallback,gpu_telemetry=candidate['gpu_telemetry'],
                interpretation='Accepted finite-region observations plus fallback controls' if accepted else
                               'Whole-frame fallback acceptance only; no approximation acceptance demonstrated')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--direct',type=Path,required=True)
    parser.add_argument('--adaptive',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expect-all-fallback',action='store_true');parser.add_argument('--expect-accepted',action='store_true')
    args=parser.parse_args()
    if args.expect_all_fallback and args.expect_accepted:parser.error('Choose one expected behavior')
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(passed=False,scope=__doc__,native_jobs_launched=0,SCFs=0,GPU_kernels=0,
                physics_gates=old.TOLERANCES,comparisons=[],execution_time_runtime_provenance_inferred=False,
                limitations=['Retrospective retained-file hashes; no historical runtime hash is inferred',
                             'Complete printed gradients, all energy/coordinate frames, and complete final checkpoint velocities checked',
                             'Intermediate velocities are not separately exported by the driver',
                             'Finite requested probes only; no ensemble or continuous total-error certification'])
    loaded=[]
    try:
        reference=load(args.direct);loaded.append(reference)
        require(not reference['control'].get('adaptive_electrostatics',False),'Direct reference enables adaptive observations')
        for path in args.adaptive:
            candidate=load(path);loaded.append(candidate)
            require(candidate['control'].get('adaptive_electrostatics',False),'Candidate does not enable adaptive observations')
            comparison=compare(reference,candidate);comparison['directory']=candidate['directory']
            if args.expect_all_fallback:require(comparison['accepted_frames']==0,'Expected only whole-frame fallback')
            if args.expect_accepted:require(comparison['accepted_frames']>0,'Expected at least one accepted approximation frame')
            report['comparisons'].append(comparison)
        report['retained_evidence_sha256']={path:digest for run in loaded for path,digest in run['source_files'].items()}
        require(all(legacy.sha256(Path(path))==digest for path,digest in report['retained_evidence_sha256'].items()),'Retained evidence changed during read-only analysis')
        report['retained_evidence_unchanged']=True;report['passed']=True
    except Exception as error:
        import traceback
        report.update(error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
    report['analysis_source_sha256']={str(p):legacy.sha256(p) for p in (Path(__file__),HERE.parent/'electrostatics_benchmark/benchmark_workflow.py',
        ROOT/'validation/oeprop_bomd/benchmark_overhead.py',ROOT/'validation/oeprop_bomd/validate_oeprop_bomd.py')}
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=report['passed'],comparisons=len(report['comparisons']),
                         accepted_frames=sum(c['accepted_frames'] for c in report['comparisons']),
                         fallback_frames=sum(c['fallback_frames'] for c in report['comparisons']),error=report.get('error'))))
    return 0 if report['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
