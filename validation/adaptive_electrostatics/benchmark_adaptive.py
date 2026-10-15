#!/usr/bin/env python3
"""Matched retained-density native VEG versus complete adaptive API calls.

One configuration per process; /usr/bin/time peak RSS includes SCF setup,
direct and adaptive calls in that process. It is not a per-phase RSS delta.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time

import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def parse_table(p):return {line.split()[0]:line.split()[1:] for line in p.read_text().splitlines()}


def coordinate_sha(points):
    return hashlib.sha256(np.ascontiguousarray(points,dtype='<f8').tobytes()).hexdigest()


def query_input(fixture_root,case,count,original,natom,boxcenter,halfwidth,allow_100k):
    """Read saved1k/10k inputs or generate opt-in100k with exact prefixes.

    This function performs no native call and never writes into fixture_root.
    Prefix equality is required before returning a generated request.
    """
    metadata={'mode':'saved','allow_100k':allow_100k,'prefix_checks':[]}
    if count==100000:
        if not allow_100k:raise ValueError('100000queries require explicit --allow-100k')
        rng=np.random.default_rng(50271)
        points=boxcenter+.5*rng.uniform(-1.,1.,(count,3))
        lines=[original[0],f'{natom} 0 {count}',*original[2:2+natom]]
        metadata.update(mode='generated',seed=50271,bit_generator=type(rng.bit_generator).__name__,
            numpy_version=np.__version__,coordinate_hash_encoding='contiguous little-endian FP64 xyz',
            formula='boxcenter + .5*default_rng(50271).uniform(-1,1,(count,3))')
        for size in (1000,10000):
            path=fixture_root/f'{case}-{size}-cpu/input.dat'
            saved_lines=path.read_text().splitlines()
            if saved_lines[:2+natom]!=[original[0],f'{natom} 0 {size}',*original[2:2+natom]]:
                raise ValueError('Frozen prefix geometry/keyword changed')
            saved=np.array([list(map(float,line.split())) for line in saved_lines[2+natom:]])
            if saved.shape!=(size,3) or not np.array_equal(points[:size],saved):
                raise ValueError(f'Generated unscaled{size}-point prefix differs from frozen input')
            scaled=boxcenter+(saved-boxcenter)*(halfwidth/.5) if halfwidth!=.5 else saved
            generated_scaled=boxcenter+(points[:size]-boxcenter)*(halfwidth/.5) if halfwidth!=.5 else points[:size]
            if not np.array_equal(generated_scaled,scaled):
                raise ValueError(f'Generated scaled{size}-point prefix differs from frozen input')
            metadata['prefix_checks'].append({'points':size,'passed':True,'saved_input_sha256':sha(path),
                'unscaled_coordinates_sha256':coordinate_sha(saved),
                'generated_unscaled_prefix_sha256':coordinate_sha(points[:size]),
                'scaled_coordinates_sha256':coordinate_sha(scaled),
                'generated_scaled_prefix_sha256':coordinate_sha(generated_scaled)})
        metadata['unscaled_coordinates_sha256']=coordinate_sha(points)
        text='\n'.join(lines)+'\n'+''.join(' '.join(format(x,'.17g') for x in point)+'\n' for point in points)
    else:
        reference=fixture_root/f'{case}-{count}-cpu/input.dat'
        text=reference.read_text();lines=text.splitlines()
        if lines[:2+natom]!=[original[0],f'{natom} 0 {count}',*original[2:2+natom]]:
            raise ValueError('Geometry/keyword changed')
        points=np.array([list(map(float,line.split())) for line in lines[2+natom:]])
        metadata.update(saved_input_path=str(reference),saved_input_sha256=sha(reference))
    if halfwidth!=.5:
        points=boxcenter+(points-boxcenter)*(halfwidth/.5)
        text='\n'.join(lines[:2+natom])+'\n'+''.join(' '.join(format(x,'.17g') for x in point)+'\n' for point in points)
    metadata['returned_coordinates_sha256']=coordinate_sha(points)
    return text,points,metadata


def runtime(exe,env):
    r=subprocess.run(['ldd',str(exe)],env=env,text=True,capture_output=True,check=True)
    if 'not found' in r.stdout:raise ValueError('Unresolved runtime dependency')
    paths={}
    for line in r.stdout.splitlines():
        match=re.search(r'(?:=>\s*)?(/\S+)\s+\(',line)
        if match:
            p=Path(match[1]).resolve(strict=True);paths[str(p)]=sha(p)
    return paths


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-root',type=Path,required=True)
    p.add_argument('--backend',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--gpu-queries',action='store_true',
        help='Explicitly select options%%gpu_queries; requires a runtime containing this control')
    p.add_argument('--compiler',type=Path,required=True)
    p.add_argument('--ddx-lib',type=Path,required=True)
    p.add_argument('--fixture-root',type=Path,required=True,
        help='Preserved direct-benchmark directory with frozen native density/basis')
    p.add_argument('--case',choices=['water-sto3g','water-tzvpd'],default='water-sto3g')
    p.add_argument('--workdir',type=Path,required=True)
    p.add_argument('--counts',type=int,nargs='+',default=[1000,10000])
    p.add_argument('--allow-100k',action='store_true',
        help='Explicit headroom opt-in for100000queries with mandatory saved1k/10k prefix equality')
    p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--prepare-seconds',type=float,default=2.)
    p.add_argument('--roi-half-width',type=float,default=.5,
        help='Explicit cubic fixed-domain halfwidth in bohr; scales the preserved nested exterior probes')
    p.add_argument('--target-tolerances',type=float,nargs=3,default=[1e-6,1e-5,1e-4],
        metavar=('V','E','G'),help='Predeclared analytic target triplet; recorded with every run')
    p.add_argument('--timeout',type=float,default=90.)
    p.add_argument('--configs',nargs='+',choices=['quadratic-loose','cubic-loose','indicator-middle','target','anchor-cap'],
        default=['quadratic-loose','cubic-loose','indicator-middle','target','anchor-cap'])
    a=p.parse_args()
    allowed_counts=(1000,10000,100000) if a.allow_100k else (1000,10000)
    if not 1<=a.repeats<=100 or len(set(a.counts))!=len(a.counts) or any(n not in allowed_counts for n in a.counts):
        p.error('counts must be unique1k/10k;100k requires --allow-100k; larger counts are unsupported')
    if not np.isfinite(a.prepare_seconds) or a.prepare_seconds<=0:p.error('positive finite prep cap required')
    if not np.isfinite(a.roi_half_width) or a.roi_half_width<=0:p.error('positive finite ROI halfwidth required')
    if not np.all(np.isfinite(a.target_tolerances)) or min(a.target_tolerances)<=0:p.error('positive finite target triplet required')
    a.workdir.mkdir(parents=True,exist_ok=False)
    repro=a.workdir/'reproduction';repro.mkdir()
    here=Path(__file__).resolve().parent
    driver=here/'quick_adaptive_benchmark.f90'
    validation_sources_before={s.name:sha(s) for s in (Path(__file__),driver)}
    for s in (Path(__file__),driver):shutil.copy2(s,repro/s.name)
    source=a.runtime_root/'source'
    supports_gpu_queries='logical :: gpu_queries=' in (source/'src/modules/quick_adaptive_electrostatics_module.f90').read_text()
    if a.gpu_queries and not supports_gpu_queries:p.error('Selected runtime predates explicit gpu_queries')
    if a.gpu_queries and a.backend!='cuda':p.error('Explicit GPU query benchmark requires --backend cuda')
    source_hashes={str(p.relative_to(source)):sha(p) for folder in ('src','quick-cmake')
        for p in sorted((source/folder).rglob('*')) if p.is_file()}
    source_hashes['CMakeLists.txt']=sha(source/'CMakeLists.txt')
    install=a.runtime_root/a.backend/'install'
    env=os.environ.copy()
    env.update(QUICK_BASIS=str(install/'basis'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    env['PATH']=str(a.compiler.parent)+':'+env.get('PATH','')
    if a.backend=='cuda':env['CUDA_VISIBLE_DEVICES']='0'
    exe=a.workdir/('benchmark-'+a.backend)
    module_folder='cuda' if a.backend=='cuda' else 'serial'
    compile_cmd=[str(a.compiler),'-cpp','-ffree-line-length-none','-I'+str(source/'src/util'),
        '-I'+str(install/'include/amber-modules/quick'/module_folder),str(repro/driver.name),'-L'+str(install/'lib'),
        '-Wl,-rpath,'+str(install/'lib'),'-Wl,-rpath,'+str(a.ddx_lib),'-Wl,-rpath-link,'+str(a.ddx_lib),
        '-lquick_cuda' if a.backend=='cuda' else '-lquick','-o',str(exe)]
    if supports_gpu_queries:compile_cmd.insert(2,'-DADAPTIVE_GPU_QUERIES')
    result=subprocess.run(compile_cmd,text=True,capture_output=True,env=env)
    (a.workdir/'compile.stdout').write_text(result.stdout);(a.workdir/'compile.stderr').write_text(result.stderr)
    result.check_returncode()
    before=runtime(exe,env)
    frozen=a.fixture_root/f'{a.case}-1000-cpu'
    density=frozen/'density.dat';native_basis=frozen/'ao_basis.dat'
    for p in (density,native_basis):p.resolve(strict=True)
    # The preserved reference defines a nested one-bohr-wide exterior cube.
    # Use its exact saved inputs, density and native AO normalization/order.
    original=(frozen/'input.dat').read_text().splitlines()
    natom,ncharge,_=map(int,original[1].split())
    if ncharge:raise ValueError('Initial profile must be isolated all-electron water')
    geometry=np.array([list(map(float,line.split()[1:])) for line in original[2:2+natom]])/.52917720859
    center=geometry.mean(axis=0)
    boxcenter=center+np.array([0.,0.,np.max(np.linalg.norm(geometry-center,axis=1))+3.])
    lower,upper=boxcenter-a.roi_half_width,boxcenter+a.roi_half_width
    configurations={
        'quadratic-loose':{'degree':2,'mode':2,'tolerances':[1e3,1e3,1e3]},
        'cubic-loose':{'degree':3,'mode':2,'tolerances':[1e3,1e3,1e3]},
        'indicator-middle':{'degree':3,'mode':1,'tolerances':[1e-3,1e-2,1e-1]},
        'target':{'degree':3,'mode':2,'tolerances':a.target_tolerances},
        'anchor-cap':{'degree':3,'mode':2,'tolerances':a.target_tolerances,'max_anchors':1},
    }
    checks=[];runs=[];start=time.perf_counter()
    def check(name,ok,**details):checks.append({'name':name,'passed':bool(ok),**details})
    build_report=a.runtime_root/'build-report.json'
    manifest_path=a.runtime_root/'source-manifest.json'
    if build_report.is_file() and manifest_path.is_file():
        build=json.loads(build_report.read_text());manifest=json.loads(manifest_path.read_text())
        check('build-source-manifest',all(manifest.get(k)==v for k,v in source_hashes.items()))
    points_previous=None
    for count in sorted(a.counts):
        text,points,query_metadata=query_input(a.fixture_root,a.case,count,original,natom,boxcenter,
            a.roi_half_width,a.allow_100k)
        for prefix in query_metadata['prefix_checks']:
            check(f'{count}:generated-prefix{prefix["points"]}',prefix['passed'])
        if points.shape!=(count,3) or np.any(points<lower) or np.any(points>upper):raise ValueError('Malformed exterior probes')
        if points_previous is not None:check('nested-probes',np.array_equal(points[:len(points_previous)],points_previous))
        points_previous=points
        for name in a.configs:
            config=dict(configurations[name]);config.update(max_depth=3,max_overlap=8,prepare_seconds=a.prepare_seconds,
                memory_mb=256.,indicator_factor=2.,lower_bohr=lower.tolist(),upper_bohr=upper.tolist(),gpu_queries=a.gpu_queries)
            config.setdefault('max_anchors',600)
            folder=a.workdir/f'{count}-{name}';folder.mkdir()
            (folder/'input.dat').write_text(text)
            (folder/'options.json').write_text(json.dumps(config,indent=2)+'\n')
            options=f"{config['degree']} {config['mode']} {config['max_depth']} {config['max_anchors']} {config['max_overlap']}\n"
            if supports_gpu_queries:options=options.rstrip()+(' T\n' if a.gpu_queries else ' F\n')
            options+=' '.join(format(x,'.17e') for x in [*config['tolerances'],a.prepare_seconds,256.,2.])+'\n'
            options+=' '.join(format(x,'.17e') for x in [*lower,*upper])+'\n'
            (folder/'options.dat').write_text(options)
            command=[str(exe),'input.dat',str(density.resolve()),'options.dat',str(a.repeats)]
            rss=folder/'peak-rss-kb.txt'
            timed_command=['/usr/bin/time','-f','%M','-o',str(rss),*command] if Path('/usr/bin/time').is_file() else command
            prelaunch={'scope':'matched retained-density VEG/API request, recorded before native execution',
                'case':a.case,'points':count,'backend':a.backend,'gpu_queries_requested':a.gpu_queries,
                'query_generation':query_metadata,'allow_100k':a.allow_100k,
                'options':config,'argv':timed_command,'repeats_after_first':a.repeats,
                'request_script_and_Fortran_source_sha256':validation_sources_before,
                'driver_sha256':sha(exe),'input_sha256':sha(folder/'input.dat'),
                'options_sha256':sha(folder/'options.dat'),'frozen_density_sha256':sha(density),
                'frozen_AOs_sha256':sha(native_basis),'runtime_libraries_sha256':before}
            (folder/'prelaunch-plan.json').write_text(json.dumps(prelaunch,indent=2)+'\n')
            tick=time.perf_counter()
            with (folder/'execution.log').open('w') as log:
                r=subprocess.run(timed_command,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout)
            process_wall=time.perf_counter()-tick
            if r.returncode:raise RuntimeError(f'Native profile failed: {folder}')
            state=parse_table(folder/'result.dat')
            check(f'{count}-{name}:state',state['converged']==['T'] and max(map(float,state['state_errors']))==0.)
            check(f'{count}-{name}:identical-D-AOs',sha(folder/'density.dat')==sha(density) and sha(folder/'ao_basis.dat')==sha(native_basis))
            with (folder/'timings.csv').open() as f:rows=list(csv.DictReader(f))
            check(f'{count}-{name}:complete-timings',len(rows)==2*(a.repeats+1))
            phases={}
            for phase in ('direct','adaptive'):
                selected=[x for x in rows if x['route']==phase]
                samples=[float(x['wall_seconds']) for x in selected[1:]]
                phases[phase]={'first_seconds':float(selected[0]['wall_seconds']),'samples_seconds':samples,
                    'median_seconds':statistics.median(samples)}
                if phase=='adaptive':
                    for iteration,row in enumerate(selected):
                        unknown=all(np.isnan(float(row['total_budget_'+x])) for x in ('V','E','G'))
                        check(f'{count}-{name}:unknown-budget{iteration}',row['certified']=='F' and row['numerical_budget_known']=='F' and unknown)
                        check(f'{count}-{name}:device-cleanup{iteration}',int(row['device_live_bytes'])==0)
                        if a.gpu_queries and row['accepted']=='T':
                            check(f'{count}-{name}:native-GPU-query{iteration}',int(row['gpu_query_launches'])>=1 and
                                state['adaptive_backends']==['cpu','cuda'])
                        elif not a.gpu_queries:
                            check(f'{count}-{name}:no-GPU-query{iteration}',int(row['gpu_query_launches'])==0)
                        error=np.array([float(row['error_'+x]) for x in ('V','E','G')])
                        budget=np.array([float(row['budget_'+x]) for x in ('V','E','G')])
                        if row['fallback']=='T' and a.backend=='cpu':
                            check(f'{count}-{name}:fallback-exact{iteration}',max(error)==0.)
                        elif row['fallback']=='T':
                            check(f'{count}-{name}:fallback-diagnostics-finite{iteration}',np.all(np.isfinite(error)) and np.all(error>=0))
                        elif config['mode']==2:check(f'{count}-{name}:truncation{iteration}',
                            np.all(budget<=config['tolerances']) and np.all(error<=budget+2e-8))
            data=np.loadtxt(folder/'properties.dat')
            check(f'{count}-{name}:all-query-output',data.shape==(count,29) and np.all(np.isfinite(data)) and
                np.array_equal(points,data[:,:3]))
            last=[x for x in rows if x['route']=='adaptive'][-1]
            # Zero-based columns in the complete native29-column table:
            # xyz0:3,adaptiveV3,adaptiveE4:7,adaptiveG7:16 (Fortran column-major), directV16,
            # directE17:20,directG20:29. RMS vector/tensor errors are RMS
            # of the pointwise Euclidean/Frobenius norm, not per-component RMS.
            pointwise_errors={'V':np.abs(data[:,3]-data[:,16]),
                'E':np.linalg.norm(data[:,4:7]-data[:,17:20],axis=1),
                'G':np.linalg.norm(data[:,7:16]-data[:,20:29],axis=1)}
            errors_max={k:float(np.max(v)) for k,v in pointwise_errors.items()}
            errors_rms={k:float(np.sqrt(np.mean(v*v))) for k,v in pointwise_errors.items()}
            fallback_component_gate=None
            if last['fallback']=='T' and a.backend=='cuda':
                component_errors=np.abs(data[:,3:16]-data[:,16:29])
                thresholds=1e-10+1e-9*np.abs(data[:,16:29])
                fallback_component_gate={'absolute':1e-10,'relative':1e-9,
                    'formula':'abs(adaptive-direct)<=1e-10+1e-9*abs(direct)',
                    'scope':'everyV/E/G component at all queries in final complete properties.dat; per-iteration CSV norm maxima retained',
                    'maximum_scaled_error':float(np.max(component_errors/thresholds))}
                check(f'{count}-{name}:CUDA-fallback-componentwise-gate',np.all(component_errors<=thresholds),
                    **fallback_component_gate)
            check(f'{count}-{name}:all-query-max-errors',all(np.isclose(errors_max[k],float(last['error_'+k]),
                rtol=1e-12,atol=1e-14) for k in ('V','E','G')))
            runs.append({'points':count,'configuration':name,'options':config,'argv':timed_command,
                'process_wall_seconds':process_wall,'process_peak_rss_kb':int(rss.read_text()) if rss.is_file() else None,
                'phases':phases,'adaptive_report':{k:(None if 'NaN' in v else v) for k,v in last.items()},
                'reason':' '.join(state['adaptive_reason']),'anchor_query_backends':state['adaptive_backends'],
                'all_query_error_metrics':{'max':errors_max,'RMS':errors_rms},
                'CUDA_fallback_componentwise_gate':fallback_component_gate,'query_generation':query_metadata,
                'prelaunch_plan_sha256':sha(folder/'prelaunch-plan.json'),
                'whole_API_speedup_vs_direct':phases['direct']['median_seconds']/phases['adaptive']['median_seconds'],
                'files_sha256':{p.name:sha(p) for p in folder.iterdir() if p.is_file()}})
            print(json.dumps({'points':count,'configuration':name,'fallback':last['fallback'],
                'direct_seconds':phases['direct']['median_seconds'],'adaptive_seconds':phases['adaptive']['median_seconds']}),flush=True)
    after=runtime(exe,env)
    check('runtime-stability',before==after)
    source_after={name:sha(source/name) for name in source_hashes}
    check('source-stability',source_after==source_hashes)
    check('validation-source-stability',validation_sources_before=={p.name:sha(p) for p in (Path(__file__),driver)})
    report={'schema':1,'status':'PASS' if all(x['passed'] for x in checks) else 'FAIL',
        'scope':'matched complete nativeVEG/API profile; ordinary FP64, no certified total error or production speed claim',
        'case':a.case,'backend':a.backend,'gpu_queries_requested':a.gpu_queries,'allow_100k':a.allow_100k,'roi_half_width_bohr':a.roi_half_width,
        'target_tolerances':a.target_tolerances,'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'counts':{'passed':sum(x['passed'] for x in checks),'total':len(checks)},'checks':checks,'runs':runs,
        'elapsed_seconds':time.perf_counter()-start,'compile_argv':compile_cmd,'driver_sha256':sha(exe),
        'validation_source_sha256':validation_sources_before,'runtime_libraries_sha256':before,
        'native_properties_columns_zero_based':{'xyz':[0,1,2],'adaptive_V':3,'adaptive_E':[4,5,6],
            'adaptive_G_column_major':list(range(7,16)),'direct_V':16,'direct_E':[17,18,19],
            'direct_G_column_major':list(range(20,29))},
        'RMS_error_definition':{'V':'sqrt(mean(deltaV**2))','E':'sqrt(mean(sum(deltaE_i**2)))',
            'G':'sqrt(mean(sum(deltaG_ij**2)))'},
        'fallback_validation_gates':{'cpu':'exactzero for every recorded iteration norm maximum',
            'cuda':{'absolute':1e-10,'relative':1e-9,'formula':'abs(adaptive-direct)<=atol+rtol*abs(direct)',
                'scope':'all13V/E/G components of final retained properties.dat at every query; iteration maxima retained as diagnostics'}},
        'production_source_root':str(source),'production_source_sha256':source_hashes,
        'frozen_density':{'path':str(density),'sha256':sha(density)},'frozen_AOs_sha256':sha(native_basis),
        'limits':['peakRSS is per process including initial SCF/direct/adaptive, not phase-delta or cumulative RUSAGE',
            'initial SCF excluded from property API times; all adaptive preparation/bounds/query/fallback included',
            'loose controls demonstrate implementation behavior; target controls may select direct fallback',
            'one frozen water density and exterior cube; no condensed-phase or general speedup claim',
            '100k requires explicit headroom opt-in and exact saved1k/10k prefixes; larger counts are rejected']}
    (a.workdir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'counts':report['counts'],'report':str(a.workdir/'report.json')}))
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
