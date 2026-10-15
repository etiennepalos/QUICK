#!/usr/bin/env python3
"""Predeclared eight-case CPU ROI viability pilot, retaining every outcome.

This is workload-dependence evidence, not a change of physics or tolerances
chosen after inspecting individual results. No GPU or10000-query runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-root',type=Path,required=True)
    p.add_argument('--compiler',type=Path,required=True)
    p.add_argument('--ddx-lib',type=Path,required=True)
    p.add_argument('--fixture-root',type=Path,required=True)
    p.add_argument('--workdir',type=Path,required=True)
    a=p.parse_args()
    a.workdir.mkdir(parents=True,exist_ok=False)
    benchmark=Path(__file__).with_name('benchmark_adaptive.py')
    driver=Path(__file__).with_name('quick_adaptive_benchmark.f90')
    frozen=a.fixture_root/'water-tzvpd-1000-cpu'
    plan={'scope':'predeclared64AO water cubic analytic-majorant CPU ROI dependence; noGPU/no10000query',
        'half_widths_bohr':[.025,.1,.25,.5],
        'tolerance_triplets':[[1e-5,1e-4,1e-3],[1e-6,1e-5,1e-4]],
        'points':1000,'repeats_after_first':1,'degree':3,'error_mode':'analytic_majorant',
        'caps':{'depth':3,'cumulative_anchors':600,'preparation_seconds':2,'memory_mb':256,'overlap':8},
        'pilot_wall_cap_seconds':120,'max_small_SCFs':8,
        'same_density_sha256':sha(frozen/'density.dat'),'same_native_AOs_sha256':sha(frozen/'ao_basis.dat'),
        'original_nested_query_input_sha256':sha(frozen/'input.dat'),
        'nested_query_seed':50271,'center':'same preserved exterior cube center for every ROI',
        'domain':'explicit fixed lower=center-halfwidth and upper=center+halfwidth, including query interpolation',
        'source_sha256':{q.name:sha(q) for q in [Path(__file__),benchmark,driver]},
        'runtime_library_sha256':sha(a.runtime_root/'cpu/install/lib/libquick.so')}
    plan_path=a.workdir/'predeclared-plan.json'
    plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    plan_hash=sha(plan_path)
    env=os.environ.copy()
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):env[k]='1'
    rows=[];start=time.monotonic()
    for index,halfwidth in enumerate(plan['half_widths_bohr']):
        for tolerance_index,tolerance in enumerate(plan['tolerance_triplets']):
            name=f'roi-{index+1}-tolerance-{tolerance_index+1}'
            folder=a.workdir/name
            cmd=[sys.executable,'-B',str(benchmark),'--runtime-root',str(a.runtime_root),'--backend','cpu',
                '--compiler',str(a.compiler),'--ddx-lib',str(a.ddx_lib),'--fixture-root',str(a.fixture_root),
                '--case','water-tzvpd','--counts','1000','--prepare-seconds','2','--repeats','1',
                '--configs','target','--roi-half-width',str(halfwidth),'--target-tolerances',*map(str,tolerance),
                '--workdir',str(folder)]
            row={'name':name,'half_width_bohr':halfwidth,'requested_tolerances':tolerance,'argv':cmd}
            remaining=plan['pilot_wall_cap_seconds']-(time.monotonic()-start)
            if remaining<=0:
                row['status']='NOT_RUN_PILOT_WALL_CAP'
            else:
                try:
                    tick=time.monotonic()
                    process=subprocess.Popen(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,start_new_session=True)
                    stdout,stderr=process.communicate(timeout=remaining)
                    (a.workdir/(name+'.stdout')).write_text(stdout)
                    (a.workdir/(name+'.stderr')).write_text(stderr)
                    row.update(returncode=process.returncode,wall_seconds=time.monotonic()-tick)
                    if process.returncode==0:
                        native=json.loads((folder/'report.json').read_text())
                        row.update(status=native['status'],report_sha256=sha(folder/'report.json'),
                            counts=native['counts'],result=native['runs'][0])
                    else:row['status']='FAILED_NATIVE_OR_VALIDATION'
                except subprocess.TimeoutExpired as ex:
                    os.killpg(process.pid,signal.SIGTERM)
                    stdout,stderr=process.communicate()
                    (a.workdir/(name+'.stdout')).write_text(stdout)
                    (a.workdir/(name+'.stderr')).write_text(stderr)
                    row['status']='FAILED_PILOT_WALL_CAP'
            rows.append(row)
            (a.workdir/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(json.dumps({'name':name,'half_width_bohr':halfwidth,'tolerances':tolerance,'status':row['status'],
                'fallback':row.get('result',{}).get('adaptive_report',{}).get('fallback')}),flush=True)
    stable=(sha(plan_path)==plan_hash and plan['runtime_library_sha256']==sha(a.runtime_root/'cpu/install/lib/libquick.so') and
        all(plan['source_sha256'][q.name]==sha(q) for q in [Path(__file__),benchmark,driver]))
    passed=stable and len(rows)==8 and all(row['status']=='PASS' for row in rows)
    report={'schema':1,'status':'PASS' if passed else 'FAIL','predeclared_plan_sha256':plan_hash,
        'predeclared_plan':plan,'all_regions_included':len(rows)==8,'source_runtime_plan_stable':stable,
        'elapsed_seconds':time.monotonic()-start,'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'rows':rows,'interpretation':'Acceptance and fallback are results; validationPASS does not mean every ROI meets its target.',
        'limits':['ordinaryFP64 full numeric error budget unknown; no certification',
            'single frozen density/exterior center; ROI dependence cannot establish a general speedup',
            'noGPU or10000query experiment; initialSCF excluded from propertyAPI times but included in processpeakRSS']}
    (a.workdir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'elapsed_seconds':report['elapsed_seconds'],'report':str(a.workdir/'report.json')}))
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
