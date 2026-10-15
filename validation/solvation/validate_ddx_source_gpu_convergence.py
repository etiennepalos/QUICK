#!/usr/bin/env python3
"""Matched CPU/CUDA checks for the explicit ddX Coulomb source option."""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import traceback
import validate_solvation as reference


def selected(case: dict, source: str | None) -> dict:
    result=copy.deepcopy(case)
    if source: result['method']+=' DDX_SOURCE='+source
    return result


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--driver',type=Path,required=True)
    ap.add_argument('--reference-driver',type=Path,required=True)
    ap.add_argument('--cpu-report',type=Path,required=True)
    ap.add_argument('--dimer-plan',type=Path,required=True)
    ap.add_argument('--basis',type=Path,required=True)
    ap.add_argument('--workdir',type=Path,required=True)
    ap.add_argument('--launcher',default='')
    ap.add_argument('--timeout',type=float,default=600)
    args=ap.parse_args()
    for key in ('driver','reference_driver','cpu_report','dimer_plan','basis'):
        setattr(args,key,getattr(args,key).resolve(strict=True))
    cpu=json.loads(args.cpu_report.read_text())
    if not cpu['passed']: raise ValueError('CPU molecular validation must pass before matched GPU campaign')
    args.workdir=args.workdir.resolve();args.workdir.mkdir(parents=True,exist_ok=False)
    args.electron_count_tolerance=1e-3;args.density_tolerance=1e-6
    env=dict(os.environ,QUICK_BASIS=str(args.basis),CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    report=dict(scope='Matched finite-grid GPU behavior and published-default controls; no refinement qualification',runtime_before=reference.runtime(args.driver,env),reference_runtime_before=reference.runtime(args.reference_driver,env),cpu_report=str(args.cpu_report),cpu_report_sha256=reference.sha256(args.cpu_report),validator_sha256=reference.sha256(Path(__file__)),runs=[],checks=[],exceptions=[],passed=False,gates=dict(cpu_gpu_energy_hartree=1e-8,cpu_gpu_gradient_hartree_per_bohr=1e-7,old_new_gpu_energy_hartree=1e-10,old_new_gpu_gradient_hartree_per_bohr=1e-9),environment={k:env[k] for k in ('CUDA_VISIBLE_DEVICES','QUICK_BASIS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')})
    def save(name): (args.workdir/name).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save('protocol-before-science.json')
    campaign=reference.Campaign(args,report,env)
    old_args=SimpleNamespace(**vars(args));old_args.driver=args.reference_driver
    old_campaign=reference.Campaign(old_args,report,env)
    grid=dict(lmax=6,nlebedev=302,nrad=50,nang=110)
    water=reference.CASES['water'];dimer=json.loads(args.dimer_plan.read_text())['case']
    try:
        for model in ('COSMO','PCM'):
            old=old_campaign.run(model+'-v5-omitted',water,model,'gradient',grid)
            omitted=campaign.run(model+'-v6-omitted',water,model,'gradient',grid)
            regular=campaign.run(model+'-v6-REGULAR',selected(water,'REGULAR'),model,'gradient',grid)
            for name,left,right in (('v5_vs_omitted',old,omitted),('omitted_vs_REGULAR',omitted,regular)):
                campaign.check(model+':'+name+':energy',abs(left['energy']-right['energy']),1e-10)
                campaign.check(model+':'+name+':gradient',reference.difference(left['gradient_hartree_per_bohr'],right['gradient_hartree_per_bohr']),1e-9)
            save('partial-report.json')
        for label,case in (('water',water),('dimer',dimer)):
            for model in ('COSMO','PCM'):
                tag=label+'-'+model+'-COULOMB'
                gpu=campaign.run(tag+'-base',selected(case,'COULOMB'),model,'gradient',grid)
                original=next(r for r in cpu['runs'] if r['label']==tag+'-base')
                current=next(r for r in report['runs'] if r['label']==tag+'-base')
                campaign.check(tag+':same_input',float(original['input_sha256']!=current['input_sha256']),0.0)
                campaign.check(tag+':cpu_gpu_energy',abs(gpu['energy']-original['result']['energy']),1e-8)
                campaign.check(tag+':cpu_gpu_gradient',reference.difference(gpu['gradient_hartree_per_bohr'],original['result']['gradient_hartree_per_bohr']),1e-7)
                campaign.frozen(tag,selected(case,'COULOMB'),model,grid)
                save('partial-report.json')
    except Exception:
        report['exceptions'].append(traceback.format_exc())
    report['runtime_after']=reference.runtime(args.driver,env);report['reference_runtime_after']=reference.runtime(args.reference_driver,env)
    campaign.check('v6_runtime_unchanged',float(report['runtime_before']!=report['runtime_after']),0.0)
    campaign.check('v5_runtime_unchanged',float(report['reference_runtime_before']!=report['reference_runtime_after']),0.0)
    report['passed']=not report['exceptions'] and bool(report['checks']) and all(c['passed'] for c in report['checks'])
    save('report.json')
    print(json.dumps(dict(passed=report['passed'],checks=len(report['checks']),runs=len(report['runs']),exceptions=report['exceptions'],failed=[c for c in report['checks'] if not c['passed']]),indent=2))
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
