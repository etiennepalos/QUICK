#!/usr/bin/env python3
"""Targeted opt-in ddX source regression, density derivatives and nuclear forces."""
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


def with_source(case: dict, source: str | None) -> dict:
    case = copy.deepcopy(case)
    if source:
        case['method'] += ' DDX_SOURCE=' + source
    return case


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--driver',type=Path,required=True)
    ap.add_argument('--reference-driver',type=Path,required=True)
    ap.add_argument('--basis',type=Path,required=True)
    ap.add_argument('--workdir',type=Path,required=True)
    ap.add_argument('--dimer-plan',type=Path,required=True)
    ap.add_argument('--launcher',default='')
    ap.add_argument('--timeout',type=float,default=600.0)
    args=ap.parse_args()
    args.driver=args.driver.resolve(strict=True)
    args.reference_driver=args.reference_driver.resolve(strict=True)
    args.basis=args.basis.resolve(strict=True)
    args.workdir=args.workdir.resolve()
    args.workdir.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,QUICK_BASIS=str(args.basis),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    args.electron_count_tolerance=1e-3
    args.density_tolerance=1e-6
    args.steps=[1e-3,5e-4]
    grid=dict(lmax=6,nlebedev=302,nrad=50,nang=110)
    water=reference.CASES['water']
    dimer=json.loads(args.dimer_plan.read_text())['case']
    report=dict(scope='Finite-grid derivative and default-preservation tests; separate refinement is required for continuum qualification',grid=grid,steps_bohr=args.steps,dimer_coordinates=[dict(atom=0,axis=0),dict(atom=1,axis=1),dict(atom=3,axis=1)],runtime_before=reference.runtime(args.driver,env),reference_runtime_before=reference.runtime(args.reference_driver,env),runs=[],checks=[],exceptions=[],passed=False,validator_sha256=reference.sha256(Path(__file__)),reference_validator_sha256=reference.sha256(Path(reference.__file__)),environment={k:env[k] for k in ('QUICK_BASIS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')})
    def save(name: str) -> None:
        (args.workdir/name).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save('protocol-before-science.json')
    campaign=reference.Campaign(args,report,env)
    old_args=SimpleNamespace(**vars(args));old_args.driver=args.reference_driver
    old_campaign=reference.Campaign(old_args,report,env)
    try:
        for model in ('COSMO','PCM'):
            old=old_campaign.run(f'{model}-v5-omitted',water,model,'gradient',grid)
            omitted=campaign.run(f'{model}-v6-omitted',water,model,'gradient',grid)
            regular=campaign.run(f'{model}-v6-REGULAR',with_source(water,'REGULAR'),model,'gradient',grid)
            for name,left,right in (('v5_vs_omitted',old,omitted),('omitted_vs_REGULAR',omitted,regular)):
                campaign.check(model+':'+name+':energy_exact',abs(left['energy']-right['energy']),0.0)
                campaign.check(model+':'+name+':gradient_exact',reference.difference(left['gradient_hartree_per_bohr'],right['gradient_hartree_per_bohr']),0.0)
            save('partial-report.json')
        for model in ('COSMO','PCM'):
            case=with_source(water,'COULOMB')
            name='water-'+model+'-COULOMB'
            base=campaign.run(name+'-base',case,model,'gradient',grid)
            campaign.frozen(name,case,model,grid)
            campaign.nuclear_fd(name,case,model,grid,base,1e-5)
            save('partial-report.json')
            campaign.frozen('water-cation-'+model+'-COULOMB',with_source(reference.CASES['water_cation'],'COULOMB'),model,grid)
            save('partial-report.json')
        for model in ('COSMO','PCM'):
            case=with_source(dimer,'COULOMB')
            base=campaign.run('dimer-'+model+'-COULOMB-base',case,model,'gradient',grid)
            campaign.frozen('dimer-'+model+'-COULOMB',case,model,grid)
            for step in args.steps:
                errors=[]
                numeric=[]
                for atom,axis in ((0,0),(1,1),(3,1)):
                    values=[]
                    for sign in (1,-1):
                        shifted=copy.deepcopy(case)
                        shifted['atoms'][atom][1][axis]+=sign*step*reference.BOHRS_TO_A
                        label=f'dimer-{model}-a{atom+1}{"xyz"[axis]}-h{step:g}-s{sign}'
                        values.append(campaign.run(label,shifted,model,'energy',grid)['energy'])
                    fd=(values[0]-values[1])/(2*step)
                    numeric.append(dict(atom=atom,axis=axis,finite_difference=fd,analytic=base['gradient_hartree_per_bohr'][3*atom+axis]))
                    errors.append(abs(fd-base['gradient_hartree_per_bohr'][3*atom+axis]))
                campaign.check(f'dimer-{model}:selected_gradient_fd:h{step:g}',max(errors),1e-5,components=numeric)
                save('partial-report.json')
    except Exception:
        report['exceptions'].append(traceback.format_exc())
    report['runtime_after']=reference.runtime(args.driver,env)
    report['reference_runtime_after']=reference.runtime(args.reference_driver,env)
    campaign.check('v6_runtime_unchanged',float(report['runtime_before']!=report['runtime_after']),0.0)
    campaign.check('v5_runtime_unchanged',float(report['reference_runtime_before']!=report['reference_runtime_after']),0.0)
    report['passed']=not report['exceptions'] and bool(report['checks']) and all(c['passed'] for c in report['checks'])
    save('report.json')
    print(json.dumps(dict(passed=report['passed'],checks=len(report['checks']),runs=len(report['runs']),exceptions=report['exceptions'],failed=[c for c in report['checks'] if not c['passed']]),indent=2))
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
