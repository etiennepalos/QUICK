#!/usr/bin/env python3
"""Check strict source parsing, cache invalidation, dependency-off and MPI behavior."""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
import traceback
from types import SimpleNamespace

import validate_solvation as reference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clients', type=Path, required=True)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--molecular-report', type=Path, required=True)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--launcher', default='taskset -c 30')
    parser.add_argument('--mpirun', type=Path, required=True)
    args = parser.parse_args()
    for key in ('clients', 'prefix', 'molecular_report', 'mpirun'):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    args.workdir = args.workdir.resolve()
    args.workdir.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               BLIS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               QUICK_BASIS=str(args.prefix/'cpu/install/basis'))
    report = dict(schema=1, scope=__doc__, runs=[], checks=[], exceptions=[], passed=False,
                  validator_sha256=reference.sha256(Path(__file__)),
                  reference_report_sha256=reference.sha256(args.molecular_report),
                  runtimes_before={}, runtimes_after={},
                  environment={k: env[k] for k in ('QUICK_BASIS', 'OMP_NUM_THREADS',
                      'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})

    def save(name='report.json'):
        (args.workdir/name).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')

    def check(name, error, tolerance=0.0, **details):
        report['checks'].append(dict(name=name, error=error, tolerance=tolerance,
                                    passed=abs(error)<=tolerance, **details))

    def run(label, driver, extra=(), text=None, mpi=False):
        folder=args.workdir/label
        folder.mkdir(exist_ok=False)
        if text is not None:
            (folder/'continuum.dat').write_text(text)
        command=shlex.split(args.launcher)
        if mpi:
            command += [str(args.mpirun), '--oversubscribe', '-np', '2', '--bind-to', 'none']
        command += [str(driver), *extra]
        start=time.monotonic()
        result=subprocess.run(command, cwd=folder, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, timeout=120)
        (folder/'execution.log').write_text(result.stdout)
        report['runs'].append(dict(label=label,command=command,returncode=result.returncode,
                                  wall_seconds=time.monotonic()-start,
                                  log_sha256=reference.sha256(folder/'execution.log')))
        if result.returncode:
            raise RuntimeError(f'{label}: returncode {result.returncode}')
        return folder, result.stdout

    names=('source-parser-cpu','source-parser-default-off','source-cache-cpu','source-mpi-cpu',
           'solvation-api-cpu','solvation-api-default-off')
    clients={name:args.clients/name for name in names}
    for name, client in clients.items():
        report['runtimes_before'][name]=reference.runtime(client, env)
    save('protocol-before-science.json')
    grid=dict(lmax=6,nlebedev=302,nrad=50,nang=110)
    try:
        for name in ('source-parser-cpu','source-parser-default-off'):
            _,log=run(name,clients[name])
            match=re.search(r'source_parser_checks\s+(\d+)\s+failures\s+(\d+)',log)
            if not match:
                raise ValueError('Missing parser check summary')
            check(name+':failures',int(match[2]),cases=int(match[1]))
        for model in ('COSMO','PCM'):
            folder,_=run('cache-'+model,clients['source-cache-cpu'],['continuum.dat'],
                reference.data_text(reference.CASES['water'],model,'energy',grid,78.3553,False))
            values={line.split()[0]:float(line.split()[1]) for line in (folder/'source_cache.dat').read_text().splitlines()}
            check(model+':cache_restored_energy',values['restored_energy_error'],1e-9)
            check(model+':cache_restored_matrix',values['restored_matrix_error'],1e-8)
            check(model+':cache_changed',float(values['changed_matrix']<=1e-10),values=values)
            for key in ('density_change','fock_change','stored_energy_change','api_step_change'):
                check(model+':cache_'+key,values[key])
        baseline=json.loads(args.molecular_report.read_text())
        for model in ('COSMO','PCM'):
            folder,_=run('mpi-'+model,clients['source-mpi-cpu'],[model],mpi=True)
            values={}; gradient=[]
            for line in (folder/'source_mpi.dat').read_text().splitlines():
                words=line.split()
                if words[0]=='gradient': gradient += [float(x) for x in words[2:]]
                else: values[words[0]]=[float(x) for x in words[1:]]
            expected=next(x['result'] for x in baseline['runs'] if x['label']=='water-'+model+'-COULOMB-base')
            check(model+':mpi_source',max(abs(x-2) for x in values['source_minmax']))
            check(model+':mpi_serial_energy',values['energy'][0]-expected['energy'],1e-9)
            check(model+':mpi_serial_gradient',reference.difference(gradient,expected['gradient_hartree_per_bohr']),1e-8)
            report['runs'][-1]['result']=dict(energy=values['energy'][0],gradient_hartree_per_bohr=gradient,
                                            source_minmax=values['source_minmax'])
        gas=[]
        for name in ('solvation-api-cpu','solvation-api-default-off'):
            local=SimpleNamespace(driver=clients[name],workdir=args.workdir/name,launcher=args.launcher,
                                  timeout=120,electron_count_tolerance=1e-3)
            local.workdir.mkdir()
            subreport=dict(runtime_before=report['runtimes_before'][name],runs=[],checks=[])
            campaign=reference.Campaign(local,subreport,env)
            gas.append(campaign.run('gas-water',reference.CASES['water'],None,'gradient',grid))
            report['runs'] += subreport['runs']
            report['checks'] += subreport['checks']
        check('dependency_off_gas_energy',gas[0]['energy']-gas[1]['energy'],1e-10)
        check('dependency_off_gas_gradient',reference.difference(gas[0]['gradient_hartree_per_bohr'],gas[1]['gradient_hartree_per_bohr']),1e-9)
    except Exception:
        report['exceptions'].append(traceback.format_exc())
    for name,client in clients.items():
        report['runtimes_after'][name]=reference.runtime(client,env)
    check('runtimes_unchanged',float(report['runtimes_before']!=report['runtimes_after']))
    report['passed']=not report['exceptions'] and bool(report['checks']) and all(x['passed'] for x in report['checks'])
    save()
    print(json.dumps(dict(passed=report['passed'],checks=len(report['checks']),runs=len(report['runs']),
                          exceptions=report['exceptions'],failed=[x for x in report['checks'] if not x['passed']]),indent=2))
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
