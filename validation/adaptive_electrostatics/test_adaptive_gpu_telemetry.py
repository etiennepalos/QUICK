#!/usr/bin/env python3
"""Fault-injection checks of telemetry acceptance; no native/GPU jobs."""
import argparse
import copy
import json
from pathlib import Path

from validate_adaptive_bomd import gpu_telemetry


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);checks=[]
    rows=[dict(step=str(k),depth='0',fallback='F',anchor_backend='cpu',query_backend='cuda',gpu_query_launches='1') for k in range(3)]
    summary=dict(adaptive_anchor_backend='cpu',adaptive_bound_backend='cpu',adaptive_cuda_queries_requested=True,
                 adaptive_gpu_query_launches=3,adaptive_device_bytes_live_after=0,adaptive_device_bytes_last_call=1024)
    def check(name,r=None,s=None,requested=True,gpu=True,valid=False):
        try:
            out=gpu_telemetry(copy.deepcopy(rows if r is None else r),copy.deepcopy(summary if s is None else s),requested,gpu)
            if not valid:raise AssertionError('Invalid telemetry accepted: '+name)
            checks.append(dict(name=name,passed=True,available=out['available']))
        except ValueError as error:
            if valid:raise
            checks.append(dict(name=name,passed=True,rejection=str(error)))
    try:
        check('actual-cuda-accepted-three-launches',valid=True)
        fallback=copy.deepcopy(rows)
        for row in fallback:row.update(fallback='T',query_backend='native direct')
        check('cuda-attempt-then-direct-fallback',r=fallback,valid=True)
        cpu=copy.deepcopy(rows);cpus=copy.deepcopy(summary)
        for row in cpu:row.update(query_backend='cpu',gpu_query_launches='0')
        cpus.update(adaptive_cuda_queries_requested=False,adaptive_gpu_query_launches=0,adaptive_device_bytes_last_call=0)
        check('explicit-cpu-telemetry',r=cpu,s=cpus,requested=False,gpu=False,valid=True)
        legacy=[dict(step='0',depth='0',fallback='T')]
        check('legacy-cpu-no-telemetry',r=legacy,s={},requested=False,gpu=False,valid=True)
        check('legacy-cannot-prove-requested-GPU',r=legacy,s={})
        for name,key,value in [('wrong-sum','adaptive_gpu_query_launches',2),('live-allocation','adaptive_device_bytes_live_after',8),
                               ('missing-allocation','adaptive_device_bytes_last_call',0),('wrong-request','adaptive_cuda_queries_requested',False),
                               ('wrong-anchor','adaptive_anchor_backend','cuda'),('wrong-bound','adaptive_bound_backend','cuda')]:
            bad=copy.deepcopy(summary);bad[key]=value;check(name,s=bad)
        for name,key,value in [('wrong-backend','query_backend','cpu'),('no-native-query','gpu_query_launches','0'),
                               ('wrong-depth-launches','depth','1'),('negative-count','gpu_query_launches','-1')]:
            bad=copy.deepcopy(rows);bad[0][key]=value;check(name,r=bad)
        bad=copy.deepcopy(fallback);bad[0]['query_backend']='cuda';check('fallback-backend-claim',r=bad)
        bad=copy.deepcopy(summary);del bad['adaptive_bound_backend'];check('partial-summary',s=bad)
        bad=copy.deepcopy(rows);del bad[0]['query_backend'];check('partial-CSV',r=bad)
        check('GPU-request-on-CPU',gpu=False)
        report=dict(passed=True,checks=checks,number_checks=len(checks),native_jobs=0,SCFs=0,GPU_kernels=0,
                    scope='Synthetic telemetry contract/fault-injection fixtures, not actual GPU execution')
    except Exception as error:report=dict(passed=False,checks=checks,error=f'{type(error).__name__}: {error}')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report.get(k) for k in ('passed','number_checks','error')}))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
