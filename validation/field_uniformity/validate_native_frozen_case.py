#!/usr/bin/env python3
"""Collect one GPU retained density, native same-D CPU queries, offline checks.

For an explicitly selected prepared case and installed runtimes. Each client
does one setup SCF; property evaluation must do no additional SCF and must
leave density/Fock state unchanged. No continuum or trajectory is assumed.
"""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

from field_uniformity import sha256,read_csv,plot_errors

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def runtime(driver,env):
    text=subprocess.check_output(["ldd",str(driver)],env=env,text=True)
    if "not found" in text:raise ValueError("Missing runtime dependency")
    files=[driver]
    for line in text.splitlines():
        match=re.search(r"(?:=>\s*)?(/[^\s]+)",line)
        if match:files.append(Path(match[1]).resolve(strict=True))
    return {str(path):sha256(path) for path in files}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for backend in ("cpu","cuda"):
        parser.add_argument("--"+backend+"-driver",type=Path,required=True);parser.add_argument("--"+backend+"-basis",type=Path,required=True)
    parser.add_argument("--prepared",type=Path,required=True);parser.add_argument("--case-id",required=True)
    parser.add_argument("--oracle-python",type=Path,required=True);parser.add_argument("--library-path",type=Path,action="append",default=[])
    parser.add_argument("--workdir",type=Path,required=True);parser.add_argument("--timeout",type=float,default=1800.)
    args=parser.parse_args()
    if not np.isfinite(args.timeout) or args.timeout<=0:parser.error("timeout must be positive finite")
    args.workdir=args.workdir.resolve();args.workdir.mkdir(parents=True,exist_ok=False)
    report={"passed":False,"checks":[],"runs":[],"gates":{"same_density_E_G_H":2e-10,"independent_Poisson_G_H":2e-8,"electron_count":1e-8,
            "state_mutation_and_extra_SCF":0.},"scope":__doc__,"continuous_bound_claim":False,"native_setup_SCFs_expected":2}
    def check(name,error,tolerance,**details):
        report["checks"].append({"name":name,"error":float(error),"tolerance":tolerance,
                                 "passed":bool(np.isfinite(error) and error<=tolerance),**details})
    def progress():
        (args.workdir/"progress.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    def execute(argv,folder,env,label):
        folder.mkdir(exist_ok=False);start=time.monotonic();row={"label":label,"argv":list(map(str,argv)),"directory":str(folder)}
        report["runs"].append(row);progress();print("RUN",label,flush=True)
        with (folder/"execution.log").open("w") as log:
            process=subprocess.run(list(map(str,argv)),cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
        row.update(returncode=process.returncode,wall_seconds=time.monotonic()-start,
                   files_sha256={p.name:sha256(p) for p in folder.iterdir() if p.is_file()});progress()
        if process.returncode:raise ValueError("Native/offline helper failed: "+str(folder))
    try:
        index_path=args.prepared/"application-index.json";index=json.loads(index_path.read_text())
        matches=[case for case in index["cases"] if case["id"]==args.case_id]
        if len(matches)!=1:raise ValueError("Expected one explicit prepared case")
        case=matches[0];folder=Path(case["directory"]);model=json.loads((folder/"model.json").read_text())
        expected_files={str(p):sha256(p) for p in [index_path,*folder.iterdir()] if p.is_file()}
        core_names=subprocess.check_output(["git","ls-files","-z","--","src"],cwd=ROOT).decode().split("\0")
        core_hashes={name:sha256(ROOT/name) for name in core_names if name and (ROOT/name).is_file()}
        report["prepared_files_sha256"]=expected_files;report["core_source_files_sha256"]=core_hashes
        drivers={b:getattr(args,b+"_driver").resolve(strict=True) for b in ("cpu","cuda")};envs={};before={}
        for backend in drivers:
            basis=getattr(args,backend+"_basis").resolve(strict=True);env=os.environ.copy()
            env.update(QUICK_BASIS=str(basis),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
            env["LD_LIBRARY_PATH"]=":".join([str(basis.parent/"lib"),*map(str,args.library_path),env.get("LD_LIBRARY_PATH","")])
            if backend=="cuda":env["CUDA_VISIBLE_DEVICES"]="0"
            envs[backend]=env;before[backend]=runtime(drivers[backend],env)
            basis_path=basis/Path(case["basis_file"]).name
            check(backend+":exact_basis_file",int(sha256(basis_path)!=sha256(case["basis_file"])),0.)
        report["runtime_before"]=before
        execute([drivers["cuda"],folder/"input.dat","-","1"],args.workdir/"cuda",envs["cuda"],"cuda-retained-density")
        execute([drivers["cpu"],folder/"input.dat",args.workdir/"cuda/density.dat","1"],args.workdir/"cpu",envs["cpu"],"cpu-replaces-same-GPU-density")
        check("Exact_exported_same_density",int(sha256(args.workdir/"cuda/density.dat")!=sha256(args.workdir/"cpu/density.dat")),0.)
        observed={}
        for backend in drivers:
            native=args.workdir/backend
            observed[backend]=np.atleast_2d(np.loadtxt(native/"properties.dat",skiprows=1))
            if observed[backend].shape!=(case["queries"],44) or not np.isfinite(observed[backend]).all():raise ValueError("Unexpected finite native properties")
            output=(native/"electrostatics_benchmark.out").read_text()
            check(backend+":one_converged_setup_SCF",abs(len(re.findall(r"REACH CONVERGENCE AFTER\s+\d+\s+CYCLES",output))-1),0.)
            timings=list(csv.DictReader((native/"timings.csv").open()))
            for route,counter in [("E","efield_launches"),("G","efg_launches"),("H","hessian_launches")]:
                launches=[int(row[counter]) for row in timings if row["route"]==route]
                check(backend+":native_"+route+"_route",int(not launches or any((n<=0 if backend=="cuda" else n!=0) for n in launches)),0.)
        for name,columns in [("V",slice(4,5)),("E",slice(5,8)),("G",slice(8,17)),("H",slice(17,44))]:
            check("same_D_CPU_CUDA:"+name,np.max(abs(observed["cpu"][:,columns]-observed["cuda"][:,columns])),2e-10)
        runs=args.workdir/"native-exports.json";runs.write_text(json.dumps({args.case_id:{b:str(args.workdir/b) for b in drivers}},indent=2)+"\n")
        execute([sys.executable,HERE/"application.py","analyze","--prepared",args.prepared,"--runs-json",runs,
                 "--oracle-python",args.oracle_python,"--output",args.workdir/"analysis","--tolerance-au","1e-4","1e-3","--plot"],
                args.workdir/"offline",envs["cpu"],"independent-AO-and-offline-analysis")
        for backend in drivers:
            analysis=args.workdir/"analysis"/(args.case_id+"-"+backend)
            centers=read_csv(analysis/"analysis/centers_analysis.csv",["poisson_g_residual","poisson_h_residual_norm"])
            for center in centers:
                check(backend+":"+center["center_id"]+":Poisson_G",abs(float(center["poisson_g_residual"])),2e-8)
                check(backend+":"+center["center_id"]+":Poisson_H",abs(float(center["poisson_h_residual_norm"])),2e-8)
            reference=json.loads((analysis/"ao-reference/reference.json").read_text())
            check(backend+":electron_count",abs(reference["electron_count_trace_DS"]-(sum(z for z,_ in model["atoms"])-model["charge"])),1e-8)
            values=read_csv(analysis/"analysis/radius_summary.csv",["center_id","radius_bohr"])
            report.setdefault("observed_Taylor_orders",{})[backend]={}
            for center_id in sorted({row["center_id"] for row in values}):
                selected=sorted([row for row in values if row["center_id"]==center_id],key=lambda row:float(row["radius_bohr"]))
                orders={}
                for approximation in ("constant","linear","quadratic"):
                    errors=[float(row[approximation+"_max_error"]) for row in selected[:2]]
                    orders[approximation]=float(np.log(errors[1]/errors[0])/np.log(float(selected[1]["radius_bohr"])/float(selected[0]["radius_bohr"]))) if min(errors)>1e-14 else None
                report["observed_Taylor_orders"][backend][center_id]=orders
                numeric_rows=[{key:(value if key=="center_id" else float(value)) for key,value in row.items()} for row in values]
                plot_errors(numeric_rows,analysis/"analysis"/("field_error_"+center_id+".pdf"),center_id=center_id)
        after={b:runtime(drivers[b],envs[b]) for b in drivers};report["runtime_after"]=after
        check("Runtime_files_unchanged",int(before!=after),0.)
        check("Prepared_files_unchanged",int(any(sha256(path)!=digest for path,digest in expected_files.items())),0.)
        check("Published_and_new_core_source_unchanged",int(any(sha256(ROOT/name)!=digest for name,digest in core_hashes.items())),0.)
        report["passed"]=all(c["passed"] for c in report["checks"])
    except Exception as error:
        import traceback
        report["error"]=type(error).__name__+": "+str(error);report["traceback"]=traceback.format_exc()
    report["analysis_sources_sha256"]={str(p):sha256(p) for p in HERE.glob("*.py")}
    (args.workdir/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"passed":report["passed"],"checks":len(report["checks"]),"native_runs":2,"error":report.get("error")}))
    return 0 if report["passed"] else 2


if __name__=="__main__":raise SystemExit(main())
