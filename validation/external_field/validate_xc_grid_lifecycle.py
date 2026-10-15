#!/usr/bin/env python3
"""Fixed-gate serial/MPI XC-grid parser and successive-job lifecycle checks.

Uses the dedicated native client; every evidence directory is new. No
acceptance gates, runtime libraries, QUICK sources or baselines are changed.
"""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
from __future__ import annotations

import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime(driver,env):
    output=subprocess.check_output(["ldd",str(driver)],env=env,text=True)
    if "not found" in output:raise ValueError("Unresolved runtime dependency")
    paths=[driver]
    for line in output.splitlines():
        match=re.search(r"(?:=>\s*)?(/[^\s]+)",line)
        if match:paths.append(Path(match[1]).resolve(strict=True))
    return {str(path):sha(path) for path in paths}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-driver",type=Path,required=True);parser.add_argument("--cpu-basis",type=Path,required=True)
    parser.add_argument("--mpi-driver",type=Path);parser.add_argument("--mpi-basis",type=Path)
    parser.add_argument("--mpiexec",type=Path);parser.add_argument("--library-path",type=Path,action="append",default=[])
    parser.add_argument("--workdir",type=Path,required=True);parser.add_argument("--timeout",type=float,default=600.)
    args=parser.parse_args()
    if bool(args.mpi_driver)!=bool(args.mpi_basis) or (args.mpi_driver and not args.mpiexec):
        parser.error("MPI requires --mpi-driver, --mpi-basis and --mpiexec")
    if not np.isfinite(args.timeout) or args.timeout<=0:parser.error("timeout must be positive finite")
    args.workdir=args.workdir.resolve();args.workdir.mkdir(parents=True,exist_ok=False)
    report={"passed":False,"started_utc":datetime.now(timezone.utc).isoformat(),"checks":[],"runs":[],"runtime":{},
            "scope":"Water PBE0/STO-3G parser and successive-job grid lifecycle; software validation, not XC convergence acceptance",
            "gates":{"same_backend_repeat_energy":2e-12,"same_backend_repeat_gradient":2e-11,
                     "same_backend_repeat_EG":2e-11,"serial_MPI_energy":2e-10,
                     "serial_MPI_gradient":2e-9,"serial_MPI_EG":2e-9,"state_mutation_or_extra_SCF":0.}}
    def check(name,error,tolerance,**details):
        report["checks"].append({"name":name,"error":float(error),"tolerance":tolerance,
                                 "passed":bool(np.isfinite(error) and error<=tolerance),**details})
    def progress():
        (args.workdir/"progress.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    def execute(command,folder,env,label):
        folder.mkdir(exist_ok=False);start=time.monotonic()
        row={"label":label,"argv":list(map(str,command)),"directory":str(folder)};report["runs"].append(row);progress()
        with (folder/"execution.log").open("w") as output:
            process=subprocess.run(list(map(str,command)),cwd=folder,env=env,stdout=output,stderr=subprocess.STDOUT,timeout=args.timeout)
        row.update(returncode=process.returncode,wall_seconds=time.monotonic()-start,
                   files_sha256={p.name:sha(p) for p in folder.iterdir() if p.is_file()});progress()
        if process.returncode:raise ValueError("Native test client failed: "+str(folder))
        return folder
    try:
        drivers={"cpu":args.cpu_driver.resolve(strict=True)};bases={"cpu":args.cpu_basis.resolve(strict=True)}
        if args.mpi_driver:drivers["mpi2"]=args.mpi_driver.resolve(strict=True);bases["mpi2"]=args.mpi_basis.resolve(strict=True)
        environments={}
        for backend in drivers:
            env=os.environ.copy();env.update(QUICK_BASIS=str(bases[backend]),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
            env["LD_LIBRARY_PATH"]=":".join([str(bases[backend].parent/"lib"),*map(str,args.library_path),env.get("LD_LIBRARY_PATH","")])
            # Open MPI permits root only when explicitly configured; these
            # variables are harmless in ordinary nonroot/Hydra execution.
            env.update(OMPI_ALLOW_RUN_AS_ROOT="1",OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1")
            environments[backend]=env;report["runtime"][backend]=runtime(drivers[backend],env)
        cases=[("default","PBE0",0,False,0,0),
               ("paired","PBE0 XC_RADIAL=75 XC_ANGULAR=302",0,True,75,302),
               ("only-radial","PBE0 XC_RADIAL=75",57,None,None,None),
               ("only-angular","PBE0 XC_ANGULAR=302",57,None,None,None),
               ("zero-radial","PBE0 XC_RADIAL=0 XC_ANGULAR=302",57,None,None,None),
               ("negative-radial","PBE0 XC_RADIAL=-1 XC_ANGULAR=302",57,None,None,None),
               ("oversize-radial","PBE0 XC_RADIAL=401 XC_ANGULAR=302",57,None,None,None),
               ("zero-angular","PBE0 XC_RADIAL=75 XC_ANGULAR=0",57,None,None,None),
               ("negative-angular","PBE0 XC_RADIAL=75 XC_ANGULAR=-6",57,None,None,None),
               ("unsupported-angular","PBE0 XC_RADIAL=75 XC_ANGULAR=12",57,None,None,None),
               ("negative-weight74","PBE0 XC_RADIAL=75 XC_ANGULAR=74",57,None,None,None),
               ("negative-weight230","PBE0 XC_RADIAL=75 XC_ANGULAR=230",57,None,None,None),
               ("negative-weight266","PBE0 XC_RADIAL=75 XC_ANGULAR=266",57,None,None,None),
               ("HF-rejected","HF XC_RADIAL=75 XC_ANGULAR=302",57,None,None,None),
               ("SG0-rejected","PBE0 SG0 XC_RADIAL=75 XC_ANGULAR=302",57,None,None,None),
               ("boundary-lookalikes","PBE0 NOT_XC_RADIAL=75 NOT_XC_ANGULAR=302",0,False,0,0),
               ("minimum-dimensions-parser-only","PBE0 XC_RADIAL=1 XC_ANGULAR=6",0,True,1,6),
               ("maximum-dimensions-parser-only","PBE0 XC_RADIAL=400 XC_ANGULAR=5810",0,True,400,5810),
               ("explicit-SG1-paired","PBE0 SG1 XC_RADIAL=75 XC_ANGULAR=302",0,True,75,302)]
        # New controls have dedicated nonterminating integer readers. Exercise
        # both fields independently; these cases never invoke an SCF.
        for control,other in [("XC_RADIAL","XC_ANGULAR=302"),("XC_ANGULAR","XC_RADIAL=75")]:
            valid_value=75 if control=="XC_RADIAL" else 302
            for label,value in [("decimal","1.5"),("nan","NaN"),("text","ABC"),("plus-only","+"),
                                ("minus-only","-"),("overflow","99999999999999999999999999999999"),
                                ("exponent","1e2"),("comma","1,2")]:
                cases.append((control.lower()+"-"+label,f"PBE0 {control}={value} {other}",57,None,None,None))
            cases.append((control.lower()+"-missing-equals",f"PBE0 {control} {valid_value} {other}",57,None,None,None))
            cases.append((control.lower()+"-missing-value",f"PBE0 {control}= {other}",57,None,None,None))
            cases.append((control.lower()+"-duplicate",f"PBE0 {control}={valid_value} {control}={valid_value} {other}",57,None,None,None))
        cases.extend([("whitespace-tabs","PBE0\tXC_RADIAL\t=\t75\tXC_ANGULAR\t=\t302",0,True,75,302),
                      ("positive-signs","PBE0 XC_RADIAL=+75 XC_ANGULAR=+302",0,True,75,302),
                      ("leading-zeros","PBE0 XC_RADIAL=00075 XC_ANGULAR=00302",0,True,75,302)])
        for label,card,expected,custom,nrad,nang in cases:
            card_path=args.workdir/("card-"+label+".txt");card_path.write_text(card+" BASIS=STO-3G ENERGY\n")
            folder=execute([drivers["cpu"],"options",card_path],args.workdir/("parser-"+label),environments["cpu"],"parser:"+label)
            text=(folder/"execution.log").read_text();match=re.search(r"options_ierr\s+(\d+)",text)
            if not match:raise ValueError("Missing parser result for "+label)
            check("parser:"+label+":status",abs(int(match[1])-expected),0.)
            if expected==0:
                got_custom=re.search(r"options_custom\s+([TF])",text);sizes=re.search(r"options_sizes\s+(\d+)\s+(\d+)",text)
                check("parser:"+label+":custom",int(not got_custom or (got_custom[1]=="T")!=custom),0.)
                check("parser:"+label+":sizes",int(not sizes or tuple(map(int,sizes.groups()))!=(nrad,nang)),0.)
        rows_by_backend={}
        gradient_names=[f"g{atom}{axis}" for atom in (1,2,3) for axis in "xyz"]
        property_names=["ex","ey","ez","gxx","gyx","gzx","gxy","gyy","gzy","gxz","gyz","gzz"]
        def values(row,names):return np.array([float(row[name]) for name in names])
        for backend,driver in drivers.items():
            command=[driver,"lifecycle"] if backend=="cpu" else [args.mpiexec,"-n","2",driver,"lifecycle"]
            folder=execute(command,args.workdir/("lifecycle-"+backend),environments[backend],"lifecycle:"+backend)
            rows=list(csv.DictReader((folder/"lifecycle.csv").open()));rows_by_backend[backend]=rows
            check(backend+":jobs",abs(len(rows)-8),0.)
            for index,row in enumerate(rows):
                check(backend+":"+row["label"]+":finite",int(not np.isfinite(values(row,["energy",*gradient_names,*property_names])).all()),0.)
                for name in ("density_error","fock_error","extra_API_calls"):
                    check(backend+":"+row["label"]+":"+name,abs(float(row[name])),0.)
                check(backend+":"+row["label"]+":one_energy_API",abs(int(row["api_steps"])-2),0.)
                output=(folder/f"xc_lifecycle_{index+1:02d}.out").read_text()
                check(backend+":"+row["label"]+":one_converged_SCF",abs(len(re.findall(r"REACH CONVERGENCE AFTER\s+\d+\s+CYCLES",output))-1),0.)
            for baseline,repeat in [(0,2),(0,5),(6,7)]:
                label=backend+":"+rows[repeat]["label"]+":repeat"
                check(label+":energy",abs(float(rows[repeat]["energy"])-float(rows[baseline]["energy"])),report["gates"]["same_backend_repeat_energy"])
                check(label+":gradient",np.max(abs(values(rows[repeat],gradient_names)-values(rows[baseline],gradient_names))),report["gates"]["same_backend_repeat_gradient"])
                check(label+":E_G",np.max(abs(values(rows[repeat],property_names)-values(rows[baseline],property_names))),report["gates"]["same_backend_repeat_EG"])
            rank_paths=sorted(folder.glob("rank-*.csv"));check(backend+":rank_files",abs(len(rank_paths)-(2 if backend=="mpi2" else 1)),0.)
            rank_rows=[list(csv.DictReader(path.open())) for path in rank_paths]
            for job in range(8):
                counts=[int(table[job]["init_ngpts"]) for table in rank_rows]
                check(backend+f":job{job+1}:rank_count_equality",max(counts)-min(counts),0.)
            progress()
        if "mpi2" in rows_by_backend:
            for cpu,mpi in zip(rows_by_backend["cpu"],rows_by_backend["mpi2"]):
                label=cpu["label"]+":serial_MPI"
                check(label+":energy",abs(float(cpu["energy"])-float(mpi["energy"])),report["gates"]["serial_MPI_energy"])
                check(label+":gradient",np.max(abs(values(cpu,gradient_names)-values(mpi,gradient_names))),report["gates"]["serial_MPI_gradient"])
                check(label+":E_G",np.max(abs(values(cpu,property_names)-values(mpi,property_names))),report["gates"]["serial_MPI_EG"])
        report["passed"]=all(row["passed"] for row in report["checks"])
    except Exception as error:
        import traceback
        report["error"]=type(error).__name__+": "+str(error);report["traceback"]=traceback.format_exc()
    report["source_files_sha256"]={str(path):sha(path) for path in [Path(__file__),Path(__file__).with_name("quick_xc_grid_lifecycle_driver.f90")]}
    (args.workdir/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"passed":report["passed"],"checks":len(report["checks"]),"runs":len(report["runs"]),"error":report.get("error")}))
    return 0 if report["passed"] else 2


if __name__=="__main__":raise SystemExit(main())
