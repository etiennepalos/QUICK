#!/usr/bin/env python3
"""Paired fresh-process BOMD overhead with source V/E/G/H disabled/enabled.

The whole-job clock includes process/launcher startup, QUICK setup, all SCFs
and gradients, property/probe work, normal trajectory/property/checkpoint IO
and teardown. Python preparation, hashing and validation are outside this
clock. Filesystem caches are not flushed and no extra fsync is introduced.
The separate synchronous property-API/force-API ratio excludes driver IO.
"""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
from __future__ import annotations

import argparse
import csv
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
LEGACY=ROOT/"validation/oeprop_bomd"
sys.path.insert(0,str(LEGACY))
import benchmark_overhead as old
from validate_oeprop_bomd import write_control,sha256,read_csv,max_difference,property_difference

BOHR_ANGSTROM=.52917720859
H_COMPONENTS=("xxx","xxy","xxz","xyy","xyz","xzz","yyy","yyz","yzz","zzz")
H_COLUMNS=["h_"+name for name in H_COMPONENTS]
SOURCE_CONVENTION="electrons+nuclei+enabled_mm; direct_applied_and_reaction_backgrounds_excluded"


def require(condition,message):
    if not condition:raise ValueError(message)


def describe(values):
    values=[float(value) for value in values]
    require(bool(values) and all(math.isfinite(x) for x in values),"Expected finite nonempty timing samples")
    return {"values":values,"median":statistics.median(values),"min":min(values),"max":max(values),
            "sample_stdev":statistics.stdev(values) if len(values)>1 else None}


def validate_expected_controls(keyword,expected):
    tokens={key:(value if separator else None) for key,separator,value in (word.partition("=") for word in keyword.split())}
    for key,wanted in expected.items():
        key=key.upper();require(key in tokens,"Missing explicit protocol control "+key);value=tokens[key]
        if wanted is None:require(value is None,"Expected bare protocol keyword "+key)
        elif isinstance(wanted,(int,float)) and not isinstance(wanted,bool):
            require(value is not None and math.isfinite(float(value.replace("D","E"))) and float(value.replace("D","E"))==wanted,"Protocol value differs: "+key)
        else:require(value==str(wanted).upper(),"Protocol value differs: "+key)


def read_inputs(geometry,probes,keyword_file,probe_units,expected_controls=None):
    lines=geometry.read_text().splitlines();natom=int(lines[0])
    require(natom>=3 and len(lines)==natom+2,"Supply one complete XYZ geometry with at least three atoms")
    coordinates=[]
    for line in lines[2:]:
        row=line.split();require(len(row)==4,"Malformed XYZ atom row")
        coordinates.append([old.finite(value,"geometry") for value in row[1:]])
    coordinates=np.asarray(coordinates)
    lines=probes.read_text().splitlines();count=int(lines[0]);points=np.atleast_2d(np.loadtxt(probes,skiprows=1))
    require(count>=1 and points.shape==(count,3) and np.isfinite(points).all(),"Malformed finite probe count/coordinates")
    if probe_units=="bohr":points=points*BOHR_ANGSTROM
    require(np.min(np.linalg.norm(points[:,None]-coordinates[None,:],axis=2))>1e-12,"Initial probe at a classical source singularity")
    keyword=keyword_file.read_text().strip().upper()
    require(bool(keyword) and len(keyword)<=256 and "\n" not in keyword and "\r" not in keyword,"Keyword must be one nonempty line of at most 256 characters")
    require("'" not in keyword and '"' not in keyword,"Keyword quotes cannot be embedded in a generated namelist")
    words=keyword.split();keys=[word.partition("=")[0] for word in words]
    require("GRADIENT" in words,"The energy/gradient API requires explicit GRADIENT in the card")
    require(any(word.startswith("BASIS=") and len(word)>6 for word in words),"Explicit BASIS=... is required")
    require("DFTD3" not in keys,"Use the public D3 keyword; DFTD3 is an internal flag and is ignored by QUICK")
    require(len(keys)==len(set(keys)),"Duplicate/ambiguous keyword controls")
    # This BOMD driver has no MM-charge input. A caller must not silently
    # benchmark an empty MM environment by supplying EXTCHARGES alone.
    require("EXTCHARGES" not in words,"This BOMD workflow driver has no external-MM-charge input")
    if expected_controls is not None:validate_expected_controls(keyword,expected_controls)
    probe_text=str(count)+"\n"+"".join(" ".join(f"{value:.17g}" for value in point)+"\n" for point in points)
    return {"natom":natom,"probe_count":count,"keyword":keyword,"probe_text":probe_text,"coordinates":coordinates,"points_angstrom":points}


def write_workflow_control(path,*,nsteps,dt_fs,keyword,stride,enabled):
    write_control(path,geometry=Path("geometry.xyz"),probes=Path("probes.probe"),prefix=Path("bomd"),
                  checkpoint=Path("bomd.chk"),mode="fixed",steps=nsteps,dt_fs=dt_fs,keyword=keyword,stride=stride)
    lines=path.read_text().splitlines()
    require(lines[-1].strip()=="/","Unexpected existing namelist writer terminator")
    lines.insert(-1," field_hessian="+(".true." if enabled else ".false."))
    path.write_text("\n".join(lines)+"\n")


def read_raw_gradients(path,natom,frames):
    data=np.atleast_2d(np.loadtxt(path))
    require(data.shape==(natom*frames,5) and np.isfinite(data).all(),"Malformed complete raw-gradient export")
    expected=np.array([(frame,atom) for frame in range(1,frames+1) for atom in range(1,natom+1)])
    require(np.array_equal(data[:,:2],expected),"Raw-gradient frame/atom identities are missing or reordered")
    return data[:,2:].reshape(frames,natom,3)


def validate_hessian(directory,data,args,stride):
    summary=data["summary"];enabled=stride>0
    sampled=[step for step in range(args.nsteps+1) if stride and step%stride==0]
    path=directory/"bomd_field_hessian.csv"
    if not enabled:
        require(not path.exists() and "field_hessian_calls" not in summary,"Disabled run enabled field-Hessian work/output")
        return []
    require(summary.get("field_hessian_calls")==len(sampled),"Field-Hessian call count does not match stride")
    gpu=summary["backend"].lower() in ("gpu","cuda","hip")
    require(summary.get("field_hessian_master_gpu_launches")== (len(sampled) if gpu else 0),"Wrong native master field-Hessian launch count")
    require(summary.get("field_hessian_convention")=="H_ijk=d2 E_i/dC_j dC_k=-d3 V/dC_i dC_j dC_k","Wrong field-Hessian sign/convention")
    require(summary.get("field_hessian_units")=="hartree/(e bohr^3); laboratory Cartesian axes","Wrong field-Hessian units/axes")
    rows=read_csv(path,["step","time_fs","probe","x_ang","y_ang","z_ang",*H_COLUMNS])
    identities=[(int(row["step"]),int(row["probe"])) for row in rows]
    require(identities==[(step,probe) for step in sampled for probe in range(1,args.probe_count+1)],"Missing/duplicated/reordered field-Hessian rows")
    require(identities==[(int(row["step"]),int(row["probe"])) for row in data["properties"]],"V/E/G and H sample identities differ")
    for h,veg in zip(rows,data["properties"]):
        require(abs(float(h["time_fs"])-int(h["step"])*args.dt_fs)<=1e-12,"Wrong field-Hessian timestamp")
        require(max_difference([float(h[name]) for name in ("x_ang","y_ang","z_ang")],
                               [float(veg[name]) for name in ("x_ang","y_ang","z_ang")])<=1e-11,"V/E/G and H probe coordinates differ")
    return rows


def validate_method_output(text,keyword,expected_basis=None):
    if "D3" in keyword.split():
        require("DISPERSION METHOD = D3, ZERO DAMPING" in text,"Requested D3 is absent from the native method output")
    matches=re.findall(r"(?m)^=+[ \t]*BASIS INFOS[ \t]*=+[ \t]*\n[ \t]*BASIS FUNCTIONS[ \t]*=[ \t]*(\d+)[ \t]*$",text)
    require(bool(matches),"Native output does not identify a total BASIS INFOS dimension")
    dimensions=list(map(int,matches));require(len(set(dimensions))==1,"Native total basis dimension changes between molecular jobs")
    nbasis=dimensions[0]
    if expected_basis is not None:require(nbasis==expected_basis,"Native total basis dimension differs from requested value")
    radial=re.search(r"(?:^|\s)XC_RADIAL\s*=\s*([+]?[0-9]+)(?=\s|$)",keyword)
    angular=re.search(r"(?:^|\s)XC_ANGULAR\s*=\s*([+]?[0-9]+)(?=\s|$)",keyword)
    if radial and angular:
        expected=(int(radial[1]),int(angular[1]))
        grids=[tuple(map(int,entry)) for entry in re.findall(r"UNPRUNED XC GRID:\s*RADIAL\s*=\s*(\d+)\s+ANGULAR\s*=\s*(\d+)",text)]
        require(expected in grids,"Requested refined XC grid is absent from native method output")
    printed_controls={}
    labels={"CUTOFF":"TWO-e INTEGRAL","GRADCUTOFF":"GRADIENT CUTOFF","MATRIXZERO":"MATRIX ELEMENTS",
            "BASISCUTOFF":"BASIS FUNCTION","XCCUTOFF":"PRUNING CUTOFF","DENSERMS":"DENSITY MATRIX MAXIMUM RMS FOR CONVERGENCE"}
    for control,label in labels.items():
        requested=re.search(r"(?:^|\s)"+control+r"=([^\s]+)",keyword)
        if not requested:continue
        wanted=float(requested[1].replace("D","E"))
        pattern=r"^[ \t|]*"+re.escape(label)+r"[ \t]*=[ \t]*([+\-0-9.eEdD]+)[ \t]*$"
        observed=[float(value.replace("D","E")) for value in re.findall(pattern,text,re.MULTILINE|re.IGNORECASE)]
        require(bool(observed),"Requested cutoff absent from native output: "+control)
        # QUICK's E10.3 tables have three significant figures. Exact input
        # protocol values are checked separately before execution.
        require(all(math.isfinite(value) and math.isclose(value,wanted,rel_tol=5.1e-3,abs_tol=0.) for value in observed),"Requested cutoff differs in native output: "+control)
        printed_controls[control]={"requested":wanted,"printed":observed,"scope":"Native table precision (three significant figures)"}
    return {"basis_functions":nbasis,"total_basis_dimensions":dimensions,"requested_D3_verified":"D3" in keyword.split(),"printed_controls":printed_controls}


def load_workflow_trial(directory,args,stride,keyword):
    data=old.load_trial(directory,args,stride,keyword)
    data["field_hessian"]=validate_hessian(directory,data,args,stride)
    output=(directory/"quick_oeprop_bomd.out").read_text(errors="replace")
    data["method_evidence"]=validate_method_output(output,keyword,args.expected_basis_functions)
    require(len(re.findall(r"REACH CONVERGENCE AFTER\s+\d+\s+CYCLES",output))==args.nsteps+1,"Unexpected additional/missing converged SCF")
    raw=directory/"bomd_forces.dat"
    if args.force_source=="raw" or (args.force_source=="auto" and raw.exists()):
        full=read_raw_gradients(raw,args.natom,args.nsteps+1)
        require(max_difference(full,data["gradients"])<=5.1e-11,"Raw and printed complete gradients disagree beyond output precision")
        data["gradients"]=full;data["gradient_source"]="Complete already-computed raw gradients, no extra force API calls"
    else:
        data["gradient_source"]="Every printed Cartesian gradient, quantized to 1e-10 Eh/bohr"
    expected_gpu=args.backend=="cuda"
    require((data["summary"]["backend"].lower() in ("gpu","cuda","hip"))==expected_gpu,"Native backend differs from requested backend")
    return data


def hessian_difference(first,second):
    require([(int(row["step"]),int(row["probe"])) for row in first]==[(int(row["step"]),int(row["probe"])) for row in second] and bool(first),"Field-Hessian repeat identity mismatch")
    return {name:max_difference([float(row[name]) for row in first],[float(row[name]) for row in second]) for name in H_COLUMNS}


def production_snapshot(root):
    names=[Path("CMakeLists.txt")]+[path.relative_to(root) for name in ("src","quick-cmake") for path in sorted((root/name).rglob("*")) if path.is_file()]
    require((root/"src").is_dir() and (root/"CMakeLists.txt").is_file(),"Supply the immutable production source used by the selected install")
    return {str(name):sha256(root/name) for name in names}


def write_table(path,rows):
    if not rows:return
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def benchmark(args):
    workdir=args.workdir.resolve();workdir.mkdir(parents=True,exist_ok=False)
    driver=args.driver.resolve(strict=True);prefix=args.install_prefix.resolve(strict=True);source=args.source_snapshot.resolve(strict=True)
    require(driver.is_file() and os.access(driver,os.X_OK),"--driver must be executable")
    protocol_path=getattr(args,"protocol_json",None)
    expected_controls=json.loads(protocol_path.read_text()) if protocol_path else None
    inputs=read_inputs(args.geometry,args.probes,args.keyword_file,args.probe_units,expected_controls)
    args.natom=inputs["natom"];args.probe_count=inputs["probe_count"];keyword=inputs["keyword"]
    env=os.environ.copy();env.update(QUICK_BASIS=str(prefix/"basis"),OMP_NUM_THREADS=str(args.threads),OPENBLAS_NUM_THREADS=str(args.threads),MKL_NUM_THREADS=str(args.threads),NUMEXPR_NUM_THREADS=str(args.threads))
    env["LD_LIBRARY_PATH"]=":".join([str(prefix/"lib"),*map(str,args.library_path),env.get("LD_LIBRARY_PATH","")])
    require((prefix/"basis").is_dir(),"Selected install has no basis directory")
    command=shlex.split(args.launcher)+[str(driver),"control.nml"]
    plans=[];input_files=[args.geometry.resolve(),args.probes.resolve(),args.keyword_file.resolve()]
    if protocol_path:input_files.append(protocol_path.resolve())
    for pair in range(args.warmup_pairs+args.repeats):
        order=("off","on") if pair%2==0 else ("on","off")
        for position,mode in enumerate(order):
            directory=workdir/f"pair_{pair:03d}"/mode;directory.mkdir(parents=True)
            shutil.copyfile(args.geometry,directory/"geometry.xyz");(directory/"probes.probe").write_text(inputs["probe_text"])
            stride=args.stride if mode=="on" else 0
            write_workflow_control(directory/"control.nml",nsteps=args.nsteps,dt_fs=args.dt_fs,keyword=keyword,stride=stride,enabled=mode=="on")
            input_files.extend(directory/name for name in ("geometry.xyz","probes.probe","control.nml"))
            plans.append({"pair":pair,"discarded_warmup":pair<args.warmup_pairs,"order_in_pair":position,"mode":mode,
                          "stride":stride,"directory":str(directory),"argv":command})
    input_files.extend(path for path in sorted((prefix/"basis").rglob("*")) if path.is_file())
    harness_files=[Path(__file__).resolve(),LEGACY/"benchmark_overhead.py",LEGACY/"validate_oeprop_bomd.py"]
    def snapshot():
        return {"driver":old.driver_snapshot(driver,env),"production_source_files_sha256":production_snapshot(source),
                "inputs_basis_harness_sha256":{str(path):sha256(path) for path in [*input_files,*harness_files]}}
    baseline=snapshot();libraries=baseline["driver"]["resolved_libraries_sha256"]
    quick=[path for path in libraries if re.fullmatch(r"libquick(?:_mpi)?(?:_cuda)?\.so(?:\.[\d.]+)?",Path(path).name)]
    require(len(quick)==1 and Path(quick[0]).parent==(prefix/"lib").resolve(),"Loaded QUICK library is not the selected install")
    if args.expected_driver_sha256:require(baseline["driver"]["sha256"]==args.expected_driver_sha256.lower(),"Driver hash differs from accepted binary")
    build_record=None
    if args.build_report:
        build_path=args.build_report.resolve(strict=True);build=json.loads(build_path.read_text());manifest_path=build_path.parent/"source-manifest.json"
        require(build.get("passed") and sha256(manifest_path)==build["source_manifest_sha256"],"Invalid retained build/source-manifest record")
        manifest=json.loads(manifest_path.read_text())
        require(all(manifest.get(name)==digest for name,digest in baseline["production_source_files_sha256"].items()),"Source snapshot differs from compiled manifest")
        built_libraries={str(Path(path).resolve()):digest for entry in build["backends"].values() for path,digest in entry["libraries_sha256"].items()}
        require(built_libraries.get(quick[0])==libraries[quick[0]],"Loaded QUICK library absent from build record")
        build_record={"path":str(build_path),"sha256":sha256(build_path),"source_manifest_sha256":sha256(manifest_path)}
    result={"schema":"quick-workflow-overhead-1","passed":False,"prepared_only":args.prepare_only,"started_utc":datetime.now(timezone.utc).isoformat(),
            "configuration":{"backend":args.backend,"keyword":keyword,"natom":args.natom,"probe_count":args.probe_count,"nsteps":args.nsteps,
                "dt_fs":args.dt_fs,"temperature_k":50.,"stride":args.stride,"threads":args.threads,"ranks":args.ranks,
                "retained_pairs":args.repeats,"discarded_pairs":args.warmup_pairs,"enabled_properties":"V,E,G,H","probe_mode":"fixed",
                "force_source":args.force_source,"expected_basis_functions":getattr(args,"expected_basis_functions",None),"expected_controls":expected_controls,"tolerances":old.TOLERANCES},
            "timing_boundary":__doc__,"source_convention":SOURCE_CONVENTION,"provenance_before":baseline,"build_record":build_record,
            "source_snapshot":str(source),"provenance_fingerprint":old.fingerprint(baseline),"plans":plans,"trials":[],"pairs":[],
            "limitations":["Finite isolated short trajectory; software/overhead illustration, no ensemble statistics",
                "All coordinates/energies/Cartesian gradients checked at every force evaluation; complete velocities checked at final checkpoint",
                "Fresh process timings include setup and IO; caches are not flushed; warmup pairs do not reuse CUDA contexts",
                "API phase ratio excludes driver IO and is not whole-job overhead; timing noise may give negative paired deltas"],
            "environment":{key:env.get(key) for key in ("QUICK_BASIS","LD_LIBRARY_PATH","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","CUDA_VISIBLE_DEVICES")}}
    output=workdir/"workflow_summary.json"
    def save():
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    save()
    if args.prepare_only:return result
    references={};saved={}
    try:
        for plan in plans:
            record=dict(plan);result["trials"].append(record);directory=Path(plan["directory"])
            require(snapshot()==baseline,"Binary/libraries/source/inputs changed before native trial")
            with (directory/"run.log").open("w") as log:
                begin=time.perf_counter_ns()
                try:
                    process=subprocess.run(command,cwd=directory,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
                    record["returncode"]=process.returncode
                finally:record["whole_job_wall_seconds"]=(time.perf_counter_ns()-begin)*1e-9
            require(snapshot()==baseline,"Binary/libraries/source/inputs changed during native trial")
            require(process.returncode==0,"Native trial failed: "+str(directory/"run.log"))
            data=load_workflow_trial(directory,args,plan["stride"],keyword)
            record.update({name:data[name] for name in ("summary","rank_evidence","phase_max_rank_seconds","invariant_max_abs","gradient_source","method_evidence")})
            record["hessian_rows"]=len(data["field_hessian"]);phase=data["phase_max_rank_seconds"]
            record["diagnostic_force_export_present"]=(directory/"bomd_forces.dat").is_file()
            record["property_api_to_force_api_percent"]=100*phase["oeprop_wall_seconds"]/phase["force_wall_seconds"]
            record["output_files"]={str(path.relative_to(directory)):{"bytes":path.stat().st_size,"sha256":sha256(path)} for path in sorted(directory.rglob("*")) if path.is_file() and path.name not in ("geometry.xyz","probes.probe","control.nml")}
            if plan["mode"] in references:
                reference=references[plan["mode"]];record["repeat_physics"]=old.compare_physics(reference,data)
                if plan["mode"]=="on":
                    veg=property_difference(reference["properties"],data["properties"]);h=hessian_difference(reference["field_hessian"],data["field_hessian"])
                    require(max(veg.values())<=old.TOLERANCES["property_au"] and max(h.values())<=old.TOLERANCES["property_au"],"Repeated V/E/G/H changed")
                    record["repeat_property_max_abs"]={**veg,**h}
            else:references[plan["mode"]]=data
            record["validated"]=True;saved[plan["mode"]]=(record,data)
            if plan["order_in_pair"]==1:
                off,on=saved["off"],saved["on"];before=off[0]["whole_job_wall_seconds"];after=on[0]["whole_job_wall_seconds"]
                require(before>0 and after>0,"Invalid whole-job elapsed time")
                require(off[1]["gradient_source"]==on[1]["gradient_source"],"Paired gradient evidence sources differ")
                result["pairs"].append({"pair":plan["pair"],"discarded_warmup":plan["discarded_warmup"],
                    "execution_order":[p["mode"] for p in plans if p["pair"]==plan["pair"]],"off_wall_seconds":before,"on_wall_seconds":after,
                    "delta_wall_seconds":after-before,"whole_job_overhead_percent":100*(after/before-1),"matched_physics":old.compare_physics(off[1],on[1])})
            save()
        result["provenance_after"]=snapshot();require(result["provenance_after"]==baseline,"Provenance changed after benchmark")
        retained=[pair for pair in result["pairs"] if not pair["discarded_warmup"]]
        require(len(retained)==args.repeats,"Missing retained timing pairs")
        result["whole_job"]={name:describe([pair[name] for pair in retained]) for name in ("off_wall_seconds","on_wall_seconds","delta_wall_seconds","whole_job_overhead_percent")}
        result["api_phase_ratio"]={"scope":"Accumulated synchronous V/E/G/H API / force API; excludes driver IO; not whole-job overhead",
            "property_api_to_force_api_percent":describe([trial["property_api_to_force_api_percent"] for trial in result["trials"] if trial["mode"]=="on" and not trial["discarded_warmup"]])}
        result["timing_sample_scope"]="At least three retained paired trials" if args.repeats>=3 else "Fewer than three retained pairs: software smoke only"
        if any(trial["diagnostic_force_export_present"] for trial in result["trials"]):
            result["limitations"].append("Whole-job timings include validation-only raw-force export IO in both paired runs")
        result["passed"]=True
    except (ValueError,OSError,RuntimeError,KeyError,subprocess.SubprocessError) as error:
        result["error"]=str(error)
        try:result["provenance_after"]=snapshot()
        except (ValueError,OSError,subprocess.SubprocessError) as snapshot_error:result["provenance_after_error"]=str(snapshot_error)
    finally:
        save()
        trials=[{"pair":t["pair"],"discarded_warmup":t["discarded_warmup"],"mode":t["mode"],"order_in_pair":t["order_in_pair"],
                 "whole_job_wall_seconds":t.get("whole_job_wall_seconds"),**t.get("phase_max_rank_seconds",{}),
                 "property_api_to_force_api_percent":t.get("property_api_to_force_api_percent"),"hessian_rows":t.get("hessian_rows")} for t in result["trials"]]
        # Every successful trial has the same columns. Failed partial trials
        # remain in JSON; emit CSV only for complete, validated timing rows.
        write_table(workdir/"trial_timings.csv",[row for row,t in zip(trials,result["trials"]) if t.get("validated")])
        write_table(workdir/"paired_overhead.csv",[{key:value for key,value in pair.items() if key not in ("execution_order","matched_physics")} for pair in result["pairs"]])
    if args.plot and result["passed"]:plot_workflow(result,workdir/"workflow_overhead.pdf")
    return result


def plot_workflow(result,path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    retained=[pair for pair in result["pairs"] if not pair["discarded_warmup"]]
    with plt.rc_context({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42}):
        fig,axes=plt.subplots(1,2,figsize=(7,3.1),layout="constrained")
        for pair in retained:axes[0].plot([0,1],[pair["off_wall_seconds"],pair["on_wall_seconds"]],"o-",alpha=.65,markersize=3)
        axes[0].set_xticks([0,1],["Properties off","V/E/G/H on"]);axes[0].set_ylabel("Fresh-process whole-job time (s)")
        values=[pair["whole_job_overhead_percent"] for pair in retained]
        axes[1].plot(range(1,len(values)+1),values,"o",markersize=4);axes[1].axhline(0,color=".6",linewidth=.8)
        axes[1].axhline(statistics.median(values),color="C1",label="Median");axes[1].set_xlabel("Retained pair");axes[1].set_ylabel("Paired whole-job overhead (%)");axes[1].legend(frameon=False)
        fig.savefig(path,metadata={"Creator":"QUICK paired workflow benchmark"});plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("driver","install-prefix","source-snapshot","geometry","probes","keyword-file","workdir"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--backend",choices=("cpu","cuda"),required=True)
    parser.add_argument("--build-report",type=Path);parser.add_argument("--library-path",type=Path,action="append",default=[])
    parser.add_argument("--protocol-json",type=Path,help="Optional exact expected bare/numeric/string controls, required in selected-protocol benchmarks")
    parser.add_argument("--probe-units",choices=("angstrom","bohr"),default="angstrom")
    parser.add_argument("--nsteps",type=int,default=2);parser.add_argument("--dt-fs",type=float,default=.025)
    parser.add_argument("--stride",type=int,default=1);parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--warmup-pairs",type=int,default=1);parser.add_argument("--threads",type=int,default=1)
    parser.add_argument("--ranks",type=int,default=1);parser.add_argument("--launcher",default="")
    parser.add_argument("--force-source",choices=("printed","raw","auto"),default="auto")
    parser.add_argument("--timeout",type=float,default=1800.);parser.add_argument("--expected-driver-sha256")
    parser.add_argument("--expected-basis-functions",type=int,help="Optional explicit active Cartesian AO dimension")
    parser.add_argument("--prepare-only",action="store_true");parser.add_argument("--plot",action="store_true")
    args=parser.parse_args()
    if any(getattr(args,key)<1 for key in ("nsteps","stride","repeats","threads","ranks")) or args.warmup_pairs<0:
        parser.error("steps/stride/repeats/threads/ranks must be positive; warmup-pairs nonnegative")
    if any(not math.isfinite(value) or value<=0 for value in (args.dt_fs,args.timeout)):parser.error("dt-fs and timeout must be finite positive")
    if args.expected_driver_sha256 and not re.fullmatch(r"[0-9a-fA-F]{64}",args.expected_driver_sha256):parser.error("Expected driver SHA256 must contain 64 hexadecimal digits")
    if args.expected_basis_functions is not None and args.expected_basis_functions<1:parser.error("Expected basis dimension must be positive")
    try:result=benchmark(args)
    except (ValueError,OSError,subprocess.SubprocessError) as error:parser.exit(1,"Workflow benchmark setup failed: "+str(error)+"\n")
    print(json.dumps({"passed":result["passed"],"prepared_only":result["prepared_only"],"summary":str(args.workdir.resolve()/"workflow_summary.json"),"whole_job":result.get("whole_job"),"error":result.get("error")},indent=2))
    return 0 if result["passed"] or result["prepared_only"] else 1


if __name__=="__main__":raise SystemExit(main())
