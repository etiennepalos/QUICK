#!/usr/bin/env python3
"""Prepare actual geometry/frame query plans and analyze retained API exports.

No native QUICK jobs are launched. Preparation is offline. Analysis invokes
only the independent AO evaluator and offline converters/plotter.
"""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

from field_uniformity import write_csv,sha256
from query_plan import fibonacci_directions,make_plan,native_input

HERE=Path(__file__).resolve().parent
BOHR_ANGSTROM=.52917720859
SYMBOLS="X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr".split()


def read_xyz_frame(path,index):
    if not isinstance(index,int) or index<0:raise ValueError("frame_index must be a nonnegative zero-based integer")
    lines=path.read_text().splitlines();position=0;frames=[]
    while position<len(lines):
        if not lines[position].strip():position+=1;continue
        count=int(lines[position]);position+=1
        if count<1 or position+1+count>len(lines):raise ValueError("Incomplete positive-count XYZ frame")
        comment=lines[position];position+=1;atoms=[]
        for line in lines[position:position+count]:
            columns=line.split()
            if len(columns)!=4:raise ValueError("Expected XYZ element and three coordinates")
            z=int(columns[0]) if columns[0].isdigit() else SYMBOLS.index(columns[0].capitalize())
            xyz=np.array(columns[1:],dtype=float)
            if z<1 or not np.isfinite(xyz).all():raise ValueError("Invalid finite atom row")
            atoms.append([z,xyz.tolist()])
        frames.append((atoms,comment));position+=count
    if index>=len(frames):raise ValueError("Selected actual XYZ frame does not exist")
    if any([z for z,_ in frame[0]]!=[z for z,_ in frames[0][0]] for frame in frames):
        raise ValueError("XYZ atom identities/order change between supplied frames")
    return *frames[index],len(frames)


def select_rows(rows,indices):
    if indices is None:return rows,list(range(1,len(rows)+1))
    if not indices or any(not isinstance(i,int) or i<1 or i>len(rows) for i in indices) or len(set(indices))!=len(indices):
        raise ValueError("Selection indices must be unique valid one-based integers")
    return [rows[i-1] for i in indices],indices


def validate_keyword(keyword,expected):
    tokens={}
    for token in keyword.upper().split():
        key,separator,value=token.partition("=")
        if key in tokens:raise ValueError("Duplicate/ambiguous keyword "+key)
        tokens[key]=value if separator else None
    for key,wanted in expected.items():
        key=key.upper()
        if key not in tokens:raise ValueError("Missing explicit protocol control "+key)
        actual=tokens[key]
        if wanted is None:
            if actual is not None:raise ValueError("Expected bare protocol keyword "+key)
        elif isinstance(wanted,(int,float)) and not isinstance(wanted,bool):
            if actual is None or not math.isfinite(float(actual.replace("D","E"))) or float(actual.replace("D","E"))!=wanted:
                raise ValueError("Numeric protocol control differs: "+key)
        elif actual!=str(wanted).upper():raise ValueError("Protocol control differs: "+key)
    return tokens


def prepare(manifest_path,output):
    manifest_path=manifest_path.resolve(strict=True);manifest=json.loads(manifest_path.read_text());base=manifest_path.parent
    if manifest.get("schema")!="quick-field-uniformity-application-1":raise ValueError("Unsupported application schema")
    protocol=manifest["protocol"]
    if not protocol.get("expected_controls"):raise ValueError("Explicit protocol expected_controls are required")
    directions=fibonacci_directions(manifest["fibonacci_directions"])
    if "extra_directions" in manifest:
        extra=np.asarray(manifest["extra_directions"],dtype=float)
        if extra.size:directions=np.vstack([directions,extra])
    cases=manifest["cases"];ids=[case["id"] for case in cases]
    if not ids or len(set(ids))!=len(ids) or any(not x or Path(x).name!=x or x in (".","..") for x in ids):
        raise ValueError("Case IDs must be unique nonempty directory-safe names")
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    index={"schema":"quick-field-uniformity-prepared-1","manifest":str(manifest_path),"cases":[],
           "source_files_sha256":{str(manifest_path):sha256(manifest_path),str(Path(__file__)):sha256(__file__)},
           "scope":"Explicit static geometries or selected actual frames; no ensemble or continuous-bound claim",
           "protocol":protocol,"native_jobs_launched":0,"continuous_bound_claim":False}
    for case in cases:
        if case.get("geometry_units")!="angstrom" or case.get("probe_units") not in ("angstrom","bohr"):
            raise ValueError("Explicit geometry_units=angstrom and probe_units are required")
        geometry=(base/case["geometry_xyz"]).resolve(strict=True)
        all_atoms,comment,nframes=read_xyz_frame(geometry,case.get("frame_index",0))
        atoms,atom_indices=select_rows(all_atoms,case.get("atom_indices_1based"))
        probes_path=(base/case["probes"]).resolve(strict=True)
        lines=probes_path.read_text().splitlines();nprobes=int(lines[0]);probes=np.atleast_2d(np.loadtxt(probes_path,skiprows=1))
        if probes.shape!=(nprobes,3) or not np.isfinite(probes).all():raise ValueError("Malformed finite probe file")
        probes,probe_indices=select_rows(probes.tolist(),case.get("probe_indices_1based"));probes=np.array(probes)
        if case["probe_units"]=="angstrom":probes=probes/BOHR_ANGSTROM
        keyword=case.get("keyword",protocol.get("keyword",""));tokens=validate_keyword(keyword,protocol["expected_controls"])
        if tokens.get("BASIS")!=case["basis"].upper():raise ValueError("Card/model basis differs")
        if int(tokens.get("CHARGE","0"))!=case["charge"] or int(tokens.get("MULT","1"))!=case["spin"]+1:
            raise ValueError("Explicit card charge/multiplicity differs from model")
        model={"atoms":atoms,"charges":case.get("charges",[]),"basis":case["basis"],"charge":case["charge"],"spin":case["spin"]}
        centers=[{"center_id":f"probe-{original:03d}",**dict(zip((x+"_bohr" for x in "xyz"),point))} for original,point in zip(probe_indices,probes)]
        rows=make_plan(centers,manifest["radii_bohr"],directions)
        # native_input checks complete keyword length and explicit MM enablement.
        text=native_input(model,keyword,rows);folder=output/case["id"];folder.mkdir()
        write_csv(folder/"centers.csv",centers);write_csv(folder/"query_plan.csv",rows)
        (folder/"input.dat").write_text(text);(folder/"keyword.txt").write_text(keyword+"\n")
        (folder/"model.json").write_text(json.dumps(model,indent=2)+"\n")
        basis_file=(base/case.get("basis_file",protocol["basis_file"])).resolve(strict=True)
        provenance={"case_id":case["id"],"geometry_source":str(geometry),"frame_index_zero_based":case.get("frame_index",0),
                    "frames_in_source":nframes,"source_comment":comment,"source_atom_indices_1based":atom_indices,
                    "source_probe_indices_1based":probe_indices,"probe_source":str(probes_path),"probe_units_in_source":case["probe_units"],
                    "source_files_sha256":{str(p):sha256(p) for p in [geometry,probes_path,basis_file]},
                    "density_policy":"Recompute or explicitly identify a new retained density for this exact model/protocol; no old-frame density is inferred",
                    "native_input_sha256":sha256(folder/"input.dat"),"radii_bohr":manifest["radii_bohr"],"directions":directions.tolist(),
                    "queries":len(rows),"centers":len(centers),"continuous_bound_claim":False}
        provenance["prepared_files_sha256"]={p.name:sha256(p) for p in [folder/"input.dat",folder/"model.json",folder/"centers.csv",folder/"query_plan.csv",folder/"keyword.txt"]}
        (folder/"case-provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
        index["cases"].append({"id":case["id"],"directory":str(folder),"basis_file":str(basis_file),"queries":len(rows),"centers":len(centers),
                               "case_provenance_sha256":sha256(folder/"case-provenance.json")})
    (output/"application-index.json").write_text(json.dumps(index,indent=2)+"\n")
    return index


def analyze_exports(prepared,runs_json,oracle_python,output,tolerances,plot):
    index=json.loads((prepared/"application-index.json").read_text());runs=json.loads(runs_json.read_text())
    if index.get("schema")!="quick-field-uniformity-prepared-1":raise ValueError("Prepared application index required")
    if not isinstance(runs,dict) or not runs or not set(runs)<=set(case["id"] for case in index["cases"]):
        raise ValueError("Run mapping must identify a nonempty subset of prepared cases")
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    report={"passed":False,"schema":"quick-field-uniformity-application-analysis-1","runs":[],"native_jobs_launched":0,
            "continuous_bound_claim":False,"scope":index["scope"],
            "source_files_sha256":{str(p):sha256(p) for p in [Path(__file__),prepared/"application-index.json",runs_json]}}
    def execute(argv,folder,label):
        started={"label":label,"argv":list(map(str,argv))};report["runs"].append(started)
        with (folder/(label+".log")).open("w") as log:
            process=subprocess.run(list(map(str,argv)),stdout=log,stderr=subprocess.STDOUT)
        started["returncode"]=process.returncode
        if process.returncode:raise ValueError("Offline helper failed: "+label)
    try:
        for case in index["cases"]:
            if case["id"] not in runs:continue
            folder=Path(case["directory"]);backends=runs[case["id"]]
            if sha256(folder/"case-provenance.json")!=case["case_provenance_sha256"]:
                raise ValueError("Prepared case provenance changed after planning")
            provenance=json.loads((folder/"case-provenance.json").read_text())
            if any(sha256(folder/name)!=digest for name,digest in provenance["prepared_files_sha256"].items()):
                raise ValueError("Prepared input/model/plan changed after planning")
            if sha256(case["basis_file"])!=provenance["source_files_sha256"][case["basis_file"]]:
                raise ValueError("Exact basis changed after planning")
            if not isinstance(backends,dict) or not backends:raise ValueError("Every selected case needs native backend exports")
            for backend,path in backends.items():
                if not backend or Path(backend).name!=backend:raise ValueError("Backend label must be directory-safe")
                native=(runs_json.parent/path).resolve(strict=True);dest=output/(case["id"]+"-"+backend);dest.mkdir()
                execute([oracle_python,HERE/"ao_density_oracle.py","--export-dir",native,"--model",folder/"model.json",
                         "--basis-file",case["basis_file"],"--centers",folder/"centers.csv","--output",dest/"ao-reference"],dest,"ao-reference")
                execute([sys.executable,HERE/"convert_api_export.py","--query-plan",folder/"query_plan.csv",
                         "--export-dir",native,"--native-input",folder/"input.dat","--density-reference",dest/"ao-reference",
                         "--model",folder/"model.json","--backend",backend,"--output",dest/"analysis-input"],dest,"convert")
                command=[sys.executable,HERE/"field_uniformity.py","--manifest",dest/"analysis-input/manifest.json","--output",dest/"analysis"]
                if tolerances:command += ["--tolerance-au",*map(str,tolerances)]
                if plot:command.append("--plot")
                execute(command,dest,"analysis")
        report["passed"]=True
    except Exception as error:
        import traceback
        report["error"]=type(error).__name__+": "+str(error);report["traceback"]=traceback.format_exc()
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="mode",required=True)
    prep=sub.add_parser("prepare");prep.add_argument("--manifest",type=Path,required=True);prep.add_argument("--output",type=Path,required=True)
    ana=sub.add_parser("analyze");ana.add_argument("--prepared",type=Path,required=True);ana.add_argument("--runs-json",type=Path,required=True)
    ana.add_argument("--oracle-python",type=Path,required=True);ana.add_argument("--output",type=Path,required=True)
    ana.add_argument("--tolerance-au",type=float,nargs="*",default=[]);ana.add_argument("--plot",action="store_true")
    args=parser.parse_args()
    if args.mode=="prepare":
        result=prepare(args.manifest,args.output);print(json.dumps({"prepared_cases":len(result["cases"]),"native_jobs_launched":0}));return 0
    result=analyze_exports(args.prepared,args.runs_json,args.oracle_python,args.output,args.tolerance_au,args.plot)
    print(json.dumps({"passed":result["passed"],"offline_helpers":len(result["runs"]),"native_jobs_launched":0,"error":result.get("error")}))
    return 0 if result["passed"] else 2


if __name__=="__main__":raise SystemExit(main())
