#!/usr/bin/env python3
"""Convert native benchmark/Hessian client exports and a query plan offline."""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from field_uniformity import (G_COMPONENTS,H_COMPONENTS,SOURCE_CONVENTION,UNITS,pack,numeric,
                              read_csv,write_csv,sha256)


def validate_plan(plan):
    centers={}
    for index,row in enumerate(plan):
        if int(row["query_index"])!=index+1:raise ValueError("Plan query order changed")
        if row["kind"]=="center":
            key=row["center_id"]
            if not key or key in centers:raise ValueError("Duplicate or empty plan center ID")
            if numeric(row,["radius_bohr"])[0]!=0 or np.any(numeric(row,["u_"+x+"_bohr" for x in "xyz"])!=0):
                raise ValueError("A center row must have exactly zero displacement")
            centers[key]=numeric(row,[x+"_bohr" for x in "xyz"])
        elif row["kind"]!="sample":raise ValueError("Unknown query kind")
    if not centers:raise ValueError("Missing query centers")
    for row in plan:
        if row["kind"]=="sample":
            if row["center_id"] not in centers:raise ValueError("Sample references unknown center")
            u=numeric(row,["u_"+x+"_bohr" for x in "xyz"]);radius=numeric(row,["radius_bohr"])[0]
            if radius<=0 or not np.isclose(np.linalg.norm(u),radius,atol=1e-12,rtol=1e-10):
                raise ValueError("Invalid sample radius or displacement")
            if not np.allclose(numeric(row,[x+"_bohr" for x in "xyz"]),centers[row["center_id"]]+u,atol=1e-12,rtol=1e-12):
                raise ValueError("Sample absolute coordinates are not center+u")
    return centers


def validate_native_state(export_dir,native_input,model,plan):
    # A supplied filename alone cannot prove a frozen-state calculation. Keep
    # the native client's zero-mutation/zero-extra-SCF assertions in evidence.
    result={line.split()[0]:line.split()[1:] for line in (export_dir/"result.dat").read_text().splitlines() if line.split()}
    if result.get("scf_converged")!=["T"]:raise ValueError("Native initial SCF did not converge")
    for key in ("extra_API_calls","density_error","fock_error"):
        if len(result.get(key,[]))!=1 or float(result[key][0])!=0:
            raise ValueError("Native state invariant failed: "+key)
    lines=native_input.read_text().splitlines()
    atoms=model["atoms"];charges=model.get("charges",[])
    if list(map(int,lines[1].split()))!=[len(atoms),len(charges),len(plan)]:
        raise ValueError("Native input/model/query dimensions disagree")
    expected=[[z,*xyz] for z,xyz in atoms]+charges
    for line,row in zip(lines[2:],expected):
        if not np.allclose(np.asarray(line.split(),dtype=float),row,atol=1e-12,rtol=1e-12):
            raise ValueError("Native input atom/MM sources differ from model")
    if charges and "EXTCHARGES" not in lines[0].upper().split():
        raise ValueError("Model MM sources are not explicitly enabled")
    probes=np.atleast_2d(np.asarray([line.split() for line in lines[2+len(expected):] if line.strip()],dtype=float))
    expected_probes=np.array([numeric(row,[x+"_bohr" for x in "xyz"]) for row in plan])
    if probes.shape!=expected_probes.shape or not np.allclose(probes,expected_probes,atol=1e-12,rtol=1e-12):
        raise ValueError("Native input probe coordinates differ from plan")


def convert_records(plan,data,refs):
    centers=[];samples=[];symmetry={"G":0.,"H":0.}
    for index,row in enumerate(plan):
        point=numeric(row,[x+"_bohr" for x in "xyz"])
        if not np.allclose(data[index,1:4],point,atol=1e-12,rtol=1e-12):raise ValueError("Native probe coordinates disagree with plan")
        if row["kind"]=="center":
            g=data[index,8:17].reshape(3,3,order="F");h=data[index,17:].reshape(3,3,3,order="F")
            symmetry["G"]=max(symmetry["G"],float(np.max(abs(g-g.T))))
            symmetry["H"]=max(symmetry["H"],max(float(np.max(abs(h-h.transpose(p)))) for p in itertools.permutations(range(3))))
            record={"center_id":row["center_id"],**dict(zip((x+"_bohr" for x in "xyz"),point)),
                    **dict(zip(("e_"+x for x in "xyz"),data[index,5:8])),
                    **dict(zip(("g_"+x for x in G_COMPONENTS),pack(g))),
                    **dict(zip(("h_"+x for x in H_COMPONENTS),pack(h)))}
            record.update({key:float(refs[row["center_id"]][key]) for key in ["n_e","grad_n_x","grad_n_y","grad_n_z"]});centers.append(record)
        elif row["kind"]=="sample":
            record={key:row[key] for key in ["center_id","sample_id","direction_id","radius_bohr","u_x_bohr","u_y_bohr","u_z_bohr"]}
            record.update(dict(zip(("e_direct_"+x for x in "xyz"),data[index,5:8])));samples.append(record)
        else:raise ValueError("Unknown query kind")
    return centers,samples,symmetry


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-plan",type=Path,required=True)
    parser.add_argument("--export-dir",type=Path,required=True)
    parser.add_argument("--density-reference",type=Path,required=True,help="Directory containing independent density_reference.csv/reference.json")
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--native-input",type=Path,help="Defaults to export-dir/input.dat; verified against model and plan")
    parser.add_argument("--backend",required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    plan=read_csv(args.query_plan,["query_index","kind","center_id","sample_id","direction_id","radius_bohr",
                                   "x_bohr","y_bohr","z_bohr","u_x_bohr","u_y_bohr","u_z_bohr"])
    plan_centers=validate_plan(plan)
    native_input=args.native_input or args.export_dir/"input.dat"
    validate_native_state(args.export_dir,native_input,json.loads(args.model.read_text()),plan)
    properties=args.export_dir/"properties.dat";data=np.atleast_2d(np.loadtxt(properties,skiprows=1))
    if data.shape!=(len(plan),44) or not np.isfinite(data).all():raise ValueError("Expected finite native properties.dat with 44 columns per query")
    if int(properties.read_text().splitlines()[0])!=len(plan):raise ValueError("Native property header count changed")
    if not np.array_equal(data[:,0],np.arange(1,len(plan)+1)):raise ValueError("Native query index order changed")
    reference=json.loads((args.density_reference/"reference.json").read_text())
    if reference.get("schema")!="quick-field-uniformity-ao-reference-1":raise ValueError("Independent AO reference metadata required")
    if reference["density_sha256"]!=sha256(args.export_dir/"density.dat") or reference["model_sha256"]!=sha256(args.model):
        raise ValueError("Property and AO-reference density/geometry provenance differ")
    if any(reference[key+"_sha256"]!=sha256(args.export_dir/filename)
           for key,filename in [("ao_basis","ao_basis.dat"),("overlap","overlap.dat")]):
        raise ValueError("Property and AO-reference basis/overlap provenance differ")
    density_rows=read_csv(args.density_reference/"density_reference.csv",["center_id","x_bohr","y_bohr","z_bohr","n_e","grad_n_x","grad_n_y","grad_n_z"])
    refs={row["center_id"]:row for row in density_rows}
    if len(refs)!=len(density_rows):raise ValueError("Duplicate density-reference centers")
    if set(refs)!=set(plan_centers):raise ValueError("Independent AO centers differ from query plan")
    for key,point in plan_centers.items():
        if not np.allclose(point,numeric(refs[key],[x+"_bohr" for x in "xyz"]),atol=1e-12,rtol=1e-12):
            raise ValueError("Independent AO center coordinates differ from query plan")
    centers,samples,symmetry=convert_records(plan,data,refs)
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    write_csv(args.output/"centers.csv",centers);write_csv(args.output/"samples.csv",samples)
    manifest={"schema":"quick-field-uniformity-input-1","input_kind":"quick-frozen-density","source_convention":SOURCE_CONVENTION,
              "units":UNITS,"centers_csv":"centers.csv","samples_csv":"samples.csv",
              "provenance":{"density_sha256":sha256(args.export_dir/"density.dat"),"geometry_sha256":sha256(args.model),
                            "basis_sha256":reference["basis_sha256"],"backend":args.backend,
                            "same_density_and_geometry_for_all_queries":True,"density_reference_method":reference["method"],
                            "native_state_assertions":"Converged initial SCF; density_error=fock_error=extra_API_calls=0",
                            "full_tensor_max_permutation_discrepancy":symmetry,
                            "packing_policy":"Validate full tensor symmetry (rtol=1e-12,atol=1e-14), then select canonical components; never symmetrize",
                            "source_files_sha256":{str(p.resolve()):sha256(p) for p in [Path(__file__),args.query_plan,properties,args.model,
                                args.export_dir/"density.dat",args.export_dir/"ao_basis.dat",args.export_dir/"overlap.dat",args.export_dir/"result.dat",native_input,
                                args.density_reference/"reference.json",args.density_reference/"density_reference.csv"]}}}
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"centers":len(centers),"samples":len(samples),"output":str(args.output),"new_SCFs":0}))


if __name__=="__main__":main()
