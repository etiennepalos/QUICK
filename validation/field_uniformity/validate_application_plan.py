#!/usr/bin/env python3
"""Validate offline plans from bundled actual WATER27 geometry, no chemistry."""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from application import read_xyz_frame,select_rows,validate_keyword,prepare,analyze_exports,BOHR_ANGSTROM
from convert_api_export import validate_plan
from field_uniformity import read_csv,sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--basis-file",type=Path,required=True);args=parser.parse_args()
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2];fixture=root/"validation/solvation/fixtures"
    report={"passed":False,"scope":__doc__,"checks":[],"SCFs":0,"GPU_launches":0}
    def check(name,error=0,tol=0):
        report["checks"].append({"name":name,"error":float(error),"tolerance":tol,"passed":bool(np.isfinite(error) and error<=tol)})
    def rejects(name,callback):
        try:callback()
        except ValueError:check(name);return
        check(name,1)
    try:
        atoms,comment,count=read_xyz_frame(fixture/"water27_h2o20es.xyz",0)
        check("Actual_WATER27_atom_count",abs(len(atoms)-60));check("Single_actual_frame",abs(count-1))
        check("Actual_first_oxygen",abs(atoms[0][0]-8))
        check("XYZ_numeric_coordinate",np.max(abs(np.array(atoms[0][1])-[1.4449507,-2.3192362,1.3852172])),1e-15)
        rejects("Reject_nonexistent_frame",lambda:read_xyz_frame(fixture/"water27_h2o20es.xyz",1))
        rejects("Reject_duplicate_atom_selection",lambda:select_rows(atoms,[1,1]))
        rejects("Reject_invalid_atom_selection",lambda:select_rows(atoms,[61]))
        expected={"PBE0":None,"XC_RADIAL":100,"XC_ANGULAR":434,"MATRIXZERO":1e-14}
        card="PBE0 BASIS=STO-3G ENERGY CHARGE=0 MULT=1 SCF=400 DENSERMS=1e-10 CUTOFF=1e-12 XC_RADIAL=100 XC_ANGULAR=434 MATRIXZERO=1e-14 XCCUTOFF=1e-12"
        validate_keyword(card,expected);check("Explicit_numerical_protocol_accepted")
        rejects("Reject_protocol_grid_change",lambda:validate_keyword(card.replace("XC_RADIAL=100","XC_RADIAL=75"),expected))
        rejects("Reject_protocol_matrix_change",lambda:validate_keyword(card.replace("MATRIXZERO=1e-14","MATRIXZERO=1e-12"),expected))
        rejects("Reject_ambiguous_duplicate_keyword",lambda:validate_keyword(card+" XC_RADIAL=75",expected))
        manifest={"schema":"quick-field-uniformity-application-1","protocol":{"keyword":card,"expected_controls":expected,"basis_file":str(args.basis_file.resolve())},
                  "radii_bohr":[.01,.1],"fibonacci_directions":6,"cases":[{"id":"water-subset","geometry_xyz":str(fixture/"water27_h2o20es.xyz"),
                  "geometry_units":"angstrom","frame_index":0,"atom_indices_1based":[1,2,3],"probes":str(fixture/"water27_h2o20es.probe"),
                  "probe_units":"angstrom","probe_indices_1based":[1],"basis":"STO-3G","charge":0,"spin":0}]}
        path=args.output/"application.json";path.write_text(json.dumps(manifest,indent=2)+"\n")
        index=prepare(path,args.output/"prepared");case=index["cases"][0];folder=Path(case["directory"])
        check("Prepared_native_jobs_exact_zero",index["native_jobs_launched"])
        check("Prepared_sample_count",abs(case["queries"]-13))
        plan=read_csv(folder/"query_plan.csv",["query_index","kind","center_id"]);centers=validate_plan(plan)
        original=np.loadtxt(fixture/"water27_h2o20es.probe",skiprows=1)[0]/BOHR_ANGSTROM
        check("Angstrom_probe_to_bohr",np.max(abs(centers["probe-001"]-original)),1e-14)
        model=json.loads((folder/"model.json").read_text());check("Actual_atom_subset_preserved",np.max(abs(np.array([xyz for _,xyz in model["atoms"]])-np.array([xyz for _,xyz in atoms[:3]]))))
        provenance=json.loads((folder/"case-provenance.json").read_text())
        check("Actual_source_hash_retained",int(provenance["source_files_sha256"][str(fixture/"water27_h2o20es.xyz")]!=sha256(fixture/"water27_h2o20es.xyz")))
        check("No_continuous_bound_claim",int(index["continuous_bound_claim"]))
        check("No_property_or_density_fabricated",int(any(p.name in ("properties.dat","density.dat","trajectory.xyz") for p in folder.iterdir())))
        runs=args.output/"unexecuted-runs.json";runs.write_text(json.dumps({"water-subset":{"cuda":str(folder)}}))
        original_input=(folder/"input.dat").read_text();(folder/"input.dat").write_text(original_input+"invalid changed input\n")
        rejected=analyze_exports(args.output/"prepared",runs,Path("unused-oracle"),args.output/"rejected-changed-input",[],False)
        check("Reject_changed_prepared_input",int(rejected["passed"]))
        check("Changed_input_rejected_before_any_oracle",len(rejected["runs"]))
        (folder/"input.dat").write_text(original_input)
        report["passed"]=all(x["passed"] for x in report["checks"])
    except Exception as error:
        import traceback
        report["error"]=type(error).__name__+": "+str(error);report["traceback"]=traceback.format_exc()
    report["source_files_sha256"]={str(p):sha256(p) for p in [Path(__file__),Path(__file__).with_name("application.py")]}
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"passed":report["passed"],"checks":len(report["checks"]),"error":report.get("error"),"SCFs":0,"GPU_launches":0}))
    return 0 if report["passed"] else 2


if __name__=="__main__":raise SystemExit(main())
