#!/usr/bin/env python3
"""Small query/provenance/ordering tests using explicitly synthetic formats.

No QUICK calls, molecular observations, SCF or GPU execution are performed.
"""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

import field_uniformity as analysis
from query_plan import fibonacci_directions,make_plan,native_input
from convert_api_export import validate_plan,validate_native_state,convert_records


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    report={"passed":False,"scope":__doc__,"checks":[]}
    def check(name,error=0,tol=0):
        report["checks"].append({"name":name,"error":float(error),"tolerance":tol,"passed":bool(np.isfinite(error) and error<=tol)})
    def rejects(name,callback):
        try:callback()
        except (ValueError,KeyError):check(name);return
        check(name,1)
    try:
        centers=[{"center_id":"a","x_bohr":.2,"y_bohr":-.3,"z_bohr":.4},
                 {"center_id":"b","x_bohr":-.7,"y_bohr":.5,"z_bohr":.1}]
        directions=fibonacci_directions(16);radii=[.1,.02,.05]
        plan=make_plan(centers,radii,directions);positions=validate_plan(plan)
        check("Fibonacci_unit_norm",np.max(abs(np.linalg.norm(directions,axis=1)-1)),3e-16)
        check("Plan_count",abs(len(plan)-(2+2*3*16)))
        check("Centers_first",int(any(row["kind"]!="center" for row in plan[:2])))
        for row in plan[2:]:
            check("Plan_center_plus_u_"+str(row["query_index"]),np.max(abs(
                analysis.numeric(row,[x+"_bohr" for x in "xyz"])-positions[row["center_id"]]
                -analysis.numeric(row,["u_"+x+"_bohr" for x in "xyz"]))),8e-17)
        rejects("Reject_duplicate_radius",lambda:make_plan(centers,[.1,.1],directions))
        rejects("Reject_negative_radius",lambda:make_plan(centers,[-.1],directions))
        rejects("Reject_unnormalized_direction",lambda:make_plan(centers,[.1],directions*2))
        rejects("Reject_empty_direction_list",lambda:make_plan(centers,[.1],np.empty((0,3))))
        rejects("Reject_duplicate_center",lambda:make_plan([centers[0]]*2,[.1],directions))
        bad=copy.deepcopy(plan);bad[-1]["x_bohr"]+=.1
        rejects("Reject_absolute_probe_not_center_plus_u",lambda:validate_plan(bad))
        model={"atoms":[[2,[0.,0.,0.]]],"charges":[[2.,1.,0.,-.1]],"charge":1,"spin":1,"basis":"STO-3G"}
        keyword="UHF BASIS=STO-3G ENERGY SCF=400 DENSERMS=1e-10 CHARGE=1 MULT=2 EXTCHARGES"
        fixture=args.output/"synthetic-format-fixture";fixture.mkdir()
        text=native_input(model,keyword,plan);(fixture/"input.dat").write_text(text)
        check("Native_input_query_count",abs(int(text.splitlines()[1].split()[2])-len(plan)))
        rejects("Reject_oversize_keyword",lambda:native_input(model,"X"*257,plan))
        rejects("Reject_missing_EXTCHARGES",lambda:native_input(model,"HF ENERGY",plan))
        state="scf_converged T\nextra_API_calls 0\ndensity_error 0\nfock_error 0\n"
        (fixture/"result.dat").write_text(state)
        validate_native_state(fixture,fixture/"input.dat",model,plan);check("Native_state_format_contract")
        for key in ("extra_API_calls","density_error","fock_error"):
            (fixture/"result.dat").write_text(state.replace(key+" 0",key+" 1"))
            rejects("Reject_mutation_"+key,lambda:validate_native_state(fixture,fixture/"input.dat",model,plan))
        (fixture/"result.dat").write_text(state)
        wrong_model=copy.deepcopy(model);wrong_model["atoms"][0][1][0]=1
        rejects("Reject_native_source_model_mismatch",lambda:validate_native_state(fixture,fixture/"input.dat",wrong_model,plan))
        # Format fixture values are labels, not an invented molecular field.
        # The independent tensor entries make Fortran-vs-C reshape mistakes
        # visible, and a full permutation violation must be rejected.
        gpack=np.arange(1.,7.);hpack=np.arange(1.,11.)
        g=analysis.unpack(gpack,2);h=analysis.unpack(hpack,3)
        data=np.zeros((len(plan),44))
        for index,row in enumerate(plan):
            data[index,0]=index+1;data[index,1:4]=analysis.numeric(row,[x+"_bohr" for x in "xyz"])
            data[index,5:8]=[.1+index,.2+index,.3+index]
            data[index,8:17]=g.flatten(order="F");data[index,17:]=h.flatten(order="F")
        refs={row["center_id"]:{"n_e":.1,"grad_n_x":.2,"grad_n_y":.3,"grad_n_z":.4} for row in centers}
        converted,samples,symmetry=convert_records(plan,data,refs)
        check("Convert_G_Fortran_order",np.max(abs(analysis.numeric(converted[0],["g_"+x for x in analysis.G_COMPONENTS])-gpack)))
        check("Convert_H_Fortran_order",np.max(abs(analysis.numeric(converted[0],["h_"+x for x in analysis.H_COMPONENTS])-hpack)))
        check("Convert_direct_fields",np.max(abs(analysis.numeric(samples[0],["e_direct_"+x for x in "xyz"])-data[2,5:8])))
        check("Full_tensor_symmetry_measure",max(symmetry.values()))
        for offset,label in [(8,"G"),(17,"H")]:
            malformed=data.copy();malformed[0,offset+1]+=.1
            rejects("Reject_full_"+label+"_asymmetry_before_pack",lambda:convert_records(plan,malformed,refs))
        check("No_QUICK_molecular_manifest_emitted",int(any(args.output.rglob("manifest.json"))))
        # Explicitly all-zero plot avoids fabricating a positive log floor.
        zero_rows=[{"radius_bohr":r,"center_id":"zero",**{a+"_max_error":0. for a in analysis.APPROXIMATIONS}} for r in [.1,.2]]
        analysis.plot_errors(zero_rows,args.output/"zero_error_plot.pdf",analytic=True);check("Zero_error_plot_finite_export")
        report["passed"]=all(row["passed"] for row in report["checks"])
    except Exception as error:
        import traceback
        report["error"]=type(error).__name__+": "+str(error);report["traceback"]=traceback.format_exc()
    report["source_files_sha256"]={str(path):analysis.sha256(path) for path in Path(__file__).parent.glob("*.py")}
    (args.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"passed":report["passed"],"checks":len(report["checks"]),"error":report.get("error")}))
    return 0 if report["passed"] else 2


if __name__=="__main__":raise SystemExit(main())
