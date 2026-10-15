#!/usr/bin/env python3
"""Plan center and displaced observation queries without running chemistry."""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from field_uniformity import read_csv, write_csv, numeric, sha256


def fibonacci_directions(count):
    if not isinstance(count,int) or count<6:
        raise ValueError("Use at least six deterministic sphere directions")
    k=np.arange(count);z=1-2*(k+.5)/count;az=k*math.pi*(3-math.sqrt(5));radial=np.sqrt(1-z*z)
    return np.column_stack([radial*np.cos(az),radial*np.sin(az),z])


def make_plan(centers,radii,directions):
    radii=np.asarray(radii,dtype=float);directions=np.asarray(directions,dtype=float)
    if radii.ndim!=1 or not len(radii) or not np.isfinite(radii).all() or np.any(radii<=0):
        raise ValueError("Query radii must be positive finite bohr values")
    if len(set(radii.tolist()))!=len(radii):raise ValueError("Duplicate query radii")
    if directions.ndim!=2 or directions.shape[1]!=3 or not len(directions) or not np.isfinite(directions).all():
        raise ValueError("Expected finite direction vectors (n,3)")
    if not np.allclose(np.linalg.norm(directions,axis=1),1,atol=1e-12,rtol=1e-12):
        raise ValueError("Direction vectors must already be normalized")
    ids=[c["center_id"] for c in centers]
    if not ids or len(set(ids))!=len(ids) or any(not x for x in ids):raise ValueError("Center IDs must be unique and nonempty")
    rows=[]
    # Centers first permits a later optimized collector to compute H only at
    # these rows; E alone is needed at all remaining displaced points.
    for center in centers:
        point=numeric(center,[x+"_bohr" for x in "xyz"])
        rows.append({"query_index":len(rows)+1,"kind":"center","center_id":center["center_id"],"sample_id":"center",
                     "direction_id":"center","radius_bohr":0.,**dict(zip((x+"_bohr" for x in "xyz"),point)),
                     **{"u_"+x+"_bohr":0. for x in "xyz"}})
    for center in centers:
        point=numeric(center,[x+"_bohr" for x in "xyz"])
        for ri,radius in enumerate(sorted(radii.tolist())):
            for di,direction in enumerate(directions):
                u=radius*direction;position=point+u
                rows.append({"query_index":len(rows)+1,"kind":"sample","center_id":center["center_id"],
                             "sample_id":f"r{ri:03d}-d{di:04d}","direction_id":f"d{di:04d}","radius_bohr":radius,
                             **dict(zip((x+"_bohr" for x in "xyz"),position)),
                             **dict(zip(("u_"+x+"_bohr" for x in "xyz"),u))})
    return rows


def native_input(model,keyword,rows):
    """Existing native client format; atoms/MM Angstrom, probes bohr."""
    keyword=keyword.strip()
    if not keyword or "\n" in keyword or "\r" in keyword or len(keyword)>256:
        raise ValueError("Native keyword card must be one nonempty line of at most 256 characters")
    atoms=model["atoms"];charges=model.get("charges",[])
    if not atoms:raise ValueError("At least one atom is required")
    if charges and "EXTCHARGES" not in keyword.upper().split():
        raise ValueError("Nonempty MM sources require the explicit EXTCHARGES keyword")
    lines=[keyword,f"{len(atoms)} {len(charges)} {len(rows)}"]
    for z,xyz in atoms:
        if int(z)!=z or z<1:raise ValueError("Expected positive atomic number")
        xyz=np.asarray(xyz,dtype=float)
        if xyz.shape!=(3,) or not np.isfinite(xyz).all():raise ValueError("Malformed finite atom coordinate")
        lines.append(str(int(z))+" "+" ".join(f"{value:.17g}" for value in xyz))
    for charge in charges:
        charge=np.asarray(charge,dtype=float)
        if charge.shape!=(4,) or not np.isfinite(charge).all():raise ValueError("Malformed finite MM row x,y,z,q")
        lines.append(" ".join(f"{value:.17g}" for value in charge))
    for row in rows:
        lines.append(" ".join(f"{value:.17g}" for value in numeric(row,[x+"_bohr" for x in "xyz"])))
    return "\n".join(lines)+"\n"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--centers",type=Path,required=True,help="CSV: center_id,x_bohr,y_bohr,z_bohr")
    parser.add_argument("--radii-bohr",type=float,nargs="+",required=True)
    parser.add_argument("--fibonacci-count",type=int,default=32)
    parser.add_argument("--directions-json",type=Path,help="Optional explicit normalized direction list")
    parser.add_argument("--model",type=Path,help="Optional model JSON to emit native input.dat; requires --keyword-file")
    parser.add_argument("--keyword-file",type=Path,help="One complete native QUICK keyword card, at most 256 characters")
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    if bool(args.model)!=bool(args.keyword_file):parser.error("--model and --keyword-file must be supplied together")
    centers=read_csv(args.centers,["center_id","x_bohr","y_bohr","z_bohr"])
    directions=np.asarray(json.loads(args.directions_json.read_text())) if args.directions_json else fibonacci_directions(args.fibonacci_count)
    rows=make_plan(centers,args.radii_bohr,directions);args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    write_csv(args.output/"query_plan.csv",rows)
    with (args.output/"probes_bohr.dat").open("w") as handle:
        handle.write(str(len(rows))+"\n")
        for row in rows:handle.write(" ".join(f"{row[x+'_bohr']:.17g}" for x in "xyz")+"\n")
    metadata={"schema":"quick-field-uniformity-query-plan-1","centers":len(centers),"queries":len(rows),
              "radii_bohr":sorted(args.radii_bohr),"directions":directions.tolist(),"coordinates":"bohr",
              "required_queries":{"center":"E3,G6,H10 at fixed density/geometry","sample":"direct E3 at center+u, same density/geometry"},
              "source_files_sha256":{str(args.centers.resolve()):sha256(args.centers),str(Path(__file__).resolve()):sha256(__file__)},
              "continuous_bound_claim":False}
    if args.model:
        text=native_input(json.loads(args.model.read_text()),args.keyword_file.read_text(),rows)
        (args.output/"input.dat").write_text(text)
        metadata["source_files_sha256"].update({str(p.resolve()):sha256(p) for p in [args.model,args.keyword_file]})
    (args.output/"query_plan.json").write_text(json.dumps(metadata,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"centers":len(centers),"direct_queries":len(rows),"output":str(args.output)}))


if __name__=="__main__":main()
