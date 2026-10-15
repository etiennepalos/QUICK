#!/usr/bin/env python3
"""Offline frozen-density field Taylor errors, Poisson checks and STF splits.

This module reads supplied direct queries. It never runs QUICK, changes a
density, or interprets finite directional sampling as a continuous bound.
"""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

import numpy as np

G_COMPONENTS = ("xx", "xy", "xz", "yy", "yz", "zz")
H_COMPONENTS = ("xxx", "xxy", "xxz", "xyy", "xyz", "xzz", "yyy", "yyz", "yzz", "zzz")
G_MULTIPLICITY = np.array([1, 2, 2, 1, 2, 1])
H_MULTIPLICITY = np.array([1, 3, 3, 3, 6, 3, 1, 3, 3, 1])
SOURCE_CONVENTION = "electrons+nuclei+enabled_mm; direct_applied_and_reaction_backgrounds_excluded"
UNITS = {"coordinates": "bohr", "field": "hartree/(e bohr)", "gradient": "hartree/(e bohr^2)",
         "hessian": "hartree/(e bohr^3)", "density": "bohr^-3", "density_gradient": "bohr^-4"}
CENTER_COLUMNS = (["center_id", "x_bohr", "y_bohr", "z_bohr"] + ["e_" + x for x in "xyz"]
                  + ["g_" + x for x in G_COMPONENTS] + ["h_" + x for x in H_COMPONENTS]
                  + ["n_e"] + ["grad_n_" + x for x in "xyz"])
SAMPLE_COLUMNS = (["center_id", "sample_id", "direction_id", "radius_bohr"]
                  + ["u_" + x + "_bohr" for x in "xyz"] + ["e_direct_" + x for x in "xyz"])
APPROXIMATIONS = ("constant", "linear", "quadratic")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unpack(packed, rank):
    names = G_COMPONENTS if rank == 2 else H_COMPONENTS if rank == 3 else None
    if names is None:
        raise ValueError("Only symmetric rank-two and rank-three tensors are supported")
    values = np.asarray(packed, dtype=float)
    if values.shape != (len(names),) or not np.isfinite(values).all():
        raise ValueError("Malformed finite packed tensor")
    result = np.zeros((3,) * rank)
    for name, value in zip(names, values):
        for indices in set(itertools.permutations(tuple("xyz".index(axis) for axis in name))):
            result[indices] = value
    return result


def pack(tensor):
    tensor = np.asarray(tensor, dtype=float)
    rank = tensor.ndim
    names = G_COMPONENTS if rank == 2 else H_COMPONENTS if rank == 3 else None
    if names is None or tensor.shape != (3,) * rank or not np.isfinite(tensor).all():
        raise ValueError("Malformed tensor")
    if any(not np.allclose(tensor, tensor.transpose(p), rtol=1e-12, atol=1e-14)
           for p in itertools.permutations(range(rank))):
        raise ValueError("Only fully symmetric tensors can be packed")
    return np.array([tensor[tuple("xyz".index(axis) for axis in name)] for name in names])


def packed_norm(values, rank):
    values = np.asarray(values, dtype=float)
    weights = G_MULTIPLICITY if rank == 2 else H_MULTIPLICITY if rank == 3 else None
    if weights is None or values.shape != weights.shape or not np.isfinite(values).all():
        raise ValueError("Malformed packed norm input")
    return float(np.sqrt(np.dot(weights, values * values)))


def stf_split(g, h):
    """Exact 3D irreducible splits; no density/source trace is discarded."""
    g, h = np.asarray(g), np.asarray(h)
    if g.shape != (3, 3) or h.shape != (3, 3, 3):
        raise ValueError("Expected G(3,3), H(3,3,3)")
    pack(g); pack(h)  # Validate the conservative/symmetric input contract.
    eye = np.eye(3)
    trace_g = float(np.trace(g))
    trace_h = np.einsum("ijj->i", h)
    g_source = trace_g * eye / 3
    h_source = (np.einsum("ij,k->ijk", eye, trace_h)
                + np.einsum("ik,j->ijk", eye, trace_h)
                + np.einsum("jk,i->ijk", eye, trace_h)) / 5
    return g - g_source, g_source, h - h_source, h_source, trace_g, trace_h


def taylor_fields(e, g, h, displacements):
    e, g, h, u = map(np.asarray, (e, g, h, displacements))
    if e.shape != (3,) or g.shape != (3, 3) or h.shape != (3, 3, 3) or u.ndim != 2 or u.shape[1] != 3:
        raise ValueError("Expected E(3), G(3,3), H(3,3,3), u(n,3)")
    if not all(np.isfinite(x).all() for x in (e, g, h, u)):
        raise ValueError("Nonfinite Taylor input")
    constant = np.broadcast_to(e, (len(u), 3)).copy()
    linear = constant + np.einsum("ij,pj->pi", g, u)
    quadratic = linear + .5 * np.einsum("ijk,pj,pk->pi", h, u, u)
    return dict(zip(APPROXIMATIONS, (constant, linear, quadratic)))


def empirical_prefix(radii, errors, tolerance):
    """Largest tested radius whose entire smaller-radius sample prefix passes."""
    radii, errors = np.asarray(radii), np.asarray(errors)
    if radii.ndim != 1 or radii.shape != errors.shape or not np.isfinite(radii).all() or not np.isfinite(errors).all():
        raise ValueError("Malformed empirical prefix input")
    if np.any(radii < 0) or np.any(errors < 0) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Radii, errors and tolerance must be nonnegative and finite")
    accepted = None
    for radius in sorted(set(radii.tolist())):
        if np.max(errors[radii == radius]) > tolerance:
            break
        accepted = radius
    return accepted


def read_csv(path, required):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError("Missing columns in " + str(path))
        rows = list(reader)
    if not rows:
        raise ValueError("Empty table " + str(path))
    return rows


def numeric(row, names):
    result = np.array([float(row[name]) for name in names])
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite input data")
    return result


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refuse to write an empty result table")
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def analyze(manifest_path, output_dir, tolerances=(), plot=False, plot_center=None):
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "quick-field-uniformity-input-1":
        raise ValueError("Unsupported input schema")
    if manifest.get("source_convention") != SOURCE_CONVENTION or manifest.get("units") != UNITS:
        raise ValueError("Explicit supported source convention and atomic units are required")
    provenance = manifest.get("provenance", {})
    if manifest.get("input_kind") == "quick-frozen-density":
        for key in ("density_sha256", "geometry_sha256", "basis_sha256"):
            if not isinstance(provenance.get(key), str) or re.fullmatch(r"[0-9a-f]{64}",provenance[key]) is None:
                raise ValueError("Missing frozen-density provenance: " + key)
        if provenance.get("same_density_and_geometry_for_all_queries") is not True:
            raise ValueError("All queries must use identical retained density and geometry")
    elif manifest.get("input_kind") != "analytic-validation":
        raise ValueError("Input kind must identify actual QUICK data or an analytic validation fixture")
    centers_path = (manifest_path.parent / manifest["centers_csv"]).resolve(strict=True)
    samples_path = (manifest_path.parent / manifest["samples_csv"]).resolve(strict=True)
    centers = read_csv(centers_path, CENTER_COLUMNS)
    samples = read_csv(samples_path, SAMPLE_COLUMNS)
    ids = [row["center_id"] for row in centers]
    if len(set(ids)) != len(ids) or any(not x for x in ids):
        raise ValueError("Center IDs must be unique and nonempty")
    sample_ids = [(r["center_id"], r["sample_id"]) for r in samples]
    if len(set(sample_ids)) != len(sample_ids) or any(not r["sample_id"] for r in samples):
        raise ValueError("Sample IDs must be unique within a center")
    if set(r["center_id"] for r in samples) != set(ids):
        raise ValueError("Every center must have samples and every sample must reference a center")
    if any(not row["direction_id"] for row in samples):
        raise ValueError("Direction IDs must be nonempty")
    tolerances = [float(x) for x in tolerances]
    if any(not math.isfinite(x) or x <= 0 for x in tolerances):
        raise ValueError("Absolute field-error tolerances must be positive and finite")
    output_dir = Path(output_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=False)
    center_output, error_output, radius_output, acceptance = [], [], [], []
    for center in centers:
        center_id = center["center_id"]
        xyz = numeric(center, [x + "_bohr" for x in "xyz"])
        e = numeric(center, ["e_" + x for x in "xyz"])
        g = unpack(numeric(center, ["g_" + x for x in G_COMPONENTS]), 2)
        h = unpack(numeric(center, ["h_" + x for x in H_COMPONENTS]), 3)
        n = numeric(center, ["n_e"])[0]
        grad_n = numeric(center, ["grad_n_" + x for x in "xyz"])
        g_stf, g_source, h_stf, h_source, trace_g, trace_h = stf_split(g, h)
        row = {"center_id": center_id, **dict(zip((x + "_bohr" for x in "xyz"), xyz)),
               "field_norm": float(np.linalg.norm(e)), "g_norm": packed_norm(pack(g), 2),
               "g_stf_norm": float(np.linalg.norm(g_stf)), "g_source_norm": float(np.linalg.norm(g_source)),
               "g_trace": trace_g, "n_e_oracle": n, "n_e_from_poisson": -trace_g/(4*np.pi),
               "poisson_g_residual": float(trace_g + 4*np.pi*n),
               "h_norm": packed_norm(pack(h), 3), "h_stf_norm": float(np.linalg.norm(h_stf)),
               "h_source_norm": float(np.linalg.norm(h_source)),
               "poisson_h_residual_norm": float(np.linalg.norm(trace_h + 4*np.pi*grad_n)),
               "g_norm_identity_residual": float(np.sum(g*g) - np.sum(g_stf*g_stf) - trace_g**2/3),
               "h_norm_identity_residual": float(np.sum(h*h) - np.sum(h_stf*h_stf) - 3*np.dot(trace_h,trace_h)/5)}
        row.update({"h_trace_"+x: float(trace_h[i]) for i,x in enumerate("xyz")})
        row.update({"grad_n_oracle_"+x: float(grad_n[i]) for i,x in enumerate("xyz")})
        row.update({"grad_n_poisson_"+x: float(-trace_h[i]/(4*np.pi)) for i,x in enumerate("xyz")})
        center_output.append(row)
        chosen = [sample for sample in samples if sample["center_id"] == center_id]
        u = np.array([numeric(sample, ["u_"+x+"_bohr" for x in "xyz"]) for sample in chosen])
        radii = np.array([numeric(sample, ["radius_bohr"])[0] for sample in chosen])
        if np.any(radii < 0) or not np.allclose(np.linalg.norm(u,axis=1),radii,rtol=1e-10,atol=1e-12):
            raise ValueError("Displacement norms do not match supplied nonnegative radii")
        directions={}
        for sample,vector,radius in zip(chosen,u,radii):
            if radius==0:
                continue
            direction=vector/radius
            key=sample["direction_id"]
            if key in directions and not np.allclose(direction,directions[key],rtol=1e-10,atol=1e-12):
                raise ValueError("A direction ID changes direction between radii")
            directions[key]=direction
        direct = np.array([numeric(sample,["e_direct_"+x for x in "xyz"]) for sample in chosen])
        predictions = taylor_fields(e,g,h,u)
        errors = {name: np.linalg.norm(prediction-direct,axis=1) for name,prediction in predictions.items()}
        for index,sample in enumerate(chosen):
            record = {"center_id":center_id,"sample_id":sample["sample_id"],"direction_id":sample["direction_id"],
                      "radius_bohr":float(radii[index]),**dict(zip(("u_"+x+"_bohr" for x in "xyz"),u[index])),
                      "direct_field_norm":float(np.linalg.norm(direct[index])),
                      "g_stf_field_increment_norm":float(np.linalg.norm(g_stf@u[index])),
                      "g_source_field_increment_norm":float(np.linalg.norm(g_source@u[index])),
                      "h_stf_field_increment_norm":float(np.linalg.norm(.5*np.einsum("ijk,j,k->i",h_stf,u[index],u[index]))),
                      "h_source_field_increment_norm":float(np.linalg.norm(.5*np.einsum("ijk,j,k->i",h_source,u[index],u[index])))}
            for name in APPROXIMATIONS:
                record[name+"_error_norm"] = float(errors[name][index])
                record.update({name+"_e_"+axis:float(predictions[name][index,i]) for i,axis in enumerate("xyz")})
            error_output.append(record)
        for radius in sorted(set(radii.tolist())):
            selected = radii == radius
            record = {"center_id":center_id,"radius_bohr":radius,"samples":int(np.sum(selected)),
                      "distinct_direction_ids":len({chosen[i]["direction_id"] for i in np.where(selected)[0]})}
            for name in APPROXIMATIONS:
                values=errors[name][selected]
                record.update({name+"_max_error":float(np.max(values)),name+"_rms_error":float(np.sqrt(np.mean(values**2))),
                               name+"_median_error":float(np.median(values)),name+"_p95_error":float(np.quantile(values,.95))})
            radius_output.append(record)
        for tolerance in tolerances:
            acceptance.append({"center_id":center_id,"absolute_field_error_tolerance_au":tolerance,
                               "largest_tested_prefix_radius_bohr":{name:empirical_prefix(radii,errors[name],tolerance) for name in APPROXIMATIONS}})
    write_csv(output_dir/"centers_analysis.csv",center_output)
    write_csv(output_dir/"sample_errors.csv",error_output)
    write_csv(output_dir/"radius_summary.csv",radius_output)
    summary={"schema":"quick-field-uniformity-analysis-1","input_kind":manifest["input_kind"],
             "source_convention":SOURCE_CONVENTION,"units":UNITS,"centers":len(centers),"samples":len(samples),
             "continuous_bound_claim":False,"empirical_radius_acceptance":acceptance,
             "interpretation":"Errors are observed only at supplied radii/directions for fixed density and geometry. Prefix acceptance does not certify a sphere, unsampled radii/directions, a trajectory, or relaxed molecular response.",
             "source_files_sha256":{str(p):sha256(p) for p in [Path(__file__).resolve(),manifest_path,centers_path,samples_path]},
             "provenance":provenance}
    (output_dir/"analysis.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    if plot:
        plot_errors(radius_output,output_dir/"field_error.pdf",center_id=plot_center,
                    analytic=manifest["input_kind"]=="analytic-validation")
    return summary


def plot_errors(rows,path,center_id=None,analytic=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected=[row for row in rows if center_id is None or row["center_id"]==center_id]
    if not selected:
        raise ValueError("Unknown plot center")
    radii=sorted({row["radius_bohr"] for row in selected if row["radius_bohr"]>0})
    with plt.rc_context({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42}):
        fig,ax=plt.subplots(figsize=(4.4,3.4),layout="constrained")
        omitted_zero=0
        for name,label in zip(APPROXIMATIONS,["Constant E","E + G u","E + G u + H[uu]/2"]):
            values=[max(row[name+"_max_error"] for row in selected if row["radius_bohr"]==radius) for radius in radii]
            nonzero=np.asarray(values)>0;omitted_zero+=int(np.sum(~nonzero))
            if np.any(nonzero):
                # Zero error has no logarithm. Omit it explicitly rather than
                # introducing an arbitrary plotting floor into the evidence.
                ax.loglog(np.asarray(radii)[nonzero],np.asarray(values)[nonzero],"o-",markersize=3,label=label)
        ax.set_xlabel("Displacement radius (bohr)");ax.set_ylabel("Absolute field error (a.u.)")
        ax.set_title("Analytic validation fixture" if analytic else "Frozen-density field reconstruction")
        if ax.lines:ax.legend(frameon=False,fontsize=8)
        else:ax.text(.5,.5,"All supplied errors are exactly zero",ha="center",transform=ax.transAxes)
        note="Maximum over supplied samples"
        if omitted_zero:note+="; zero values omitted"
        ax.text(.02,.02,note,transform=ax.transAxes,fontsize=7)
        fig.savefig(path,metadata={"Creator":"QUICK offline field-uniformity analysis"});plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--tolerance-au",type=float,nargs="*",default=[])
    parser.add_argument("--plot",action="store_true");parser.add_argument("--plot-center")
    args=parser.parse_args();result=analyze(args.manifest,args.output,args.tolerance_au,args.plot,args.plot_center)
    print(json.dumps({"centers":result["centers"],"samples":result["samples"],"continuous_bound_claim":False,"output":str(args.output)}))


if __name__=="__main__":
    main()
