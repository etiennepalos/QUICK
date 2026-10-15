#!/usr/bin/env python3
"""Independent AO n and grad(n) from an exported QUICK density; no SCF/GPU.

Run with PySCF/NumPy. This uses Cartesian AO values/derivatives, not electric
field traces, so Poisson comparisons remain independent.
"""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import gto, lib
from pyscf.gto.basis import parse_gaussian

from field_uniformity import read_csv,numeric,write_csv,sha256

ROOT=Path(__file__).resolve().parents[2]
BOHR_ANGSTROM=.52917720859


def evaluate_density(molecule,density,points,mapping,phases):
    overlap=molecule.intor("int1e_ovlp",hermi=1);normalization=np.sqrt(np.diag(overlap))
    values=molecule.eval_gto("GTOval_cart",points)/normalization
    derivatives=molecule.eval_gto("GTOval_ip_cart",points)/normalization
    values=values[:,mapping]*phases;derivatives=derivatives[:,:,mapping]*phases
    # Averaging is exact for rho from an arbitrary stored matrix because the
    # product chi_mu*chi_nu is symmetric. Record the original asymmetry below.
    symmetric=(density+density.T)/2
    n=np.einsum("pm,mn,pn->p",values,symmetric,values)
    grad=2*np.einsum("pm,mn,xpn->px",values,symmetric,derivatives)
    normalized_s=overlap/normalization[:,None]/normalization[None,:]
    normalized_s=normalized_s[np.ix_(mapping,mapping)]*phases[:,None]*phases[None,:]
    return n,grad,normalized_s


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir",type=Path,required=True,help="density.dat,ao_basis.dat,overlap.dat from native API client")
    parser.add_argument("--model",type=Path,required=True,help="JSON atoms in Angstrom, charge,spin,basis; existing validation model schema")
    parser.add_argument("--basis-file",type=Path,required=True,help="Exact installed QUICK Gaussian .BAS")
    parser.add_argument("--centers",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();lib.num_threads(1)
    helper=ROOT/"validation/field_hessian/reference_field_hessian.py"
    spec=importlib.util.spec_from_file_location("exact_quick_cartesian_mapping",helper);reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    model=json.loads(args.model.read_text());symbols={int(z):gto.mole._symbol(int(z)) for z,_ in model["atoms"]}
    basis={symbol:parse_gaussian.load(str(args.basis_file),symbol,optimize=False) for symbol in symbols.values()}
    molecule=gto.M(atom=[[symbols[int(z)],(np.asarray(xyz)/BOHR_ANGSTROM).tolist()] for z,xyz in model["atoms"]],
                   basis=basis,unit="Bohr",cart=True,charge=model.get("charge",0),spin=model.get("spin",0),verbose=0)
    mapping,phases=reference.ao_mapping(molecule,reference.read_descriptors(args.export_dir/"ao_basis.dat"))
    lines=(args.export_dir/"density.dat").read_text().splitlines();n,unrestricted=lines[0].split();n=int(n)
    if n!=molecule.nao_nr() or unrestricted not in ("T","F"):
        raise ValueError("Density/basis dimension or spin-header mismatch")
    flat=np.array([float(x) for line in lines[1:] for x in line.split()]);expected=n*n*(2 if unrestricted=="T" else 1)
    if len(flat)!=expected or not np.isfinite(flat).all():raise ValueError("Malformed finite exported density")
    density=flat[:n*n].reshape(n,n,order="F")
    if unrestricted=="T":density=density+flat[n*n:].reshape(n,n,order="F")
    asymmetry=float(np.max(abs(density-density.T)))
    if asymmetry>1e-10:raise ValueError("Exported density is not symmetric within the native contract")
    centers=read_csv(args.centers,["center_id","x_bohr","y_bohr","z_bohr"])
    ids=[row["center_id"] for row in centers]
    if len(set(ids))!=len(ids) or any(not x for x in ids):raise ValueError("Center IDs must be unique and nonempty")
    points=np.array([numeric(row,[x+"_bohr" for x in "xyz"]) for row in centers])
    density_n,gradient,overlap=evaluate_density(molecule,density,points,mapping,phases)
    quick_overlap=np.loadtxt(args.export_dir/"overlap.dat",skiprows=1).reshape(n,n,order="F")
    overlap_error=float(np.max(abs(quick_overlap-overlap)))
    if overlap_error>2e-10:raise ValueError("Independent AO normalization/order overlap gate failed")
    if not np.isfinite(density_n).all() or not np.isfinite(gradient).all():raise ValueError("Nonfinite AO density reference")
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    rows=[{"center_id":row["center_id"],**{x+"_bohr":float(row[x+"_bohr"]) for x in "xyz"},
           "n_e":float(density_n[i]),**{"grad_n_"+x:float(gradient[i,j]) for j,x in enumerate("xyz")}}
          for i,row in enumerate(centers)]
    write_csv(args.output/"density_reference.csv",rows)
    metadata={"schema":"quick-field-uniformity-ao-reference-1","method":"Independent PySCF Cartesian AO values and first derivatives; exported spin-summed QUICK density; no SCF and no Poisson-derived density",
              "pyscf_version":pyscf.__version__,"basis_functions":n,"unrestricted":unrestricted=="T",
              "overlap_error":overlap_error,"density_asymmetry":asymmetry,"electron_count_trace_DS":float(np.einsum("ij,ji->",density,overlap)),
              "density_sha256":sha256(args.export_dir/"density.dat"),"basis_sha256":sha256(args.basis_file),"model_sha256":sha256(args.model),
              "ao_basis_sha256":sha256(args.export_dir/"ao_basis.dat"),"overlap_sha256":sha256(args.export_dir/"overlap.dat"),
              "AO_mapping_0based":mapping.tolist(),"AO_phases":phases.tolist(),
              "centers_sha256":sha256(args.centers),
              "source_files_sha256":{str(p.resolve()):sha256(p) for p in [Path(__file__),helper,args.model,args.basis_file,args.centers,
                                                                         args.export_dir/"density.dat",args.export_dir/"ao_basis.dat",args.export_dir/"overlap.dat"]}}
    (args.output/"reference.json").write_text(json.dumps(metadata,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"centers":len(rows),"basis_functions":n,"overlap_error":overlap_error,"SCFs":0}))


if __name__=="__main__":main()
