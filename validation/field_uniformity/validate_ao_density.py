#!/usr/bin/env python3
"""Validate independent Cartesian AO n/grad(n) against primitive polynomials.

No SCF, QUICK call or GPU launch. Includes s/p/d/f values, AO permutation and
phases, spin-summed density linearity, and two-step density finite differences.
"""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from pyscf import gto,lib
from scipy.special import factorial2

from ao_density_oracle import evaluate_density
from field_uniformity import sha256


def polynomial_values(points):
    values=[];derivatives=[]
    for ell,alpha in enumerate([.7,.5,.4,.3]):
        for lx in range(ell,-1,-1):
            for ly in range(ell-lx,-1,-1):
                angular=(lx,ly,ell-lx-ly)
                denom=math.prod(1 if power==0 else factorial2(2*power-1) for power in angular)
                norm=(2*alpha/math.pi)**.75*math.sqrt((4*alpha)**ell/denom)
                def term(powers):
                    return np.prod(points**np.asarray(powers),axis=1)
                exponential=np.exp(-alpha*np.einsum("pi,pi->p",points,points))
                value=norm*term(angular)*exponential;values.append(value)
                gradient=[]
                for axis,power in enumerate(angular):
                    powers=list(angular);powers[axis]+=1
                    derivative=-2*alpha*term(powers)
                    if power:
                        powers=list(angular);powers[axis]-=1;derivative+=power*term(powers)
                    gradient.append(norm*exponential*derivative)
                derivatives.append(gradient)
    return np.asarray(values).T,np.asarray(derivatives).transpose(1,2,0)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False);lib.num_threads(1)
    report={"passed":False,"scope":__doc__,"checks":[]}
    def check(name,error,tol):
        report["checks"].append({"name":name,"error":float(error),"tolerance":tol,"passed":bool(np.isfinite(error) and error<=tol)})
    try:
        molecule=gto.M(atom="He 0 0 0",basis={"He":[[ell,[alpha,1.]] for ell,alpha in enumerate([.7,.5,.4,.3])]},
                       unit="Bohr",cart=True,charge=1,spin=1,verbose=0)
        points=np.array([[0.,0.,0.],[.3,-.2,.7],[-.8,.4,.1],[1.1,-.7,.6]])
        rng=np.random.default_rng(20261015);n=molecule.nao_nr();mapping=rng.permutation(n);phases=rng.choice([-1,1],n)
        a=rng.normal(size=(n,3));b=rng.normal(size=(n,2));alpha=a@a.T*.01;beta=b@b.T*.007;density=alpha+beta
        observed,gradient,overlap=evaluate_density(molecule,density,points,mapping,phases)
        phi,dphi=polynomial_values(points);phi=phi[:,mapping]*phases;dphi=dphi[:,:,mapping]*phases
        expected=np.einsum("pm,mn,pn->p",phi,density,phi)
        expected_gradient=2*np.einsum("pm,mn,xpn->px",phi,density,dphi)
        check("Cartesian_spdf_density_vs_normalized_polynomials",np.max(abs(observed-expected)),5e-15)
        check("Cartesian_spdf_gradient_vs_normalized_polynomials",np.max(abs(gradient-expected_gradient)),8e-15)
        check("Normalized_AO_overlap_diagonal",np.max(abs(np.diag(overlap)-1)),5e-16)
        da,ga,_=evaluate_density(molecule,alpha,points,mapping,phases);db,gb,_=evaluate_density(molecule,beta,points,mapping,phases)
        check("Unrestricted_spin_sum_density",np.max(abs(observed-da-db)),5e-16)
        check("Unrestricted_spin_sum_gradient",np.max(abs(gradient-ga-gb)),1e-15)
        for step in [1e-3,5e-4]:
            fd=np.zeros_like(gradient)
            for axis in range(3):
                delta=np.eye(3)[axis]*step
                plus=evaluate_density(molecule,density,points+delta,mapping,phases)[0]
                minus=evaluate_density(molecule,density,points-delta,mapping,phases)[0]
                fd[:,axis]=(plus-minus)/(2*step)
            check("Independent_density_gradient_FD_h"+str(step),np.max(abs(fd-gradient)),3e-7)
        zero_n,zero_gradient,_=evaluate_density(molecule,np.zeros_like(density),points,mapping,phases)
        check("Zero_density_n",np.max(abs(zero_n)),0.)
        check("Zero_density_gradient",np.max(abs(zero_gradient)),0.)
        # Antisymmetric stored density does not change the scalar AO product.
        skew=rng.normal(size=(n,n));skew=skew-skew.T
        sn,sg,_=evaluate_density(molecule,density+skew,points,mapping,phases)
        check("Stored_density_antisymmetric_part_has_zero_rho",np.max(abs(sn-observed)),3e-16)
        check("Stored_density_antisymmetric_part_has_zero_grad_rho",np.max(abs(sg-gradient)),6e-16)
        report["basis_functions"]=n;report["SCFs"]=0;report["GPU_launches"]=0
        report["passed"]=all(row["passed"] for row in report["checks"])
    except Exception as error:
        import traceback
        report["error"]=type(error).__name__+": "+str(error);report["traceback"]=traceback.format_exc()
    report["source_files_sha256"]={str(path):sha256(path) for path in [Path(__file__),Path(__file__).with_name("ao_density_oracle.py")]}
    (args.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"passed":report["passed"],"checks":len(report["checks"]),"error":report.get("error")}))
    return 0 if report["passed"] else 2


if __name__=="__main__":raise SystemExit(main())
