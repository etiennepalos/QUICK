#!/usr/bin/env python3
"""Meaningful small analytic tests; no QUICK, SCF, GPU or trajectory jobs."""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import hyp1f1

import field_uniformity as analysis


def gaussian(points, beta=1.4):
    """Electron charge -1 with n=(beta/pi)^(3/2) exp(-beta*r²)."""
    r=np.asarray(points,dtype=float);t=beta*np.einsum("pi,pi->p",r,r)
    f1=hyp1f1(1.5,2.5,-t)/3;f2=hyp1f1(2.5,3.5,-t)/5;f3=hyp1f1(3.5,4.5,-t)/7
    k=4*beta**1.5/math.sqrt(math.pi);eye=np.eye(3)
    e=-k*r*f1[:,None]
    g=-k*np.einsum("ij,p->pij",eye,f1)+2*k*beta*np.einsum("pi,pj,p->pij",r,r,f2)
    h=(2*k*beta*(np.einsum("ij,pk,p->pijk",eye,r,f2)+np.einsum("ik,pj,p->pijk",eye,r,f2)
                  +np.einsum("jk,pi,p->pijk",eye,r,f2))
       -4*k*beta**2*np.einsum("pi,pj,pk,p->pijk",r,r,r,f3))
    n=(beta/math.pi)**1.5*np.exp(-t);grad_n=-2*beta*r*n[:,None]
    return e,g,h,n,grad_n


def combined(points):
    e,g,h,n,grad_n=gaussian(points)
    for center,charge in [([1.8,-.6,.2],.35),([-.8,1.9,-.4],-.2)]:
        r=np.asarray(points)-center;r2=np.einsum("pi,pi->p",r,r)
        if np.any(r2==0):raise ValueError("Analytic validation source singularity")
        eye=np.eye(3);e+=charge*r*r2[:,None]**-1.5
        g+=charge*(np.einsum("ij,p->pij",eye,r2**-1.5)-3*np.einsum("pi,pj,p->pij",r,r,r2**-2.5))
        h+=charge*(15*np.einsum("pi,pj,pk,p->pijk",r,r,r,r2**-3.5)
            -3*(np.einsum("ij,pk,p->pijk",eye,r,r2**-2.5)+np.einsum("ik,pj,p->pijk",eye,r,r2**-2.5)
                +np.einsum("jk,pi,p->pijk",eye,r,r2**-2.5)))
    return e,g,h,n,grad_n


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    report={"passed":False,"scope":"Analytic mathematical validation only; no molecular calculation or manufactured trajectory", "checks":[]}
    def check(name,error,tol,**detail):
        report["checks"].append({"name":name,"error":float(error),"tolerance":tol,
                                 "passed":bool(np.isfinite(error) and error<=tol),**detail})
    try:
        rng=np.random.default_rng(20261015)
        # Independent dense-versus-packed counts explicitly catch omitted
        # factors of2,3,6 rather than repeating the projection implementation.
        gxy=np.zeros(6);gxy[1]=1;hxyz=np.zeros(10);hxyz[4]=1
        check("Gxy_multiplicity2",abs(analysis.packed_norm(gxy,2)**2-2),1e-14)
        check("Hxyz_multiplicity6",abs(analysis.packed_norm(hxyz,3)**2-6),1e-14)
        for trial in range(24):
            g=analysis.unpack(rng.normal(size=6),2);h=analysis.unpack(rng.normal(size=10),3)
            gs,g0,hs,h1,tg,th=analysis.stf_split(g,h)
            check(f"{trial}:packedG_norm",abs(analysis.packed_norm(analysis.pack(g),2)-np.linalg.norm(g)),2e-14)
            check(f"{trial}:packedH_norm",abs(analysis.packed_norm(analysis.pack(h),3)-np.linalg.norm(h)),3e-14)
            check(f"{trial}:G_exact_reconstruction",np.max(abs(g-gs-g0)),1e-15)
            check(f"{trial}:H_exact_reconstruction",np.max(abs(h-hs-h1)),1e-15)
            check(f"{trial}:G_STF_trace",abs(np.trace(gs)),2e-15)
            check(f"{trial}:H_STF_trace",np.max(abs(np.einsum("ijj->i",hs))),3e-15)
            check(f"{trial}:G_orthogonality",abs(np.sum(gs*g0)),3e-14)
            check(f"{trial}:H_orthogonality",abs(np.sum(hs*h1)),5e-14)
            check(f"{trial}:G_norm_identity",abs(np.sum(g*g)-np.sum(gs*gs)-tg*tg/3),5e-14)
            check(f"{trial}:H_norm_identity",abs(np.sum(h*h)-np.sum(hs*hs)-3*np.dot(th,th)/5),8e-14)
            q,_=np.linalg.qr(rng.normal(size=(3,3)))
            if np.linalg.det(q)<0:q[:,0]*=-1
            gr=q@g@q.T;hr=np.einsum("ia,jb,kc,abc->ijk",q,q,q,h)
            gsr,_,hsr,_,_,_=analysis.stf_split(gr,hr)
            check(f"{trial}:rotation_G_norm",abs(np.linalg.norm(gr)-np.linalg.norm(g)),3e-14)
            check(f"{trial}:rotation_H_norm",abs(np.linalg.norm(hr)-np.linalg.norm(h)),4e-14)
            check(f"{trial}:rotation_G_projection",np.max(abs(gsr-q@gs@q.T)),4e-14)
            check(f"{trial}:rotation_H_projection",np.max(abs(hsr-np.einsum("ia,jb,kc,abc->ijk",q,q,q,hs))),5e-14)
            e=rng.normal(size=3);u=rng.normal(size=(7,3))*.1
            ordinary=analysis.taylor_fields(e,g,h,u);rotated=analysis.taylor_fields(q@e,gr,hr,u@q.T)
            for name in analysis.APPROXIMATIONS:
                check(f"{trial}:rotation_Taylor_{name}",np.max(abs(rotated[name]-ordinary[name]@q.T)),3e-14)
            # The trace H contribution must have its exact 1/10 contraction.
            explicit=(np.einsum("pj,pj->p",u,u)[:,None]*th+2*u*(u@th)[:,None])/10
            check(f"{trial}:H_source_directional_factor",np.max(abs(.5*np.einsum("ijk,pj,pk->pi",h1,u,u)-explicit)),2e-15)
        center=np.array([.43,-.27,.35]);direction=np.array([.37,-.51,.78]);direction/=np.linalg.norm(direction)
        e,g,h,n,grad_n=combined(center[None,:]);e,g,h=e[0],g[0],h[0]
        check("Poisson_G_analytic_density",abs(np.trace(g)+4*np.pi*n[0]),2e-14)
        check("Poisson_H_analytic_density_gradient",np.max(abs(np.einsum("ijj->i",h)+4*np.pi*grad_n[0])),3e-14)
        # Verify analytic derivatives against displaced direct fields, not
        # against projections or the polynomial being fitted.
        for step in [1e-3,5e-4]:
            dg=np.empty((3,3));dh=np.empty((3,3,3))
            for axis in range(3):
                offset=np.eye(3)[axis]*step;plus=combined((center+offset)[None,:]);minus=combined((center-offset)[None,:])
                dg[:,axis]=(plus[0][0]-minus[0][0])/(2*step);dh[:,:,axis]=(plus[1][0]-minus[1][0])/(2*step)
            check(f"analytic_G_vs_directE_FD_h{step}",np.max(abs(dg-g)),3e-6)
            check(f"analytic_H_vs_directG_FD_h{step}",np.max(abs(dh-h)),1e-5)
        radii=np.array([.08,.04,.02,.01,.005]);u=radii[:,None]*direction
        direct=combined(center+u)[0];predictions=analysis.taylor_fields(e,g,h,u)
        errors={name:np.linalg.norm(predictions[name]-direct,axis=1) for name in analysis.APPROXIMATIONS}
        slopes={name:np.log2(values[-2]/values[-1]) for name,values in errors.items()}
        for name,order in zip(analysis.APPROXIMATIONS,[1,2,3]):
            check("Taylor_asymptotic_order_"+name,abs(slopes[name]-order),.12,observed_order=float(slopes[name]))
        # Center H=0 cannot establish a finite-radius error bound: here G is
        # nonzero, quadratic=linear, yet the cubic residual remains finite.
        ec,gc,hc,_,_=gaussian([[0.,0.,0.]])
        uc=np.array([[.6,0.,0.]])
        prediction=analysis.taylor_fields(ec[0],gc[0],hc[0],uc);actual=gaussian(uc)[0]
        counter_error=float(np.linalg.norm(prediction["quadratic"]-actual))
        check("Gaussian_center_H_exact_zero",np.max(abs(hc)),0.)
        check("Gaussian_center_linear_equals_quadratic",np.max(abs(prediction["linear"]-prediction["quadratic"])),0.)
        check("Gaussian_zero_H_nonzero_finite_radius_error",max(0.,.05-counter_error),0.,radius_bohr=.6,absolute_error=counter_error)
        check("empirical_prefix_cannot_skip_failure",abs(analysis.empirical_prefix([.1,.2,.3],[.001,.1,.001],.01)-.1),0.)
        check("empirical_prefix_none",int(analysis.empirical_prefix([.1,.2],[.1,.01],.001) is not None),0.)
        # End-to-end schema/CSV/JSON/plot validation uses a clearly tagged
        # analytic fixture, not invented QUICK molecular observations.
        fixture=args.output/"analytic-input";fixture.mkdir()
        centers=[]
        for name,point in [("off-center",center),("gaussian-center",np.zeros(3))]:
            ev,gv,hv,nv,dv=combined(point[None,:])
            row={"center_id":name,**dict(zip((x+"_bohr" for x in "xyz"),point)),**dict(zip(("e_"+x for x in "xyz"),ev[0])),
                 **dict(zip(("g_"+x for x in analysis.G_COMPONENTS),analysis.pack(gv[0]))),
                 **dict(zip(("h_"+x for x in analysis.H_COMPONENTS),analysis.pack(hv[0]))),"n_e":nv[0],
                 **dict(zip(("grad_n_"+x for x in "xyz"),dv[0]))}
            centers.append(row)
        samples=[]
        directions=np.vstack([np.eye(3),-np.eye(3),direction])
        for row in centers:
            point=np.array([row[x+"_bohr"] for x in "xyz"])
            for ri,radius in enumerate([.005,.01,.02,.04,.08,.16]):
                for di,vector in enumerate(directions):
                    offset=radius*vector
                    direct=combined((point+offset)[None,:])[0][0]
                    samples.append({"center_id":row["center_id"],"sample_id":f"r{ri}-d{di}","direction_id":f"d{di}","radius_bohr":radius,
                                    **dict(zip(("u_"+x+"_bohr" for x in "xyz"),offset)),
                                    **dict(zip(("e_direct_"+x for x in "xyz"),direct))})
        analysis.write_csv(fixture/"centers.csv",centers);analysis.write_csv(fixture/"samples.csv",samples)
        manifest={"schema":"quick-field-uniformity-input-1","input_kind":"analytic-validation","source_convention":analysis.SOURCE_CONVENTION,
                  "units":analysis.UNITS,"centers_csv":"centers.csv","samples_csv":"samples.csv",
                  "provenance":{"description":"Closed normalized electron Gaussian plus fixed classical charges; static analytic validation only"}}
        (fixture/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        summary=analysis.analyze(fixture/"manifest.json",args.output/"analytic-analysis",[1e-4,1e-3],True,"off-center")
        check("analysis_never_claims_continuous_bound",int(summary["continuous_bound_claim"]),0.)
        rows=analysis.read_csv(args.output/"analytic-analysis/centers_analysis.csv",["poisson_g_residual","poisson_h_residual_norm"])
        check("analysis_csv_Poisson",max(abs(float(row["poisson_g_residual"]))+float(row["poisson_h_residual_norm"]) for row in rows),1e-13)
        # Deliberately incorrect units and radius metadata must be rejected.
        bad=copy.deepcopy(manifest);bad["units"]["coordinates"]="Angstrom";(fixture/"bad-units.json").write_text(json.dumps(bad))
        rejected=False
        try:analysis.analyze(fixture/"bad-units.json",args.output/"rejected-units")
        except ValueError:rejected=True
        check("reject_ambiguous_units",int(not rejected),0.)
        report["Taylor_orders"]=dict(zip(slopes,map(float,slopes.values())))
        report["Gaussian_counterexample"]={"center_H_norm":0.,"radius_bohr":.6,"quadratic_field_error":counter_error,
                                           "interpretation":"A zero center H does not bound finite-radius residuals."}
        report["passed"]=all(x["passed"] for x in report["checks"])
    except Exception as error:
        import traceback
        report["error"]=type(error).__name__+": "+str(error);report["traceback"]=traceback.format_exc()
    (args.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"passed":report["passed"],"checks":len(report["checks"]),"error":report.get("error")}),flush=True)
    return 0 if report["passed"] else 2


if __name__=="__main__":
    raise SystemExit(main())
