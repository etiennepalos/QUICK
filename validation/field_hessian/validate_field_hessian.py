#!/usr/bin/env python3
"""Fixed-gate CPU/CUDA field-Hessian accuracy, lifecycle and BOMD campaign.

Run with NumPy. An independent worker requires PySCF/SciPy. Every evidence
directory is new; no QUICK source, basis, baseline or acceptance gate changes.
"""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BOHR_ANGSTROM = .52917720859
WATER = [[8, [-.33840, .00380, .23923]], [1, [-.33510, -.00190, -.83277]], [1, [.67350, -.00190, .59353]]]
PROBES = [[0., 0., 3.8], [2.8, 0., 0.], [0., 2.8, 0.], [-2., -2., 1.]]
UNIQUE = [(0,0,0),(0,0,1),(0,0,2),(0,1,1),(0,1,2),(0,2,2),(1,1,1),(1,1,2),(1,2,2),(2,2,2)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value,float) and not math.isfinite(value):
        return str(value)
    if isinstance(value,list):
        return [json_safe(item) for item in value]
    if isinstance(value,dict):
        return {key:json_safe(item) for key,item in value.items()}
    return value


def runtime(driver, env):
    loaded = subprocess.run(["ldd", str(driver)], env=env, capture_output=True, text=True, check=True)
    if "not found" in loaded.stdout:
        raise ValueError("Unresolved runtime library")
    paths = [driver]; quick = []
    for line in loaded.stdout.splitlines():
        match = re.search(r"(?:=>\s*)?(/[^\s]+)", line)
        if match:
            path = Path(match.group(1)).resolve(strict=True); paths.append(path)
            if re.fullmatch(r"libquick(?:_mpi)?(?:_cuda)?\.so(?:\.[\d.]+)?", path.name):
                quick.append(path)
    if len(quick) != 1:
        raise ValueError("Expected one QUICK host library")
    return {"quick_library": str(quick[0]), "files_sha256": {str(p): sha(p) for p in paths}}


def parse_result(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        words = line.split(); key = words[0]
        if key in ("unrestricted", "scf_converged"):
            result[key] = words[1] == "T"
        elif key == "error_statuses":
            result[key] = list(map(int, words[1:]))
        elif key == "old_property_errors":
            result[key] = list(map(float, words[1:]))
        else:
            result[key] = float(words[1])
    return result


def h_values(path, raw=False):
    rows = np.loadtxt(path, skiprows=1)
    if raw:
        return np.asarray([row.reshape(3,3,3,order="F") for row in np.atleast_1d(rows).reshape(-1,27)])
    rows = np.atleast_2d(rows)
    return np.asarray([row[17:].reshape(3,3,3,order="F") for row in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for backend in ("cpu", "cuda"):
        parser.add_argument(f"--{backend}-driver", type=Path, required=True)
        parser.add_argument(f"--{backend}-basis", type=Path, required=True)
        parser.add_argument(f"--{backend}-bomd-driver", type=Path)
    parser.add_argument("--oracle-python", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800.)
    parser.add_argument("--skip-bomd", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be positive finite")
    if not args.skip_bomd and (not args.cpu_bomd_driver or not args.cuda_bomd_driver):
        parser.error("provide both BOMD drivers or use --skip-bomd")
    args.workdir = args.workdir.resolve(); args.workdir.mkdir(parents=True, exist_ok=False)
    drivers = {backend: getattr(args, backend + "_driver").resolve(strict=True) for backend in ("cpu", "cuda")}
    bases = {backend: getattr(args, backend + "_basis").resolve(strict=True) for backend in drivers}
    envs = {}
    for backend in drivers:
        env = os.environ.copy()
        env.update(QUICK_BASIS=str(bases[backend]), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        env["LD_LIBRARY_PATH"] = str(bases[backend].parent / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
        envs[backend] = env
    report = {"passed": False, "started_utc": datetime.now(timezone.utc).isoformat(), "checks": [], "runs": [],
              "scope": "All27 source-only spatial derivatives H_ijk=d²Ei/dCj dCk=-V_ijk: exact-density independent analytic integral oracle and native CPU/CUDA",
              "gates": {"independent_H_atomic_units": 2e-8, "CPU_CUDA_same_density_H": 2e-10,
                        "EFG_FD_H": 2e-6, "old_properties_preserved": 2e-11, "permutation_symmetry": 2e-11,
                        "Poisson_density_gradient_trace": 2e-8, "BOMD_energy_hartree": 1e-7,
                        "BOMD_geometry_Angstrom": 1e-7, "BOMD_sampling_preserves_forces": 1e-8},
              "limitations": ["Single GPU architecture; finite isolated all-electron molecules through Cartesian f functions",
                              "Electronic and classical source contributions only; direct applied/solvent backgrounds excluded",
                              "Public property API, not nuclear energy Hessians; no self-excluded nuclear derivative observable",
                              "Short BOMD software checks do not establish production sampling or long-time stability"]}
    oracle_request = {"molecular": {}, "raw": {}}

    def check(name, error, tolerance, **details):
        report["checks"].append({"name": name, "error": float(error), "tolerance": tolerance,
                                 "passed": bool(np.isfinite(error) and error <= tolerance), **details})

    def progress():
        (args.workdir / "progress.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    def execute(backend, mode, label, text, density="-"):
        folder = args.workdir / f"{len(report['runs']):03d}-{backend}-{label}"; folder.mkdir(exist_ok=False)
        path = folder / "input.dat"; path.write_text(text)
        command = [str(drivers[backend]), mode, path.name] + ([str(density)] if mode == "api" else [])
        row = {"backend": backend, "mode": mode, "label": label, "folder": str(folder), "command": command,
               "input_sha256": sha(path), "density_input": str(density)}
        report["runs"].append(row); print("RUN", backend, label, flush=True)
        started = time.monotonic()
        with (folder / "execution.log").open("w") as log:
            process = subprocess.run(command, cwd=folder, env=envs[backend], stdout=log, stderr=subprocess.STDOUT,
                                     timeout=args.timeout)
        row.update(returncode=process.returncode, wall_seconds=time.monotonic()-started)
        if process.returncode:
            raise ValueError(f"Test client failed: {folder}")
        row["files_sha256"] = {p.name: sha(p) for p in folder.iterdir() if p.is_file()}
        return folder, row

    def molecular_case(name, model, points, density="-"):
        card = (f"{'UHF' if model['spin'] else 'HF'} BASIS={model['basis']} ENERGY SCF=400 DENSERMS=1e-10 "
                f"CUTOFF=1e-12 CHARGE={model['charge']} MULT={model['spin']+1}"
                + (" EXTCHARGES" if model["charges"] else ""))
        if len(card)>256:
            raise ValueError("Oversized API card")
        text = card + f"\n{len(model['atoms'])} {len(model['charges'])} {len(points)}\n"
        text += "".join(str(z)+" "+" ".join(f"{v:.17g}" for v in xyz)+"\n" for z,xyz in model["atoms"])
        text += "".join(" ".join(f"{v:.17g}" for v in row)+"\n" for row in model["charges"])
        text += "".join(" ".join(f"{v:.17g}" for v in point)+"\n" for point in points)
        results = {}
        for backend in drivers:
            frozen = density if backend == "cpu" or density == "ZERO" else results["cpu"][1] / "density.dat"
            folder, row = execute(backend,"api",name,text,frozen)
            result = parse_result(folder / "result.dat"); observed = h_values(folder / "properties.dat")
            if not np.isfinite(observed).all():
                raise ValueError("Nonfinite accepted H")
            row.update(result=result, model=model)
            check(f"{backend}:{name}:SCF_converged", int(not result["scf_converged"]), 0.)
            output = (folder / "field_hessian_api.out").read_text(errors="replace")
            check(f"{backend}:{name}:one_converged_SCF", abs(len(re.findall(r"REACH CONVERGENCE AFTER\s+\d+\s+CYCLES",output))-1), 0.)
            check(f"{backend}:{name}:native_GPU_H", abs(result["main_native_gpu_launches"]-(backend=="cuda" and density!="ZERO")), 0.)
            for key in ("extra_API_calls","density_error","fock_error","error_output_error","post_delete_output_error"):
                check(f"{backend}:{name}:{key}", result[key], 0.)
            check(f"{backend}:{name}:old_ESP_E_G_unchanged", max(result["old_property_errors"]), 2e-11)
            check(f"{backend}:{name}:repeat", result["repeat_error"], 2e-11)
            check(f"{backend}:{name}:chunk17", result["chunk_error"], 2e-11)
            check(f"{backend}:{name}:error_statuses", int(result["error_statuses"] != [47,46,46,55,55]), 0.)
            check(f"{backend}:{name}:post_delete", abs(result["post_delete_status"]-47), 0.)
            for permutation in itertools.permutations((0,1,2)):
                check(f"{backend}:{name}:permutation{permutation}",
                      np.max(np.abs(observed-observed.transpose((0,)+tuple(1+i for i in permutation)))),2e-11)
            fd = np.loadtxt(folder / "finite_difference.dat")
            for step in (1e-3,5e-4):
                errors=[]
                for r in fd[np.isclose(fd[:,0],step,rtol=1e-12,atol=0.)]:
                    axis,point = int(r[1])-1,int(r[2])-1
                    errors.append(np.max(np.abs(r[3:].reshape(3,3,order="F")-observed[point,:,:,axis])))
                check(f"{backend}:{name}:EFG_FD_h{step:g}",max(errors),2e-6)
            basis_name = {"STO-3G":"STO-3G.BAS","6-31G**":"6-31GDP.BAS","DEF2-TZVPD":"DEF2-TZVPD.BAS"}[model["basis"]]
            oracle_request["molecular"][backend+":"+name] = {"model":model,"folder":str(folder),"basis_file":str(bases[backend]/basis_name)}
            results[backend]=(observed,folder)
            progress()
        check(name+":CPU_CUDA_same_D_all27",np.max(np.abs(results["cpu"][0]-results["cuda"][0])),2e-10)

    def raw_case(name, model, expected_status=0, expected_launch=None):
        nb,na,mp,np_ = len(model["density"]),len(model["atoms"]),len(model["exponents"][0]),len(model["points"])
        text=f"{nb} {na} {mp} {np_}\n{model.get('cutoff',0.):.17g} {model.get('seed',0.):.17g}\n"
        text += "".join(" ".join(f"{x:.17g}" for x in atom)+"\n" for atom in model["atoms"])
        for mu in range(nb):
            text += f"{model['centers'][mu]} "+" ".join(str(v) for v in model["angular"][mu])+f" {model['nc'][mu]} {model['owned'][mu]}\n"
            text += "".join(f"{a:.17g} {c:.17g}\n" for a,c in zip(model["exponents"][mu],model["coefficients"][mu]))
        text += "".join(f"{x:.17g}\n" for x in np.asarray(model["density"]).ravel(order="F"))
        text += "".join(" ".join(f"{x:.17g}" for x in point)+"\n" for point in model["points"])
        output={}
        for backend in drivers:
            folder,row=execute(backend,"raw",name,text)
            lines=(folder/"raw_result.dat").read_text().splitlines();status,count,npoint=map(int,lines[0].split())
            values=np.asarray([float(v) for line in lines[1:] for v in line.split()])
            check(f"{backend}:raw:{name}:status",abs(status-expected_status),0.)
            if status!=expected_status:
                raise ValueError(f"Unexpected raw status {status}, expected {expected_status}: {folder}")
            if expected_launch is not None:
                check(f"{backend}:raw:{name}:native_launch",abs(count-(expected_launch if backend=="cuda" else 0)),0.)
            if status:
                check(f"{backend}:raw:{name}:transactional_error",np.max(np.abs(values-model.get("seed",0.))) if len(values) else 0.,0.)
            elif npoint:
                observed=np.asarray([v.reshape(3,3,3,order="F") for v in values.reshape(npoint,27)])
                output[backend]=observed
                oracle_request["raw"][backend+":"+name]={"model":model,"observed":observed.tolist()}
                for permutation in itertools.permutations((0,1,2)):
                    check(f"{backend}:raw:{name}:permutation{permutation}",
                          np.max(np.abs(observed-observed.transpose((0,)+tuple(1+i for i in permutation)))),2e-11)
            row.update(status=status,native_launches=count,raw_model=json_safe(model))
            progress()
        if len(output)==2:
            scale=max(np.max(np.abs(output["cpu"])),1e-300)
            check("raw:"+name+":CPU_CUDA",np.max(np.abs(output["cpu"]-output["cuda"])),max(2e-12*scale,1e-300))

    try:
        sources=[Path(__file__).resolve(),HERE/"reference_field_hessian.py",HERE/"quick_field_hessian_api_driver.f90"]
        sources.extend(p for basis in bases.values() for p in basis.glob("*.BAS") if p.name in ("STO-3G.BAS","6-31GDP.BAS","DEF2-TZVPD.BAS"))
        report["inputs_before"]={str(p):sha(p) for p in sources}
        report["runtime_before"]={backend:runtime(driver,envs[backend]) for backend,driver in drivers.items()}
        for backend in drivers:
            expected=(bases[backend].parent/"lib"/("libquick_cuda.so" if backend=="cuda" else "libquick.so")).resolve(strict=True)
            if Path(report["runtime_before"][backend]["quick_library"])!=expected:
                raise ValueError("Unexpected QUICK runtime")
        model={"atoms":copy.deepcopy(WATER),"charges":[],"charge":0,"spin":0,"basis":"STO-3G"}
        molecular_case("water_hf_s",model,PROBES)
        d=copy.deepcopy(model);d["basis"]="6-31G**";molecular_case("water_hf_d",d,PROBES)
        u=copy.deepcopy(model);u.update(charge=1,spin=1);molecular_case("water_cation_uhf",u,PROBES)
        f=copy.deepcopy(model);f["basis"]="DEF2-TZVPD";molecular_case("water_hf_f64",f,PROBES)
        mm=copy.deepcopy(model);mm["charges"]=[[2.2,-.7,1.6,.35],[-1.8,1.9,-.8,-.2]]
        molecular_case("water_signed_MM",mm,PROBES)
        molecular_case("zero_density_classical",mm,PROBES,"ZERO")
        dense=[]
        for k in range(257):
            z=1-2*(k+.5)/257.;az=k*math.pi*(3-math.sqrt(5));radius=3.5+.17*(k%7)
            dense.append([radius*math.sqrt(1-z*z)*math.cos(az)+.2,radius*math.sqrt(1-z*z)*math.sin(az)+.3,radius*z-.1])
        molecular_case("dense257",model,dense)
        for exponent in (.001,.7,1000.):
            probes=(np.asarray([[0.,0.,0.],[.41,-.33,.19],[1e-9,-2e-9,3e-9],[8.,-3.,2.],[1000.,-2000.,3000.]])/math.sqrt(exponent)).tolist()
            gaussian={"reference":"gaussian","ao_exponent":exponent,"atoms":[[0.,0.,0.]],"centers":[1],"angular":[[0,0,0]],"nc":[1],
                      "owned":[1],"exponents":[[exponent]],"coefficients":[[(2*exponent/math.pi)**.75]],"density":[[1.]],"points":probes}
            raw_case(f"sGaussian_a{exponent:g}",gaussian,expected_launch=1)
        angular=[];exponents=[];coefficients=[]
        for l,a in enumerate([.7,.5,.4,.3]):
            for x in range(l,-1,-1):
                for y in range(l-x,-1,-1):
                    powers=[x,y,l-x-y];angular.append(powers);exponents.append([a])
                    denominator=math.prod(math.prod(range(1,2*p,2)) for p in powers)
                    coefficients.append([(2*a/math.pi)**.75*math.sqrt((4*a)**l/denominator)])
        n=len(angular);rng=np.random.default_rng(20261015);matrix=rng.normal(size=(n,n));density=(matrix+matrix.T)*.03
        spdf={"reference":"spdf","atoms":[[0.,0.,0.]],"centers":[1]*n,"angular":angular,"nc":[1]*n,"owned":[1]*n,
              "exponents":exponents,"coefficients":coefficients,"density":density.tolist(),"points":[[0.,0.,0.],*PROBES,[.41,-.33,.19]]}
        raw_case("all_spdf_pairs_signed_D",spdf,expected_launch=1)
        nearly_symmetric=copy.deepcopy(spdf);nearly_symmetric["density"][1][0]+=5e-11
        raw_case("upper_triangle_nearly_symmetric_D",nearly_symmetric,expected_launch=1)
        partition=copy.deepcopy(spdf);partition["owned"]=[i%2 for i in range(n)];raw_case("higher_AO_partition",partition,expected_launch=1)
        seeded=copy.deepcopy(spdf);seeded["seed"]=.125;raw_case("additive_seed",seeded,expected_launch=1)
        zero=copy.deepcopy(spdf);zero["density"]=np.zeros((n,n)).tolist();raw_case("zero_D",zero,expected_launch=0)
        no_owned=copy.deepcopy(spdf);no_owned["owned"]=[0]*n;raw_case("no_owned",no_owned,expected_launch=0)
        for label,mutate in [("nonsymmetric_D",lambda x:x["density"][0].__setitem__(1,.8)),
                             ("angular_g",lambda x:x["angular"][0].__setitem__(0,4)),
                             ("bad_owner",lambda x:x["owned"].__setitem__(0,2)),
                             ("zero_exponent",lambda x:x["exponents"][0].__setitem__(0,0.)),
                             ("NaN_probe",lambda x:x["points"][0].__setitem__(0,float("nan"))),
                             ("overflow_probe",lambda x:x["points"][0].__setitem__(0,1e200)),
                             ("NaN_density",lambda x:x["density"][0].__setitem__(0,float("nan"))),
                             ("negative_cutoff",lambda x:x.__setitem__("cutoff",-1.))]:
            invalid=copy.deepcopy(spdf);invalid["seed"]=.125;mutate(invalid);raw_case("reject_"+label,invalid,55)
        invalid=copy.deepcopy(spdf);invalid["points"]=[];raw_case("reject_zero_np",invalid,46,expected_launch=0)
        request_path=args.workdir/"oracle-request.json";request_path.write_text(json.dumps(oracle_request,indent=2)+"\n")
        output_path=args.workdir/"oracle-report.json"
        command=[str(args.oracle_python.resolve(strict=True)),str(HERE/"reference_field_hessian.py"),"--request",str(request_path),"--output",str(output_path)]
        with (args.workdir/"oracle-execution.log").open("w") as log:
            oracle_process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=envs["cpu"],timeout=7200)
        oracle=json.loads(output_path.read_text()) if output_path.exists() else {}
        check("independent_analytic_libcint_campaign",int(oracle_process.returncode!=0 or not oracle.get("passed",False)),0.)
        report["checks"].extend({**item,"name":"oracle:"+item["name"]} for item in oracle.get("checks",[]))
        report["oracle"]={"command":command,"returncode":oracle_process.returncode,"report":str(output_path)}
        if not args.skip_bomd:
            run_bomd(args,report,envs,check)
        report["runtime_after"]={backend:runtime(driver,envs[backend]) for backend,driver in drivers.items()}
        report["inputs_after"]={str(p):sha(p) for p in sources}
        check("runtime_unchanged",int(report["runtime_before"]!=report["runtime_after"]),0.)
        check("harness_basis_sources_unchanged",int(report["inputs_before"]!=report["inputs_after"]),0.)
        report["passed"]=all(item["passed"] for item in report["checks"])
    except Exception as error:
        import traceback
        report["error"]=f"{type(error).__name__}: {error}";report["traceback"]=traceback.format_exc()
    report["finished_utc"]=datetime.now(timezone.utc).isoformat();report["failed_checks"]=sum(not x["passed"] for x in report["checks"])
    path=args.workdir/"report.json";path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"passed":report["passed"],"checks":len(report["checks"]),"runs":len(report["runs"]),"failed_checks":report["failed_checks"],"error":report.get("error"),"report":str(path)}),flush=True)
    return 0 if report["passed"] else 2


def run_bomd(args,report,envs,check):
    import csv
    sys.path.insert(0,str(ROOT/"validation/oeprop_bomd"))
    from validate_oeprop_bomd import write_control,read_trajectory,read_checkpoint,property_difference,max_difference
    keyword="HF BASIS=STO-3G GRADIENT CUTOFF=1e-12 GRADCUTOFF=1e-12 DENSERMS=1e-10"
    report["bomd"]={}
    for backend in ("cpu","cuda"):
        driver=getattr(args,backend+"_bomd_driver").resolve(strict=True);results={}
        for label,enabled,stride in [("old_properties",False,1),("H_enabled",True,1),("all_properties_off",False,0)]:
            folder=args.workdir/"bomd"/backend/label;folder.mkdir(parents=True,exist_ok=False)
            control=folder/"control.nml"
            write_control(control,geometry=ROOT/"validation/oeprop_bomd/water.xyz",probes=ROOT/"validation/oeprop_bomd/water_fixed.probe",
                          prefix=Path("trajectory"),mode="fixed",steps=4,dt_fs=.025,keyword=keyword,stride=stride)
            if enabled:
                control.write_text(control.read_text().replace("\n/","\n field_hessian=.true.\n/"))
            print("BOMD",backend,label,flush=True);start=time.monotonic()
            with (folder/"execution.log").open("w") as log:
                process=subprocess.run([str(driver),control.name],cwd=folder,env=envs[backend],stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
            if process.returncode:
                raise ValueError("BOMD failed: "+str(folder))
            summary=json.loads((folder/"trajectory_summary.json").read_text());text=(folder/"quick_oeprop_bomd.out").read_text(errors="replace")
            with (folder/"trajectory_energy.csv").open() as handle:energies=list(csv.DictReader(handle))
            with (folder/"trajectory_oeprop.csv").open() as handle:properties=list(csv.DictReader(handle))
            trajectory=read_trajectory(folder/"trajectory_trajectory.xyz",list(range(5)),.025)
            forces=np.loadtxt(folder/"trajectory_forces.dat")
            if forces.shape!=(15,5):raise ValueError("Incomplete BOMD force history")
            check(f"{backend}:BOMD:{label}:five_force_SCF",max(abs(summary[k]-5) for k in ["force_calls","api_energy_gradient_calls"]),0.)
            check(f"{backend}:BOMD:{label}:five_converged_SCF",abs(len(re.findall(r"REACH CONVERGENCE AFTER\s+\d+\s+CYCLES",text))-5),0.)
            check(f"{backend}:BOMD:{label}:no_extra_SCF",summary["properties_extra_api_calls"],0.)
            check(f"{backend}:BOMD:{label}:old_property_calls",abs(summary["property_calls"]-(5 if stride else 0)),0.)
            check(f"{backend}:BOMD:{label}:short_drift",summary["max_abs_drift_au"],1e-5)
            if enabled:
                check(f"{backend}:BOMD:H_calls",abs(summary["field_hessian_calls"]-5),0.)
                check(f"{backend}:BOMD:H_native_launches",abs(summary["field_hessian_master_gpu_launches"]-(5 if backend=="cuda" else 0)),0.)
                with (folder/"trajectory_field_hessian.csv").open() as handle:hrows=list(csv.DictReader(handle))
                check(f"{backend}:BOMD:H_step_probe_sequence",int([(int(r["step"]),int(r["probe"])) for r in hrows]!=[(s,p) for s in range(5) for p in range(1,5)]),0.)
                hnumeric=np.asarray([[float(row['h_'+''.join('xyz'[i] for i in comp)]) for comp in UNIQUE] for row in hrows])
                if not np.isfinite(hnumeric).all():raise ValueError("Nonfinite sampled BOMD H")
            else:
                check(f"{backend}:BOMD:{label}:no_H_output",int((folder/"trajectory_field_hessian.csv").exists() or "field_hessian_calls" in summary),0.)
                hnumeric=None
            results[label]={"summary":summary,"trajectory":trajectory,"forces":forces[:,2:],"properties":properties,
                            "energy":np.asarray([[float(row[c]) for c in ["potential_au","kinetic_au","total_au"]] for row in energies]),
                            "H":hnumeric,"wall_seconds":time.monotonic()-start,"folder":str(folder)}
        for label in ("H_enabled","all_properties_off"):
            for key,tolerance in [("energy",1e-7),("trajectory",1e-7),("forces",1e-8)]:
                check(f"{backend}:BOMD:{label}:preserves_{key}",max_difference(results[label][key],results["old_properties"][key]),tolerance)
        for name,error in property_difference(results["H_enabled"]["properties"],results["old_properties"]["properties"]).items():
            check(f"{backend}:BOMD:H_preserves_old_{name}",error,2e-11)
        report["bomd"][backend]={label:{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in result.items()} for label,result in results.items()}
    for key,tolerance in [("energy",1e-7),("trajectory",1e-7),("forces",1e-7),("H",2e-8)]:
        check("BOMD:CPU_CUDA:"+key,max_difference(report["bomd"]["cpu"]["H_enabled"][key],report["bomd"]["cuda"]["H_enabled"][key]),tolerance)


if __name__=="__main__":
    raise SystemExit(main())
