#!/usr/bin/env python3
"""Meaningful workflow schema/control tests; no QUICK, SCF or GPU execution."""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
from __future__ import annotations

import argparse
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import benchmark_workflow as workflow


class WorkflowIO(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix="quick-workflow-unit-")
        self.root=Path(self.temporary.name)
        self.args=argparse.Namespace(nsteps=2,dt_fs=.025,probe_count=2)
        self.rows=[{"step":step,"time_fs":step*.025,"probe":probe,"x_ang":probe+.1,"y_ang":.2,"z_ang":.3,
                    **{name:step+probe+.1*index for index,name in enumerate(workflow.H_COLUMNS)}}
                   for step in range(3) for probe in (1,2)]
        self.data={"summary":{"backend":"cpu","field_hessian_calls":3,"field_hessian_master_gpu_launches":0,
             "field_hessian_convention":"H_ijk=d2 E_i/dC_j dC_k=-d3 V/dC_i dC_j dC_k",
             "field_hessian_units":"hartree/(e bohr^3); laboratory Cartesian axes"},
             "properties":[{key:value for key,value in row.items() if key not in workflow.H_COLUMNS} for row in self.rows]}
        self.write_h()

    def tearDown(self):self.temporary.cleanup()

    def write_h(self,rows=None,columns=None):
        with (self.root/"bomd_field_hessian.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=columns or list(self.rows[0]),extrasaction="ignore")
            writer.writeheader();writer.writerows(rows if rows is not None else self.rows)

    def test_complete_ten_components_and_stride_alignment(self):
        values=workflow.validate_hessian(self.root,self.data,self.args,1)
        self.assertEqual(len(values),6)
        self.assertEqual(float(values[0]["h_xyz"]),1.4)
        selected=[row for row in self.rows if row["step"] in (0,2)];self.write_h(selected)
        data=copy.deepcopy(self.data);data["summary"]["field_hessian_calls"]=2;data["properties"]=selected
        self.assertEqual(len(workflow.validate_hessian(self.root,data,self.args,2)),4)

    def test_gpu_must_prove_native_H_launches(self):
        data=copy.deepcopy(self.data);data["summary"]["backend"]="GPU"
        with self.assertRaisesRegex(ValueError,"launch count"):workflow.validate_hessian(self.root,data,self.args,1)
        data["summary"]["field_hessian_master_gpu_launches"]=3
        self.assertEqual(len(workflow.validate_hessian(self.root,data,self.args,1)),6)

    def test_reject_missing_component_and_nonfinite_value(self):
        self.write_h(columns=[name for name in self.rows[0] if name!="h_xyz"])
        with self.assertRaises(ValueError):workflow.validate_hessian(self.root,self.data,self.args,1)
        malformed=copy.deepcopy(self.rows);malformed[0]["h_xyz"]="NaN";self.write_h(malformed)
        with self.assertRaises(ValueError):workflow.validate_hessian(self.root,self.data,self.args,1)

    def test_reject_missing_and_duplicated_sample(self):
        for rows in (self.rows[:-1],[self.rows[0],*self.rows[:-1]]):
            with self.subTest(rows=len(rows)):
                self.write_h(rows)
                with self.assertRaisesRegex(ValueError,"rows"):workflow.validate_hessian(self.root,self.data,self.args,1)

    def test_reject_wrong_timestamp_or_probe_position(self):
        for key in ("time_fs","x_ang"):
            rows=copy.deepcopy(self.rows);rows[0][key]+=.1;self.write_h(rows)
            with self.subTest(key=key):
                with self.assertRaises(ValueError):workflow.validate_hessian(self.root,self.data,self.args,1)

    def test_reject_wrong_sign_units_or_call_count(self):
        for key,value in [("field_hessian_calls",4),("field_hessian_convention","H=+V'''"),
                          ("field_hessian_units","Angstrom")]:
            data=copy.deepcopy(self.data);data["summary"][key]=value
            with self.subTest(key=key):
                with self.assertRaises(ValueError):workflow.validate_hessian(self.root,data,self.args,1)

    def test_off_does_no_H_work_or_output(self):
        data={"summary":{"backend":"cpu"},"properties":[]}
        with self.assertRaises(ValueError):workflow.validate_hessian(self.root,data,self.args,0)
        (self.root/"bomd_field_hessian.csv").unlink()
        self.assertEqual(workflow.validate_hessian(self.root,data,self.args,0),[])
        data["summary"]["field_hessian_calls"]=0
        with self.assertRaises(ValueError):workflow.validate_hessian(self.root,data,self.args,0)

    def test_complete_raw_force_vectors_and_missing_atom(self):
        rows=np.array([[frame,atom,.1*frame,.2*atom,-.3] for frame in (1,2,3) for atom in (1,2,3)])
        path=self.root/"forces.dat";np.savetxt(path,rows,fmt="%.17g")
        parsed=workflow.read_raw_gradients(path,3,3)
        self.assertEqual(parsed.shape,(3,3,3));self.assertAlmostEqual(parsed[2,1,1],.4)
        np.savetxt(path,rows[:-1])
        with self.assertRaises(ValueError):workflow.read_raw_gradients(path,3,3)
        rows[[0,1]]=rows[[1,0]];np.savetxt(path,rows)
        with self.assertRaises(ValueError):workflow.read_raw_gradients(path,3,3)

    def test_timing_statistics_preserve_negative_overhead(self):
        value=workflow.describe([-2.,1.,3.]);self.assertEqual(value["median"],1.);self.assertEqual(value["min"],-2.)
        self.assertIsNone(workflow.describe([1.])["sample_stdev"])
        with self.assertRaises(ValueError):workflow.describe([float("nan")])

    def test_H_repeat_requires_identical_sample_ids(self):
        same=[{key:str(value) for key,value in row.items()} for row in self.rows]
        self.assertEqual(max(workflow.hessian_difference(same,same).values()),0.)
        wrong=copy.deepcopy(same);wrong[0]["probe"]="3"
        with self.assertRaises(ValueError):workflow.hessian_difference(same,wrong)

    def inputs(self):
        geometry=self.root/"geometry.xyz";geometry.write_text("3\nStatic I/O fixture; no dynamics or property observations\nO 0 0 0\nH 0 0 1\nH 1 0 0\n")
        probes=self.root/"probes.probe";probes.write_text("1\n0 0 3\n")
        keyword=self.root/"keyword.txt";keyword.write_text("PBE0 D3 BASIS=DEF2-SVP GRADIENT CHARGE=0 XC_RADIAL=150 XC_ANGULAR=590 MATRIXZERO=1e-14\n")
        return geometry,probes,keyword

    def test_explicit_keyword_and_unit_contract(self):
        geometry,probes,keyword=self.inputs()
        value=workflow.read_inputs(geometry,probes,keyword,"bohr")
        self.assertAlmostEqual(value["points_angstrom"][0,2],3*.52917720859)
        self.assertIn(" D3 ",value["keyword"])
        keyword.write_text(keyword.read_text().replace(" D3 "," DFTD3 "))
        with self.assertRaisesRegex(ValueError,"internal flag"):workflow.read_inputs(geometry,probes,keyword,"angstrom")

    def test_requested_dispersion_grid_and_basis_require_native_evidence(self):
        keyword="PBE0 D3 BASIS=DEF2-SVP GRADIENT XC_RADIAL=150 XC_ANGULAR=590"
        output=(" BASIS FUNCTIONS = 15\n BASIS FUNCTIONS = 5\n============== BASIS INFOS ==============\n BASIS FUNCTIONS =   25\n"
                " NSHELL = 12 NPRIM = 39\n| SIGNIFICANT NUMBER OF BASIS FUNCTIONS = 64723\n"
                " DISPERSION METHOD = D3, ZERO DAMPING\n UNPRUNED XC GRID: RADIAL = 150 ANGULAR = 590\n")
        evidence=workflow.validate_method_output(output,keyword,25)
        self.assertEqual(evidence["basis_functions"],25);self.assertTrue(evidence["requested_D3_verified"])
        for malformed in (output.replace("DISPERSION METHOD","SOME OTHER METHOD"),output.replace("ANGULAR = 590","ANGULAR = 434")):
            with self.assertRaises(ValueError):workflow.validate_method_output(malformed,keyword,25)
        with self.assertRaises(ValueError):workflow.validate_method_output(output,keyword,24)
        with self.assertRaises(ValueError):workflow.validate_method_output(output+"============== BASIS INFOS ==============\n BASIS FUNCTIONS = 75\n",keyword,25)

    def test_explicit_screening_protocol_and_printed_cutoffs(self):
        keyword="PBE0 D3 BASIS=DEF2-SVP GRADIENT CUTOFF=1e-12 GRADCUTOFF=1e-12 MATRIXZERO=1e-14 BASISCUTOFF=1e-12 XCCUTOFF=1e-12 DENSERMS=1e-10".upper()
        expected={"PBE0":None,"D3":None,"BASIS":"DEF2-SVP","BASISCUTOFF":1e-12,"MATRIXZERO":1e-14}
        workflow.validate_expected_controls(keyword,expected)
        with self.assertRaises(ValueError):workflow.validate_expected_controls(keyword.replace("BASISCUTOFF=1E-12",""),expected)
        output=("============== BASIS INFOS ==============\n BASIS FUNCTIONS = 25\n DISPERSION METHOD = D3, ZERO DAMPING\n"
                " TWO-e INTEGRAL = 0.100E-11\n GRADIENT CUTOFF = 0.100E-11\n MATRIX ELEMENTS = 0.100E-13\n"
                " BASIS FUNCTION = 0.100E-11\n PRUNING CUTOFF = 0.100E-11\n DENSITY MATRIX MAXIMUM RMS FOR CONVERGENCE = 0.100E-09\n")
        evidence=workflow.validate_method_output(output,keyword,25)
        self.assertEqual(set(evidence["printed_controls"]),{"CUTOFF","GRADCUTOFF","MATRIXZERO","BASISCUTOFF","XCCUTOFF","DENSERMS"})
        with self.assertRaisesRegex(ValueError,"BASISCUTOFF"):workflow.validate_method_output(output.replace("BASIS FUNCTION = 0.100E-11","BASIS FUNCTION = 0.100E-05"),keyword,25)

    def test_reject_missing_gradient_and_nuclear_singularity(self):
        geometry,probes,keyword=self.inputs();text=keyword.read_text()
        keyword.write_text(text.replace("GRADIENT","ENERGY"))
        with self.assertRaisesRegex(ValueError,"GRADIENT"):workflow.read_inputs(geometry,probes,keyword,"angstrom")
        keyword.write_text(text);probes.write_text("1\n0 0 0\n")
        with self.assertRaisesRegex(ValueError,"singularity"):workflow.read_inputs(geometry,probes,keyword,"angstrom")

    def test_matched_controls_differ_only_in_sampling(self):
        card="HF BASIS=STO-3G GRADIENT CHARGE=0"
        for enabled in (False,True):
            workflow.write_workflow_control(self.root/(str(enabled)+".nml"),nsteps=2,dt_fs=.025,keyword=card,stride=1 if enabled else 0,enabled=enabled)
        def physical(path):return [line for line in path.read_text().splitlines() if "field_hessian=" not in line and "oeprop_stride=" not in line]
        self.assertEqual(physical(self.root/"False.nml"),physical(self.root/"True.nml"))
        self.assertIn("field_hessian=.true.",(self.root/"True.nml").read_text())

    def test_prepare_never_runs_a_native_process_and_alternates_pairs(self):
        geometry,probes,keyword=self.inputs();prefix=self.root/"install";(prefix/"lib").mkdir(parents=True);(prefix/"basis").mkdir()
        library=prefix/"lib/libquick.so";library.write_text("Explicit synthetic runtime-metadata unit fixture")
        (prefix/"basis/DEF2-SVP.BAS").write_text("Explicit synthetic basis metadata fixture")
        source=self.root/"source";(source/"src").mkdir(parents=True);(source/"CMakeLists.txt").write_text("# synthetic unit fixture")
        (source/"src/fixture.f90").write_text("! synthetic provenance fixture")
        driver=self.root/"driver";driver.write_text("#!/bin/sh\nexit 89\n");driver.chmod(0o700)
        args=argparse.Namespace(workdir=self.root/"prepared",driver=driver,install_prefix=prefix,source_snapshot=source,
            geometry=geometry,probes=probes,keyword_file=keyword,probe_units="angstrom",threads=1,ranks=1,library_path=[],launcher="",
            warmup_pairs=1,repeats=3,stride=2,nsteps=2,dt_fs=.025,expected_driver_sha256=None,build_report=None,backend="cpu",
            force_source="auto",prepare_only=True,plot=False,timeout=10.)
        snapshot={"path":str(driver),"sha256":workflow.sha256(driver),"resolved_libraries_sha256":{str(library):workflow.sha256(library)},"ldd":{"unit_fixture":True}}
        with patch.object(workflow.old,"driver_snapshot",return_value=snapshot),patch.object(workflow.subprocess,"run",side_effect=AssertionError("Preparation launched a process")):
            result=workflow.benchmark(args)
        self.assertFalse(result["passed"]);self.assertTrue(result["prepared_only"]);self.assertEqual(len(result["trials"]),0)
        self.assertEqual([row["mode"] for row in result["plans"]],["off","on","on","off","off","on","on","off"])
        self.assertEqual(sum(row["discarded_warmup"] for row in result["plans"]),2)
        self.assertEqual(len(list(args.workdir.rglob("control.nml"))),8)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);log=io.StringIO()
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(WorkflowIO))
    (args.output/"tests.log").write_text(log.getvalue())
    report={"passed":result.wasSuccessful(),"tests":result.testsRun,"scope":__doc__,"native_QUICK_runs":0,"SCFs":0,"GPU_launches":0,
            "failures":[trace for _,trace in result.failures],"errors":[trace for _,trace in result.errors],
            "source_files_sha256":{str(path):workflow.sha256(path) for path in [Path(__file__),HERE/"benchmark_workflow.py",workflow.LEGACY/"benchmark_overhead.py",workflow.LEGACY/"validate_oeprop_bomd.py"]}}
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(log.getvalue(),end="")
    print(json.dumps({"passed":report["passed"],"tests":result.testsRun,"native_QUICK_runs":0}))
    return 0 if report["passed"] else 2


HERE=Path(__file__).resolve().parent
if __name__=="__main__":raise SystemExit(main())
