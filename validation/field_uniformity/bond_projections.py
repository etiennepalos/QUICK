#!/usr/bin/env python3
"""Offline fixed-probe projections on explicitly associated water bond axes.

No geometry/density response, native query, SCF, GPU work or continuous-bound
claim. Probe-to-water association is geometric nearest oxygen, not a claim
that the supplied observation point is an atom or bond center.
"""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import re

import numpy as np

from field_uniformity import (SOURCE_CONVENTION,UNITS,G_COMPONENTS,H_COMPONENTS,
                              read_csv,write_csv,sha256,unpack,stf_split,taylor_fields)

BOHR_ANGSTROM=.52917720859  # Match the retained QUICK/native AO protocol.


def require(condition,message):
    if not condition:raise ValueError(message)


def unit_axis(axis):
    axis=np.asarray(axis,float)
    require(axis.shape==(3,) and np.isfinite(axis).all(),'Require a finite three-component unit axis')
    require(abs(np.linalg.norm(axis)-1)<=1e-12,'Axis must already be normalized')
    return axis


def project(e,g,h,axis):
    axis=unit_axis(axis);e=np.asarray(e,float);g=np.asarray(g,float);h=np.asarray(h,float)
    require(e.shape==(3,) and g.shape==(3,3) and h.shape==(3,3,3),'Malformed E/G/H projection shapes')
    require(all(np.isfinite(x).all() for x in (e,g,h)),'Nonfinite E/G/H projection input')
    gstf,gsource,hstf,hsource,_,_=stf_split(g,h)
    parallel=float(e@axis)
    q=lambda matrix:float(axis@matrix@axis)
    c=lambda tensor:float(np.einsum('i,j,k,ijk',axis,axis,axis,tensor))
    return dict(e_parallel=parallel,e_perpendicular_norm=float(np.linalg.norm(e-parallel*axis)),
                g_axis=q(g),h_axis=c(h),g_axis_stf=q(gstf),g_axis_trace_source=q(gsource),
                h_axis_stf=c(hstf),h_axis_trace_source=c(hsource))


def xyz_waters(path):
    lines=path.read_text().splitlines();count=int(lines[0])
    require(count>0 and count%3==0 and len(lines)==count+2,'Require exactly one complete OHH XYZ frame')
    atoms=[];coordinates=[]
    for line in lines[2:]:
        words=line.split();require(len(words)==4,'Malformed XYZ atom row')
        symbol=words[0].capitalize();z=int(symbol) if symbol.isdigit() else {'O':8,'H':1}.get(symbol,0)
        atoms.append(z);coordinates.append(list(map(float,words[1:])))
    coordinates=np.array(coordinates);require(np.isfinite(coordinates).all(),'Nonfinite XYZ geometry')
    require(atoms==[8,1,1]*(count//3),'Require explicit repeated O,H,H atom grouping')
    oxygen=coordinates[::3]
    for k in range(count//3):
        for hydrogen in (3*k+1,3*k+2):
            distances=np.linalg.norm(oxygen-coordinates[hydrogen],axis=1)
            require(distances[k]>0 and int(np.argmin(distances))==k,'Grouped H is not uniquely associated with its nearest O')
            require(np.count_nonzero(abs(distances-distances[k])<1e-10)==1,'Ambiguous grouped H/O association')
    return np.array(atoms),coordinates


def geometry_identity(geometry,native_input,model,manifest):
    atoms,xyz=xyz_waters(geometry)
    specification=json.loads(model.read_text());require(sha256(model)==manifest['provenance']['geometry_sha256'],'Model hash differs from frozen-density provenance')
    require([int(z) for z,_ in specification['atoms']]==atoms.tolist(),'Model/XYZ atom identity differs')
    model_xyz=np.array([p for _,p in specification['atoms']],float)
    require(np.allclose(xyz,model_xyz,atol=1e-12,rtol=0),'Model/XYZ coordinates differ')
    lines=native_input.read_text().splitlines();na,nq,np_=map(int,lines[1].split())
    require(na==len(atoms) and nq==len(specification.get('charges',[])),'Native input/model source counts differ')
    records=np.array([line.split() for line in lines[2:2+na]],float)
    require(np.array_equal(records[:,0],atoms),'Native input/XYZ atom identity differs')
    require(np.allclose(records[:,1:],xyz,atol=1e-12,rtol=0),'Native input/XYZ coordinates differ')
    if nq:
        charges=np.array([line.split() for line in lines[2+na:2+na+nq]],float)
        require(np.allclose(charges,specification['charges'],atol=1e-12,rtol=0) and 'EXTCHARGES' in lines[0].upper().split(),'Native input/model enabled MM charges differ')
    keywords={k:v for k,sep,v in (word.partition('=') for word in lines[0].upper().split()) if sep}
    require(int(keywords.get('CHARGE','0'))==specification['charge'] and int(keywords.get('MULT','1'))==specification['spin']+1,'Native input/model charge or multiplicity differs')
    return atoms,xyz,dict(atoms=na,water_groups=na//3,enabled_MM_charges=nq,native_probes=np_,
        native_XYZ_max_coordinate_error_angstrom=float(np.max(abs(records[:,1:]-xyz))),
        model_XYZ_max_coordinate_error_angstrom=float(np.max(abs(model_xyz-xyz))),
        identity='Same atom identities/order and coordinates as accepted native calculation; fixed nuclei and retained density')


POLY={(0,0,0):.19,(1,0,0):.31,(0,1,0):.12,(0,0,1):-.23,(2,0,0):.17,
      (1,1,0):-.11,(0,1,1):.08,(0,0,2):-.15,(3,0,0):.07,(1,1,1):-.05,
      (0,1,2):.03,(4,0,0):.021,(2,1,1):.013}


def polynomial(point,orders=(0,0,0)):
    total=0.
    for powers,coefficient in POLY.items():
        if any(p<d for p,d in zip(powers,orders)):continue
        factor=coefficient
        for x,p,d in zip(point,powers,orders):factor*=math.factorial(p)/math.factorial(p-d)*x**(p-d)
        total+=factor
    return total


def polynomial_tensor(point,rank):
    out=np.empty((3,)*rank)
    for axes in itertools.product(range(3),repeat=rank):out[axes]=-polynomial(point,tuple(axes.count(k) for k in range(3)))
    return out


def math_validation():
    checks=[]
    def close(name,value,expected,gate):
        error=float(np.max(abs(np.asarray(value)-expected)));require(error<=gate,name+': '+str(error))
        checks.append(dict(name=name,maximum_absolute_error=error,absolute_gate=gate))
    rng=np.random.default_rng(491);point=np.array([.23,-.31,.42]);e,g,h=[polynomial_tensor(point,r) for r in (1,2,3)]
    for k in range(12):
        axis=rng.normal(size=3);axis/=np.linalg.norm(axis);p=project(e,g,h,axis)
        close(f'decomposition-G-{k}',p['g_axis_stf']+p['g_axis_trace_source'],p['g_axis'],2e-14)
        close(f'decomposition-H-{k}',p['h_axis_stf']+p['h_axis_trace_source'],p['h_axis'],2e-14)
        for step in (.02,.01):
            v0=polynomial(point);vp=polynomial(point+step*axis);vm=polynomial(point-step*axis)
            vpp=polynomial(point+2*step*axis);vmm=polynomial(point-2*step*axis)
            close(f'potential-directional-E-{k}-{step}',-(vp-vm)/(2*step),p['e_parallel'],2e-4)
            close(f'potential-directional-G-{k}-{step}',-(vp-2*v0+vm)/step**2,p['g_axis'],2e-4)
            close(f'potential-directional-H-{k}-{step}',-(vpp-2*vp+2*vm-vmm)/(2*step**3),p['h_axis'],2e-8)
        rotation,_=np.linalg.qr(rng.normal(size=(3,3)))
        if np.linalg.det(rotation)<0:rotation[:,0]*=-1
        transformed=project(rotation@e,rotation@g@rotation.T,np.einsum('ia,jb,kc,abc->ijk',rotation,rotation,rotation,h),rotation@axis)
        close(f'rotation-invariants-{k}',list(transformed.values()),np.array(list(p.values())),2e-13)
        reversed_=project(e,g,h,-axis)
        close(f'axis-parity-{k}',[reversed_['e_parallel'],reversed_['g_axis'],reversed_['h_axis'],reversed_['e_perpendicular_norm']],
              [-p['e_parallel'],p['g_axis'],-p['h_axis'],p['e_perpendicular_norm']],2e-14)
        residual=rng.normal(size=3)
        require(abs(residual@axis)<=np.linalg.norm(residual)+1e-14,'Projected residual exceeds full-vector norm')
        checks.append(dict(name=f'projection-contraction-{k}',passed=True))
    for bad in ([0,0,0],[2,0,0],[np.nan,0,1],[1,0]):
        try:unit_axis(bad)
        except ValueError:checks.append(dict(name='reject-nonunit-or-malformed-axis',passed=True));continue
        raise AssertionError('Invalid axis accepted')
    return dict(passed=True,checks=checks,number_checks=len(checks),native_jobs=0,SCFs=0,GPU_kernels=0,
                scope='Independent analytic quartic scalar-potential directional finite differences, rotations, unit-axis/parity and norm inequalities')


def analyze(args):
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    summary={'passed':False,'scope':__doc__,'native_jobs_launched':0,'SCFs':0,'GPU_kernels':0,'continuous_bound_claim':False}
    try:
        tests=math_validation();(output/'math-validation.json').write_text(json.dumps(tests,indent=2)+'\n');summary['math_checks']=tests['number_checks']
        if args.self_test:
            summary.update(passed=True,self_test_only=True)
        else:
            for name in ('manifest','geometry','native_input','model','native_report'):require(getattr(args,name) is not None,'Required application argument: --'+name.replace('_','-'))
            manifest_path=args.manifest.resolve(strict=True);manifest=json.loads(manifest_path.read_text())
            require(manifest.get('schema')=='quick-field-uniformity-input-1' and manifest.get('input_kind')=='quick-frozen-density','Require actual frozen-density analysis input')
            require(manifest['source_convention']==SOURCE_CONVENTION and manifest['units']==UNITS,'Unsupported source convention or units')
            require(manifest['provenance']['same_density_and_geometry_for_all_queries'] is True,'Require common fixed density/geometry')
            center_path=(manifest_path.parent/manifest['centers_csv']).resolve(strict=True);sample_path=(manifest_path.parent/manifest['samples_csv']).resolve(strict=True)
            report_path=args.native_report.resolve(strict=True);native_report=json.loads(report_path.read_text())
            require(native_report.get('passed') is True and native_report.get('provenance_unchanged') is True,'Accepted native/provenance report required')
            for path in (manifest_path,center_path,sample_path,args.native_input.resolve(strict=True)):
                relative=str(path.relative_to(report_path.parent));require(native_report['retained_artifacts_sha256'].get(relative)==sha256(path),'Input differs from accepted native report: '+relative)
            source_paths={Path(p):digest for p,digest in manifest['provenance']['source_files_sha256'].items()}
            require(all(path.is_file() and sha256(path)==digest for path,digest in source_paths.items()),'Frozen native/conversion provenance file changed')
            native_state=native_report['native_validation']['result']
            require(native_state.get('scf_converged')==['T'] and all(float(native_state[k][0])==0 for k in ('extra_API_calls','density_error','fock_error')),'Native retained-state assertions failed')
            paths={manifest_path,center_path,sample_path,args.geometry.resolve(strict=True),args.native_input.resolve(strict=True),args.model.resolve(strict=True),report_path,Path(__file__).resolve(),Path(__file__).with_name('field_uniformity.py').resolve(),*source_paths}
            before={str(path):sha256(path) for path in paths}
            atoms,xyz,identity=geometry_identity(args.geometry,args.native_input,args.model,manifest)
            centers=read_csv(center_path,['center_id','x_bohr','y_bohr','z_bohr',*['e_'+x for x in 'xyz'],*['g_'+x for x in G_COMPONENTS],*['h_'+x for x in H_COMPONENTS]])
            samples=read_csv(sample_path,['center_id','sample_id','direction_id','radius_bohr',*['u_'+x+'_bohr' for x in 'xyz'],*['e_direct_'+x for x in 'xyz']])
            require(len({r['center_id'] for r in centers})==len(centers),'Duplicate probe center IDs')
            require(len({(r['center_id'],r['sample_id']) for r in samples})==len(samples),'Duplicate sample IDs')
            require({r['center_id'] for r in centers}=={r['center_id'] for r in samples},'Center/sample association differs')
            oxygen=xyz[::3]/BOHR_ANGSTROM;associations=[];center_rows=[];sample_rows=[];radius_rows=[];max_excess=0.;direction_counts=set()
            for center in centers:
                cid=center['center_id'];position=np.array([float(center[x+'_bohr']) for x in 'xyz']);distance=np.linalg.norm(oxygen-position,axis=1);order=np.argsort(distance);water=int(order[0]);gap=float(distance[order[1]]-distance[order[0]]) if len(order)>1 else None
                require(gap is None or gap>1e-10,'Ambiguous nearest-oxygen probe association')
                association=dict(center_id=cid,water_index_1based=water+1,oxygen_atom_1based=3*water+1,
                    **{x+'_bohr':float(position[k]) for k,x in enumerate('xyz')},probe_O_distance_bohr=float(distance[water]),
                    probe_O_distance_angstrom=float(distance[water]*BOHR_ANGSTROM),next_O_distance_bohr=float(distance[order[1]]) if len(order)>1 else '',
                    nearest_O_gap_bohr=gap if gap is not None else '',association='Nearest oxygen by supplied geometry; observation probe is not relabeled as atom/bond center')
                associations.append(association)
                bond_vectors=[xyz[3*water+k]-xyz[3*water] for k in (1,2)];lengths=[np.linalg.norm(v) for v in bond_vectors]
                axes=[(f'O-H{k+1}',v/lengths[k],3*water+k+2,float(lengths[k])) for k,v in enumerate(bond_vectors)]
                if args.bisector:
                    vector=axes[0][1]+axes[1][1];require(np.linalg.norm(vector)>1e-12,'Degenerate O-H bisector')
                    axes.append(('OH-unit-axis-bisector',vector/np.linalg.norm(vector),0,0.))
                e=np.array([float(center['e_'+x]) for x in 'xyz']);g=unpack([float(center['g_'+x]) for x in G_COMPONENTS],2);h=unpack([float(center['h_'+x]) for x in H_COMPONENTS],3)
                selected=[r for r in samples if r['center_id']==cid];displacements=np.array([[float(r['u_'+x+'_bohr']) for x in 'xyz'] for r in selected]);direct=np.array([[float(r['e_direct_'+x]) for x in 'xyz'] for r in selected]);radii=np.array([float(r['radius_bohr']) for r in selected])
                require(np.all(radii>0) and np.allclose(np.linalg.norm(displacements,axis=1),radii,atol=1e-12,rtol=1e-10),'Malformed sample radius/displacement')
                prediction=taylor_fields(e,g,h,displacements)
                for axis_name,axis,hydrogen,length in axes:
                    projection=project(e,g,h,axis)
                    center_rows.append({**association,'axis':axis_name,'hydrogen_atom_1based':hydrogen,'O_H_length_angstrom':length,
                        **{'axis_'+x:float(axis[k]) for k,x in enumerate('xyz')},**projection})
                    direct_parallel=direct@axis;residuals={name:(value-direct) for name,value in prediction.items()}
                    for index,row in enumerate(selected):
                        record=dict(center_id=cid,water_index_1based=water+1,axis=axis_name,sample_id=row['sample_id'],direction_id=row['direction_id'],radius_bohr=float(radii[index]),
                            **{'u_'+x+'_bohr':float(displacements[index,k]) for k,x in enumerate('xyz')},cos_displacement_to_axis=float(displacements[index]@axis/radii[index]),
                            direct_parallel=float(direct_parallel[index]),direct_perpendicular_norm=float(np.linalg.norm(direct[index]-direct_parallel[index]*axis)))
                        for name,value in prediction.items():
                            scalar=float(residuals[name][index]@axis);norm=float(np.linalg.norm(residuals[name][index]));excess=abs(scalar)-norm;max_excess=max(max_excess,excess)
                            require(excess<=1e-14*max(1.,norm),'Projected reconstruction error exceeds full vector norm')
                            record.update({name+'_parallel':float(value[index]@axis),name+'_parallel_residual':scalar,
                                           name+'_parallel_abs_error':abs(scalar),name+'_full_vector_error_norm':norm})
                        sample_rows.append(record)
                    for radius in sorted(set(radii)):
                        take=np.flatnonzero(radii==radius);count=len({selected[k]['direction_id'] for k in take});require(count==len(take),'Duplicated direction within radius');direction_counts.add(count)
                        record=dict(center_id=cid,axis=axis_name,radius_bohr=float(radius),supplied_directions=count)
                        for name,residual in residuals.items():
                            parallel=residual[take]@axis
                            record.update({name+'_max_abs_parallel_error':float(max(abs(parallel))),
                                           name+'_rms_parallel_error':float(np.sqrt(np.mean(parallel**2))),
                                           name+'_max_full_vector_error':float(max(np.linalg.norm(residual[take],axis=1)))})
                        radius_rows.append(record)
            write_csv(output/'probe_water_associations.csv',associations);write_csv(output/'center_axis_projections.csv',center_rows)
            write_csv(output/'sample_axis_residuals.csv',sample_rows);write_csv(output/'radius_axis_summary.csv',radius_rows)
            after={str(path):sha256(path) for path in paths};require(after==before,'Frozen source/input/native files changed during offline analysis')
            summary.update(passed=True,geometry_identity=identity,centers=len(centers),axes_per_center=3 if args.bisector else 2,
                center_axis_rows=len(center_rows),sample_axis_rows=len(sample_rows),radius_axis_rows=len(radius_rows),supplied_direction_counts=sorted(direction_counts),
                associated_distinct_waters=len({r['water_index_1based'] for r in associations}),probe_O_distance_angstrom_range=[min(r['probe_O_distance_angstrom'] for r in associations),max(r['probe_O_distance_angstrom'] for r in associations)],
                source_convention=SOURCE_CONVENTION,units={**UNITS,'axis':'dimensionless O-to-H unit vector; optional normalized sum bisector'},
                convention='At original fixed probe C: E.a, a_i G_ij a_j, a_i a_j a_k H_ijk; G=dE/dC=-Vsecond,H=d2E/dC2=-Vthird. Sample residual=prediction-direct; all axes held fixed.',
                interpretation='O-H directions belong to the geometrically nearest water. Probes remain their supplied observation positions; no atom/bond-center identity, nuclear response, vibrational/Stark calibration or continuous sampled-sphere bound is inferred.',
                density_sha256=manifest['provenance']['density_sha256'],geometry_model_sha256=manifest['provenance']['geometry_sha256'],
                native_report=str(report_path),native_report_sha256=sha256(report_path),inputs_and_sources_sha256=before,inputs_unchanged=True,
                maximum_projection_residual_minus_full_vector_norm=max_excess,projection_inequality_absolute_slack=1e-14,
                mathematical_projection_checks=tests['number_checks'],new_native_queries=0)
    except Exception as error:
        import traceback
        summary.update(error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
    summary['outputs_sha256']={p.name:sha256(p) for p in output.iterdir() if p.is_file()}
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:summary.get(k) for k in ('passed','centers','center_axis_rows','sample_axis_rows','math_checks','error')}))
    return 0 if summary['passed'] else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','geometry','native-input','model','native-report'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--bisector',action='store_true');parser.add_argument('--self-test',action='store_true')
    return analyze(parser.parse_args())


if __name__=='__main__':raise SystemExit(main())
