#!/usr/bin/env python3
"""Bounded independent PBE0 reference using each geometry's exported QUICK SG1.

No QUICK/PySCF source is modified. Independent SCFs use full PySCF AOs on
the actual QUICK coordinates/weights. Native bin/primitive AO screening is
not reproduced; native weight<DMCutoff skipping is reproduced explicitly.
Finite differences export a separate grid at each displaced geometry, so
moving-grid and partition-weight response is included in the energy change.
"""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
import pyscf
from pyscf import dft,gto,lib
from pyscf.gto.basis import parse_gaussian

BOHR=0.52917720859


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--driver',type=Path,required=True)
    parser.add_argument('--workdir',type=Path,required=True)
    parser.add_argument('--quick-report',type=Path,required=True)
    parser.add_argument('--higher-grid-report',type=Path,required=True)
    parser.add_argument('--case',default='water_pbe0_sto3g')
    parser.add_argument('--component',default='1z',help='One-based atom and Cartesian axis, e.g.1z')
    parser.add_argument('--steps',type=float,nargs='+',default=[.001,.0005])
    args=parser.parse_args()
    if (not re.fullmatch(r'[1-9][0-9]*[xyz]',args.component) or len(set(args.steps))!=len(args.steps)
            or any(not math.isfinite(h) or h<=0 for h in args.steps)):
        parser.error('Require a valid atom/axis component and finite positive displacement sizes')
    component_atom=int(args.component[:-1])-1;component_axis='xyz'.index(args.component[-1])
    args.repo=args.repo.resolve(strict=True);args.driver=args.driver.resolve(strict=True)
    args.quick_report=args.quick_report.resolve(strict=True);args.higher_grid_report=args.higher_grid_report.resolve(strict=True);args.workdir=args.workdir.resolve()
    args.workdir.mkdir(parents=True,exist_ok=False)
    lib.num_threads(1)
    native=json.loads(args.quick_report.read_text())
    if not native['passed']:raise ValueError('Require an accepted native field report')
    field=next(r for r in native['runs'] if r['label']==args.case+'-mixed_plus' and not r['reference'])
    inputfile=Path(field['workdir'])/'field.in'
    if sha(inputfile)!=field['input_sha256']:raise ValueError('Frozen input changed')
    sys.path.insert(0,str(args.repo/'validation/solvation'))
    sys.path.insert(0,str(args.repo/'validation/external_field'))
    from validate_solvation import runtime
    from validate_pyscf_external_field import parse_model
    model=parse_model(inputfile)
    if model['method']!='PBE0' or model['basis'].upper()!='STO-3G' or model['charge']!=0 or model['spin']!=0 or model['unrestricted']:
        raise ValueError('This bounded diagnostic supports neutral restricted PBE0/STO-3G only')
    if component_atom>=len(model['atoms_bohr']):raise ValueError('Selected component atom is outside molecule')
    higher=json.loads(args.higher_grid_report.read_text())
    higher_output=args.higher_grid_report.parent/'oracle-output.json'
    if sha(higher_output)!=higher['oracle_output_sha256']:raise ValueError('Higher-grid raw oracle changed')
    higher_worker=json.loads(higher_output.read_text())
    if not higher.get('exact_quick_basis',False):raise ValueError('Higher-grid reference must use exact QUICK basis contractions')
    higher_result=higher_worker['models'][field['input_sha256']]
    higher_components={x['step_bohr']:x['gradient_hartree_per_bohr'][3*component_atom+component_axis] for x in higher_result['finite_difference_gradients']}
    field_vector=np.array(model['field_au'])

    atomlines=re.split(r'\n\s*\n',inputfile.read_text().strip())[1].splitlines()
    symbols=[line.split()[0] for line in atomlines]
    base=np.array([[float(v)/BOHR for v in line.split()[1:]] for line in atomlines])
    basisfile=args.repo/'basis/STO-3G.BAS';text=basisfile.read_text();basis={}
    native_basis=(Path(native['arguments']['basis'])/'STO-3G.BAS').resolve()
    if sha(basisfile)!=native['source_inputs_before'].get(str(native_basis)) or sha(basisfile)!=native['source_inputs_after'].get(str(native_basis)):
        raise ValueError('Reference basis differs from native frozen basis definition')
    previous_basis_hashes=[value for path,value in higher_worker['independent_runtime_before'].items() if Path(path).name==basisfile.name]
    if previous_basis_hashes!=[sha(basisfile)]:raise ValueError('Higher-grid reference used a different basis definition')

    for symbol in set(symbols):
        blocks=[b for b in text.split('****') if re.search(r'^\s*'+symbol+r'\s+0\s*$',b,re.M|re.I)]
        if len(blocks)!=1:raise ValueError('Incomplete QUICK basis block')
        basis[symbol]=parse_gaussian.parse(blocks[0],optimize=False)
    core=[args.repo/'src/modules/quick_gridpoints_module.f90',args.repo/'src/modules/quick_scf_operator_module.f90',
        args.repo/'src/modules/quick_method_module.f90',args.driver,Path(__file__).with_name('quick_sg1_export_driver.f90'),
        basisfile,Path(__file__).resolve(),Path(sys.executable).resolve(),args.quick_report,inputfile,args.higher_grid_report,higher_output]
    env=os.environ.copy();env.update(QUICK_BASIS=str(args.repo/'basis'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    quick_runtime_before=runtime(args.driver,env)
    if '_cuda' in quick_runtime_before['quick_library'] or '_mpi' in quick_runtime_before['quick_library']:
        raise ValueError('Use the serial CPU exporter for this bounded diagnostic')
    package=Path(pyscf.__file__).resolve().parent
    independent_paths=[package/p for p in ('gto/mole.py','gto/basis/parse_gaussian.py','scf/hf.py','dft/rks.py','dft/gen_grid.py','dft/numint.py','dft/libxc.py')]
    independent_paths+=sorted({p.resolve() for p in (package/'lib').rglob('*.so*') if p.is_file()})
    report={'started_utc':datetime.now(timezone.utc).isoformat(),'scope':__doc__,
        'source_runtime_before':{str(p):sha(p) for p in core},'runs':[],
        'component':args.component,'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'native_basis_sha256':sha(basisfile),'quick_runtime_before':quick_runtime_before,
        'independent_runtime_before':{str(p):sha(p) for p in independent_paths},
        'higher_grid_reference':{'path':str(args.higher_grid_report),'sha256':sha(args.higher_grid_report),'passed':higher['passed'],
            'failed_checks':[c for c in higher['checks'] if not c['passed']]},
        'field_origin':'Diagnostic origin0; neutral restricted molecule total energy/gradient are origin-invariant.',
        'units':{'points':'bohr','weights':'bohr^3','gradient':'hartree/bohr'},
        'pyscf_version':pyscf.__version__,'comparisons':[],
        'original_absolute_failures_remain_visible':True,
        'settings':{'direct_scf_tol':1e-14,'scf_energy_tolerance':1e-12,'orbital_gradient_tolerance':1e-9,
            'additional_density_pruning':False,'native_weight_skip_reproduced':True,
            'native_bin_AO_primitive_screening_reproduced':False}}
    for name,F in [('zero',np.zeros(3)),('field',field_vector)]:
        for step,sign in [(0.,0)]+[(h,sgn) for h in args.steps for sgn in (1,-1)]:
            label=f'{name}-h{step:g}-s{sign:+d}';folder=args.workdir/label;folder.mkdir()
            coords=base.copy();coords[component_atom,component_axis]+=sign*step
            card='DFT PBE0 BASIS=STO-3G CHARGE=0 MULT=1 GRADIENT SCF=250 DENSERMS=1D-10 CUTOFF=1D-12 GRADCUTOFF=1D-12 XCCUTOFF=1D-12 BASISCUTOFF=1D-12'
            if name=='field':card+=''.join(f' EXTERNAL_EFIELD_{axis.upper()}={value:.12g}' for axis,value in zip('xyz',F) if value)
            if len(card)>256:raise ValueError('API keyword too long')
            inp=folder/'input.dat';inp.write_text(card+f'\n{len(symbols)}\n'+''.join(f'{gto.charge(s)} '+' '.join(f'{v:.17g}' for v in c*BOHR)+'\n' for s,c in zip(symbols,coords)))
            with(folder/'execution.log').open('w') as log:
                process=subprocess.run([str(args.driver),'input.dat'],cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
            if process.returncode:raise RuntimeError('Native export failed: '+label)
            raw=(folder/'result.dat').read_text().splitlines();values={line.split()[0]:float(line.split()[1]) for line in raw if line.split()[0]!='gradient'}
            qgradient=np.array([[float(x) for x in line.split()[2:]] for line in raw if line.startswith('gradient')])
            grid=np.loadtxt(folder/'grid.dat',skiprows=1);weights=grid[:,3].copy()
            ngrid=int((folder/'grid.dat').read_text().splitlines()[0])
            if grid.shape!=(ngrid,6) or not np.all(np.isfinite(grid)) or qgradient.shape!=(len(symbols),3):
                raise ValueError('Malformed/nonfinite native grid or gradient')
            skipped=weights<values['matrix_cutoff'];weights[skipped]=0
            mol=gto.M(atom=list(zip(symbols,coords)),unit='Bohr',basis=basis,cart=True,verbose=0)
            mf=dft.RKS(mol);mf.xc='PBE0';mf.conv_tol=1e-12;mf.conv_tol_grad=1e-9;mf.direct_scf_tol=1e-14;mf.max_cycle=250
            mf.small_rho_cutoff=0;mf.grids.coords=np.ascontiguousarray(grid[:,:3]);mf.grids.weights=np.ascontiguousarray(weights)
            mf.grids.non0tab=mf.grids.make_mask(mol,mf.grids.coords)
            moments=mol.intor('int1e_r',comp=3,hermi=1);hcore=mf.get_hcore()+np.einsum('x,xij->ij',F,moments)
            enuc=mol.energy_nuc()-np.einsum('a,ax,x->',mol.atom_charges(),coords,F)
            mf.get_hcore=lambda *unused:hcore;mf.energy_nuc=lambda *unused:enuc
            pe=float(mf.kernel())
            if not mf.converged or not math.isfinite(pe):raise RuntimeError('Independent SCF failed: '+label)
            dm=mf.make_rdm1();ao=dft.numint.eval_ao(mol,grid[:,:3]);rho=np.einsum('pi,ij,pj->p',ao,dm,ao)
            record={'label':label,'field':name,'step_bohr':step,'sign':sign,'quick_energy':values['energy'],
                'pyscf_energy':pe,'energy_difference':values['energy']-pe,'quick_gradient':qgradient.tolist(),
                'grid_points':len(grid),'native_weight_skips':int(skipped.sum()),
                'integrated_electrons_full_AOs':float(np.dot(weights,rho)),
                'density_threshold_skipped_electrons_bound':float(np.dot(np.abs(weights[rho<2*values['matrix_cutoff']]),np.abs(rho[rho<2*values['matrix_cutoff']]))),
                'quick_settings':values,'source_grid_hash':sha(folder/'grid.dat'),
                'raw_hashes':{p.name:sha(p) for p in folder.iterdir() if p.is_file()}}
            report['runs'].append(record);print(label,record['energy_difference'],flush=True)
    for name in ('zero','field'):
        runs=[r for r in report['runs'] if r['field']==name];base_run=next(r for r in runs if not r['step_bohr'])
        for step in args.steps:
            plus=next(r for r in runs if r['step_bohr']==step and r['sign']==1)
            minus=next(r for r in runs if r['step_bohr']==step and r['sign']==-1)
            pq=(plus['quick_energy']-minus['quick_energy'])/(2*step)
            pp=(plus['pyscf_energy']-minus['pyscf_energy'])/(2*step)
            analytic=base_run['quick_gradient'][component_atom][component_axis]
            report['comparisons'].append({'field':name,'step_bohr':step,'quick_analytic':analytic,
                'quick_finite_difference':pq,'matched_grid_pyscf_finite_difference':pp,
                'quick_analytic_minus_quick_FD':analytic-pq,'quick_analytic_minus_pyscf_matched_FD':analytic-pp,
                'higher_grid_pyscf_component':higher_components.get(step) if name=='field' else None,
                'quick_analytic_minus_higher_grid_FD':analytic-higher_components[step] if name=='field' and step in higher_components else None})
    report['source_runtime_after']={str(p):sha(p) for p in core}
    report['quick_runtime_after']=runtime(args.driver,env)
    report['independent_runtime_after']={str(p):sha(p) for p in independent_paths}
    report['quick_runtime_unchanged']=report['quick_runtime_before']==report['quick_runtime_after']
    report['independent_runtime_unchanged']=report['independent_runtime_before']==report['independent_runtime_after']
    report['source_runtime_unchanged']=report['source_runtime_before']==report['source_runtime_after']
    report['max_abs_energy_difference']=max(abs(r['energy_difference']) for r in report['runs'])
    report['finished_utc']=datetime.now(timezone.utc).isoformat()
    (args.workdir/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:report[k] for k in ('max_abs_energy_difference','source_runtime_unchanged','quick_runtime_unchanged','independent_runtime_unchanged','comparisons')},indent=2))


if __name__=='__main__':main()
