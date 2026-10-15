#!/usr/bin/env python3
"""Validate independent absolute and incremental applied-field forces.

Reads accepted QUICK field reports, uses their exact Cartesian .BAS
contractions, and preserves both absolute DFT discrepancies and separately
reported zero-to-field response gates. The orchestrator uses stdlib only;
--python selects a scientific environment with NumPy and PySCF.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import sys


DEFAULT_REPO = Path(__file__).resolve().parents[2]
CORE_CASES = ('water_hf_sto3g', 'water_hf_631gss',
              'water_cation_uhf_sto3g', 'water_pbe0_sto3g')
BASIS_ALIASES = {'6-31G*': '6-31GD', '6-31G**': '6-31GDP',
                 '6-31G(D)': '6-31GD', '6-31G(D,P)': '6-31GDP'}


def load_oracle(repo):
    source = repo / 'validation/external_field/validate_pyscf_external_field.py'
    spec = importlib.util.spec_from_file_location('quick_existing_field_oracle', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, source


def basis_path(repo, name):
    name = BASIS_ALIASES.get(name.upper(), name.upper())
    return (repo / 'basis' / (name + '.BAS')).resolve(strict=True)


def option_values(arguments, name):
    if name not in arguments:
        return []
    values = []
    for argument in arguments[arguments.index(name)+1:]:
        if argument.startswith('--'):
            break
        values.append(argument)
    return values


def run_worker(reference, source, repo, input_path, output_path):
    # All scientific imports are confined to the explicitly launched worker.
    import pyscf
    from pyscf import gto
    from pyscf.gto.basis import parse_gaussian

    request = json.loads(input_path.read_text())
    source_paths = {Path(__file__).resolve(), source.resolve()}
    source_paths.update(basis_path(repo, model['basis'])
                        for model in request['models'].values())
    package = Path(pyscf.__file__).resolve().parent
    for relative in ('gto/mole.py', 'gto/basis/parse_gaussian.py',
                     'scf/hf.py', 'scf/uhf.py', 'dft/rks.py', 'dft/uks.py',
                     'dft/gen_grid.py', 'dft/numint.py', 'dft/libxc.py'):
        source_paths.add(package / relative)
    # Actual integral/SCF/XC shared objects, including bundled dependencies.
    for candidate in (package / 'lib').rglob('*.so*'):
        if candidate.is_file():
            source_paths.add(candidate.resolve())
    before = {str(path): reference.sha256(path) for path in sorted(source_paths)}
    original_factory = gto.M
    basis_cache = {}

    def exact_factory(*args, **kwargs):
        if args or not isinstance(kwargs.get('basis'), str):
            raise ValueError('Expected the repository oracle named-basis gto.M call')
        name = kwargs['basis']
        elements = sorted({gto.mole._std_symbol(gto.charge(atom[0]))
                           for atom in kwargs['atom']})
        key = (name.upper(), tuple(elements))
        if key not in basis_cache:
            path = basis_path(repo, name)
            text = path.read_text()
            basis = {}
            for symbol in elements:
                matches = [block for block in text.split('****')
                           if re.search(r'^\s*' + re.escape(symbol) + r'\s+0\s*$',
                                        block, re.M | re.I)]
                if len(matches) != 1:
                    raise ValueError(f'Expected one complete {symbol} block in {path}')
                basis[symbol] = parse_gaussian.parse(matches[0], optimize=False)
            basis_cache[key] = basis
        kwargs['basis'] = basis_cache[key]
        return original_factory(**kwargs)

    gto.M = exact_factory
    try:
        status = reference.worker(input_path, output_path)
    finally:
        gto.M = original_factory
    after = {str(path): reference.sha256(path) for path in sorted(source_paths)}
    result = json.loads(output_path.read_text())
    result['exact_quick_basis'] = True
    result['basis_definition_sources'] = [str(path) for path in sorted(source_paths)
                                          if path.suffix.upper() == '.BAS']
    result['independent_runtime_before'] = before
    result['independent_runtime_after'] = after
    result['independent_runtime_unchanged'] = before == after
    result['basis_override_scope'] = 'Process-local gto.M factory; restored after worker; installed packages untouched'
    if before != after:
        result['passed'] = False
        result['error'] = 'Independent source/basis/runtime fingerprints changed'
        status = 2
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return status


def absolute_main():
    arguments = sys.argv[1:]
    repo = Path(os.environ.get('QUICK_FIELD_REFERENCE_REPO', str(DEFAULT_REPO)))
    if '--repo' in arguments:
        index = arguments.index('--repo')
        repo = Path(arguments[index+1])
        arguments = arguments[:index] + arguments[index+2:]
    repo = repo.resolve(strict=True)
    reference, source = load_oracle(repo)
    if len(arguments) == 3 and arguments[0] == '--worker':
        return run_worker(reference, source, repo,
                          Path(arguments[1]), Path(arguments[2]))
    if '--cases' not in arguments:
        arguments += ['--cases', *CORE_CASES]
    before = reference.sha256(source)
    os.environ['QUICK_FIELD_REFERENCE_REPO'] = str(repo)
    # Existing main launches __file__ --worker. Route that child through this
    # wrapper while preserving all reference numerical gates without edits.
    reference.__file__ = str(Path(__file__).resolve())
    sys.argv = [str(Path(__file__).resolve()), *arguments]
    status = reference.main()
    if '--workdir' not in arguments:
        return status  # argparse has already printed the appropriate error.
    report_path = Path(arguments[arguments.index('--workdir')+1]).resolve() / 'report.json'
    report = json.loads(report_path.read_text())
    after = reference.sha256(source)
    report['upstream_reference_source'] = {'path': str(source),
                                          'sha256_before': before, 'sha256_after': after}
    report['exact_quick_basis'] = True
    report['repository'] = str(repo)
    report['scope'] += '; exact QUICK .BAS contractions and strict per-report selected-case coverage'
    selected = set(option_values(arguments, '--cases'))
    report['selected_cases_required_per_quick_report'] = sorted(selected)
    for path in report['quick_reports']:
        covered = {entry['case'] for entry in report['comparisons'] if entry['quick_report'] == path}
        missing = sorted(selected-covered)
        report['checks'].append({'name': str(path)+':all_selected_cases_compared',
                                 'passed': not missing, 'max_abs_error': len(missing),
                                 'tolerance': 0, 'missing_cases': missing})
    report['limitations'].append('PBE0 retains independent PySCF XC quadrature; QUICK SG1 is different. The declared DFT comparison tolerances are unchanged.')
    report['checks'].append({'name': 'upstream_oracle_source_unchanged',
                             'passed': before == after, 'max_abs_error': int(before != after),
                             'tolerance': 0})
    oracle_output = report_path.parent / 'oracle-output.json'
    if oracle_output.exists():
        worker = json.loads(oracle_output.read_text())
        report['independent_runtime_unchanged'] = worker.get('independent_runtime_unchanged', False)
        report['checks'].append({'name': 'independent_basis_source_runtime_unchanged',
                                 'passed': report['independent_runtime_unchanged'],
                                 'max_abs_error': int(not report['independent_runtime_unchanged']),
                                 'tolerance': 0})
        for path in report['quick_reports']:
            quick_report = json.loads(Path(path).read_text())
            for entry in quick_report['runs']:
                suffix = '-mixed_plus'
                case = entry['label'][:-len(suffix)]
                if (entry['reference'] or not entry['analytic_gradient_requested']
                        or not entry['label'].endswith(suffix) or case not in selected):
                    continue
                key = entry['input_sha256']
                if key not in worker.get('models', {}):
                    continue  # The upstream campaign already records failure.
                quick_output = Path(entry['workdir']) / 'field.out'
                matches = re.findall(r'^\s*BASIS FUNCTIONS\s*=\s*(\d+)\s*$',
                                     quick_output.read_text(), re.M)
                expected = worker['models'][key]['basis_functions']
                matched = bool(matches) and int(matches[-1]) == expected
                intact = reference.sha256(quick_output) == entry['output_sha256']
                report['checks'].append({'name': str(path)+':'+case+':cartesian_AO_count',
                                         'passed': matched and intact,
                                         'max_abs_error': int(not (matched and intact)),
                                         'tolerance': 0, 'oracle_basis_functions': expected,
                                         'quick_basis_functions': int(matches[-1]) if matches else None,
                                         'quick_output_fingerprint_matches': intact})
    report['passed'] = (report['passed'] and all(row['passed'] for row in report['checks']))
    report['passed_checks'] = sum(row['passed'] for row in report['checks'])
    report['failed_checks'] = sum(not row['passed'] for row in report['checks'])
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    return 0 if report['passed'] else 2


def portable_main():
    import argparse
    from datetime import datetime, timezone
    import math
    import signal
    import subprocess

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick-reports', type=Path, nargs='+', required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', default=list(CORE_CASES))
    parser.add_argument('--steps', type=float, nargs='+', default=[1e-3, 5e-4])
    parser.add_argument('--dft-grid-level', type=int, default=5)
    parser.add_argument('--timeout', type=float, default=7200)
    parser.add_argument('--repo', type=Path, default=DEFAULT_REPO)
    args = parser.parse_args()
    if (not math.isfinite(args.timeout) or args.timeout <= 0
            or args.dft_grid_level < 0
            or any(not math.isfinite(h) or h <= 0 for h in args.steps)):
        parser.error('Require finite positive timeout/displacements and nonnegative grid level')
    args.repo = args.repo.resolve(strict=True)
    args.python = args.python.resolve(strict=True)
    args.quick_reports = [p.resolve(strict=True) for p in args.quick_reports]
    args.workdir = args.workdir.resolve()
    args.workdir.mkdir(parents=True, exist_ok=False)
    reference, source = load_oracle(args.repo)
    absolute_dir = args.workdir / 'absolute-field'
    inputs = [Path(__file__).resolve(), source, args.python, *args.quick_reports]
    report = {'passed': False, 'absolute_comparisons_passed': False,
        'incremental_field_validation_passed': False,
        'started_utc': datetime.now(timezone.utc).isoformat(), 'checks': [],
        'comparisons': [], 'scope': 'Independent exact-QUICK-basis Cartesian SCF '
            'energies/forces; preserve absolute DFT discrepancies and separately '
            'validate zero-to-field response. QUICK SG1 and PySCF level grids differ.',
        'absolute_field_report': str(absolute_dir / 'report.json'),
        'tolerances': {'hf_absolute_energy_hartree': 1e-6,
            'hf_absolute_gradient_hartree_per_bohr': 1e-5,
            'dft_absolute_energy_hartree': 5e-5,
            'dft_absolute_gradient_hartree_per_bohr': 5e-5,
            'field_response_energy_hartree': 1e-6,
            'field_response_gradient_hartree_per_bohr': 1e-6},
        'limitations': ['An incremental response pass does not establish absolute DFT force agreement.',
            'The original absolute report and all of its failed checks remain unchanged.']}
    commands = []

    def check(name, error, tolerance, **details):
        report['checks'].append({'name': name, 'passed': math.isfinite(error) and error <= tolerance,
            'max_abs_error': float(error), 'tolerance': tolerance, **details})

    def execute(command, logfile):
        commands.append(command)
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with logfile.open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                env=env, start_new_session=True)
            try:
                return process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError('Independent reference timeout; inspect ' + str(logfile))

    try:
        report['source_inputs_before'] = {str(p): reference.sha256(p) for p in inputs}
        # Establish the actual basis used by each native calculation before
        # launching any independent SCFs. Matching a basis name is insufficient.
        for path in args.quick_reports:
            quick = json.loads(path.read_text())
            for case in args.cases:
                rows = [r for r in quick['runs'] if r['label'] == case+'-mixed_plus'
                    and r['analytic_gradient_requested'] and not r['reference']]
                if len(rows) != 1:
                    raise ValueError('Require one mixed-field gradient per selected case/report')
                model = reference.parse_model(Path(rows[0]['workdir'])/'field.in')
                selected_basis = basis_path(args.repo, model['basis'])
                native_basis = (Path(quick['arguments']['basis'])/selected_basis.name).resolve()
                recorded_before = quick['source_inputs_before'].get(str(native_basis))
                recorded_after = quick['source_inputs_after'].get(str(native_basis))
                matching = (recorded_before is not None and recorded_before == recorded_after
                    and reference.sha256(selected_basis) == recorded_before)
                check(str(path)+':'+case+':native_basis_definition_matches_reference', int(not matching), 0)
                if not matching:
                    raise ValueError('Selected reference basis differs from the frozen native basis definition')
        command = [str(args.python), '-B', str(Path(__file__).resolve()), '--absolute',
            '--repo', str(args.repo), '--quick-reports', *map(str, args.quick_reports),
            '--python', str(args.python), '--workdir', str(absolute_dir),
            '--cases', *args.cases, '--steps', *map(str, args.steps),
            '--dft-grid-level', str(args.dft_grid_level), '--timeout', str(args.timeout)]
        status = execute(command, args.workdir / 'absolute-execution.log')
        if status not in (0, 2):
            raise RuntimeError('Absolute reference failed to produce scientific evidence')
        absolute_path = absolute_dir / 'report.json'
        absolute = json.loads(absolute_path.read_text())
        if 'error' in absolute or not absolute.get('comparisons'):
            raise RuntimeError('Absolute reference incomplete: ' + str(absolute.get('error')))
        report['absolute_field_report_sha256'] = reference.sha256(absolute_path)
        report['absolute_field_report_passed'] = absolute['passed']
        report['absolute_field_failed_checks'] = [c for c in absolute['checks'] if not c['passed']]
        request = json.loads((absolute_dir / 'oracle-input.json').read_text())
        field_results = json.loads((absolute_dir / 'oracle-output.json').read_text())
        zero_models, targets = {}, []
        dft_absolute_names = set()
        for path in args.quick_reports:
            if reference.sha256(path) != absolute['source_inputs_before'][str(path)]:
                raise ValueError('QUICK report changed during reference')
            quick = json.loads(path.read_text())
            for case in args.cases:
                field = [r for r in quick['runs'] if r['label'] == case+'-mixed_plus'
                    and r['analytic_gradient_requested'] and not r['reference']]
                zero = [r for r in quick['runs'] if r['label'] == case+'-explicit-zero'
                    and r['analytic_gradient_requested'] and not r['reference']]
                if len(field) != 1 or len(zero) != 1:
                    raise ValueError('Require one mixed-field and explicit-zero gradient per case/report')
                field, zero = field[0], zero[0]
                inp, out = Path(zero['workdir'])/'field.in', Path(zero['workdir'])/'field.out'
                if reference.sha256(inp) != zero['input_sha256'] or reference.sha256(out) != zero['output_sha256']:
                    raise ValueError('Zero-field QUICK raw fingerprints changed')
                inputs.extend((inp, out))
                zmodel = reference.parse_model(inp)
                fmodel = request['models'][field['input_sha256']]
                if zmodel['field_au'] != [0.0]*3 or any(zmodel[k] != fmodel[k]
                        for k in ('atoms_bohr','method','basis','charge','spin','unrestricted')):
                    raise ValueError('Zero/field Hamiltonians differ beyond the applied field/origin')
                key = zero['input_sha256']
                zero_models[key] = zmodel
                targets.append({'case': case, 'report': str(path), 'field': field,
                    'zero': zero, 'model': zmodel, 'zero_key': key})
                label = str(path)+':'+case
                if zmodel['method'] == 'PBE0':
                    dft_absolute_names.add(label+':independent_energy')
                    dft_absolute_names.update(label+':independent_gradient:h='+format(h,'g') for h in args.steps)
                legacy = [r for r in quick['runs'] if r['label'] == case+'-legacy-reference'
                    and r['analytic_gradient_requested'] and r['reference']]
                if len(legacy) == 1:
                    legacy_input = Path(legacy[0]['workdir'])/'field.in'
                    legacy_output = Path(legacy[0]['workdir'])/'field.out'
                    if (reference.sha256(legacy_input) != legacy[0]['input_sha256']
                            or reference.sha256(legacy_output) != legacy[0]['output_sha256']):
                        raise ValueError('Pre-feature zero-field raw fingerprints changed')
                    inputs.extend((legacy_input,legacy_output))
                    check(label+':zero_matches_pre_feature_reference',
                        max(abs(a-b) for a,b in zip(zero['result']['gradient_hartree_per_bohr'],
                            legacy[0]['result']['gradient_hartree_per_bohr'])), 0)
        report['zero_raw_before'] = {str(p): reference.sha256(p)
            for p in inputs[len(report['source_inputs_before']):]}
        zero_request = dict(request)
        zero_request['models'] = zero_models
        zero_input, zero_output = args.workdir/'oracle-zero-input.json', args.workdir/'oracle-zero-output.json'
        zero_input.write_text(json.dumps(zero_request, indent=2, sort_keys=True)+'\n')
        command = [str(args.python), '-B', str(Path(__file__).resolve()), '--repo',
            str(args.repo), '--worker', str(zero_input), str(zero_output)]
        if execute(command, args.workdir/'zero-execution.log') != 0:
            raise RuntimeError('Zero-field independent reference failed')
        zero_results = json.loads(zero_output.read_text())
        check('zero_worker_complete', int(not zero_results['passed']), 0)
        check('zero_runtime_unchanged', int(not zero_results['independent_runtime_unchanged']), 0)
        check('field_zero_share_independent_runtime', int(field_results['independent_runtime_after']
            != zero_results['independent_runtime_before']), 0)
        zero_absolute_names = set()
        for target in targets:
            label = target['report']+':'+target['case']
            qzero, qfield = target['zero']['result'], target['field']['result']
            pzero = zero_results['models'][target['zero_key']]
            pfield = field_results['models'][target['field']['input_sha256']]
            dft = target['model']['method'] == 'PBE0'
            energy_tolerance, gradient_tolerance = (5e-5,5e-5) if dft else (1e-6,1e-5)
            name = label+':zero_absolute_energy'
            check(name, abs(qzero['energy_hartree']-pzero['energy_hartree']), energy_tolerance)
            if dft:
                zero_absolute_names.add(name)
            check(label+':field_response_energy', abs((qfield['energy_hartree']-qzero['energy_hartree'])
                -(pfield['energy_hartree']-pzero['energy_hartree'])), 1e-6)
            if len(pzero['finite_difference_gradients']) != len(pfield['finite_difference_gradients']):
                raise ValueError('Zero/field finite-difference step counts differ')
            for zg, fg in zip(pzero['finite_difference_gradients'],pfield['finite_difference_gradients']):
                step = zg['step_bohr']
                if step != fg['step_bohr']:
                    raise ValueError('Zero/field finite-difference steps differ')
                vectors = [qzero['gradient_hartree_per_bohr'], qfield['gradient_hartree_per_bohr'],
                    zg['gradient_hartree_per_bohr'], fg['gradient_hartree_per_bohr']]
                if any(len(v) != 3*len(target['model']['atoms_bohr']) for v in vectors):
                    raise ValueError('Gradient component counts differ')
                zero_error = [a-b for a,b in zip(vectors[0],vectors[2])]
                response_error = [(b-a)-(d-c) for a,b,c,d in zip(*vectors)]
                name = label+':zero_absolute_gradient:h='+format(step,'g')
                check(name, max(map(abs,zero_error)), gradient_tolerance)
                if dft:
                    zero_absolute_names.add(name)
                check(label+':field_response_gradient:h='+format(step,'g'),
                    max(map(abs,response_error)), 1e-6)
                report['comparisons'].append({'case': target['case'], 'quick_report': target['report'],
                    'step_bohr': step, 'zero_absolute_error': zero_error,
                    'field_response_error': response_error})
        report['source_inputs_after'] = {str(p): reference.sha256(p) for p in inputs[:len(report['source_inputs_before'])]}
        check('source_inputs_unchanged', int(report['source_inputs_before'] != report['source_inputs_after']), 0)
        report['zero_raw_after'] = {path: reference.sha256(Path(path)) for path in report['zero_raw_before']}
        check('zero_raw_unchanged', int(report['zero_raw_before'] != report['zero_raw_after']), 0)
        check('absolute_report_unchanged', int(reference.sha256(absolute_path) != report['absolute_field_report_sha256']), 0)
        report['zero_artifact_sha256'] = {str(p): reference.sha256(p) for p in (zero_input,zero_output)}
        report['absolute_comparisons_passed'] = (absolute['passed'] and all(c['passed'] for c in report['checks']
            if c['name'] in zero_absolute_names or ':zero_absolute_' in c['name']))
        # Keep all HF/UHF absolute gates and every source/structure/oracle gate;
        # exclude only explicitly enumerated DFT absolute comparisons here.
        report['incremental_field_validation_passed'] = (all(c['passed'] for c in absolute['checks']
            if c['name'] not in dft_absolute_names) and all(c['passed'] for c in report['checks']
            if c['name'] not in zero_absolute_names))
        report['passed'] = absolute['passed'] and all(c['passed'] for c in report['checks'])
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
    report['commands'] = commands
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    report['failed_checks'] = sum(not c['passed'] for c in report['checks'])
    (args.workdir/'report.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps({k: report.get(k) for k in ('passed','absolute_comparisons_passed',
        'incremental_field_validation_passed','failed_checks','error')}))
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--absolute':
        del sys.argv[1]
        raise SystemExit(absolute_main())
    if '--worker' in sys.argv:
        raise SystemExit(absolute_main())
    raise SystemExit(portable_main())
