#!/usr/bin/env python3
"""Export retained complete-API costs and observed errors without rerunning QUICK."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', action='append', default=None,
                        help='Figure configuration; repeatable, default target. CSV retains all.')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    fingerprints = {}
    for path in args.report:
        report = json.loads(path.read_text())
        if report['status'] != 'PASS':
            raise ValueError(f'Only completed PASS measurements are plotted: {path}')
        fingerprints[str(path.resolve())] = sha(path)
        case = report.get('case')
        if case is None:
            # The first retained schema predates the explicit case field.
            card = (Path(report['frozen_density']['path']).parent / 'input.dat').read_text().splitlines()[0]
            if 'BASIS=STO-3G' not in card.upper():
                raise ValueError('Legacy report needs an unambiguous native basis card')
            case = 'water-sto3g'
        for run in report['runs']:
            native = run['adaptive_report']
            folder = path.parent / f"{run['points']}-{run['configuration']}"
            values = np.loadtxt(folder / 'properties.dat', ndmin=2)
            if values.shape != (run['points'], 29):
                raise ValueError('Expected full direct/adaptive V/E/G export')
            # Columns: xyz, adaptive V/E/G(1+3+9), direct V/E/G(1+3+9).
            errors = [np.abs(values[:, 3] - values[:, 16]),
                      np.linalg.norm(values[:, 4:7] - values[:, 17:20], axis=1),
                      np.linalg.norm(values[:, 7:16] - values[:, 20:29], axis=1)]
            record = {
                'report': str(path.resolve()), 'case': case,
                'backend': report['backend'], 'configuration': run['configuration'],
                'queries': run['points'],
                'roi_half_width_bohr': run['options']['upper_bohr'][0] / 2
                    - run['options']['lower_bohr'][0] / 2,
                'mode': run['options']['mode'], 'degree': run['options']['degree'],
                'fallback': native['fallback'] == 'T',
                'accepted': native['accepted'] == 'T',
                'anchor_backend': run['anchor_query_backends'][0],
                'query_backend': run['anchor_query_backends'][1],
                'anchors': int(native['anchors']), 'bound_count': int(native['bound_count']),
                'depth': int(native['depth']),
                'direct_seconds': run['phases']['direct']['median_seconds'],
                'adaptive_seconds': run['phases']['adaptive']['median_seconds'],
                'preparation_seconds_last_sample': float(native['preparation_seconds']),
                'query_seconds_last_sample': float(native['query_seconds']),
                'process_peak_rss_kb': run['process_peak_rss_kb'],
                'device_last_bytes': int(native.get('device_last_bytes', 0)),
                'total_numerical_budget_known': False,
            }
            for channel, err, tol in zip('VEG', errors, run['options']['tolerances']):
                record[f'target_{channel}'] = tol
                record[f'max_error_{channel}'] = float(np.max(err))
                record[f'rms_error_{channel}'] = float(np.sqrt(np.mean(err * err)))
                raw = native.get(f'budget_{channel}')
                record[f'truncation_budget_{channel}'] = float(raw) if raw is not None else np.nan
                if not np.isclose(record[f'max_error_{channel}'], float(native[f'error_{channel}']),
                                  rtol=2e-6, atol=2e-13):
                    raise ValueError(f'Full export and retained maximum error disagree: {folder}')
            rows.append(record)
    with (args.output / 'adaptive_measurements.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selected = [r for r in rows if r['configuration'] in (args.config or ['target'])]
    if not selected:
        raise ValueError('No selected figure configurations')
    plt.rcParams.update({'font.size': 9, 'savefig.dpi': 220})
    labels = [f"{r['case']} {r['backend']}; {r['queries']:,} points\n"
              f"r={r['roi_half_width_bohr']:.3g}; "
              f"{'fallback' if r['fallback'] else 'accepted'}\n"
              f"tau=({r['target_V']:.0e},{r['target_E']:.0e},{r['target_G']:.0e})"
              for r in selected]
    fig, axes = plt.subplots(1, 3, figsize=(13, max(3.8, .78 * len(selected))))
    for axis, channel in zip(axes, 'VEG'):
        for i, record in enumerate(selected):
            target = record[f'target_{channel}']
            actual = record[f'max_error_{channel}'] / target
            bound = record[f'truncation_budget_{channel}'] / target
            if actual > 0:
                axis.scatter(actual, i, color='#1877b8', s=24)
            else:
                axis.annotate('exact direct fallback', (0.03, i),
                              xycoords=('axes fraction', 'data'), fontsize=7)
            if np.isfinite(bound) and bound > 0:
                axis.scatter(bound, i, marker='x', color='#cd6c16', s=30)
        axis.axvline(1, color='0.4', linewidth=.8, linestyle='--')
        axis.set_xscale('log')
        axis.set_yticks(range(len(selected)))
        axis.set_yticklabels(labels if channel == 'V' else [])
        axis.invert_yaxis()
        axis.set_xlabel(f'{channel}: error / requested tolerance')
        axis.grid(axis='x', alpha=.2)
    axes[0].scatter([], [], color='#1877b8', label='observed maximum')
    axes[0].scatter([], [], marker='x', color='#cd6c16',
                    label='Taylor budget (last attempted on fallback)')
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='upper center',
               bbox_to_anchor=(.5, -.01), ncol=2, fontsize=8)
    fig.suptitle('Finite-query errors and truncation budgets; total numerical error is uncertified')
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(args.output / f'adaptive_error_budget.{suffix}', bbox_inches='tight')
    plt.close(fig)

    cases = sorted({r['case'] for r in selected})
    fig, axes = plt.subplots(1, len(cases), figsize=(7 * len(cases), 7), squeeze=False)
    for axis, case in zip(axes[0], cases):
        matching = [r for r in selected if r['case'] == case]
        groups = {}
        for record in matching:
            key = (record['backend'], record['roi_half_width_bohr'], record['configuration'],
                   tuple(record[f'target_{c}'] for c in 'VEG'))
            groups.setdefault(key, []).append(record)
        for key, group in groups.items():
            group.sort(key=lambda r: r['queries'])
            x = [r['queries'] for r in group]
            targets = ','.join(f'{x:.0e}' for x in key[3])
            label = f'{key[0]}, r={key[1]:.3g} bohr, tau=({targets})'
            line = axis.plot(x, [r['adaptive_seconds'] for r in group], 'o-',
                             label=f'adaptive API: {label}')[0]
            axis.plot(x, [r['direct_seconds'] for r in group], 'x--',
                      color=line.get_color(), label=f'direct VEG: {label}')
        axis.set_xscale('log')
        axis.set_yscale('log')
        axis.set_xlabel('Property queries')
        axis.set_ylabel('Median complete property API time (s)')
        axis.set_title(case)
        axis.grid(alpha=.2)
        axis.legend(fontsize=7, loc='upper center', bbox_to_anchor=(.5, -.2))
    fig.suptitle('Initial SCF excluded; all adaptive preparation, transfers and fallback included')
    fig.subplots_adjust(bottom=.43, top=.88, wspace=.25)
    for suffix in ('png', 'pdf'):
        fig.savefig(args.output / f'adaptive_total_time.{suffix}', bbox_inches='tight')
    plt.close(fig)
    notes = ('Direct VEG uses three existing native property routes, not a new fused kernel.\n'
             'Only completed measurements are plotted; incomplete resource-capped pilots remain in their reports.\n'
             'Figure labels distinguish ROIs and actual accepted/fallback classification.\n'
             'V/E/G errors use absolute V, vector E and Frobenius G norms at retained probes.\n'
             'Unknown floating-point/anchor errors are not included in Taylor truncation budgets.\n'
             'Peak RSS includes setup SCF and both routes; device_last_bytes describes one adaptive query call.\n')
    (args.output / 'interpretation.txt').write_text(notes)
    provenance = {'script_sha256': sha(Path(__file__)), 'input_reports_sha256': fingerprints,
                  'native_jobs_started': 0, 'rows': len(rows), 'figure_rows': len(selected)}
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    main()
