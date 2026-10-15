#!/usr/bin/env python3
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
"""Export matched direct-call timing CSV and standalone figures from a report."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if not report.get('passed'):
        raise ValueError('Numerical/native-path acceptance must pass before plotting timing comparisons')
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for run in report['runs']:
        for route, phase in run['phases'].items():
            rows.append({'case': run['case'], 'backend': run['backend'], 'basis': run['basis_name'],
                         'basis_functions': int(run['result']['basis_functions']), 'points': run['points'],
                         'outputs': route, 'median_seconds': phase['median_seconds'],
                         'min_seconds': phase['min_seconds'], 'max_seconds': phase['max_seconds'],
                         'first_route_call_seconds': phase['first_route_call_seconds'],
                         'repeat_count': len(phase['samples_seconds']),
                         'density_sha256': run['files_sha256']['density.dat'],
                         'timing_scope': 'native_API_allocations_uploads_kernels_downloads_classical;initial_SCF_excluded'})
    with (args.output/'direct_timings.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for case in sorted({row['case'] for row in rows}):
        with plt.rc_context({'font.size': 10, 'axes.spines.top': False,
                             'axes.spines.right': False, 'pdf.fonttype': 42}):
            fig, ax = plt.subplots(figsize=(4.8, 3.6), layout='constrained')
            for backend, color in [('cpu', '#0072B2'), ('cuda', '#D55E00')]:
                for route, style in [('VEG', '-'), ('VEGH', '--')]:
                    selected = sorted([r for r in rows if r['case'] == case and r['backend'] == backend
                                       and r['outputs'] == route], key=lambda row: row['points'])
                    if not selected:
                        continue
                    x = [r['points'] for r in selected]
                    y = [r['median_seconds'] for r in selected]
                    errors = [[r['median_seconds']-r['min_seconds'] for r in selected],
                              [r['max_seconds']-r['median_seconds'] for r in selected]]
                    label = ('CPU' if backend == 'cpu' else 'CUDA') + ', ' + '/'.join(route)
                    ax.errorbar(x, y, yerr=errors, fmt='o'+style, color=color, label=label,
                                markersize=4, capsize=3)
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlabel('Observation points'); ax.set_ylabel('Direct property API time (s)')
            ax.set_title(case.replace('-', ' ')); ax.legend(frameon=False, fontsize=8)
            for suffix in ('pdf', 'png'):
                fig.savefig(args.output/(case+'.'+suffix), dpi=400)
            plt.close(fig)
    (args.output/'interpretation.txt').write_text(
        'All curves use the same retained density within each molecular case. V/E/G dispatches separate native routes.\n'
        'Points show medians; error bars span observed minimum/maximum repeated-call times, not confidence intervals.\n'
        'Initial SCF, process startup and file output are excluded from the property clock. No whole-BOMD speedup follows.\n'
        'Interpret hardware, runtime, precision, source and workload limitations in the adjacent source report.\n')
    print(args.output/'direct_timings.csv')


if __name__ == '__main__':
    main()
