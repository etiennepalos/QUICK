#!/usr/bin/env python3
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
"""Export paired whole-BOMD timing variation and separate property/force ratios."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows, reports = [], []
    for path in args.reports:
        report = json.loads(path.read_text())
        if not report.get('passed') or report.get('prepared_only'):
            raise ValueError('Executed numerical/native-path acceptance must pass before plotting')
        config = report['configuration']
        if reports:
            previous = reports[0]['configuration']
            for key in ('keyword', 'natom', 'probe_count', 'nsteps', 'dt_fs', 'temperature_k',
                        'stride', 'threads', 'ranks', 'enabled_properties', 'probe_mode'):
                if config[key] != previous[key]:
                    raise ValueError('Selected reports use different protocols: '+key)
        reports.append(report)
        pairs = [pair for pair in report['pairs'] if not pair['discarded_warmup']]
        ratios = report['api_phase_ratio']['property_api_to_force_api_percent']['values']
        if len(pairs) != len(ratios):
            raise ValueError('Retained paired trials and API timing ratios differ')
        for pair, ratio in zip(pairs, ratios):
            rows.append({'backend': config['backend'], 'pair': pair['pair'],
                         'off_wall_seconds': pair['off_wall_seconds'],
                         'on_wall_seconds': pair['on_wall_seconds'],
                         'delta_wall_seconds': pair['delta_wall_seconds'],
                         'whole_job_overhead_percent': pair['whole_job_overhead_percent'],
                         'property_API_to_force_API_percent': ratio,
                         'probes': config['probe_count'], 'forces': config['nsteps']+1,
                         'source_report_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output/'workflow_timings.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    with plt.rc_context({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42}):
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), layout='constrained')
        backends = list(dict.fromkeys(row['backend'] for row in rows))
        for index, backend in enumerate(backends):
            selected = [row for row in rows if row['backend'] == backend]
            jitter = np.linspace(-.10, .10, len(selected))
            axes[0].scatter(index+jitter, [1000*row['delta_wall_seconds'] for row in selected],
                            s=25, color=('#0072B2' if backend == 'cpu' else '#D55E00'))
            values = np.array([row['property_API_to_force_API_percent'] for row in selected])
            median = np.median(values)
            axes[1].errorbar(index, median, yerr=[[median-values.min()], [values.max()-median]],
                             fmt='o', color=('#0072B2' if backend == 'cpu' else '#D55E00'), capsize=4)
        axes[0].axhline(0, color='.6', linewidth=.8)
        axes[0].set_ylabel('Paired whole-job difference (ms)')
        axes[1].set_ylabel('Property API / force API (%)')
        for axis in axes:
            axis.set_xticks(range(len(backends)), [backend.upper() for backend in backends])
        fig.suptitle(f"{rows[0]['probes']} probes, {rows[0]['forces']} forces; V/E/G/H")
        for suffix in ('pdf', 'png'):
            fig.savefig(args.output/('workflow_overhead.'+suffix), dpi=400)
        plt.close(fig)
    (args.output/'interpretation.txt').write_text(
        'Left: retained paired fresh-process differences, including setup and output; negative deltas reflect timing variation.\n'
        'Right: median synchronous property/force API ratios, with observed min/max ranges; excludes driver output.\n'
        'The phase ratio is not whole-job overhead, and observed ranges are not confidence intervals.\n'
        'These short fixed-probe runs establish lifecycle/overhead behavior, not trajectory statistics or dense-grid overhead.\n')
    print(args.output/'workflow_timings.csv')


if __name__ == '__main__':
    main()
