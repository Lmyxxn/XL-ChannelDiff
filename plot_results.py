"""Plot algorithm comparisons from CSV or NPY result files."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_series(path, metric='nmse', mask_ratios=None, protocol=None, dataset=None):
    path = Path(path)
    if path.suffix.lower() == '.csv':
        with path.open(newline='', encoding='utf-8-sig') as handle:
            rows = list(csv.DictReader(handle))
    elif path.suffix.lower() == '.npy':
        values = np.load(path, allow_pickle=False)
        if values.dtype.names:
            rows = [{key: row[key] for key in values.dtype.names} for row in values]
        else:
            if mask_ratios is None:
                raise ValueError('Numeric NPY arrays require --mask_ratios; values are linear metrics.')
            if values.ndim not in (1, 2) or len(values) != len(mask_ratios):
                raise ValueError('Numeric NPY shape must be [ratios] or [ratios, samples].')
            means = values if values.ndim == 1 else values.mean(axis=1)
            rows = [dict(mask_ratio=x, **{f'avg_{metric}': y}) for x, y in zip(mask_ratios, means)]
    else:
        raise ValueError('Result files must be CSV or NPY.')
    if protocol is not None:
        rows = [row for row in rows if row.get('protocol', row.get('mask_protocol')) == protocol]
    if dataset is not None:
        rows = [row for row in rows if row.get('dataset') == dataset]
    if not rows:
        raise ValueError(f'No matching results in {path}.')
    points = []
    keys = {'nmse': ('avg_nmse', 'nmse'), 'l1': ('avg_l1', 'l1'), 'cosine': ('avg_cosine', 'cosine')}[metric]
    for row in rows:
        x = float(row['mask_ratio']) if 'mask_ratio' in row else 1-float(row['gamma'])
        key = next((key for key in keys if key in row), None)
        if key:
            y = float(row[key])
            if metric == 'nmse':
                if y < 0:
                    raise ValueError('Linear NMSE cannot be negative.')
                y = 10*np.log10(max(y, 1e-12))
        elif metric == 'nmse':
            key = next((key for key in ('nmse_db', 'avg_nmse_db') if key in row), None)
            if key is None:
                raise ValueError(f'No NMSE column in {path}.')
            y = float(row[key])
        else:
            raise ValueError(f'No {metric} column in {path}.')
        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError(f'Non-finite result in {path}.')
        points.append((x, y))
    points.sort()
    if len({x for x, _ in points}) != len(points):
        raise ValueError('Duplicate mask ratios: select one dataset/protocol or use separate result files.')
    return np.asarray(points).T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--series', action='append', required=True, metavar='LABEL=FILE')
    parser.add_argument('--metric', choices=['nmse', 'l1', 'cosine'], default='nmse')
    parser.add_argument('--mask_ratios', type=float, nargs='+')
    parser.add_argument('--protocol')
    parser.add_argument('--dataset')
    parser.add_argument('--x', choices=['mask_ratio', 'gamma'], default='mask_ratio')
    parser.add_argument('--output', required=True, help='Output filename stem.')
    parser.add_argument('--formats', nargs='+', choices=['png', 'pdf', 'eps'], default=['png', 'pdf', 'eps'])
    args = parser.parse_args()
    plt.rcParams.update({'font.family': 'serif', 'font.size': 14})
    figure, ax = plt.subplots(figsize=(8, 5.6))
    markers = ('o', '^', 's', 'D', 'v', 'h', 'x', '+')
    styles = {'OMP': ('magenta', 'h', '--'), 'WCGAN': ('darkorange', 'v', '-'),
              'CDDPM': ('blue', 'o', '-'), 'Proposed': ('red', '^', '-')}
    for index, specification in enumerate(args.series):
        label, separator, filename = specification.partition('=')
        if not separator or not label or not filename:
            parser.error('Each --series must have the form LABEL=FILE.')
        x, y = load_series(filename, args.metric, args.mask_ratios, args.protocol, args.dataset)
        if args.x == 'gamma':
            x = 1-x
        order = np.argsort(x)
        color, marker, linestyle = styles.get(label, (None, markers[index % len(markers)], '-'))
        ax.plot(x[order], y[order], label=label, color=color, marker=marker, linestyle=linestyle,
                linewidth=2, markersize=7)
    ax.set_xlabel(r'Mask ratio $\gamma$' if args.x == 'mask_ratio' else 'Observed fraction')
    ax.set_ylabel({'nmse': 'NMSE (dB)', 'l1': 'Relative L1 distance', 'cosine': 'Cosine distance'}[args.metric])
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(frameon=False)
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in args.formats:
        figure.savefig(str(output) + '.' + extension, dpi=300, bbox_inches='tight')
    plt.close(figure)


if __name__ == '__main__':
    main()
