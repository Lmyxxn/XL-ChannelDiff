"""Evaluate 2D DFT-OMP channel reconstruction and export CSV/NPY results."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from cs_baselines import omp_dft2_completion, nmse, relative_l1, cosine_distance
from loaders import load_first_subcarrier_from_revised_mat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--test_file', help='Override the default seed4321 MAT file.')
    parser.add_argument('--output_root', required=True)
    parser.add_argument('--mask_ratios', type=float, nargs='+', default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    parser.add_argument('--num_channels', type=int, default=200)
    parser.add_argument('--sparsity', type=int, default=64)
    parser.add_argument('--tol', type=float, default=1e-6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mask_file', help='Boolean NPY masks with shape [ratios, samples, height, width].')
    args = parser.parse_args()
    if args.num_channels <= 0 or args.sparsity <= 0:
        parser.error('num_channels and sparsity must be positive.')
    if any(not 0 <= ratio < 1 for ratio in args.mask_ratios):
        parser.error('mask_ratios must be in [0, 1).')
    path = Path(args.test_file) if args.test_file else Path(args.data_root) / 'CDL-C_Nt1_Nr1024_QuaRIGA_UPA0.50_seed4321.mat'
    channels = load_first_subcarrier_from_revised_mat(path)[:args.num_channels]
    if len(channels) != args.num_channels:
        raise ValueError('The test file contains fewer channels than requested.')
    masks = np.load(args.mask_file, allow_pickle=False) if args.mask_file else None
    expected_shape = (len(args.mask_ratios),) + channels.shape
    if masks is not None and masks.shape != expected_shape:
        raise ValueError(f'Expected mask shape {expected_shape}, received {masks.shape}.')
    np.random.seed(args.seed)
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    rows, all_nmse, all_masks = [], [], []
    for index, ratio in enumerate(args.mask_ratios):
        current_masks = []
        estimates = []
        start = time.perf_counter()
        for sample_index, channel in enumerate(channels):
            np.random.choice([ratio])
            mask = masks[index, sample_index].astype(bool) if masks is not None else np.random.rand(*channel.shape) > ratio
            current_masks.append(mask)
            estimates.append(omp_dft2_completion(channel * mask, mask, args.sparsity, args.tol))
        estimate = np.stack(estimates)
        errors = nmse(estimate, channels)
        average = float(errors.mean())
        row = dict(mask_ratio=ratio, gamma=1-ratio, num_channels=len(channels), sparsity=args.sparsity,
                   avg_nmse=average, nmse_db=float(10*np.log10(max(average, 1e-12))),
                   avg_l1=float(relative_l1(estimate, channels).mean()),
                   avg_cosine=float(cosine_distance(estimate, channels).mean()),
                   elapsed_seconds=time.perf_counter()-start)
        rows.append(row)
        all_nmse.append(errors)
        all_masks.append(np.stack(current_masks))
        print(f"mask_ratio={ratio:.1f}, NMSE={row['nmse_db']:.4f} dB", flush=True)
    with (output / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    structured = np.array([tuple(row.values()) for row in rows], dtype=[(key, 'f8') for key in rows[0]])
    np.save(output / 'summary.npy', structured)
    np.save(output / 'nmse_per_sample.npy', np.stack(all_nmse))
    np.save(output / 'masks.npy', np.stack(all_masks))
    with (output / 'settings.json').open('w') as handle:
        json.dump(dict(vars(args), resolved_test_file=str(path.resolve())), handle, indent=2)


if __name__ == '__main__':
    main()
