"""Evaluate Random, Partial, and 2D DFT-OMP reconstruction baselines."""

import argparse
import csv
import json
import time
from pathlib import Path
from functools import lru_cache
from typing import Tuple

import numpy as np
from loaders import load_first_subcarrier_from_revised_mat


@lru_cache(maxsize=8)
def dft2_synthesis_dictionary(shape: Tuple[int, int]) -> np.ndarray:
    """Return the unitary inverse-DFT dictionary in C-order antenna layout."""
    height, width = shape
    num_entries = height * width
    eye = np.eye(num_entries, dtype=np.complex64).reshape(num_entries, height, width)
    atoms = [np.fft.ifft2(eye[idx], norm="ortho").reshape(-1)
             for idx in range(num_entries)]
    return np.stack(atoms, axis=1).astype(np.complex64)


def omp_complex(y: np.ndarray, phi: np.ndarray, sparsity: int,
                tol: float = 1e-6) -> np.ndarray:
    """Solve y ~= phi @ x with complex-valued orthogonal matching pursuit."""
    y = np.asarray(y, dtype=np.complex64).reshape(-1)
    phi = np.asarray(phi, dtype=np.complex64)
    num_atoms = phi.shape[1]
    sparsity = int(max(1, min(sparsity, num_atoms, phi.shape[0])))
    column_norms = np.linalg.norm(phi, axis=0)
    valid = column_norms > 0
    safe_norms = np.where(valid, column_norms, 1.0).astype(np.float32)
    phi_normalized = phi / safe_norms[None, :]
    residual = y.copy()
    selected = []
    selected_mask = np.zeros(num_atoms, dtype=bool)
    stop_norm = tol * max(np.linalg.norm(y), 1e-12)
    for _ in range(sparsity):
        correlations = phi_normalized.conj().T @ residual
        correlations[selected_mask] = 0
        correlations[~valid] = 0
        atom_idx = int(np.argmax(np.abs(correlations)))
        if selected_mask[atom_idx] or np.abs(correlations[atom_idx]) == 0:
            break
        selected.append(atom_idx)
        selected_mask[atom_idx] = True
        active_phi = phi_normalized[:, selected]
        active_coef, *_ = np.linalg.lstsq(active_phi, y, rcond=None)
        residual = y - active_phi @ active_coef
        if np.linalg.norm(residual) <= stop_norm:
            break
    coef = np.zeros(num_atoms, dtype=np.complex64)
    if selected:
        coef[np.asarray(selected)] = active_coef / safe_norms[np.asarray(selected)]
    return coef


def omp_dft2_completion(partial_channel: np.ndarray, mask: np.ndarray,
                        sparsity: int, tol: float = 1e-6) -> np.ndarray:
    """Recover observed antenna samples using OMP in the 2D DFT domain."""
    partial_channel = np.asarray(partial_channel, dtype=np.complex64)
    mask = np.asarray(mask, dtype=bool)
    if partial_channel.shape != mask.shape:
        raise ValueError(f"partial_channel shape {partial_channel.shape} != mask shape {mask.shape}")
    dictionary = dft2_synthesis_dictionary(tuple(partial_channel.shape))
    flat_mask = mask.reshape(-1)
    if not np.any(flat_mask):
        return np.zeros_like(partial_channel)
    y = partial_channel.reshape(-1)[flat_mask]
    phi = dictionary[flat_mask, :]
    angular_coef = omp_complex(y, phi, sparsity=sparsity, tol=tol)
    recovered = dictionary @ angular_coef
    return recovered.reshape(partial_channel.shape).astype(np.complex64)


def nmse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-sample normalized mean-squared error for complex channels."""
    numerator = np.sum(np.abs(pred - target) ** 2, axis=(-1, -2))
    denominator = np.sum(np.abs(target) ** 2, axis=(-1, -2))
    return numerator / np.maximum(denominator, 1e-12)


def relative_l1(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-sample relative L1 distance for complex channels."""
    numerator = np.sum(np.abs(pred - target), axis=(-1, -2))
    denominator = np.sum(np.abs(target), axis=(-1, -2))
    return numerator / np.maximum(denominator, 1e-12)


def cosine_distance(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-sample complex cosine distance, 1 - absolute cosine similarity."""
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    inner = np.sum(pred_flat * np.conj(target_flat), axis=1)
    pred_norm = np.sqrt(np.sum(np.abs(pred_flat) ** 2, axis=1))
    target_norm = np.sqrt(np.sum(np.abs(target_flat) ** 2, axis=1))
    similarity = np.abs(inner) / np.maximum(pred_norm * target_norm, 1e-12)
    return 1 - similarity


def reference_completions(channels, masks, training_std, rng):
    normalized = channels / training_std
    partial = normalized * masks
    # Match the historical Random baseline's noise scale in normalized space.
    noise = (rng.standard_normal(channels.shape).astype(np.float32)
             + 1j * rng.standard_normal(channels.shape).astype(np.float32)) * training_std
    random = np.where(masks, normalized, noise)
    return normalized, partial, random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--test_file', help='Override the default seed4321 MAT file.')
    parser.add_argument('--train_file', help='Training MAT file supplying the historical normalization scale.')
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
    train_path = Path(args.train_file) if args.train_file else Path(args.data_root) / 'CDL-C_Nt1_Nr1024_QuaRIGA_UPA0.50_seed1234.mat'
    train_channels = load_first_subcarrier_from_revised_mat(train_path)
    training_std = float(np.std(train_channels))
    del train_channels
    if not np.isfinite(training_std) or training_std <= 0:
        raise ValueError('Training standard deviation must be positive and finite.')
    masks = np.load(args.mask_file, allow_pickle=False) if args.mask_file else None
    expected_shape = (len(args.mask_ratios),) + channels.shape
    if masks is not None and masks.shape != expected_shape:
        raise ValueError(f'Expected mask shape {expected_shape}, received {masks.shape}.')
    np.random.seed(args.seed)
    random_rng = np.random.RandomState(args.seed)
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    rows, all_nmse, all_masks = [], [], []
    baseline_rows, baseline_errors = [], []
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
        normalized, partial, random = reference_completions(channels, all_masks[-1], training_std, random_rng)
        for method, method_errors in [('OMP', errors), ('Random', nmse(random, normalized)), ('Partial', nmse(partial, normalized))]:
            method_average = float(method_errors.mean())
            baseline_rows.append(dict(method=method, mask_ratio=ratio, gamma=1-ratio,
                                      num_channels=len(channels), sparsity=args.sparsity if method == 'OMP' else 0,
                                      avg_nmse=method_average, nmse_db=float(10*np.log10(max(method_average, 1e-12)))))
            baseline_errors.append(method_errors)
            print(f"{method}: mask_ratio={ratio:.1f}, NMSE={baseline_rows[-1]['nmse_db']:.4f} dB", flush=True)
    with (output / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    structured = np.array([tuple(row.values()) for row in rows], dtype=[(key, 'f8') for key in rows[0]])
    np.save(output / 'summary.npy', structured)
    np.save(output / 'nmse_per_sample.npy', np.stack(all_nmse))
    np.save(output / 'masks.npy', np.stack(all_masks))
    with (output / 'baseline_summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(baseline_rows[0]))
        writer.writeheader()
        writer.writerows(baseline_rows)
    np.save(output / 'baseline_nmse_per_sample.npy', np.stack(baseline_errors))
    with (output / 'settings.json').open('w') as handle:
        json.dump(dict(vars(args), resolved_test_file=str(path.resolve()),
                       resolved_train_file=str(train_path.resolve()), training_std=training_std,
                       random_definition='Gaussian missing-entry fill with training_std in normalized space'), handle, indent=2)


if __name__ == '__main__':
    main()
