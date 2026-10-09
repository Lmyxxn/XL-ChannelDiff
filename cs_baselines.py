#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Conventional compressed-sensing baselines for channel completion."""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np


@lru_cache(maxsize=8)
def dft2_synthesis_dictionary(shape: Tuple[int, int]) -> np.ndarray:
    """Return the 2D unitary inverse-DFT synthesis dictionary.

    The dictionary maps sparse angular-domain coefficients to spatial antenna
    samples using numpy's C-order flattening convention.
    """

    height, width = shape
    num_entries = height * width
    eye = np.eye(num_entries, dtype=np.complex64).reshape(
        num_entries, height, width
    )
    atoms = [
        np.fft.ifft2(eye[idx], norm="ortho").reshape(-1)
        for idx in range(num_entries)
    ]
    return np.stack(atoms, axis=1).astype(np.complex64)


def omp_complex(
    y: np.ndarray,
    phi: np.ndarray,
    sparsity: int,
    tol: float = 1e-6,
) -> np.ndarray:
    """Complex-valued orthogonal matching pursuit.

    Solves y ~= phi @ x with at most ``sparsity`` nonzero entries in x.
    """

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
    y_norm = np.linalg.norm(y)
    stop_norm = tol * max(y_norm, 1e-12)

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


def omp_dft2_completion(
    partial_channel: np.ndarray,
    mask: np.ndarray,
    sparsity: int,
    tol: float = 1e-6,
) -> np.ndarray:
    """Recover a 2D channel from observed entries via OMP in 2D DFT domain."""

    partial_channel = np.asarray(partial_channel, dtype=np.complex64)
    mask = np.asarray(mask, dtype=bool)
    if partial_channel.shape != mask.shape:
        raise ValueError(
            f"partial_channel shape {partial_channel.shape} != mask shape {mask.shape}"
        )

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
