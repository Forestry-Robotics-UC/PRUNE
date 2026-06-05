#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: GNU General Public License v3.0 (GPL-3.0)
# Repository: PRUNE
#
# Description:
#   Shared projected-patch sampling helpers for semantic fusion.

"""Shared projected-patch sampling helpers for semantic fusion."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from prune_core.utils.semantics import packed_rgb_to_triplets


def _normalize_patch_size(patch_size: int) -> int:
    patch_size = int(patch_size)
    if patch_size < 1:
        raise ValueError("patch_size must be >= 1")
    if patch_size % 2 == 0:
        raise ValueError("patch_size must be odd")
    return patch_size


def _gather_patch_samples(
    image: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    patch_size: int,
    confidence: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    image = np.asarray(image)
    u = np.asarray(u, dtype=np.int32).reshape(-1)
    v = np.asarray(v, dtype=np.int32).reshape(-1)
    if u.shape != v.shape:
        raise ValueError("u and v must have the same shape")
    if image.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {image.shape}")
    if confidence is not None:
        confidence = np.asarray(confidence, dtype=np.float32)
        if confidence.shape != image.shape:
            raise ValueError("confidence must match image shape")

    patch_size = _normalize_patch_size(patch_size)
    radius = patch_size // 2
    h, w = image.shape

    # Build all (dy, dx) offsets vectorised; ordering matches the original
    # dy-outer dx-inner loop so column indices stay identical.
    offsets = np.arange(-radius, radius + 1, dtype=np.int32)
    dy_offsets, dx_offsets = np.meshgrid(offsets, offsets, indexing="ij")
    dy_offsets = dy_offsets.ravel()
    dx_offsets = dx_offsets.ravel()

    vv = v[:, None] + dy_offsets[None, :]
    uu = u[:, None] + dx_offsets[None, :]
    valid = (vv >= 0) & (vv < h) & (uu >= 0) & (uu < w)

    # Clamp for safe gather; downstream code always masks with `valid`
    # so values at out-of-bounds positions are never used.
    vv_c = np.clip(vv, 0, h - 1)
    uu_c = np.clip(uu, 0, w - 1)

    samples = image[vv_c, uu_c]
    conf_samples = (
        confidence[vv_c, uu_c].astype(np.float32, copy=False)
        if confidence is not None
        else None
    )

    return samples, valid, conf_samples


def sample_projected_label_patches(
    labels_img: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    confidence: Optional[np.ndarray] = None,
    patch_size: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample label patches around projected pixels using a robust majority vote."""
    labels_img = np.asarray(labels_img)
    if labels_img.ndim != 2:
        raise ValueError(f"labels_img must be 2D, got shape {labels_img.shape}")
    if labels_img.dtype.kind not in ("i", "u"):
        raise ValueError(f"labels_img must be integer, got dtype {labels_img.dtype}")

    u = np.asarray(u, dtype=np.int32).reshape(-1)
    v = np.asarray(v, dtype=np.int32).reshape(-1)
    if u.shape != v.shape:
        raise ValueError("u and v must have the same shape")

    patch_size = _normalize_patch_size(patch_size)
    if patch_size == 1:
        labels = labels_img[v, u].astype(np.int64, copy=False)
        if confidence is None:
            patch_conf = np.ones(labels.shape[0], dtype=np.float32)
        else:
            patch_conf = np.clip(
                np.asarray(confidence[v, u], dtype=np.float32),
                0.0,
                1.0,
            )
        return labels, patch_conf

    samples, valid, conf_samples = _gather_patch_samples(
        labels_img,
        u,
        v,
        patch_size=patch_size,
        confidence=confidence,
    )

    n = int(u.size)
    no_valid = ~valid.any(axis=1)

    row_ids, col_ids = np.nonzero(valid)
    flat_vals = samples[row_ids, col_ids].astype(np.int64)

    if flat_vals.size == 0:
        return np.full(n, -1, dtype=np.int64), np.zeros(n, dtype=np.float32)

    if conf_samples is not None:
        flat_w = np.clip(conf_samples[row_ids, col_ids].astype(np.float64), 0.0, None)
        row_wsum = np.zeros(n, dtype=np.float64)
        np.add.at(row_wsum, row_ids, flat_w)
        fallback = row_wsum[row_ids] == 0.0
        if fallback.any():
            flat_w = flat_w.copy()
            flat_w[fallback] = 1.0
    else:
        flat_w = np.ones(len(row_ids), dtype=np.float64)

    label_min = int(flat_vals.min())
    vals_shifted = flat_vals - label_min
    n_bins = int(vals_shifted.max()) + 1

    linear_idx = row_ids * n_bins + vals_shifted
    flat_scores = np.bincount(linear_idx, weights=flat_w, minlength=n * n_bins)
    scores = flat_scores.reshape(n, n_bins)

    best = np.argmax(scores, axis=1)
    labels_out = (best + label_min).astype(np.int64)
    labels_out[no_valid] = -1

    total_w = scores.sum(axis=1)
    conf_out = np.where(total_w > 0.0, scores[np.arange(n), best] / total_w, 0.0).astype(
        np.float32
    )
    conf_out[no_valid] = 0.0

    return labels_out, conf_out


def sample_projected_rgb_patches(
    packed_img: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    confidence: Optional[np.ndarray] = None,
    patch_size: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample RGB patches around projected pixels using a robust channel-wise median."""
    packed_img = np.asarray(packed_img, dtype=np.uint32)
    if packed_img.ndim != 2:
        raise ValueError(f"packed_img must be 2D, got shape {packed_img.shape}")

    u = np.asarray(u, dtype=np.int32).reshape(-1)
    v = np.asarray(v, dtype=np.int32).reshape(-1)
    if u.shape != v.shape:
        raise ValueError("u and v must have the same shape")

    patch_size = _normalize_patch_size(patch_size)
    if patch_size == 1:
        colors = packed_img[v, u].astype("<u4", copy=False)
        rgb_values = colors.view("<f4")
        if confidence is None:
            patch_conf = np.ones(colors.shape[0], dtype=np.float32)
        else:
            patch_conf = np.clip(
                np.asarray(confidence[v, u], dtype=np.float32),
                0.0,
                1.0,
            )
        return rgb_values, patch_conf

    samples, valid, conf_samples = _gather_patch_samples(
        packed_img,
        u,
        v,
        patch_size=patch_size,
        confidence=confidence,
    )
    rgb_triplets = packed_rgb_to_triplets(samples).astype(np.float32, copy=False)

    # Fast path: use partition-based median for fully-valid patches (no NaN overhead).
    # Points within patch_radius pixels of the image border have partial patches (~3%
    # of projected points at typical resolutions); those use nanmedian as fallback.
    n = rgb_triplets.shape[0]
    k = (patch_size * patch_size) // 2
    median_rgb = np.empty((n, 3), dtype=np.float32)
    all_valid = valid.all(axis=1)
    fv_idx = np.nonzero(all_valid)[0]
    if fv_idx.size:
        part = np.partition(rgb_triplets[fv_idx], k, axis=1)
        median_rgb[fv_idx] = part[:, k, :]
    pv_idx = np.nonzero(~all_valid)[0]
    if pv_idx.size:
        masked = np.where(valid[pv_idx, :, None], rgb_triplets[pv_idx], np.nan)
        median_rgb[pv_idx] = np.nan_to_num(np.nanmedian(masked, axis=1), nan=0.0)

    median_rgb = np.nan_to_num(median_rgb, nan=0.0)
    rgb_u8 = np.clip(np.rint(median_rgb), 0.0, 255.0).astype(np.uint8, copy=False)
    packed = (
        (rgb_u8[:, 0].astype(np.uint32) << 16)
        | (rgb_u8[:, 1].astype(np.uint32) << 8)
        | rgb_u8[:, 2].astype(np.uint32)
    )

    diff = np.abs(rgb_triplets - median_rgb[:, None, :]).sum(axis=2)
    diff = np.where(valid, diff / (255.0 * 3.0), 0.0)
    counts = np.maximum(np.count_nonzero(valid, axis=1), 1)
    patch_conf = 1.0 - np.clip(diff.sum(axis=1) / counts, 0.0, 1.0)
    patch_conf = patch_conf.astype(np.float32, copy=False)

    if conf_samples is not None:
        counts_f = counts.astype(np.float32, copy=False)
        conf_mean = conf_samples.sum(axis=1) / counts_f
        patch_conf = patch_conf * np.clip(conf_mean, 0.0, 1.0)

    return packed.astype("<u4", copy=False).view("<f4"), patch_conf
