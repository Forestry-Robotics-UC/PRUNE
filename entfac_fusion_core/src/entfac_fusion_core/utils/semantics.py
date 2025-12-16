#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Derived from Semantic SLAM
#
# Original Author:
#   Xuan Zhang
#
# Subsequent Contributions:
#   David Russell
#
# Modified by:
#   Duda Andrada (ENTFAC Sensor Fusion)
#
# Original project:
#   https://github.com/floatlazer/semantic_slam
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: GNU General Public License v3.0 (GPL-3.0)
# Repository: ENTFAC-Sensor-Fusion
#
# Description:
#   Lightweight utilities for inspecting semantic label or palette arrays.

"""Utilities for inspecting semantic labels and color palettes."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


def unique_label_ids(
    labels: np.ndarray, *, ignore_negative: bool = True
) -> np.ndarray:
    """Return sorted unique label IDs.

    Args:
        labels: Label image (H, W) or label vector (N,).
        ignore_negative: If true, ignore labels < 0 (e.g. unlabeled = -1).

    Returns:
        Sorted array of unique label IDs (dtype inferred from input).
    """
    labels = np.asarray(labels)
    if labels.ndim not in (1, 2):
        raise ValueError(f"labels must be 1D or 2D, got shape {labels.shape}")

    flat = labels.reshape(-1)
    if ignore_negative:
        flat = flat[flat >= 0]
    unique = np.unique(flat)
    LOGGER.debug(
        "unique_label_ids: shape=%s ignore_negative=%s unique=%d",
        labels.shape,
        ignore_negative,
        int(unique.size),
    )
    return unique


def count_unique_labels(labels: np.ndarray, *, ignore_negative: bool = True) -> int:
    """Count unique label IDs in a label array."""
    return int(unique_label_ids(labels, ignore_negative=ignore_negative).size)


def quantize_rgb(
    semantic_rgb: np.ndarray, *, step: int, sample_stride: int = 1
) -> np.ndarray:
    """Quantize an RGB image to reduce palette noise (e.g., JPEG artifacts).

    Palette semantic images transported via JPEG/PNG can introduce small per-pixel
    color deviations and inflate the number of unique colors.
    """
    rgb = np.asarray(semantic_rgb)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(
            f"semantic_rgb must be (H, W, 3/4), got shape {rgb.shape}"
        )
    step = int(step)
    if step < 1:
        raise ValueError("step must be >= 1")
    sample_stride = int(sample_stride)
    if sample_stride < 1:
        raise ValueError("sample_stride must be >= 1")

    rgb = rgb[::sample_stride, ::sample_stride, :3]
    if step == 1:
        return rgb.astype(np.uint8, copy=False)

    vals = rgb.astype(np.int16, copy=False)
    half = step // 2
    quant = ((vals + half) // step) * step
    return np.clip(quant, 0, 255).astype(np.uint8, copy=False)


def _pack_rgb(
    rgb: np.ndarray, *, quantize_step: int = 1, sample_stride: int = 1
) -> np.ndarray:
    rgb_q = quantize_rgb(rgb, step=quantize_step, sample_stride=sample_stride)
    r = rgb_q[:, :, 0].astype(np.uint32)
    g = rgb_q[:, :, 1].astype(np.uint32)
    b = rgb_q[:, :, 2].astype(np.uint32)
    return (r << 16) | (g << 8) | b


def packed_rgb_to_triplets(packed_rgb: np.ndarray) -> np.ndarray:
    """Convert packed RGB uint32 values into RGB triplets.

    Args:
        packed_rgb: Packed RGB values as uint32 (r<<16 | g<<8 | b). Any shape.

    Returns:
        Array with shape (..., 3) and dtype uint8 holding [r, g, b].
    """
    packed = np.asarray(packed_rgb, dtype=np.uint32)
    r = (packed >> 16) & 0xFF
    g = (packed >> 8) & 0xFF
    b = packed & 0xFF
    return np.stack((r, g, b), axis=-1).astype(np.uint8, copy=False)


def _cluster_packed_colors(
    packed_colors: np.ndarray,
    counts: np.ndarray,
    *,
    merge_distance: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster packed RGB colors by Euclidean distance in RGB space.

    This is a lightweight heuristic to merge JPEG palette noise for semantic
    palette images.
    """
    packed_colors = np.asarray(packed_colors, dtype=np.uint32).reshape(-1)
    counts = np.asarray(counts, dtype=np.int64).reshape(-1)
    if packed_colors.shape != counts.shape:
        raise ValueError("packed_colors and counts must have the same shape")
    merge_distance = int(merge_distance)
    if merge_distance <= 0 or packed_colors.size == 0:
        order = np.argsort(packed_colors)
        return packed_colors[order], counts[order]

    dist2_max = int(merge_distance) * int(merge_distance)

    # Process most frequent colors first for stable cluster centers.
    order = np.argsort(counts)[::-1]
    colors_sorted = packed_colors[order]
    counts_sorted = counts[order]
    colors_rgb = packed_rgb_to_triplets(colors_sorted).astype(np.int16, copy=False)

    centers_rgb = []
    centers_packed = []
    centers_counts = []

    for rgb, packed, cnt in zip(colors_rgb, colors_sorted, counts_sorted):
        if not centers_rgb:
            centers_rgb.append(rgb)
            centers_packed.append(int(packed))
            centers_counts.append(int(cnt))
            continue
        centers_arr = np.asarray(centers_rgb, dtype=np.int16)
        diff = centers_arr - rgb.reshape(1, 3)
        dist2 = np.sum(diff * diff, axis=1)
        nearest = int(np.argmin(dist2))
        if int(dist2[nearest]) <= dist2_max:
            centers_counts[nearest] += int(cnt)
        else:
            centers_rgb.append(rgb)
            centers_packed.append(int(packed))
            centers_counts.append(int(cnt))

    centers_packed_arr = np.asarray(centers_packed, dtype=np.uint32)
    centers_counts_arr = np.asarray(centers_counts, dtype=np.int64)
    packed_order = np.argsort(centers_packed_arr)
    return centers_packed_arr[packed_order], centers_counts_arr[packed_order]


def dominant_packed_colors(
    packed_rgb: np.ndarray,
    *,
    min_count: int = 1,
    min_fraction: float = 0.0,
    max_colors: Optional[int] = None,
    merge_distance: int = 0,
) -> np.ndarray:
    """Estimate semantic palette colors from packed RGB values.

    Args:
        packed_rgb: Packed RGB image (H, W) or flat vector (N,).
        min_count: Minimum count for a color to be kept (>=1).
        min_fraction: Minimum fraction (0..1) of pixels for a color to be kept.
        max_colors: Optional cap; if exceeded, keep the most frequent colors.
        merge_distance: Optional RGB Euclidean distance threshold used to merge
            similar colors (helps with JPEG artifacts). 0 disables.

    Returns:
        Sorted packed RGB colors as uint32 (r<<16 | g<<8 | b).
    """
    packed_flat = np.asarray(packed_rgb, dtype=np.uint32).reshape(-1)
    total = int(packed_flat.size)
    unique, counts = np.unique(packed_flat, return_counts=True)

    min_count = int(min_count)
    if min_count < 1:
        raise ValueError("min_count must be >= 1")
    min_fraction = float(min_fraction)
    if not (0.0 <= min_fraction <= 1.0):
        raise ValueError("min_fraction must be in [0, 1]")

    min_count_eff = max(min_count, int(np.ceil(min_fraction * total)))
    mask = counts >= min_count_eff
    filtered = unique[mask]
    filtered_counts = counts[mask]

    raw_unique = int(unique.size)
    if filtered.size == 0:
        LOGGER.debug(
            "dominant_packed_colors: filter removed all colors (raw_unique=%d min_count_eff=%d); falling back to raw unique colors.",
            raw_unique,
            int(min_count_eff),
        )
        filtered = unique
        filtered_counts = counts

    merged, merged_counts = _cluster_packed_colors(
        filtered, filtered_counts, merge_distance=int(merge_distance)
    )

    if max_colors is not None:
        max_colors = int(max_colors)
        if max_colors < 1:
            raise ValueError("max_colors must be >= 1")
        if merged.size > max_colors:
            idx = np.argsort(merged_counts)[::-1][:max_colors]
            merged = merged[idx]
            merged_counts = merged_counts[idx]

    merged = np.sort(merged.astype(np.uint32))
    LOGGER.debug(
        "dominant_packed_colors: raw_unique=%d filtered=%d merged=%d "
        "(min_count=%d min_fraction=%.6f min_count_eff=%d merge_distance=%d max_colors=%s total=%d)",
        raw_unique,
        int(filtered.size),
        int(merged.size),
        int(min_count),
        float(min_fraction),
        int(min_count_eff),
        int(merge_distance),
        str(max_colors),
        total,
    )
    return merged


def unique_color_triplets(
    semantic_rgb: np.ndarray,
    *,
    quantize_step: int = 1,
    sample_stride: int = 1,
) -> np.ndarray:
    """Return unique RGB colors from a semantic palette image.

    Args:
        semantic_rgb: Semantic image in RGB order (H, W, 3/4). Alpha is ignored.
        quantize_step: Quantization step (>=1). step=1 counts exact colors.
        sample_stride: Optional stride for subsampling before counting.

    Returns:
        Sorted array of packed RGB colors as uint32: (r<<16 | g<<8 | b).
    """
    packed = _pack_rgb(
        semantic_rgb, quantize_step=quantize_step, sample_stride=sample_stride
    ).reshape(-1)
    unique = np.unique(packed)
    LOGGER.debug(
        "unique_color_triplets: shape=%s quantize_step=%d sample_stride=%d unique=%d",
        np.asarray(semantic_rgb).shape,
        int(quantize_step),
        int(sample_stride),
        int(unique.size),
    )
    if unique.size:
        preview = unique[: min(10, int(unique.size))].tolist()
        LOGGER.debug("unique_color_triplets: first_colors_packed=%s", preview)
    return unique


def count_unique_colors(
    semantic_rgb: np.ndarray, *, quantize_step: int = 1, sample_stride: int = 1
) -> int:
    """Count unique RGB colors in a semantic palette image."""
    return int(
        unique_color_triplets(
            semantic_rgb, quantize_step=quantize_step, sample_stride=sample_stride
        ).size
    )


def dominant_color_triplets(
    semantic_rgb: np.ndarray,
    *,
    quantize_step: int = 1,
    sample_stride: int = 1,
    min_count: int = 1,
    min_fraction: float = 0.0,
    max_colors: Optional[int] = None,
    merge_distance: int = 0,
) -> np.ndarray:
    """Estimate palette colors by filtering low-frequency RGB triplets.

    Lossy transports (e.g., JPEG) can introduce spurious colors in what should be
    a small semantic palette. This helper quantizes and then keeps only colors
    that appear frequently enough to plausibly represent a semantic class.

    Args:
        semantic_rgb: Semantic image in RGB order (H, W, 3/4). Alpha is ignored.
        quantize_step: Quantization step (>=1).
        sample_stride: Optional stride for subsampling before counting.
        min_count: Minimum pixel count for a color to be kept (>=1).
        min_fraction: Minimum fraction (0..1) of pixels for a color to be kept.
        max_colors: Optional cap; if exceeded, keep the most frequent colors.

    Returns:
        Sorted packed RGB colors as uint32 (r<<16 | g<<8 | b).
    """
    packed = _pack_rgb(
        semantic_rgb, quantize_step=quantize_step, sample_stride=sample_stride
    )
    palette = dominant_packed_colors(
        packed,
        min_count=min_count,
        min_fraction=min_fraction,
        max_colors=max_colors,
        merge_distance=merge_distance,
    )
    LOGGER.debug(
        "dominant_color_triplets: shape=%s quantize_step=%d sample_stride=%d palette=%d merge_distance=%d",
        np.asarray(semantic_rgb).shape,
        int(quantize_step),
        int(sample_stride),
        int(palette.size),
        int(merge_distance),
    )
    return palette


def count_semantic_groups(
    semantic: np.ndarray,
    *,
    color_quantize_step: int = 1,
    color_sample_stride: int = 1,
    color_min_count: int = 1,
    color_min_fraction: float = 0.0,
    color_max_colors: Optional[int] = None,
    color_merge_distance: int = 0,
) -> Tuple[str, int]:
    """Count semantic groups for either label or palette arrays.

    Args:
        semantic: Either a label image (H, W) or a palette image (H, W, 3/4).

    Returns:
        (kind, count) where kind is "labels" or "colors".
    """
    semantic = np.asarray(semantic)
    if semantic.ndim == 2:
        count = count_unique_labels(semantic)
        LOGGER.debug("count_semantic_groups: labels=%d", count)
        return "labels", count
    if semantic.ndim == 3 and semantic.shape[2] in (3, 4):
        palette = dominant_color_triplets(
            semantic[:, :, :3],
            quantize_step=color_quantize_step,
            sample_stride=color_sample_stride,
            min_count=color_min_count,
            min_fraction=color_min_fraction,
            max_colors=color_max_colors,
            merge_distance=color_merge_distance,
        )
        count = int(palette.size)
        LOGGER.debug(
            "count_semantic_groups: colors=%d (quantize_step=%d sample_stride=%d)",
            int(count),
            int(color_quantize_step),
            int(color_sample_stride),
        )
        return "colors", count
    raise ValueError(
        f"semantic must be (H,W) labels or (H,W,3/4) colors, got shape {semantic.shape}"
    )
