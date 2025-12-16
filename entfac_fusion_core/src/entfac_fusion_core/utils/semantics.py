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
from typing import Tuple

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


def count_semantic_groups(
    semantic: np.ndarray,
    *,
    color_quantize_step: int = 1,
    color_sample_stride: int = 1,
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
        count = count_unique_colors(
            semantic[:, :, :3],
            quantize_step=color_quantize_step,
            sample_stride=color_sample_stride,
        )
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
