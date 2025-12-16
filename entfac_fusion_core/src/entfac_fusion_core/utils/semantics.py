#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: MIT License (open source, free to modify and redistribute)
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
    return np.unique(flat)


def count_unique_labels(labels: np.ndarray, *, ignore_negative: bool = True) -> int:
    """Count unique label IDs in a label array."""
    return int(unique_label_ids(labels, ignore_negative=ignore_negative).size)


def _pack_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"rgb must be (H, W, 3/4), got shape {rgb.shape}")
    r = rgb[:, :, 0].astype(np.uint32)
    g = rgb[:, :, 1].astype(np.uint32)
    b = rgb[:, :, 2].astype(np.uint32)
    return (r << 16) | (g << 8) | b


def unique_color_triplets(semantic_rgb: np.ndarray) -> np.ndarray:
    """Return unique RGB colors from a semantic palette image.

    Args:
        semantic_rgb: Semantic image in RGB order (H, W, 3/4). Alpha is ignored.

    Returns:
        Sorted array of packed RGB colors as uint32: (r<<16 | g<<8 | b).
    """
    packed = _pack_rgb(semantic_rgb).reshape(-1)
    return np.unique(packed)


def count_unique_colors(semantic_rgb: np.ndarray) -> int:
    """Count unique RGB colors in a semantic palette image."""
    return int(unique_color_triplets(semantic_rgb).size)


def count_semantic_groups(semantic: np.ndarray) -> Tuple[str, int]:
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
        count = count_unique_colors(semantic[:, :, :3])
        LOGGER.debug("count_semantic_groups: colors=%d", count)
        return "colors", count
    raise ValueError(
        f"semantic must be (H,W) labels or (H,W,3/4) colors, got shape {semantic.shape}"
    )
