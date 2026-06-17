#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: GNU General Public License v3.0 (GPL-3.0)
# Repository: PRUNE
#
# Description:
#   Unit tests for semantic utility helpers (labels and palette inspection).

import sys
from pathlib import Path

import numpy as np

CORE_SRC = Path(__file__).resolve().parents[1] / "prune_core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from prune_core.utils.semantics import (  # noqa: E402
    count_semantic_groups,
    count_unique_colors,
    count_unique_labels,
    unique_color_triplets,
    unique_label_ids,
)


def test_label_stats():
    labels = np.array([[0, 1, 1], [-1, 2, 0]], dtype=np.int32)
    assert count_unique_labels(labels) == 3
    assert unique_label_ids(labels).tolist() == [0, 1, 2]
    assert count_semantic_groups(labels) == ("labels", 3)


def test_palette_stats():
    # Colors are in RGB order.
    rgb = np.array(
        [
            [[0, 0, 0], [0, 0, 128]],
            [[0, 0, 0], [0, 51, 0]],
        ],
        dtype=np.uint8,
    )
    assert count_unique_colors(rgb) == 3
    assert count_semantic_groups(rgb) == ("colors", 3)

    packed = unique_color_triplets(rgb).tolist()
    # packed: (r<<16|g<<8|b)
    assert (0 << 16) | (0 << 8) | 0 in packed
    assert (0 << 16) | (0 << 8) | 128 in packed
    assert (0 << 16) | (51 << 8) | 0 in packed


def test_palette_merge_distance_reduces_noise():
    # Two very similar blues should be merged when merge_distance is enabled.
    rgb = np.array([[[0, 0, 128], [0, 0, 129]]], dtype=np.uint8)
    assert count_semantic_groups(rgb) == ("colors", 2)
    assert count_semantic_groups(rgb, color_merge_distance=1) == ("colors", 1)


def test_palette_count_denoises_jpeg_like_noise():
    # Palette semantic images transported via lossy codecs (e.g., JPEG) can gain
    # many near-colors around each class color. count_semantic_groups should
    # return the semantic palette size, not the raw unique-color count.
    rng = np.random.default_rng(0)

    # Example palette (RGB) with 6 semantic classes.
    base_colors = np.array(
        [
            [0, 0, 0],
            [0, 0, 128],
            [0, 50, 100],
            [0, 213, 255],
            [163, 0, 128],
            [0, 51, 0],
        ],
        dtype=np.uint8,
    )
    h, w = 200, 200
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    stripe = w // int(base_colors.shape[0])
    for idx, color in enumerate(base_colors):
        start = idx * stripe
        end = w if idx == int(base_colors.shape[0]) - 1 else (idx + 1) * stripe
        rgb[:, start:end, :] = color

    # Inject near-color noise into ~30% of pixels.
    num = int(h * w * 0.3)
    flat = rgb.reshape(-1, 3).astype(np.int16, copy=False)
    pix_idx = rng.choice(flat.shape[0], size=num, replace=False)
    deltas = rng.integers(-8, 9, size=(num, 3), dtype=np.int16)
    flat[pix_idx] = np.clip(flat[pix_idx] + deltas, 0, 255)
    rgb_noisy = flat.astype(np.uint8, copy=False).reshape(h, w, 3)

    kind, count = count_semantic_groups(rgb_noisy)
    assert kind == "colors"
    assert count == int(base_colors.shape[0])
