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
#   Unit tests for semantic utility helpers (labels and palette inspection).

import sys
from pathlib import Path

import numpy as np

CORE_SRC = Path(__file__).resolve().parents[1] / "entfac_fusion_core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from entfac_fusion_core.utils.semantics import (  # noqa: E402
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
