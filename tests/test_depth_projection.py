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
#   Unit tests for depth back-projection helpers.

import sys
from pathlib import Path

import numpy as np

CORE_SRC = Path(__file__).resolve().parents[1] / "entfac_fusion_core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from entfac_fusion_core.projection.depth import depth_to_points  # noqa: E402


def test_depth_to_points_returns_mask_in_image_shape_when_empty():
    depth = np.zeros((2, 3), dtype=np.float32)
    intrinsics = np.eye(3, dtype=float)
    pts, mask = depth_to_points(depth, intrinsics)
    assert pts.shape == (0, 3)
    assert mask.shape == depth.shape

