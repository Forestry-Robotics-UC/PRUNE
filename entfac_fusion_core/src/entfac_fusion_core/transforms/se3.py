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
#   SE(3) transform utilities (apply, invert) for numpy Sensor Fusion pipelines.

"""SE(3) helpers for Sensor Fusion (numpy-only)."""

import numpy as np

from entfac_fusion_core.utils.validation import require_homogeneous_transform


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to a set of points (N, 3)."""
    transform = require_homogeneous_transform(transform)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got shape {points.shape}")
    rot = transform[:3, :3]
    trans = transform[:3, 3]
    return (points @ rot.T) + trans


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a 4x4 homogeneous transform."""
    transform = require_homogeneous_transform(transform)
    rot = transform[:3, :3]
    trans = transform[:3, 3]
    inv = np.eye(4, dtype=float)
    inv[:3, :3] = rot.T
    inv[:3, 3] = -rot.T @ trans
    return inv
