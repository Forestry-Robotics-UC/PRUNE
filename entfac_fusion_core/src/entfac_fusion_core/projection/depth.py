#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Adapted from Semantic SLAM (substantially refactored for ENTFAC).
#
# Original Author:
#   Xuan Zhang
#
# Subsequent Contributions:
#   David Russell
#
# Upstream reference:
#   https://github.com/floatlazer/semantic_slam
#
# Modified by:
#   Duda Andrada (ENTFAC Sensor Fusion)
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: GNU General Public License v3.0 (GPL-3.0)
# Repository: ENTFAC-Sensor-Fusion
#
# Description:
#   Depth back-projection utilities for creating point clouds from depth images.

"""Depth image back-projection utilities."""

import logging
from functools import lru_cache

from typing import Optional

import numpy as np

from entfac_fusion_core.utils.validation import ensure_float_matrix

LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _meshgrid(shape):
    h, w = shape
    return np.meshgrid(np.arange(w), np.arange(h))


def depth_to_points(
    depth: np.ndarray, intrinsics: np.ndarray, max_depth_m: Optional[float] = None
):
    """Convert a depth image into 3D points in the camera frame.

    Args:
        depth: Depth image with shape ``(H, W)``.
        intrinsics: Camera intrinsics matrix ``K`` with shape ``(3, 3)``.
        max_depth_m: Optional max depth in meters; values beyond are dropped.

    Returns:
        Tuple ``(points_xyz, valid_mask)``:
        - ``points_xyz`` has shape ``(N, 3)`` in the camera frame.
        - ``valid_mask`` has shape ``(H, W)`` and marks pixels with finite depth
          values greater than zero.

    Notes:
        The internal pixel meshgrid is cached per image shape to reduce
        allocations in high-rate pipelines.
    """
    if depth.ndim != 2:
        raise ValueError(f"depth must be 2D (H, W), got shape {depth.shape}")
    intrinsics = ensure_float_matrix(np.asarray(intrinsics), (3, 3))

    u_coord, v_coord = _meshgrid(depth.shape)
    depth_flat = depth.reshape(-1)
    valid = np.isfinite(depth_flat) & (depth_flat > 0)
    if max_depth_m is not None and max_depth_m > 0:
        valid &= depth_flat <= float(max_depth_m)

    if not np.any(valid):
        LOGGER.warning("No valid depth values found; returning empty point cloud")
        return np.empty((0, 3), dtype=float), valid.reshape(depth.shape)

    u = u_coord.reshape(-1)[valid]
    v = v_coord.reshape(-1)[valid]
    z = depth_flat[valid]

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.column_stack((x, y, z))
    LOGGER.debug("Back-projected %d depth pixels to points", points.shape[0])
    return points, valid.reshape(depth.shape)
