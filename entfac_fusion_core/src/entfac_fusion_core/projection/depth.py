#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: MIT License (open source, free to modify and redistribute)
# Repository: ENTFAC-Sensor-Fusion
#
# Description:
#   Depth back-projection utilities for creating point clouds from depth images.

"""Depth image back-projection utilities."""

import logging
from functools import lru_cache

import numpy as np

from entfac_fusion_core.utils.validation import ensure_float_matrix

LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _meshgrid(shape):
    h, w = shape
    return np.meshgrid(np.arange(w), np.arange(h))


def depth_to_points(depth: np.ndarray, intrinsics: np.ndarray):
    """Convert a depth image into 3D points in the camera frame.

    Returns (points, mask) where mask selects valid (finite, >0) depth entries.
    Meshgrid is cached per shape to reduce allocations.
    """
    if depth.ndim != 2:
        raise ValueError(f"depth must be 2D (H, W), got shape {depth.shape}")
    intrinsics = ensure_float_matrix(np.asarray(intrinsics), (3, 3))

    u_coord, v_coord = _meshgrid(depth.shape)
    depth_flat = depth.reshape(-1)
    valid = np.isfinite(depth_flat) & (depth_flat > 0)

    if not np.any(valid):
        LOGGER.warning("No valid depth values found; returning empty point cloud")
        return np.empty((0, 3), dtype=float), valid

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
