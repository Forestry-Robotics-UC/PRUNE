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
#   LiDAR-to-camera projection helpers for sampling semantics from images.

"""LiDAR-to-camera projection helpers."""

import logging
import numpy as np

from entfac_fusion_core.transforms.se3 import transform_points
from entfac_fusion_core.utils.validation import (
    ensure_float_matrix,
    require_homogeneous_transform,
)

LOGGER = logging.getLogger(__name__)


def project_points_to_image(
    lidar_points: np.ndarray,
    camera_intrinsics: np.ndarray,
    camera_T_lidar: np.ndarray,
    image_size,
):
    """Project lidar points into camera pixel coordinates.

    Args:
        lidar_points: 3D points with shape ``(N, 3)`` in the LiDAR frame.
        camera_intrinsics: Camera intrinsics matrix ``K`` with shape ``(3, 3)``.
        camera_T_lidar: Homogeneous transform with shape ``(4, 4)`` mapping LiDAR
            points into the camera frame.
        image_size: Tuple ``(width, height)`` in pixels.

    Returns:
        Tuple ``(uv, inside_mask)``:
        - ``uv`` has shape ``(N, 2)`` in pixel coordinates ``(u, v)``.
        - ``inside_mask`` has shape ``(N,)`` and is true for points that are in
          front of the camera and project inside the image bounds.

    Raises:
        ValueError: If shapes/dtypes are invalid.
    """
    camera_intrinsics = ensure_float_matrix(
        np.asarray(camera_intrinsics), (3, 3)
    )
    camera_T_lidar = require_homogeneous_transform(np.asarray(camera_T_lidar))
    if lidar_points.ndim != 2 or lidar_points.shape[1] != 3:
        raise ValueError(
            f"lidar_points must be (N, 3), got shape {lidar_points.shape}"
        )

    points_cam = transform_points(camera_T_lidar, lidar_points)
    z = points_cam[:, 2]
    in_front = z > 0

    uv = np.zeros((lidar_points.shape[0], 2), dtype=float)
    uv[in_front, 0] = (
        points_cam[in_front, 0] * camera_intrinsics[0, 0] / z[in_front]
    ) + camera_intrinsics[0, 2]
    uv[in_front, 1] = (
        points_cam[in_front, 1] * camera_intrinsics[1, 1] / z[in_front]
    ) + camera_intrinsics[1, 2]

    if not isinstance(image_size, (tuple, list)) or len(image_size) != 2:
        raise ValueError("image_size must be (width, height)")
    w, h = int(image_size[0]), int(image_size[1])
    if w <= 0 or h <= 0:
        raise ValueError("image_size must be positive")
    inside = (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < w)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < h)
        & in_front
    )

    LOGGER.debug(
        "Projected %d LiDAR points; %d in front, %d inside image",
        lidar_points.shape[0],
        np.count_nonzero(in_front),
        np.count_nonzero(inside),
    )
    return uv, inside
