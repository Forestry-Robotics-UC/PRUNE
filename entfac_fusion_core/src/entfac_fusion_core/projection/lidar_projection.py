#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: MIT License (open source, free to modify and redistribute)
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

    Returns (uv, mask) where uv has shape (N, 2) and mask marks points
    that fall inside the image bounds.
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

    w, h = image_size
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
