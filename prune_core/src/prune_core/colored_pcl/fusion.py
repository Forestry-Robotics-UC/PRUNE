#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: GNU General Public License v3.0 (GPL-3.0)
# Repository: PRUNE
#
# Description:
#   Single-frame semantic fusion pipelines (depth-based and LiDAR projection).

"""Single-frame semantic fusion pipelines."""

import logging
from typing import Optional, Tuple

import numpy as np

from prune_core.projection.depth import depth_to_points
from prune_core.projection.lidar_projection import project_points_to_image
from prune_core.transforms.se3 import transform_points
from prune_core.types.observations import (
    DepthObservation,
    PointObservation,
    SemanticObservation,
    SemanticPointCloud,
)
from prune_core.colored_pcl.sampling import (
    sample_projected_label_patches,
    sample_projected_rgb_patches,
)
from prune_core.utils.validation import (
    flatten_masked,
    require_homogeneous_transform,
)

LOGGER = logging.getLogger(__name__)


def fuse_depth_semantics(
    semantic: SemanticObservation,
    depth: DepthObservation,
    intrinsics: np.ndarray,
    target_T_depth: np.ndarray,
    include_unlabeled: bool = False,
    max_depth_m: Optional[float] = None,
) -> SemanticPointCloud:
    """Fuse aligned semantic + depth into a semantic point cloud in the target frame.

    This function is stateless and processes a single frame. It does not perform
    any temporal fusion or mapping updates.

    Args:
        semantic: Semantic labels (and optional confidence) with shape ``(H, W)``.
        depth: Depth image aligned with ``semantic.labels`` and shape ``(H, W)``.
            Depth values are expected in meters.
        intrinsics: Camera intrinsics matrix ``K`` with shape ``(3, 3)``.
        target_T_depth: Homogeneous transform with shape ``(4, 4)`` mapping points
            from the depth frame into the desired target/output frame.
        include_unlabeled: If true, keep points whose semantic label is negative
            (e.g. ``-1``). If false, drop them.
        max_depth_m: Optional max depth in meters; values beyond are dropped.

    Returns:
        SemanticPointCloud in the target frame:
        - ``points_xyz`` has shape ``(N, 3)`` in target coordinates (meters)
        - ``labels`` has shape ``(N,)`` and dtype ``int64``
        - ``confidence`` has shape ``(N,)`` (float32) if provided, else ``None``

    Raises:
        ValueError: If inputs have invalid shapes/dtypes or transforms are invalid.
    """
    LOGGER.debug(
        "fuse_depth_semantics: labels=%s depth=%s include_unlabeled=%s",
        semantic.labels.shape,
        depth.depth.shape,
        include_unlabeled,
    )
    semantic.validate()
    depth.validate()
    target_T_depth = require_homogeneous_transform(target_T_depth)

    points_cam, valid_mask = depth_to_points(
        depth.depth, intrinsics, max_depth_m=max_depth_m
    )
    if points_cam.shape[0] == 0:
        LOGGER.warning("Depth fusion received no valid points; returning empty PCL")
        return SemanticPointCloud(
            np.empty((0, 3)), np.empty((0,), dtype=np.int64), None
        )

    labels = flatten_masked(semantic.labels, valid_mask)
    if not include_unlabeled:
        keep = labels >= 0
        points_cam = points_cam[keep]
        labels = labels[keep]
        conf = (
            flatten_masked(semantic.confidence, valid_mask)[keep]
            if semantic.confidence is not None
            else None
        )
        LOGGER.debug(
            "Depth fusion keeping %d labeled points (filtered unlabeled)",
            points_cam.shape[0],
        )
    else:
        conf = (
            flatten_masked(semantic.confidence, valid_mask)
            if semantic.confidence is not None
            else None
        )
        LOGGER.debug(
            "Depth fusion keeping %d points (including unlabeled)",
            points_cam.shape[0],
        )

    if conf is not None:
        conf = np.asarray(conf, dtype=np.float32)

    points_target = transform_points(target_T_depth, points_cam)
    pcl = SemanticPointCloud(points_target, labels.astype(np.int64), conf)
    pcl.validate()
    return pcl


def fuse_lidar_semantics(
    semantic: SemanticObservation,
    lidar_points: PointObservation,
    intrinsics: np.ndarray,
    camera_T_lidar: np.ndarray,
    target_T_lidar: np.ndarray,
    include_unlabeled: bool = False,
    patch_size: int = 1,
) -> SemanticPointCloud:
    """Project LiDAR into an image, sample semantics, and emit a semantic point cloud.

    This function is stateless and processes a single frame. It does not perform
    any temporal fusion or mapping updates.

    Args:
        semantic: Semantic labels (and optional confidence) with shape ``(H, W)``.
        lidar_points: LiDAR points with shape ``(N, 3)`` in the LiDAR frame.
        intrinsics: Camera intrinsics matrix ``K`` with shape ``(3, 3)``.
        camera_T_lidar: Homogeneous transform with shape ``(4, 4)`` mapping LiDAR
            points into the camera frame.
        target_T_lidar: Homogeneous transform with shape ``(4, 4)`` mapping LiDAR
            points into the desired target/output frame.
        include_unlabeled: If true, also output points that project outside the
            image bounds with label ``-1`` and confidence ``0`` (if confidence is
            enabled). If false, drop them.
        patch_size: Odd patch size for robust label sampling around each projected
            pixel (1=center pixel, 3=3x3, 5=5x5, ...).

    Returns:
        SemanticPointCloud in the target frame:
        - ``points_xyz`` has shape ``(M, 3)`` in target coordinates (meters)
        - ``labels`` has shape ``(M,)`` and dtype ``int64``
        - ``confidence`` has shape ``(M,)`` (float32) if provided, else ``None``

    Raises:
        ValueError: If inputs have invalid shapes/dtypes or transforms are invalid.
        RuntimeError: If internal projection bookkeeping becomes inconsistent.
    """
    LOGGER.debug(
        "fuse_lidar_semantics: labels=%s points=%s include_unlabeled=%s",
        semantic.labels.shape,
        lidar_points.points_xyz.shape,
        include_unlabeled,
    )
    semantic.validate()
    lidar_points.validate()
    camera_T_lidar = require_homogeneous_transform(camera_T_lidar)
    target_T_lidar = require_homogeneous_transform(target_T_lidar)

    h, w = semantic.labels.shape
    uv, inside = project_points_to_image(
        lidar_points.points_xyz, intrinsics, camera_T_lidar, (w, h)
    )

    uv_inside = uv[inside]
    # Use truncation (floor for positive coords) to keep indices in-bounds.
    # Rounding can produce u==w or v==h for points close to the image border
    # (e.g., v=479.6 with h=480 -> round(v)=480), causing IndexError.
    u = uv_inside[:, 0].astype(int)
    v = uv_inside[:, 1].astype(int)
    in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not np.all(in_bounds):
        dropped = int(np.count_nonzero(~in_bounds))
        LOGGER.debug(
            "LiDAR projection produced %d/%d borderline indices out of bounds after truncation; dropping them",
            dropped,
            int(u.size),
        )
        inside_idx = np.nonzero(inside)[0]
        inside = inside.copy()
        inside[inside_idx[~in_bounds]] = False
        uv_inside = uv_inside[in_bounds]
        u = u[in_bounds]
        v = v[in_bounds]

    labeled_points = lidar_points.points_xyz[inside]
    if labeled_points.shape[0] == 0 and not include_unlabeled:
        LOGGER.warning("LiDAR fusion found no points inside image bounds")
        return SemanticPointCloud(
            np.empty((0, 3)), np.empty((0,), dtype=np.int64), None
        )

    if labeled_points.shape[0] != u.shape[0]:
        raise RuntimeError(
            "internal error: projected uv count does not match filtered point count"
        )
    labels, confidences = sample_projected_label_patches(
        semantic.labels,
        u,
        v,
        confidence=semantic.confidence,
        patch_size=patch_size,
    )

    points_target_labeled = transform_points(target_T_lidar, labeled_points)

    if include_unlabeled:
        unlabeled_points = lidar_points.points_xyz[~inside]
        points_all = np.vstack(
            (points_target_labeled, transform_points(target_T_lidar, unlabeled_points))
        )
        labels_all = np.concatenate(
            (
                labels.astype(np.int64),
                np.full(unlabeled_points.shape[0], -1, dtype=np.int64),
            )
        )
        if confidences is not None:
            conf_all = np.concatenate(
                (
                    confidences,
                    np.zeros(
                        unlabeled_points.shape[0], dtype=confidences.dtype
                    ),
                )
            )
        else:
            conf_all = None
    else:
        points_all = points_target_labeled
        labels_all = labels.astype(np.int64)
        conf_all = confidences

    pcl = SemanticPointCloud(points_all, labels_all, conf_all)
    pcl.validate()
    LOGGER.debug(
        "LiDAR fusion produced %d labeled points (%d unlabeled kept=%s)",
        labels.shape[0],
        points_all.shape[0] - labels.shape[0],
        include_unlabeled,
    )
    return pcl
