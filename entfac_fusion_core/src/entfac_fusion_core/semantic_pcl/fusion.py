#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: MIT License (open source, free to modify and redistribute)
# Repository: ENTFAC-Sensor-Fusion
#
# Description:
#   Single-frame semantic fusion pipelines (depth-based and LiDAR projection).

"""Single-frame semantic fusion pipelines."""

import logging
import numpy as np

from entfac_fusion_core.projection.depth import depth_to_points
from entfac_fusion_core.projection.lidar_projection import project_points_to_image
from entfac_fusion_core.transforms.se3 import transform_points
from entfac_fusion_core.types.observations import (
    DepthObservation,
    PointObservation,
    SemanticObservation,
    SemanticPointCloud,
)
from entfac_fusion_core.utils.validation import (
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
) -> SemanticPointCloud:
    """Fuse aligned semantic + depth into a semantic point cloud in target frame."""
    semantic.validate()
    depth.validate()
    target_T_depth = require_homogeneous_transform(target_T_depth)

    points_cam, valid_mask = depth_to_points(depth.depth, intrinsics)
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

    points_target = transform_points(target_T_depth, points_cam)
    pcl = SemanticPointCloud(points_target, labels.astype(np.int64), conf)
    pcl.validate()
    LOGGER.info("Depth fusion produced %d points", points_target.shape[0])
    return pcl


def fuse_lidar_semantics(
    semantic: SemanticObservation,
    lidar_points: PointObservation,
    intrinsics: np.ndarray,
    camera_T_lidar: np.ndarray,
    target_T_lidar: np.ndarray,
    include_unlabeled: bool = False,
) -> SemanticPointCloud:
    """Project LiDAR into image, sample semantics, and emit semantic point cloud."""
    semantic.validate()
    lidar_points.validate()
    camera_T_lidar = require_homogeneous_transform(camera_T_lidar)
    target_T_lidar = require_homogeneous_transform(target_T_lidar)

    h, w = semantic.labels.shape
    uv, inside = project_points_to_image(
        lidar_points.points_xyz, intrinsics, camera_T_lidar, (w, h)
    )

    labeled_points = lidar_points.points_xyz[inside]
    if labeled_points.shape[0] == 0 and not include_unlabeled:
        LOGGER.warning("LiDAR fusion found no points inside image bounds")
        return SemanticPointCloud(
            np.empty((0, 3)), np.empty((0,), dtype=np.int64), None
        )

    uv_valid = uv[inside]
    u = np.round(uv_valid[:, 0]).astype(int)
    v = np.round(uv_valid[:, 1]).astype(int)
    labels = semantic.labels[v, u]

    if semantic.confidence is not None:
        confidences = semantic.confidence[v, u]
    else:
        confidences = None

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
    LOGGER.info(
        "LiDAR fusion produced %d labeled points (%d unlabeled kept=%s)",
        labels.shape[0],
        points_all.shape[0] - labels.shape[0],
        include_unlabeled,
    )
    return pcl
