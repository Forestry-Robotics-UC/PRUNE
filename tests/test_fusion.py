#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: MIT License (open source, free to modify and redistribute)
# Repository: fruc_ros_utils
#
# Description:
#   Unit tests for single-frame Sensor Fusion numpy core.

import numpy as np

from entfac_fusion_core.semantic_pcl import (
    fuse_depth_semantics,
    fuse_lidar_semantics,
)
from entfac_fusion_core.types import (
    DepthObservation,
    PointObservation,
    SemanticObservation,
)


def test_depth_fusion_filters_invalid_depth():
    labels = np.array([[1, 2], [3, 4]], dtype=np.int32)
    depth = np.array([[1.0, 0.0], [2.0, np.nan]], dtype=float)
    intrinsics = np.eye(3)
    target_T_depth = np.eye(4)

    pcl = fuse_depth_semantics(
        SemanticObservation(labels=labels),
        DepthObservation(depth=depth),
        intrinsics,
        target_T_depth,
    )

    assert pcl.points_xyz.shape[0] == 2
    assert np.all(pcl.labels == np.array([1, 3]))


def test_lidar_fusion_projects_and_samples():
    labels = np.array([[5, 6], [7, 8]], dtype=np.int32)
    intrinsics = np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]])
    camera_T_lidar = np.eye(4)
    target_T_lidar = np.eye(4)

    # One point at (u=0, v=0) with z=1, one behind camera (z negative)
    lidar_points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])

    pcl = fuse_lidar_semantics(
        SemanticObservation(labels=labels),
        PointObservation(points_xyz=lidar_points),
        intrinsics,
        camera_T_lidar,
        target_T_lidar,
    )

    assert pcl.points_xyz.shape[0] == 1
    assert pcl.labels[0] == 5
