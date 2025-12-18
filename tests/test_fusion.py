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
#   Unit tests for single-frame Sensor Fusion numpy core.

import sys
from pathlib import Path

import numpy as np

CORE_SRC = Path(__file__).resolve().parents[1] / "entfac_fusion_core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

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


def test_lidar_fusion_uses_truncation_to_avoid_border_rounding():
    # Regression test: rounding can produce v==h or u==w for points near the
    # border (e.g. v=1.9 with h=2 -> round(v)=2), which would be out-of-bounds.
    labels = np.array([[5, 6], [7, 8]], dtype=np.int32)
    intrinsics = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    pcl = fuse_lidar_semantics(
        SemanticObservation(labels=labels),
        PointObservation(points_xyz=np.array([[0.0, 1.9, 1.0]])),
        intrinsics,
        np.eye(4),
        np.eye(4),
    )

    assert pcl.points_xyz.shape[0] == 1
    # v=1.9 should sample row 1 (floor), not crash by rounding to row 2.
    assert pcl.labels[0] == 7
