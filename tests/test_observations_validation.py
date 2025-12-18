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
#   Unit tests for v1 public core dataclasses (shape and dtype validation).

import sys
from pathlib import Path

import numpy as np
import pytest

CORE_SRC = Path(__file__).resolve().parents[1] / "entfac_fusion_core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from entfac_fusion_core.types import (  # noqa: E402
    DepthObservation,
    PointObservation,
    SemanticObservation,
    SemanticPointCloud,
)


def test_semantic_observation_validate_rejects_wrong_shape():
    obs = SemanticObservation(labels=np.zeros((10,), dtype=np.int32))
    with pytest.raises(ValueError):
        obs.validate()


def test_semantic_observation_validate_rejects_non_integer_labels():
    obs = SemanticObservation(labels=np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        obs.validate()


def test_semantic_observation_validate_rejects_confidence_shape_mismatch():
    obs = SemanticObservation(
        labels=np.zeros((2, 2), dtype=np.int32),
        confidence=np.zeros((2, 3), dtype=np.float32),
    )
    with pytest.raises(ValueError):
        obs.validate()


def test_depth_observation_validate_rejects_non_numeric_depth():
    obs = DepthObservation(depth=np.array([["x"]], dtype=object))
    with pytest.raises(ValueError):
        obs.validate()


def test_point_observation_validate_rejects_wrong_shape():
    obs = PointObservation(points_xyz=np.zeros((10, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        obs.validate()


def test_semantic_pointcloud_validate_rejects_label_dtype():
    pcl = SemanticPointCloud(
        points_xyz=np.zeros((3, 3), dtype=np.float32),
        labels=np.zeros((3,), dtype=np.float32),
    )
    with pytest.raises(ValueError):
        pcl.validate()

