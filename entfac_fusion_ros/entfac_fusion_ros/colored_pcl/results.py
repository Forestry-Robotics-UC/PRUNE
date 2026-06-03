#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ENTFAC Sensor Fusion implementation.
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
#   Shared dataclasses for the colored point-cloud node refactor.

"""Shared dataclasses for the colored point-cloud node refactor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

try:
    import rospy
except ImportError:  # pragma: no cover - optional for non-ROS import paths
    rospy = None  # type: ignore[assignment]


@dataclass
class PipelineResult:
    cloud: SemanticPointCloud
    stamp: rospy.Time
    frame_id: str
    callback_sec: float
    debug: Dict[str, Any]


@dataclass
class LastPcl:
    stamp: rospy.Time
    points_xyz: np.ndarray
    labels: np.ndarray
    confidence: Optional[np.ndarray]
    rgb_packed_float: Optional[np.ndarray]


@dataclass
class SemanticInputs:
    labels: Optional[np.ndarray]
    packed_rgb: Optional[np.ndarray]
    confidence: Optional[np.ndarray]
    projection_invalid_mask: Optional[np.ndarray]
    rgb_lut: Optional[np.ndarray]


__all__ = ["PipelineResult", "LastPcl", "SemanticInputs"]
