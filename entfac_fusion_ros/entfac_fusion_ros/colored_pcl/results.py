"""Shared dataclasses for the colored point-cloud pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import rospy

from entfac_fusion_core.types import SemanticPointCloud


@dataclass
class PipelineResult:
    """Lightweight result object for future pipeline extraction phases."""

    cloud: SemanticPointCloud
    stamp: rospy.Time
    frame_id: str
    callback_sec: float
    debug: dict


@dataclass
class LastPcl:
    """Last published cloud payload used by the PLY recorder."""

    stamp: rospy.Time
    points_xyz: np.ndarray
    labels: np.ndarray
    confidence: Optional[np.ndarray]
    rgb_packed_float: Optional[np.ndarray]


@dataclass
class SemanticInputs:
    """Parsed semantic inputs shared by the current and future pipeline stages."""

    labels: Optional[np.ndarray]
    packed_rgb: Optional[np.ndarray]
    confidence: Optional[np.ndarray]
    projection_invalid_mask: Optional[np.ndarray]
    rgb_lut: Optional[np.ndarray]
