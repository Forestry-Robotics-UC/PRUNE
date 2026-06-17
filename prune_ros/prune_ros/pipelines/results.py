"""Shared dataclasses for the prune node refactor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

if TYPE_CHECKING:
    from prune_core.types import SemanticPointCloud

try:
    import rospy
except ImportError:  # pragma: no cover
    rospy = None  # type: ignore[assignment]


@dataclass
class PipelineResult:
    cloud: 'SemanticPointCloud'
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


__all__ = ['PipelineResult', 'LastPcl', 'SemanticInputs']
