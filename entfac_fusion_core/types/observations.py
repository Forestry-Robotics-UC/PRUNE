#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: MIT License (open source, free to modify and redistribute)
# Repository: fruc_ros_utils
#
# Description:
#   Numpy-only dataclasses describing per-frame semantic and geometric observations.

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SemanticObservation:
    """Per-frame semantic labels (and optional confidence) from perception."""

    labels: np.ndarray  # (H, W)
    confidence: Optional[np.ndarray] = None  # (H, W)

    def validate(self) -> None:
        if self.labels.ndim != 2:
            raise ValueError(
                f"labels must be 2D (H, W), got shape {self.labels.shape}"
            )
        if self.confidence is not None:
            if self.confidence.shape != self.labels.shape:
                raise ValueError(
                    "confidence shape "
                    f"{self.confidence.shape} must match labels {self.labels.shape}"
                )


@dataclass
class DepthObservation:
    """Depth image aligned to the semantic frame."""

    depth: np.ndarray  # (H, W) float depth in meters

    def validate(self) -> None:
        if self.depth.ndim != 2:
            raise ValueError(
                f"depth must be 2D (H, W), got shape {self.depth.shape}"
            )


@dataclass
class PointObservation:
    """Unordered 3D points (e.g., LiDAR) in a single frame."""

    points_xyz: np.ndarray  # (N, 3)

    def validate(self) -> None:
        if self.points_xyz.ndim != 2 or self.points_xyz.shape[1] != 3:
            raise ValueError(
                f"points_xyz must be (N, 3), got shape {self.points_xyz.shape}"
            )


@dataclass
class SemanticPointCloud:
    """Semantic point cloud measurement ready for mapping."""

    points_xyz: np.ndarray  # (N, 3)
    labels: np.ndarray  # (N,)
    confidence: Optional[np.ndarray] = None  # (N,)

    def validate(self) -> None:
        if self.points_xyz.ndim != 2 or self.points_xyz.shape[1] != 3:
            raise ValueError(
                f"points_xyz must be (N, 3), got shape {self.points_xyz.shape}"
            )
        if (
            self.labels.ndim != 1
            or self.labels.shape[0] != self.points_xyz.shape[0]
        ):
            raise ValueError("labels must be (N,) and aligned with points_xyz")
        if self.confidence is not None:
            if self.confidence.shape != self.labels.shape:
                raise ValueError("confidence must be (N,) and aligned with labels")
