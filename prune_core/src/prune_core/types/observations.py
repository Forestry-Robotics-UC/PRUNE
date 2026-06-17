#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: GNU General Public License v3.0 (GPL-3.0)
# Repository: PRUNE
#
# Description:
#   Numpy-only dataclasses describing per-frame semantic and geometric observations.

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SemanticObservation:
    """Per-frame semantic labels (and optional confidence) from perception.

    This is a single-frame, stateless observation. It must not contain any map
    state or temporal history.

    Attributes:
        labels: Integer label IDs with shape ``(H, W)``.
            Convention: unlabeled/unknown may be encoded as ``-1``.
        confidence: Optional per-pixel confidence/probability aligned with
            ``labels`` and shape ``(H, W)``. Any numeric dtype is accepted; fusion
            outputs convert confidence to float.
    """

    labels: np.ndarray  # (H, W)
    confidence: Optional[np.ndarray] = None  # (H, W)

    def validate(self) -> None:
        """Validate shapes and dtypes.

        Raises:
            ValueError: If shapes/dtypes are invalid.
        """
        labels = np.asarray(self.labels)
        if labels.ndim != 2:
            raise ValueError(
                f"labels must be 2D (H, W), got shape {labels.shape}"
            )
        if labels.dtype.kind not in ("i", "u"):
            raise ValueError(
                f"labels must be an integer array, got dtype {labels.dtype}"
            )
        if self.confidence is not None:
            conf = np.asarray(self.confidence)
            if conf.shape != labels.shape:
                raise ValueError(
                    "confidence shape "
                    f"{conf.shape} must match labels {labels.shape}"
                )
            if conf.dtype.kind not in ("f", "i", "u"):
                raise ValueError(
                    f"confidence must be numeric, got dtype {conf.dtype}"
                )


@dataclass
class DepthObservation:
    """Depth image aligned to the semantic frame.

    Attributes:
        depth: Depth image with shape ``(H, W)``. Values are expected in meters
            for fusion functions in this repository.
    """

    depth: np.ndarray  # (H, W) float depth in meters

    def validate(self) -> None:
        """Validate shapes and dtypes.

        Raises:
            ValueError: If shapes/dtypes are invalid.
        """
        depth = np.asarray(self.depth)
        if depth.ndim != 2:
            raise ValueError(
                f"depth must be 2D (H, W), got shape {depth.shape}"
            )
        if depth.dtype.kind not in ("f", "i", "u"):
            raise ValueError(f"depth must be numeric, got dtype {depth.dtype}")


@dataclass
class PointObservation:
    """Unordered 3D points (e.g., LiDAR) in a single frame.

    Attributes:
        points_xyz: 3D points with shape ``(N, 3)`` in the source sensor frame
            (e.g., LiDAR frame).
    """

    points_xyz: np.ndarray  # (N, 3)

    def validate(self) -> None:
        """Validate shapes and dtypes.

        Raises:
            ValueError: If shapes/dtypes are invalid.
        """
        points = np.asarray(self.points_xyz)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"points_xyz must be (N, 3), got shape {points.shape}"
            )
        if points.dtype.kind not in ("f", "i", "u"):
            raise ValueError(
                f"points_xyz must be numeric, got dtype {points.dtype}"
            )


@dataclass
class SemanticPointCloud:
    """Semantic point cloud measurement ready for mapping.

    This is a single-frame measurement (not a map). It is intended to be handed
    off to a mapping layer that performs any temporal accumulation.

    Attributes:
        points_xyz: 3D points with shape ``(N, 3)`` in the chosen target frame.
        labels: Integer label IDs with shape ``(N,)`` aligned with ``points_xyz``.
            Convention: unlabeled/unknown may be encoded as ``-1``.
        confidence: Optional per-point confidence aligned with ``labels`` and
            shape ``(N,)``.
    """

    points_xyz: np.ndarray  # (N, 3)
    labels: np.ndarray  # (N,)
    confidence: Optional[np.ndarray] = None  # (N,)

    def validate(self) -> None:
        """Validate shapes and dtypes.

        Raises:
            ValueError: If shapes/dtypes are invalid.
        """
        points = np.asarray(self.points_xyz)
        labels = np.asarray(self.labels)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"points_xyz must be (N, 3), got shape {points.shape}"
            )
        if points.dtype.kind not in ("f", "i", "u"):
            raise ValueError(
                f"points_xyz must be numeric, got dtype {points.dtype}"
            )
        if (
            labels.ndim != 1
            or labels.shape[0] != points.shape[0]
        ):
            raise ValueError("labels must be (N,) and aligned with points_xyz")
        if labels.dtype.kind not in ("i", "u"):
            raise ValueError(
                f"labels must be an integer array, got dtype {labels.dtype}"
            )
        if self.confidence is not None:
            conf = np.asarray(self.confidence)
            if conf.shape != labels.shape:
                raise ValueError("confidence must be (N,) and aligned with labels")
            if conf.dtype.kind not in ("f", "i", "u"):
                raise ValueError(
                    f"confidence must be numeric, got dtype {conf.dtype}"
                )
