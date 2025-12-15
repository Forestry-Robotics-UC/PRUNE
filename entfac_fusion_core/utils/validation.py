#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: MIT License (open source, free to modify and redistribute)
# Repository: fruc_ros_utils
#
# Description:
#   Validation helpers for matrices, transforms, and masked flattening.

"""Lightweight validation helpers for numpy Sensor Fusion pipelines."""

import numpy as np


def ensure_float_matrix(mat: np.ndarray, shape: tuple) -> np.ndarray:
    """Validate shape and return float64 view."""
    if mat.shape != shape:
        raise ValueError(f"expected shape {shape}, got {mat.shape}")
    return np.asarray(mat, dtype=float)


def require_homogeneous_transform(transform: np.ndarray) -> np.ndarray:
    """Validate 4x4 SE(3) matrix with proper rotation."""
    transform = ensure_float_matrix(transform, (4, 4))
    rot = transform[:3, :3]
    det = np.linalg.det(rot)
    if not np.isfinite(det) or np.abs(det - 1.0) > 1e-3:
        raise ValueError("rotation block of transform is not a proper rotation")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("transform last row must be [0, 0, 0, 1]")
    return transform


def flatten_masked(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Flatten 2D array using a boolean mask."""
    if values.shape != mask.shape:
        raise ValueError("mask shape must match values")
    return values[mask]
