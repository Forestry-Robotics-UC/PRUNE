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
#   Validation helpers for matrices, transforms, and masked flattening.

"""Lightweight validation helpers for numpy Sensor Fusion pipelines."""

import numpy as np


def ensure_float_matrix(mat: np.ndarray, shape: tuple) -> np.ndarray:
    """Ensure a numeric matrix has the expected shape and float dtype.

    Args:
        mat: Input array-like matrix.
        shape: Expected matrix shape.

    Returns:
        Matrix as a ``float64`` NumPy array.

    Raises:
        ValueError: If the shape is wrong or values are not finite.
    """
    arr = np.asarray(mat, dtype=float)
    if arr.shape != shape:
        raise ValueError(f"expected shape {shape}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("matrix contains non-finite values")
    return arr


def require_homogeneous_transform(transform: np.ndarray) -> np.ndarray:
    """Validate a 4x4 SE(3) homogeneous transform.

    The transform is expected to represent a rigid-body pose in 3D:

    - last row is ``[0, 0, 0, 1]``
    - top-left ``3x3`` block is an orthonormal rotation matrix with ``det(R) = +1``

    Args:
        transform: Homogeneous transform with shape ``(4, 4)``.

    Returns:
        Transform as a ``float64`` NumPy array.

    Raises:
        ValueError: If the transform is not a valid SE(3) matrix.
    """
    tf = ensure_float_matrix(transform, (4, 4))
    if not np.allclose(tf[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("transform last row must be [0, 0, 0, 1]")

    rot = tf[:3, :3]
    det = float(np.linalg.det(rot))
    if not np.isfinite(det) or det < 0.0 or np.abs(det - 1.0) > 1e-3:
        raise ValueError("rotation block must be a proper rotation (det=+1)")

    rtr = rot.T @ rot
    if not np.allclose(rtr, np.eye(3), atol=1e-3):
        raise ValueError("rotation block must be orthonormal (R^T R = I)")

    return tf


def flatten_masked(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Flatten an array using a boolean mask of the same shape.

    Args:
        values: Input array.
        mask: Boolean mask with the same shape as ``values``.

    Returns:
        1D array with the elements of ``values`` where ``mask`` is true.

    Raises:
        ValueError: If shapes do not match or mask is not boolean.
    """
    values_arr = np.asarray(values)
    mask_arr = np.asarray(mask)
    if values_arr.shape != mask_arr.shape:
        raise ValueError("mask shape must match values")
    if mask_arr.dtype != np.bool_:
        raise ValueError("mask must be a boolean array")
    return values_arr[mask_arr]
