#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ENTFAC Sensor Fusion implementation.
#
# Note:
#   This file was developed specifically for ENTFAC Sensor Fusion.
#   Project-level upstream attribution is documented in README.md.
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
#   TF conversion helpers for ROS TransformStamped -> numpy 4x4 transforms.

"""TF conversion helpers."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

from entfac_fusion_core.utils.validation import require_homogeneous_transform


def transform_stamped_to_matrix(transform_stamped) -> np.ndarray:
    """Convert geometry_msgs/TransformStamped to 4x4 numpy matrix."""
    t = transform_stamped.transform.translation
    q = transform_stamped.transform.rotation
    translation = np.array([t.x, t.y, t.z], dtype=float)
    quat = np.array([q.x, q.y, q.z, q.w], dtype=float)

    rot_obj = SciRotation.from_quat(quat)
    # SciPy < 1.4 uses as_dcm(); newer versions use as_matrix().
    rot = rot_obj.as_matrix() if hasattr(rot_obj, "as_matrix") else rot_obj.as_dcm()

    mat = np.eye(4, dtype=float)
    mat[:3, :3] = rot
    mat[:3, 3] = translation
    return require_homogeneous_transform(mat.astype(float))


def format_matrix(mat: np.ndarray) -> str:
    """Pretty-format a matrix for logs (precision=3, suppress_small=True)."""
    return np.array2string(
        np.asarray(mat), precision=3, suppress_small=True, max_line_width=120
    )

