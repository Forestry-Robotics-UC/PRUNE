#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Adapted from Semantic SLAM (substantially refactored for ENTFAC).
#
# Original Author:
#   Xuan Zhang
#
# Subsequent Contributions:
#   David Russell
#
# Upstream reference:
#   https://github.com/floatlazer/semantic_slam
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
#   PointCloud2 creation utilities (labels/confidence/rgb fields) using numpy.

"""PointCloud2 construction helpers."""

from __future__ import annotations

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from typing import Optional


def labels_to_uint16(labels: np.ndarray) -> np.ndarray:
    """Convert int labels to uint16, mapping negatives to 65535 (unknown)."""
    labels_arr = np.asarray(labels)
    if labels_arr.ndim != 1:
        labels_arr = labels_arr.reshape(-1)
    if labels_arr.dtype.kind not in ("i", "u"):
        raise ValueError("labels must be an integer array")
    if np.any(labels_arr > 65535):
        raise ValueError("label must fit into uint16 (0..65535)")
    if labels_arr.dtype.kind == "u":
        return labels_arr.astype(np.uint16, copy=False)
    labels_u16 = labels_arr.astype(np.uint16, copy=True)
    neg_mask = labels_arr < 0
    if np.any(neg_mask):
        labels_u16[neg_mask] = 65535
    return labels_u16


def build_label_rgb_float_lut(
    *,
    color_map=None,
    num_labels: Optional[int] = None,
    seed: int = 1,
) -> np.ndarray:
    """Build a uint16-label -> packed RGB float32 lookup table.

    The packed RGB float encoding is the common ROS/PCL convention where a 24-bit
    RGB integer is reinterpreted as float32.

    If color_map is not provided, a deterministic random palette is generated for
    labels [0..num_labels-1] (if num_labels is provided). Remaining labels fall
    back to a stable hash palette.
    """
    labels = np.arange(65536, dtype=np.uint32)
    packed = np.zeros_like(labels, dtype=np.uint32)

    def _hash_palette(ids: np.ndarray) -> np.ndarray:
        r = (ids * 37) & 0xFF
        g = (ids * 17) & 0xFF
        b = (ids * 73) & 0xFF
        return (r << 16) | (g << 8) | b

    if num_labels is not None:
        n = int(num_labels)
        if n < 0 or n > 65536:
            raise ValueError("num_labels must be in [0, 65536]")
        rng = np.random.default_rng(int(seed))
        pal = rng.integers(0, 256, size=(n, 3), dtype=np.uint32)
        packed[:n] = (pal[:, 0] << 16) | (pal[:, 1] << 8) | pal[:, 2]
        if n < 65536:
            packed[n:] = _hash_palette(labels[n:])
    else:
        packed[:] = _hash_palette(labels)

    packed[65535] = 0xFFFFFF
    if color_map:
        for label_id, rgb in color_map.items():
            if not (0 <= int(label_id) <= 65535):
                continue
            if not isinstance(rgb, (list, tuple)) or len(rgb) != 3:
                continue
            rr, gg, bb = int(rgb[0]), int(rgb[1]), int(rgb[2])
            packed[int(label_id)] = ((rr & 0xFF) << 16) | ((gg & 0xFF) << 8) | (
                bb & 0xFF
            )

    packed_le = packed.astype("<u4", copy=False)
    return packed_le.view("<f4")


def semantic_pointcloud_to_msg(
    pcl,
    frame_id,
    stamp,
    *,
    colorize_labels: bool = False,
    rgb_lut: Optional[np.ndarray] = None,
    rgb_values: Optional[np.ndarray] = None,
):
    """Convert SemanticPointCloud dataclass to PointCloud2."""
    has_conf = pcl.confidence is not None
    has_rgb = bool(colorize_labels)
    num_points = int(pcl.points_xyz.shape[0])

    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
        PointField("label", 12, PointField.UINT16, 1),
    ]
    point_step = 16
    if has_conf:
        fields.append(PointField("confidence", point_step, PointField.FLOAT32, 1))
        point_step += 4
    if has_rgb:
        fields.append(PointField("rgb", point_step, PointField.FLOAT32, 1))
        point_step += 4

    dtype_fields = [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("label", "<u2"),
        ("_pad", "<u2"),
    ]
    if has_conf:
        dtype_fields.append(("confidence", "<f4"))
    if has_rgb:
        dtype_fields.append(("rgb", "<f4"))
    dtype = np.dtype(dtype_fields)
    if dtype.itemsize != point_step:
        raise RuntimeError(
            f"internal dtype mismatch: itemsize={dtype.itemsize} point_step={point_step}"
        )

    cloud = np.empty(num_points, dtype=dtype)
    points = np.asarray(pcl.points_xyz)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must be (N, 3)")
    cloud["x"] = points[:, 0]
    cloud["y"] = points[:, 1]
    cloud["z"] = points[:, 2]
    cloud["_pad"] = 0

    labels_u16 = labels_to_uint16(pcl.labels)
    if labels_u16.shape[0] != num_points:
        raise ValueError("labels must be (N,) and aligned with points_xyz")
    cloud["label"] = labels_u16

    if has_conf:
        conf = np.asarray(pcl.confidence, dtype=np.float32)
        if conf.shape[0] != num_points:
            raise ValueError("confidence must be (N,) and aligned with points_xyz")
        cloud["confidence"] = conf

    if has_rgb:
        if rgb_values is not None:
            rgb_arr = np.asarray(rgb_values, dtype=np.float32)
            if rgb_arr.shape[0] != num_points:
                raise ValueError("rgb_values must be (N,) and aligned with points_xyz")
            cloud["rgb"] = rgb_arr
        else:
            if rgb_lut is None:
                raise ValueError(
                    "rgb output requested but no rgb_values or rgb_lut was provided"
                )
            cloud["rgb"] = rgb_lut[labels_u16]

    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.height = 1
    msg.width = num_points
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = point_step
    msg.row_step = point_step * num_points
    msg.is_dense = True
    msg.data = cloud.tobytes()
    return msg
