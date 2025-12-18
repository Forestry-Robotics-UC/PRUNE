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
#   Fast numpy conversions for ROS sensor_msgs/Image and sensor_msgs/PointCloud2.

"""Fast numpy conversions for ROS messages."""

from __future__ import annotations

import numpy as np


def image_to_numpy(msg):
    """Convert sensor_msgs/Image to numpy without copies (when possible)."""
    if msg is None:
        return None
    encoding = msg.encoding.lower()
    if encoding in ("bgr8", "rgb8"):
        dtype = np.uint8
        channels = 3
    elif encoding in ("bgra8", "rgba8"):
        dtype = np.uint8
        channels = 4
    elif encoding in ("32fc1", "32fc"):
        dtype = np.float32
        channels = 1
    elif encoding in ("32sc1",):
        dtype = np.int32
        channels = 1
    elif encoding in ("16uc1", "16sc1", "mono16"):
        dtype = np.uint16
        channels = 1
    elif encoding in ("8uc1", "mono8"):
        dtype = np.uint8
        channels = 1
    else:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")
    expected_step = msg.width * channels * dtype().nbytes
    if msg.step != expected_step:
        raise ValueError(
            f"Unsupported step {msg.step} for {msg.encoding}; expected {expected_step}"
        )
    arr = np.frombuffer(msg.data, dtype=dtype)
    if channels == 1:
        return arr.reshape(msg.height, msg.width)
    return arr.reshape(msg.height, msg.width, channels)


def pointcloud2_to_xyz(msg):
    """Extract xyz float32 points from sensor_msgs/PointCloud2."""
    if msg.is_bigendian:
        raise ValueError("big-endian PointCloud2 not supported in fast path")
    field_offsets = {f.name: f.offset for f in msg.fields}
    for needed in ("x", "y", "z"):
        if needed not in field_offsets:
            raise ValueError("PointCloud2 missing xyz fields")
    dtype = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": ["<f4", "<f4", "<f4"],
            "offsets": [
                field_offsets["x"],
                field_offsets["y"],
                field_offsets["z"],
            ],
            "itemsize": msg.point_step,
        }
    )
    cloud = np.frombuffer(msg.data, dtype=dtype, count=msg.width * msg.height)
    points = np.zeros((cloud.shape[0], 3), dtype=np.float32)
    points[:, 0] = cloud["x"]
    points[:, 1] = cloud["y"]
    points[:, 2] = cloud["z"]
    return points


def _quantize_u8(arr_u8: np.ndarray, step: int) -> np.ndarray:
    step = int(step)
    if step <= 1:
        return arr_u8.astype(np.uint32, copy=False)
    vals = arr_u8.astype(np.int16, copy=False)
    half = step // 2
    quant = ((vals + half) // step) * step
    return np.clip(quant, 0, 255).astype(np.uint32, copy=False)


def rgb_to_packed_u32(data: np.ndarray, encoding: str, *, quantize_step: int = 1):
    """Pack rgb/bgr/rgba/bgra image into uint32 (r<<16|g<<8|b)."""
    encoding = str(encoding).lower()
    if encoding == "bgr8":
        b, g, r = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    elif encoding == "rgb8":
        r, g, b = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    elif encoding == "bgra8":
        b, g, r = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    elif encoding == "rgba8":
        r, g, b = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    else:
        raise ValueError(f"Unsupported 3/4-channel encoding: {encoding}")
    step = int(quantize_step)
    r_u32 = _quantize_u8(r, step)
    g_u32 = _quantize_u8(g, step)
    b_u32 = _quantize_u8(b, step)
    return (r_u32 << 16) | (g_u32 << 8) | b_u32

