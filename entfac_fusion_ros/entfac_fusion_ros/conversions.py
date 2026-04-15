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
#   Fast numpy conversions for ROS sensor_msgs/Image and sensor_msgs/PointCloud2.

"""Fast numpy conversions for ROS messages."""

from __future__ import annotations

import numpy as np


def image_to_numpy(msg):
    """Convert ``sensor_msgs/Image`` to a NumPy array without copies when possible.

    Args:
        msg: ROS Image message. Supported encodings include ``mono8``,
            ``mono16``, ``16UC1``, ``32FC1``, ``rgb8``, ``bgr8``, ``rgba8``,
            and ``bgra8``.

    Returns:
        NumPy array with shape ``(H, W)`` for single-channel images or
        ``(H, W, C)`` for 3/4-channel images. The dtype matches the encoding.

    Raises:
        ValueError: If the encoding is unsupported or the row step does not
            match the expected packed layout.
    """
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
    """Extract XYZ points from ``sensor_msgs/PointCloud2`` into a float32 array.

    Args:
        msg: ROS PointCloud2 message containing ``x``, ``y``, and ``z`` fields.

    Returns:
        ``(N, 3)`` array of XYZ points in the message frame (float32).

    Raises:
        ValueError: If the message is big-endian or missing XYZ fields.
    """
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


def pointcloud2_to_xyz_t(msg, *, time_field: str = "t"):
    """Extract XYZ points and per-point time from ``sensor_msgs/PointCloud2``.

    Args:
        msg: ROS PointCloud2 message containing ``x``, ``y``, ``z``, and time field.
        time_field: Field name for per-point time (default: "t").

    Returns:
        Tuple (points_xyz, t_raw):
        - points_xyz: (N, 3) float32 array of XYZ points.
        - t_raw: (N,) array of raw time values (dtype per field, typically uint32).

    Raises:
        ValueError: If the message is big-endian or missing required fields.
    """
    if msg.is_bigendian:
        raise ValueError("big-endian PointCloud2 not supported in fast path")
    field_offsets = {f.name: f.offset for f in msg.fields}
    for needed in ("x", "y", "z", time_field):
        if needed not in field_offsets:
            raise ValueError(f"PointCloud2 missing field: {needed}")
    dtype = np.dtype(
        {
            "names": ["x", "y", "z", "t"],
            "formats": ["<f4", "<f4", "<f4", "<u4"],
            "offsets": [
                field_offsets["x"],
                field_offsets["y"],
                field_offsets["z"],
                field_offsets[time_field],
            ],
            "itemsize": msg.point_step,
        }
    )
    cloud = np.frombuffer(msg.data, dtype=dtype, count=msg.width * msg.height)
    points = np.zeros((cloud.shape[0], 3), dtype=np.float32)
    points[:, 0] = cloud["x"]
    points[:, 1] = cloud["y"]
    points[:, 2] = cloud["z"]
    t_raw = cloud["t"].copy()
    return points, t_raw


def _quantize_u8(arr_u8: np.ndarray, step: int) -> np.ndarray:
    step = int(step)
    if step <= 1:
        return arr_u8.astype(np.uint32, copy=False)
    vals = arr_u8.astype(np.int16, copy=False)
    half = step // 2
    quant = ((vals + half) // step) * step
    return np.clip(quant, 0, 255).astype(np.uint32, copy=False)


def rgb_to_packed_u32(data: np.ndarray, encoding: str, *, quantize_step: int = 1):
    """Pack RGB/BGR/RGBA/BGRA image into uint32 (``r<<16|g<<8|b``).

    Args:
        data: Image array with shape ``(H, W, 3)`` or ``(H, W, 4)``.
        encoding: One of ``rgb8``, ``bgr8``, ``rgba8``, or ``bgra8``.
        quantize_step: Optional quantization step for 8-bit channels to reduce
            noise (use ``8`` or ``16`` for JPEG artifacts; ``1`` disables).

    Returns:
        ``(H, W)`` array of packed RGB values as ``uint32``.
    """
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
