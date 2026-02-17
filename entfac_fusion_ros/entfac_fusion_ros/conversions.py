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


_POINTFIELD_TO_DTYPE = {
    1: np.dtype("<i1"),  # INT8
    2: np.dtype("<u1"),  # UINT8
    3: np.dtype("<i2"),  # INT16
    4: np.dtype("<u2"),  # UINT16
    5: np.dtype("<i4"),  # INT32
    6: np.dtype("<u4"),  # UINT32
    7: np.dtype("<f4"),  # FLOAT32
    8: np.dtype("<f8"),  # FLOAT64
}


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


def _pointcloud_struct_view(msg, fields):
    if msg.is_bigendian:
        raise ValueError("big-endian PointCloud2 not supported in fast path")
    available = {f.name: f for f in msg.fields}
    names = []
    formats = []
    offsets = []
    for alias, src_name in fields:
        if src_name not in available:
            raise ValueError(f"PointCloud2 missing field: {src_name}")
        field = available[src_name]
        if int(getattr(field, "count", 1)) not in (0, 1):
            raise ValueError(
                f"PointCloud2 field '{src_name}' has unsupported count={field.count}"
            )
        dtype = _POINTFIELD_TO_DTYPE.get(int(field.datatype))
        if dtype is None:
            raise ValueError(
                f"PointCloud2 field '{src_name}' has unsupported datatype={field.datatype}"
            )
        names.append(alias)
        formats.append(dtype)
        offsets.append(int(field.offset))
    struct_dtype = np.dtype(
        {
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": int(msg.point_step),
        }
    )
    return np.frombuffer(
        msg.data, dtype=struct_dtype, count=int(msg.width) * int(msg.height)
    )


def pointcloud2_to_xyz(msg):
    """Extract XYZ points from ``sensor_msgs/PointCloud2`` into a float32 array.

    Args:
        msg: ROS PointCloud2 message containing ``x``, ``y``, and ``z`` fields.

    Returns:
        ``(N, 3)`` array of XYZ points in the message frame (float32).

    Raises:
        ValueError: If the message is big-endian or missing XYZ fields.
    """
    cloud = _pointcloud_struct_view(msg, (("x", "x"), ("y", "y"), ("z", "z")))
    points = np.empty((cloud.shape[0], 3), dtype=np.float32)
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
    cloud = _pointcloud_struct_view(
        msg,
        (("x", "x"), ("y", "y"), ("z", "z"), ("t", time_field)),
    )
    points = np.empty((cloud.shape[0], 3), dtype=np.float32)
    points[:, 0] = cloud["x"]
    points[:, 1] = cloud["y"]
    points[:, 2] = cloud["z"]
    t_raw = np.asarray(cloud["t"]).copy()
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
