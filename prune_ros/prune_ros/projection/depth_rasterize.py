#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: GNU General Public License v3.0 (GPL-3.0)
# Repository: PRUNE

"""Depth-map rasterization and depth-edge map computation.

No ROS imports. Each function takes its persistent reuse buffer in and
returns it back out (resized/reallocated only on a shape change), so
callers own the buffer lifetime and these stay plain, testable functions.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from prune_core.transforms.se3 import transform_points


def rasterize_depth_map(
    points: np.ndarray,
    intrinsics: np.ndarray,
    camera_T_lidar: np.ndarray,
    image_shape: Tuple[int, int],
    buffer: Optional[np.ndarray],
    buffer_shape: Optional[Tuple[int, int]],
    *,
    points_cam: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Fill a per-pixel min-depth buffer from projected LiDAR points.

    Uses a sort-based reduceat pattern for better cache locality than
    ``np.minimum.at``. Returns ``(depth_map, buffer, buffer_shape)`` —
    callers store ``buffer``/``buffer_shape`` back to reuse across frames
    and eliminate per-frame malloc overhead.

    Args:
        points_cam: if provided, skips the ``transform_points`` call
            (caller has already computed camera-frame coordinates).
    """
    h, w = int(image_shape[0]), int(image_shape[1])
    shape = (h, w)
    if buffer is None or buffer_shape != shape:
        buffer = np.empty(shape, dtype=np.float32)
        buffer_shape = shape
    buffer.fill(np.inf)
    depth = buffer

    if points_cam is None:
        points_cam = transform_points(camera_T_lidar, points)
    z = points_cam[:, 2]
    in_front = z > 0.0
    if not np.any(in_front):
        return depth, buffer, buffer_shape

    pts = points_cam[in_front]
    z = z[in_front]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    u = (pts[:, 0] * fx / z + cx).astype(np.int32, copy=False)
    v = (pts[:, 1] * fy / z + cy).astype(np.int32, copy=False)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not np.any(inside):
        return depth, buffer, buffer_shape

    u = u[inside]
    v = v[inside]
    z = z[inside].astype(np.float32, copy=False)

    idx = v * w + u
    sort_order = np.argsort(idx)
    idx_sorted = idx[sort_order]
    z_sorted = z[sort_order]
    segment_starts = np.concatenate(([0], np.where(np.diff(idx_sorted) != 0)[0] + 1))
    min_values = np.minimum.reduceat(z_sorted, segment_starts)
    unique_idx = idx_sorted[segment_starts]
    flat = depth.ravel()
    flat[unique_idx] = np.minimum(flat[unique_idx], min_values)
    return flat.reshape(h, w), buffer, buffer_shape


def range_image_depth_map(
    points_all_cam: np.ndarray,
    cloud_height: int,
    cloud_width: int,
    out_shape: Tuple[int, int],
    buffer: Optional[np.ndarray],
    buffer_shape: Optional[Tuple[int, int]],
    intrinsics: Optional[np.ndarray],
    current_subsample: int,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Build a min-depth buffer from an organized point cloud's range image.

    Operates in ring/column space (O(H_lidar * W_lidar) ~= 131K for a 128-beam
    Ouster) rather than camera-image space (O(H_cam * W_cam) ~= 2.76M for MAPIR
    1440p). Resolution-agnostic: cost does not grow with camera resolution.
    Returns ``(depth_map, buffer, buffer_shape)``, see :func:`rasterize_depth_map`.

    Args:
        points_all_cam: (cloud_height * cloud_width, 3) all points in camera
            frame, including those outside the FOV.
        out_shape: (sh, sw) output buffer shape — (H//s, W//s) for subsample s.
    """
    sh, sw = out_shape
    if buffer is None or buffer_shape != (sh, sw):
        buffer = np.empty((sh, sw), dtype=np.float32)
        buffer_shape = (sh, sw)
    buffer.fill(np.inf)
    depth = buffer

    z_all = points_all_cam[:, 2]
    in_front = z_all > 0.0
    if not np.any(in_front):
        return depth, buffer, buffer_shape

    if intrinsics is None:
        return depth, buffer, buffer_shape

    pts = points_all_cam[in_front]
    z = z_all[in_front]
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    u = (pts[:, 0] * fx / z + cx).astype(np.int32, copy=False) // max(1, current_subsample)
    v = (pts[:, 1] * fy / z + cy).astype(np.int32, copy=False) // max(1, current_subsample)
    inside = (u >= 0) & (u < sw) & (v >= 0) & (v < sh)
    if not np.any(inside):
        return depth, buffer, buffer_shape

    u, v, z = u[inside], v[inside], z[inside].astype(np.float32, copy=False)
    idx = v * sw + u
    sort_order = np.argsort(idx)
    idx_sorted = idx[sort_order]
    z_sorted = z[sort_order]
    seg_starts = np.concatenate(([0], np.where(np.diff(idx_sorted) != 0)[0] + 1))
    min_vals = np.minimum.reduceat(z_sorted, seg_starts)
    flat = depth.ravel()
    flat[idx_sorted[seg_starts]] = np.minimum(flat[idx_sorted[seg_starts]], min_vals)
    return flat.reshape(sh, sw), buffer, buffer_shape


def depth_to_edge_map(
    depth_map: np.ndarray,
    buffer: Optional[np.ndarray],
    buffer_shape: Optional[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Compute a normalised depth-gradient edge map.

    Works in sparse pixel coordinates (~1-2% occupancy from a single
    LiDAR scan) to avoid computing ``inf - inf = nan`` over millions of
    empty pixels at high resolutions. Returns ``(edge_map, buffer,
    buffer_shape)``, see :func:`rasterize_depth_map`.

    The edge value for pixel ``(r, c)`` is the maximum absolute depth
    difference with its valid 4-connected neighbours, normalised by the
    global maximum edge value in the frame.

    Note on the scatter pattern: ``np.maximum(a[idx], b, out=a[idx])``
    silently discards writes because fancy indexing returns a copy.
    The correct pattern is to compute the result first, then scatter.
    """
    depth_map = np.asarray(depth_map, dtype=np.float32)
    h, w = depth_map.shape
    if buffer is None or buffer_shape != (h, w):
        buffer = np.empty((h, w), dtype=np.float32)
        buffer_shape = (h, w)
    buffer.fill(0.0)
    edges = buffer

    vy, vx = np.nonzero(np.isfinite(depth_map) & (depth_map > 0.0))
    if vy.size == 0:
        return edges, buffer, buffer_shape

    # Horizontal pairs → attribute edge to the right pixel.
    r_ok = vx < (w - 1)
    yr, xr = vy[r_ok], vx[r_ok]
    rn_ok = np.isfinite(depth_map[yr, xr + 1]) & (depth_map[yr, xr + 1] > 0.0)
    if rn_ok.any():
        yy, xx = yr[rn_ok], xr[rn_ok]
        edges[yy, xx + 1] = np.abs(depth_map[yy, xx] - depth_map[yy, xx + 1])

    # Vertical pairs → take max with any horizontal edge already written.
    b_ok = vy < (h - 1)
    yb, xb = vy[b_ok], vx[b_ok]
    bn_ok = np.isfinite(depth_map[yb + 1, xb]) & (depth_map[yb + 1, xb] > 0.0)
    if bn_ok.any():
        yy, xx = yb[bn_ok], xb[bn_ok]
        edges[yy + 1, xx] = np.maximum(
            edges[yy + 1, xx], np.abs(depth_map[yy, xx] - depth_map[yy + 1, xx])
        )

    max_val = float(np.max(edges))
    if max_val > 0.0:
        edges /= max_val
    return edges, buffer, buffer_shape
