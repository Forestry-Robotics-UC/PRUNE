#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ENTFAC Sensor Fusion implementation.
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
#   Pure numpy helpers for LiDAR projection quality masks.

"""Pure numpy helpers for LiDAR projection quality masks."""

from __future__ import annotations

import numpy as np

try:
    from scipy import ndimage as _scipy_ndimage
except ImportError:
    _scipy_ndimage = None


def query_neighborhood_reduce(
    image: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    radius_px: int,
    op: str,
) -> np.ndarray:
    """Min or max of ``image`` values in a ``(2r+1)²`` box per query point."""
    h, w = image.shape[:2]
    r = int(radius_px)
    if r <= 0:
        return image[v, u].astype(np.float32, copy=False)
    offsets = np.arange(-r, r + 1, dtype=np.int32)
    dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
    dy = dy.ravel()
    dx = dx.ravel()
    vv = np.clip(v[:, None] + dy[None, :], 0, h - 1)
    uu = np.clip(u[:, None] + dx[None, :], 0, w - 1)
    samples = image[vv, uu]
    if op == "min":
        return samples.min(axis=1).astype(np.float32, copy=False)
    if op == "max":
        return samples.max(axis=1).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported op: {op!r}")


def reduce_image_neighborhood(
    image: np.ndarray,
    *,
    radius_px: int,
    op: str,
) -> np.ndarray:
    """Dense H×W neighbourhood min/max filter (scipy fast path or numpy fallback)."""
    image = np.asarray(image, dtype=np.float32)
    radius_px = int(radius_px)
    if radius_px <= 0:
        return image
    if op not in ("min", "max"):
        raise ValueError(f"Unsupported op: {op!r}")
    size = 2 * radius_px + 1
    if _scipy_ndimage is not None:
        fn = (
            _scipy_ndimage.minimum_filter if op == "min"
            else _scipy_ndimage.maximum_filter
        )
        return fn(image, size=size, mode="nearest")
    h, w = image.shape[:2]
    if op == "min":
        out = np.full_like(image, np.inf, dtype=np.float32)
        reducer = np.minimum
    else:
        out = np.zeros_like(image, dtype=np.float32)
        reducer = np.maximum
    for dy in range(-radius_px, radius_px + 1):
        dst_y0, dst_y1 = max(0, dy), min(h, h + dy)
        src_y0, src_y1 = max(0, -dy), min(h, h - dy)
        if dst_y0 >= dst_y1:
            continue
        for dx in range(-radius_px, radius_px + 1):
            dst_x0, dst_x1 = max(0, dx), min(w, w + dx)
            src_x0, src_x1 = max(0, -dx), min(w, w - dx)
            if dst_x0 >= dst_x1:
                continue
            reducer(
                out[dst_y0:dst_y1, dst_x0:dst_x1],
                image[src_y0:src_y1, src_x0:src_x1],
                out=out[dst_y0:dst_y1, dst_x0:dst_x1],
            )
    return out
