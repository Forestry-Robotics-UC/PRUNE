#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: GNU General Public License v3.0 (GPL-3.0)
# Repository: PRUNE

"""Label/depth colorization helpers for LiDAR projection debug output.

No ROS imports; pure numpy. Stateless — the RGB LUT cache itself stays on
:class:`LidarProjector` since it persists across frames.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def labels_to_uint16(labels: np.ndarray) -> np.ndarray:
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
    labels = np.arange(65536, dtype=np.uint32)
    packed = np.zeros_like(labels, dtype=np.uint32)

    def _hash_palette(ids: np.ndarray) -> np.ndarray:
        r = (ids * 37) & 0xFF
        g = (ids * 17) & 0xFF
        b = (ids * 73) & 0xFF
        return (r << 16) | (g << 8) | b

    if num_labels is not None:
        n = int(num_labels)
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
            packed[int(label_id)] = ((rr & 0xFF) << 16) | ((gg & 0xFF) << 8) | (bb & 0xFF)

    return packed.astype("<u4", copy=False).view("<f4")


def infer_num_labels(labels_img: np.ndarray) -> int:
    flat = np.asarray(labels_img).reshape(-1)
    flat = flat[flat >= 0]
    return 0 if flat.size == 0 else int(flat.max()) + 1


def build_gate_debug_colors(
    n: int,
    keep: np.ndarray,
    invalid_reject: np.ndarray,
    depth_edge_reject: np.ndarray,
    occlusion_reject: np.ndarray,
    geometric_reject: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-point gate-status colours for the live debug overlay.

    Priority (last write wins): accepted=green, geometric (G5)=cyan,
    occlusion (G3)=magenta, depth-edge (G2)=orange, invalid-mask
    (G1)=red. The reject masks are would-hit masks, so suppressed gates
    still show their hits in the overlay.
    """
    colors = np.full((n, 3), 60, dtype=np.uint8)
    truly_accepted = keep & ~invalid_reject
    colors[truly_accepted] = (0, 255, 0)
    if geometric_reject is not None and geometric_reject.shape[0] == n:
        colors[geometric_reject] = (0, 255, 255)
    colors[occlusion_reject] = (255, 0, 255)
    colors[depth_edge_reject] = (255, 165, 0)
    colors[invalid_reject] = (255, 0, 0)
    return colors


def depth_to_debug_colors(depths: np.ndarray) -> Optional[np.ndarray]:
    """Map per-point depth values to a red-to-blue colour gradient."""
    if depths is None or depths.size == 0:
        return None
    depths = np.asarray(depths, dtype=np.float32).reshape(-1)
    valid = np.isfinite(depths) & (depths > 0)
    if not np.any(valid):
        return None
    dmin = float(np.nanmin(depths[valid]))
    dmax = float(np.nanpercentile(depths[valid], 95))
    if dmax <= dmin:
        dmax = dmin + 1e-3
    t = np.clip((depths - dmin) / (dmax - dmin), 0.0, 1.0)
    r = ((1.0 - t) * 255.0).astype(np.uint8, copy=False)  # near=red
    g = np.zeros_like(r)
    b = (t * 255.0).astype(np.uint8, copy=False)           # far=blue
    return np.stack((r, g, b), axis=-1)
