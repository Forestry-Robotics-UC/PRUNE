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
#   Single-frame semantic fusion pipelines (depth-based and LiDAR projection).

"""Single-frame semantic fusion pipelines."""

import logging
from typing import Optional, Tuple

import numpy as np

from entfac_fusion_core.projection.depth import depth_to_points
from entfac_fusion_core.projection.lidar_projection import project_points_to_image
from entfac_fusion_core.transforms.se3 import transform_points
from entfac_fusion_core.types.observations import (
    DepthObservation,
    PointObservation,
    SemanticObservation,
    SemanticPointCloud,
)
from entfac_fusion_core.utils.validation import (
    flatten_masked,
    require_homogeneous_transform,
)
from entfac_fusion_core.utils.semantics import packed_rgb_to_triplets

LOGGER = logging.getLogger(__name__)


def _normalize_patch_size(patch_size: int) -> int:
    patch_size = int(patch_size)
    if patch_size < 1:
        raise ValueError("patch_size must be >= 1")
    if patch_size % 2 == 0:
        raise ValueError("patch_size must be odd")
    return patch_size


def _gather_patch_samples(
    image: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    patch_size: int,
    confidence: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    image = np.asarray(image)
    u = np.asarray(u, dtype=np.int32).reshape(-1)
    v = np.asarray(v, dtype=np.int32).reshape(-1)
    if u.shape != v.shape:
        raise ValueError("u and v must have the same shape")
    if image.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {image.shape}")
    if confidence is not None:
        confidence = np.asarray(confidence, dtype=np.float32)
        if confidence.shape != image.shape:
            raise ValueError("confidence must match image shape")

    patch_size = _normalize_patch_size(patch_size)
    radius = patch_size // 2
    h, w = image.shape

    # Build all (dy, dx) offsets vectorised; ordering matches the original
    # dy-outer dx-inner loop so column indices stay identical.
    offsets = np.arange(-radius, radius + 1, dtype=np.int32)
    dy_offsets, dx_offsets = np.meshgrid(offsets, offsets, indexing="ij")
    dy_offsets = dy_offsets.ravel()  # (k,)
    dx_offsets = dx_offsets.ravel()  # (k,)

    vv = v[:, None] + dy_offsets[None, :]  # (n, k)
    uu = u[:, None] + dx_offsets[None, :]  # (n, k)
    valid = (vv >= 0) & (vv < h) & (uu >= 0) & (uu < w)  # (n, k)

    # Clamp for safe gather; downstream code always masks with `valid`
    # so values at out-of-bounds positions are never used.
    vv_c = np.clip(vv, 0, h - 1)
    uu_c = np.clip(uu, 0, w - 1)

    samples = image[vv_c, uu_c]  # (n, k)
    conf_samples = (
        confidence[vv_c, uu_c].astype(np.float32, copy=False)
        if confidence is not None
        else None
    )

    return samples, valid, conf_samples


def sample_projected_label_patches(
    labels_img: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    confidence: Optional[np.ndarray] = None,
    patch_size: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample label patches around projected pixels using a robust majority vote."""
    labels_img = np.asarray(labels_img)
    if labels_img.ndim != 2:
        raise ValueError(f"labels_img must be 2D, got shape {labels_img.shape}")
    if labels_img.dtype.kind not in ("i", "u"):
        raise ValueError(f"labels_img must be integer, got dtype {labels_img.dtype}")

    u = np.asarray(u, dtype=np.int32).reshape(-1)
    v = np.asarray(v, dtype=np.int32).reshape(-1)
    if u.shape != v.shape:
        raise ValueError("u and v must have the same shape")

    patch_size = _normalize_patch_size(patch_size)
    if patch_size == 1:
        labels = labels_img[v, u].astype(np.int64, copy=False)
        if confidence is None:
            patch_conf = np.ones(labels.shape[0], dtype=np.float32)
        else:
            patch_conf = np.clip(
                np.asarray(confidence[v, u], dtype=np.float32),
                0.0,
                1.0,
            )
        return labels, patch_conf

    samples, valid, conf_samples = _gather_patch_samples(
        labels_img,
        u,
        v,
        patch_size=patch_size,
        confidence=confidence,
    )

    n = int(u.size)
    no_valid = ~valid.any(axis=1)  # (n,) — points with no valid neighbours

    row_ids, col_ids = np.nonzero(valid)  # (m,)
    flat_vals = samples[row_ids, col_ids].astype(np.int64)

    if flat_vals.size == 0:
        return np.full(n, -1, dtype=np.int64), np.zeros(n, dtype=np.float32)

    if conf_samples is not None:
        flat_w = np.clip(conf_samples[row_ids, col_ids].astype(np.float64), 0.0, None)
        # Fall back to uniform weights for rows where every weight is zero.
        row_wsum = np.zeros(n, dtype=np.float64)
        np.add.at(row_wsum, row_ids, flat_w)
        fallback = row_wsum[row_ids] == 0.0
        if fallback.any():
            flat_w = flat_w.copy()
            flat_w[fallback] = 1.0
    else:
        flat_w = np.ones(len(row_ids), dtype=np.float64)

    # Shift labels to [0, n_bins) so we can use bincount on a flat 2-D index.
    label_min = int(flat_vals.min())
    vals_shifted = flat_vals - label_min
    n_bins = int(vals_shifted.max()) + 1

    linear_idx = row_ids * n_bins + vals_shifted
    flat_scores = np.bincount(linear_idx, weights=flat_w, minlength=n * n_bins)
    scores = flat_scores.reshape(n, n_bins)

    best = np.argmax(scores, axis=1)  # (n,)
    labels_out = (best + label_min).astype(np.int64)
    labels_out[no_valid] = -1

    total_w = scores.sum(axis=1)
    conf_out = np.where(total_w > 0.0, scores[np.arange(n), best] / total_w, 0.0).astype(
        np.float32
    )
    conf_out[no_valid] = 0.0

    return labels_out, conf_out


def sample_projected_rgb_patches(
    packed_img: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    confidence: Optional[np.ndarray] = None,
    patch_size: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample RGB patches around projected pixels using a robust channel-wise median."""
    packed_img = np.asarray(packed_img, dtype=np.uint32)
    if packed_img.ndim != 2:
        raise ValueError(f"packed_img must be 2D, got shape {packed_img.shape}")

    u = np.asarray(u, dtype=np.int32).reshape(-1)
    v = np.asarray(v, dtype=np.int32).reshape(-1)
    if u.shape != v.shape:
        raise ValueError("u and v must have the same shape")

    patch_size = _normalize_patch_size(patch_size)
    if patch_size == 1:
        colors = packed_img[v, u].astype("<u4", copy=False)
        rgb_values = colors.view("<f4")
        if confidence is None:
            patch_conf = np.ones(colors.shape[0], dtype=np.float32)
        else:
            patch_conf = np.clip(
                np.asarray(confidence[v, u], dtype=np.float32),
                0.0,
                1.0,
            )
        return rgb_values, patch_conf

    samples, valid, conf_samples = _gather_patch_samples(
        packed_img,
        u,
        v,
        patch_size=patch_size,
        confidence=confidence,
    )
    rgb_triplets = packed_rgb_to_triplets(samples).astype(np.float32, copy=False)
    masked = np.where(valid[:, :, None], rgb_triplets, np.nan)
    median_rgb = np.nanmedian(masked, axis=1)
    median_rgb = np.nan_to_num(median_rgb, nan=0.0)
    rgb_u8 = np.clip(np.rint(median_rgb), 0.0, 255.0).astype(np.uint8, copy=False)
    packed = (
        (rgb_u8[:, 0].astype(np.uint32) << 16)
        | (rgb_u8[:, 1].astype(np.uint32) << 8)
        | rgb_u8[:, 2].astype(np.uint32)
    )

    diff = np.abs(rgb_triplets - median_rgb[:, None, :]).sum(axis=2)
    diff = np.where(valid, diff / (255.0 * 3.0), 0.0)
    counts = np.maximum(np.count_nonzero(valid, axis=1), 1)
    patch_conf = 1.0 - np.clip(diff.sum(axis=1) / counts, 0.0, 1.0)
    patch_conf = patch_conf.astype(np.float32, copy=False)

    if conf_samples is not None:
        counts_f = counts.astype(np.float32, copy=False)
        conf_mean = conf_samples.sum(axis=1) / counts_f
        patch_conf = patch_conf * np.clip(conf_mean, 0.0, 1.0)

    return packed.astype("<u4", copy=False).view("<f4"), patch_conf


def fuse_depth_semantics(
    semantic: SemanticObservation,
    depth: DepthObservation,
    intrinsics: np.ndarray,
    target_T_depth: np.ndarray,
    include_unlabeled: bool = False,
    max_depth_m: Optional[float] = None,
) -> SemanticPointCloud:
    """Fuse aligned semantic + depth into a semantic point cloud in the target frame.

    This function is stateless and processes a single frame. It does not perform
    any temporal fusion or mapping updates.

    Args:
        semantic: Semantic labels (and optional confidence) with shape ``(H, W)``.
        depth: Depth image aligned with ``semantic.labels`` and shape ``(H, W)``.
            Depth values are expected in meters.
        intrinsics: Camera intrinsics matrix ``K`` with shape ``(3, 3)``.
        target_T_depth: Homogeneous transform with shape ``(4, 4)`` mapping points
            from the depth frame into the desired target/output frame.
        include_unlabeled: If true, keep points whose semantic label is negative
            (e.g. ``-1``). If false, drop them.
        max_depth_m: Optional max depth in meters; values beyond are dropped.

    Returns:
        SemanticPointCloud in the target frame:
        - ``points_xyz`` has shape ``(N, 3)`` in target coordinates (meters)
        - ``labels`` has shape ``(N,)`` and dtype ``int64``
        - ``confidence`` has shape ``(N,)`` (float32) if provided, else ``None``

    Raises:
        ValueError: If inputs have invalid shapes/dtypes or transforms are invalid.
    """
    LOGGER.debug(
        "fuse_depth_semantics: labels=%s depth=%s include_unlabeled=%s",
        semantic.labels.shape,
        depth.depth.shape,
        include_unlabeled,
    )
    semantic.validate()
    depth.validate()
    target_T_depth = require_homogeneous_transform(target_T_depth)

    points_cam, valid_mask = depth_to_points(
        depth.depth, intrinsics, max_depth_m=max_depth_m
    )
    if points_cam.shape[0] == 0:
        LOGGER.warning("Depth fusion received no valid points; returning empty PCL")
        return SemanticPointCloud(
            np.empty((0, 3)), np.empty((0,), dtype=np.int64), None
        )

    labels = flatten_masked(semantic.labels, valid_mask)
    if not include_unlabeled:
        keep = labels >= 0
        points_cam = points_cam[keep]
        labels = labels[keep]
        conf = (
            flatten_masked(semantic.confidence, valid_mask)[keep]
            if semantic.confidence is not None
            else None
        )
        LOGGER.debug(
            "Depth fusion keeping %d labeled points (filtered unlabeled)",
            points_cam.shape[0],
        )
    else:
        conf = (
            flatten_masked(semantic.confidence, valid_mask)
            if semantic.confidence is not None
            else None
        )
        LOGGER.debug(
            "Depth fusion keeping %d points (including unlabeled)",
            points_cam.shape[0],
        )

    if conf is not None:
        conf = np.asarray(conf, dtype=np.float32)

    points_target = transform_points(target_T_depth, points_cam)
    pcl = SemanticPointCloud(points_target, labels.astype(np.int64), conf)
    pcl.validate()
    return pcl


def fuse_lidar_semantics(
    semantic: SemanticObservation,
    lidar_points: PointObservation,
    intrinsics: np.ndarray,
    camera_T_lidar: np.ndarray,
    target_T_lidar: np.ndarray,
    include_unlabeled: bool = False,
    patch_size: int = 1,
) -> SemanticPointCloud:
    """Project LiDAR into an image, sample semantics, and emit a semantic point cloud.

    This function is stateless and processes a single frame. It does not perform
    any temporal fusion or mapping updates.

    Args:
        semantic: Semantic labels (and optional confidence) with shape ``(H, W)``.
        lidar_points: LiDAR points with shape ``(N, 3)`` in the LiDAR frame.
        intrinsics: Camera intrinsics matrix ``K`` with shape ``(3, 3)``.
        camera_T_lidar: Homogeneous transform with shape ``(4, 4)`` mapping LiDAR
            points into the camera frame.
        target_T_lidar: Homogeneous transform with shape ``(4, 4)`` mapping LiDAR
            points into the desired target/output frame.
        include_unlabeled: If true, also output points that project outside the
            image bounds with label ``-1`` and confidence ``0`` (if confidence is
            enabled). If false, drop them.
        patch_size: Odd patch size for robust label sampling around each projected
            pixel (1=center pixel, 3=3x3, 5=5x5, ...).

    Returns:
        SemanticPointCloud in the target frame:
        - ``points_xyz`` has shape ``(M, 3)`` in target coordinates (meters)
        - ``labels`` has shape ``(M,)`` and dtype ``int64``
        - ``confidence`` has shape ``(M,)`` (float32) if provided, else ``None``

    Raises:
        ValueError: If inputs have invalid shapes/dtypes or transforms are invalid.
        RuntimeError: If internal projection bookkeeping becomes inconsistent.
    """
    LOGGER.debug(
        "fuse_lidar_semantics: labels=%s points=%s include_unlabeled=%s",
        semantic.labels.shape,
        lidar_points.points_xyz.shape,
        include_unlabeled,
    )
    semantic.validate()
    lidar_points.validate()
    camera_T_lidar = require_homogeneous_transform(camera_T_lidar)
    target_T_lidar = require_homogeneous_transform(target_T_lidar)

    h, w = semantic.labels.shape
    uv, inside = project_points_to_image(
        lidar_points.points_xyz, intrinsics, camera_T_lidar, (w, h)
    )

    uv_inside = uv[inside]
    # Use truncation (floor for positive coords) to keep indices in-bounds.
    # Rounding can produce u==w or v==h for points close to the image border
    # (e.g., v=479.6 with h=480 -> round(v)=480), causing IndexError.
    u = uv_inside[:, 0].astype(int)
    v = uv_inside[:, 1].astype(int)
    in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not np.all(in_bounds):
        dropped = int(np.count_nonzero(~in_bounds))
        LOGGER.debug(
            "LiDAR projection produced %d/%d borderline indices out of bounds after truncation; dropping them",
            dropped,
            int(u.size),
        )
        inside_idx = np.nonzero(inside)[0]
        inside = inside.copy()
        inside[inside_idx[~in_bounds]] = False
        uv_inside = uv_inside[in_bounds]
        u = u[in_bounds]
        v = v[in_bounds]

    labeled_points = lidar_points.points_xyz[inside]
    if labeled_points.shape[0] == 0 and not include_unlabeled:
        LOGGER.warning("LiDAR fusion found no points inside image bounds")
        return SemanticPointCloud(
            np.empty((0, 3)), np.empty((0,), dtype=np.int64), None
        )

    if labeled_points.shape[0] != u.shape[0]:
        raise RuntimeError(
            "internal error: projected uv count does not match filtered point count"
        )
    labels, confidences = sample_projected_label_patches(
        semantic.labels,
        u,
        v,
        confidence=semantic.confidence,
        patch_size=patch_size,
    )

    points_target_labeled = transform_points(target_T_lidar, labeled_points)

    if include_unlabeled:
        unlabeled_points = lidar_points.points_xyz[~inside]
        points_all = np.vstack(
            (points_target_labeled, transform_points(target_T_lidar, unlabeled_points))
        )
        labels_all = np.concatenate(
            (
                labels.astype(np.int64),
                np.full(unlabeled_points.shape[0], -1, dtype=np.int64),
            )
        )
        if confidences is not None:
            conf_all = np.concatenate(
                (
                    confidences,
                    np.zeros(
                        unlabeled_points.shape[0], dtype=confidences.dtype
                    ),
                )
            )
        else:
            conf_all = None
    else:
        points_all = points_target_labeled
        labels_all = labels.astype(np.int64)
        conf_all = confidences

    pcl = SemanticPointCloud(points_all, labels_all, conf_all)
    pcl.validate()
    LOGGER.debug(
        "LiDAR fusion produced %d labeled points (%d unlabeled kept=%s)",
        labels.shape[0],
        points_all.shape[0] - labels.shape[0],
        include_unlabeled,
    )
    return pcl
