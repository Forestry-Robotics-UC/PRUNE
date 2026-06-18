#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: GNU General Public License v3.0 (GPL-3.0)
# Repository: PRUNE

"""Per-point projection quality gates: occlusion, depth-edge, geometric, confidence.

No ROS imports. Persistent depth/edge buffers stay owned by
:class:`LidarProjector` (avoids per-frame malloc); this module receives the
rasterize/edge-map operations as callables and the edge-map cache as
explicit in/out state, so it can be unit-tested without an instance.
"""

from __future__ import annotations

import time
from typing import Callable, Optional, Tuple

import numpy as np

from prune_core.geometry import GeometricReliabilityParams, evaluate_geometric_reliability
from prune_core.transforms.se3 import transform_points

from .gate_utils import query_neighborhood_reduce
from .types import LidarProjectorParams, ProjectionQualityResult


def projection_health_from_counts(
    *,
    total_points: int,
    in_front_points: int,
    projected_points: int,
    rejection_ratio: float,
) -> float:
    """Conservative [0, 1] projection-health proxy from projection/gate counts."""
    total = max(int(total_points), 1)
    in_front = max(int(in_front_points), 0)
    projected = max(int(projected_points), 0)
    in_front_ratio = float(np.clip(in_front / float(total), 0.0, 1.0))
    in_image_ratio = float(np.clip(projected / float(max(in_front, 1)), 0.0, 1.0))
    rejection_term = 1.0 - float(np.clip(rejection_ratio, 0.0, 1.0))
    return float(np.clip(in_front_ratio * in_image_ratio * rejection_term, 0.0, 1.0))


def compute_quality_mask(
    p: LidarProjectorParams,
    *,
    points_all: np.ndarray,
    points_selected: np.ndarray,
    intrinsics: np.ndarray,
    camera_T_lidar: np.ndarray,
    image_shape: Tuple[int, int],
    u: np.ndarray,
    v: np.ndarray,
    point_confidence: Optional[np.ndarray],
    points_cam_all: Optional[np.ndarray] = None,
    points_selected_cam: Optional[np.ndarray] = None,
    point_labels: Optional[np.ndarray] = None,
    up_lidar: Optional[np.ndarray] = None,
    frame_cloud_height: int = 0,
    frame_cloud_width: int = 0,
    frame_points_all_cam: Optional[np.ndarray] = None,
    frame_stamp: float = 0.0,
    cached_edge_map: Optional[np.ndarray] = None,
    cached_edge_stamp: float = -np.inf,
    rasterize_depth_map: Optional[Callable[..., np.ndarray]] = None,
    range_image_depth_map: Optional[Callable[..., np.ndarray]] = None,
    depth_to_edge_map: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[ProjectionQualityResult, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], float]:
    """Compute occlusion, depth-edge, geometric, and confidence rejection masks.

    Returns ``(quality_result, depth_map, edge_map, new_cached_edge_map,
    new_cached_edge_stamp)``. The maps are ``None`` when the corresponding
    gates are disabled. ``point_labels`` and ``up_lidar`` feed the optional
    semantic-normal consistency check of the geometric reliability gate.
    Callers own the persistent depth/edge buffers; pass them in via
    ``rasterize_depth_map``/``range_image_depth_map``/``depth_to_edge_map``
    and store the returned edge-cache values back for the next frame.
    """
    u = np.asarray(u, dtype=np.int32).reshape(-1)
    v = np.asarray(v, dtype=np.int32).reshape(-1)
    keep = np.ones(u.shape[0], dtype=bool)
    empty = np.zeros(u.shape[0], dtype=bool)
    if keep.size == 0:
        result = ProjectionQualityResult(
            keep, empty, empty, empty, None, geometric_reject=empty
        )
        return result, None, None, cached_edge_map, cached_edge_stamp

    runtime_rasterize_ms = 0.0
    depth_map = None
    s = max(1, int(p.depth_map_subsample))  # opt 1: subsample factor
    h_img, w_img = image_shape
    sh, sw = h_img // s, w_img // s

    if p.projection_occlusion_epsilon_m > 0.0 or p.projection_reject_depth_edges:
        _rast_t0 = time.perf_counter()

        # Opt 3: structured LiDAR range-image path avoids full-res rasterization.
        _is_org = frame_cloud_height > 1 and frame_cloud_width > 0 and frame_points_all_cam is not None
        _use_ri = p.use_range_image_edges
        _want_ri = (_use_ri == "always") or (_use_ri == "auto" and _is_org)

        if _want_ri and frame_points_all_cam is not None:
            depth_map = range_image_depth_map(
                frame_points_all_cam, frame_cloud_height, frame_cloud_width, (sh, sw),
            )
        else:
            depth_map = rasterize_depth_map(
                points_all, intrinsics, camera_T_lidar, (sh, sw),
                points_cam=points_cam_all,
            )
        runtime_rasterize_ms = 1000.0 * (time.perf_counter() - _rast_t0)

    # Subsampled UV coordinates used for all depth-buffer queries.
    us = (u // s).clip(0, sw - 1) if s > 1 else u
    vs = (v // s).clip(0, sh - 1) if s > 1 else v

    effective_conf = (
        np.asarray(point_confidence, dtype=np.float32).reshape(-1)
        if point_confidence is not None else None
    )

    pre_health = projection_health_from_counts(
        total_points=int(points_all.shape[0]),
        in_front_points=(
            int(np.count_nonzero(points_cam_all[:, 2] > 0.0))
            if points_cam_all is not None and points_cam_all.shape[0]
            else int(points_selected.shape[0])
        ),
        projected_points=int(u.shape[0]),
        rejection_ratio=0.0,
    )
    adaptive_bad = bool(
        p.enable_adaptive_projection_health
        and pre_health < float(p.projection_health_bad_threshold)
    )
    effective_confidence_min = float(p.projection_confidence_min)
    effective_depth_edge_thresh = float(p.projection_depth_edge_thresh)
    if adaptive_bad:
        effective_confidence_min = min(
            1.0,
            effective_confidence_min + float(p.adaptive_confidence_threshold_offset),
        )
        effective_depth_edge_thresh = float(
            np.clip(
                effective_depth_edge_thresh * float(p.adaptive_depth_edge_threshold_scale),
                0.0,
                1.0,
            )
        )

    # --- Occlusion gate ---
    runtime_occlusion_ms = 0.0
    occlusion_reject = np.zeros(keep.shape[0], dtype=bool)
    if p.projection_occlusion_epsilon_m > 0.0 and depth_map is not None:
        _occ_t0 = time.perf_counter()
        point_depth = (
            np.asarray(points_selected_cam[:, 2], dtype=np.float32)
            if points_selected_cam is not None
            else np.asarray(
                transform_points(camera_T_lidar, points_selected)[:, 2],
                dtype=np.float32,
            )
        )
        nearest_depth = query_neighborhood_reduce(
            depth_map, us, vs, p.projection_occlusion_radius_px, "min"
        )
        depth_margin = np.asarray(point_depth - nearest_depth, dtype=np.float32)
        occlusion_reject = (
            ~np.isfinite(nearest_depth)
            | (depth_margin > float(p.projection_occlusion_epsilon_m))
        )
        if p.use_occlusion_gate:
            keep &= ~occlusion_reject
        occ_conf = np.clip(
            1.0 - np.maximum(depth_margin, 0.0) / float(p.projection_occlusion_epsilon_m),
            0.0, 1.0,
        ).astype(np.float32, copy=False)
        effective_conf = (
            occ_conf if effective_conf is None else np.minimum(effective_conf, occ_conf)
        )
        runtime_occlusion_ms = 1000.0 * (time.perf_counter() - _occ_t0)

    # --- Depth-edge gate ---
    edge_map = None
    runtime_depth_edge_ms = 0.0
    depth_edge_reject = np.zeros(keep.shape[0], dtype=bool)
    if p.projection_reject_depth_edges and depth_map is not None:
        _edge_t0 = time.perf_counter()
        # Opt 2: reuse cached edge map if within max age window.
        _max_age = float(p.edge_cache_max_age_sec)
        _cache_valid = (
            _max_age > 0.0
            and cached_edge_map is not None
            and abs(frame_stamp - cached_edge_stamp) < _max_age
            and cached_edge_map.shape == depth_map.shape
        )
        if _cache_valid:
            edge_map = cached_edge_map
        else:
            edge_map = depth_to_edge_map(depth_map)
            cached_edge_map = edge_map
            cached_edge_stamp = frame_stamp
        edge_values = query_neighborhood_reduce(
            edge_map, us, vs, p.projection_depth_edge_radius_px, "max"
        )
        depth_edge_reject = edge_values >= effective_depth_edge_thresh
        if p.use_depth_edge_rejection:
            keep &= ~depth_edge_reject
        edge_conf = np.clip(1.0 - edge_values, 0.0, 1.0).astype(np.float32, copy=False)
        effective_conf = (
            edge_conf if effective_conf is None else np.minimum(effective_conf, edge_conf)
        )
        runtime_depth_edge_ms = 1000.0 * (time.perf_counter() - _edge_t0)

    # --- Geometric reliability gate (GLIM-inspired local surface cues) ---
    runtime_geometric_ms = 0.0
    geometric_reject = np.zeros(keep.shape[0], dtype=bool)
    surface_normals: Optional[np.ndarray] = None
    geometric_reliability: Optional[np.ndarray] = None
    if p.projection_geometric_enable and points_selected.shape[0]:
        _geo_t0 = time.perf_counter()
        geo = evaluate_geometric_reliability(
            points_selected,
            params=GeometricReliabilityParams(
                k_neighbors=int(p.geometric_k_neighbors),
                radius_m=float(p.geometric_radius_m),
                min_neighbors=int(p.geometric_min_neighbors),
                curvature_max=float(p.geometric_curvature_max),
                up_labels=tuple(p.geometric_up_labels),
                up_max_angle_deg=float(p.geometric_up_max_angle_deg),
                score_min=float(p.geometric_score_min),
            ),
            reference_points=points_all,
            labels=point_labels,
            up_vector=up_lidar,
        )
        geometric_reject = geo.reject
        surface_normals = geo.normals
        geometric_reliability = geo.reliability
        if p.use_geometric_gate:
            keep &= ~geometric_reject
        if p.geometric_fold_into_confidence and p.use_geometric_gate:
            # Optional coupling (default off): fold reliability into the
            # confidence evidence only where the normal is valid;
            # unestimable normals stay neutral instead of zeroing points
            # through the confidence gate. Kept behind its own flag, and
            # inert in suppression mode (~use_geometric_gate=false), so
            # the G5 ablation row isolates the geometric gate and
            # suppression mode never changes the output cloud.
            geo_conf = np.where(
                geo.normal_valid, geo.reliability, 1.0
            ).astype(np.float32, copy=False)
            effective_conf = (
                geo_conf if effective_conf is None else np.minimum(effective_conf, geo_conf)
            )
        runtime_geometric_ms = 1000.0 * (time.perf_counter() - _geo_t0)

    # --- Confidence gate ---
    confidence_reject = np.zeros(keep.shape[0], dtype=bool)
    if effective_confidence_min > 0.0 and effective_conf is not None:
        if effective_conf.shape[0] == keep.shape[0]:
            confidence_reject = effective_conf < effective_confidence_min
            keep &= ~confidence_reject

    active_rejection_ratio = 0.0
    if keep.shape[0] > 0:
        active_rejection_ratio = float(np.count_nonzero(~keep)) / float(keep.shape[0])
    health_score = projection_health_from_counts(
        total_points=int(points_all.shape[0]),
        in_front_points=(
            int(np.count_nonzero(points_cam_all[:, 2] > 0.0))
            if points_cam_all is not None and points_cam_all.shape[0]
            else int(points_selected.shape[0])
        ),
        projected_points=int(u.shape[0]),
        rejection_ratio=active_rejection_ratio,
    )

    result = ProjectionQualityResult(
        keep=keep,
        confidence_reject=confidence_reject,
        depth_edge_reject=depth_edge_reject,
        occlusion_reject=occlusion_reject,
        depth_edge_map=edge_map,
        runtime_rasterize_ms=runtime_rasterize_ms,
        runtime_depth_edge_ms=runtime_depth_edge_ms,
        runtime_occlusion_ms=runtime_occlusion_ms,
        projection_health_score=health_score,
        geometric_reject=geometric_reject,
        runtime_geometric_ms=runtime_geometric_ms,
        surface_normals=surface_normals,
        geometric_reliability=geometric_reliability,
    )

    return result, depth_map, edge_map, cached_edge_map, cached_edge_stamp
