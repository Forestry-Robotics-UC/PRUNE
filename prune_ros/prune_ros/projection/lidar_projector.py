#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: GNU General Public License v3.0 (GPL-3.0)
# Repository: PRUNE

"""Pure-numpy LiDAR-to-image projection and semantic fusion.

No ROS imports.  The only public entry point is :class:`LidarProjector`.
All stateful buffers are owned by the projector instance and reused
across frames to avoid per-frame allocation overhead.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from prune_core.colored_pcl.sampling import (
    sample_projected_label_patches,
    sample_projected_rgb_patches,
)
from prune_core.geometry import (
    GeometricReliabilityParams,
    evaluate_geometric_reliability,
)
from prune_core.projection.lidar_projection import project_points_to_image
from prune_core.transforms.se3 import transform_points
from prune_core.types import SemanticPointCloud
from prune_core.utils.masks import (
    apply_invalid_projection_samples,
    filter_invalid_projection_samples,
    sample_invalid_mask,
)
from prune_core.utils.semantics import packed_rgb_to_triplets
from .gate_utils import query_neighborhood_reduce
from .results_overlays import save_frame_overlays

# ---------------------------------------------------------------------------
# Jet-reversed depth LUT: index 0 = red (near 0 m), index 255 = blue (far).
# Built at module level to avoid Python-3 class-scope/comprehension issues.
# ---------------------------------------------------------------------------
_JET_REV_WP = np.array([
    [1.00, 0.00, 0.00],  # red   (nearest)
    [1.00, 0.50, 0.00],  # orange
    [1.00, 1.00, 0.00],  # yellow
    [0.50, 1.00, 0.00],  # yellow-green
    [0.00, 1.00, 0.00],  # green
    [0.00, 1.00, 0.50],  # spring
    [0.00, 1.00, 1.00],  # cyan
    [0.00, 0.50, 1.00],  # sky-blue
    [0.00, 0.00, 1.00],  # blue  (farthest)
], dtype=np.float32)
_JET_REV_T = np.linspace(0.0, 1.0, 9, dtype=np.float32)
_JET_LUT_T = np.linspace(0.0, 1.0, 256, dtype=np.float32)
_JET_REV_LUT: np.ndarray = np.clip(
    np.column_stack([
        np.interp(_JET_LUT_T, _JET_REV_T, _JET_REV_WP[:, c]) for c in range(3)
    ]) * 255, 0, 255,
).astype(np.uint8)
# Copied verbatim from pc2.py — pure numpy, no ROS dependency.
# Keep in sync with pc2.py if either changes.


def _labels_to_uint16(labels: np.ndarray) -> np.ndarray:
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


def __build_label_rgb_float_lut(
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

_log = logging.getLogger(__name__)



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

# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ProjectionQualityResult:
    """Per-point quality gate results for one frame."""

    keep: np.ndarray
    confidence_reject: np.ndarray
    depth_edge_reject: np.ndarray
    occlusion_reject: np.ndarray
    depth_edge_map: Optional[np.ndarray]
    runtime_rasterize_ms: float = 0.0
    runtime_depth_edge_ms: float = 0.0
    runtime_occlusion_ms: float = 0.0
    projection_health_score: float = 0.0
    geometric_reject: Optional[np.ndarray] = None
    runtime_geometric_ms: float = 0.0
    # Per-point enrichment computed by the geometric gate; available for
    # future Tier 2 (per-point reliability state) export to ENTFAC-Mapping.
    surface_normals: Optional[np.ndarray] = None
    geometric_reliability: Optional[np.ndarray] = None


@dataclass
class GateMetrics:
    """Counters and runtimes for one projection frame."""

    num_points_in_front: int = 0
    num_points_projected_in_image: int = 0
    num_rejected_invalid_mask: int = 0
    num_rejected_confidence: int = 0
    num_rejected_depth_edge: int = 0
    num_rejected_occlusion: int = 0
    num_rejected_geometric: int = 0
    num_rejected_other: int = 0
    num_would_hit_invalid_mask: int = 0
    num_would_hit_depth_edge: int = 0
    num_would_fail_occlusion: int = 0
    num_would_hit_geometric: int = 0
    runtime_projection_ms: float = 0.0
    runtime_mask_ms: float = 0.0
    runtime_rasterize_ms: float = 0.0
    runtime_depth_edge_ms: float = 0.0
    runtime_occlusion_ms: float = 0.0
    runtime_geometric_ms: float = 0.0
    runtime_publish_ms: float = 0.0
    projection_health_score: float = 0.0


@dataclass
class ProjectionResult:
    """All outputs of a single :meth:`LidarProjector.process_frame` call."""

    cloud: SemanticPointCloud
    metrics: GateMetrics
    image_shape: Tuple[int, int]
    # FOV-gated points in their original frames (for calibration and debug)
    points_fov: np.ndarray          # LiDAR frame, shape (M, 3)
    points_cam_all: np.ndarray      # camera frame, shape (M, 3)
    # Optional outputs populated when corresponding gates are enabled
    depth_map: Optional[np.ndarray] = None
    edge_map: Optional[np.ndarray] = None
    # Optional outputs for debug publishers
    rgb_values: Optional[np.ndarray] = None
    debug_colors: Optional[np.ndarray] = None
    rolling_shutter_active: bool = False


# ---------------------------------------------------------------------------
# Parameter dataclass
# ---------------------------------------------------------------------------


@dataclass
class LidarProjectorParams:
    """All parameters consumed by :class:`LidarProjector`.

    Build from node parameters; pass to the constructor or :meth:`update_params`.
    """

    # FOV gate
    max_depth_m: Optional[float] = None
    camera_fov_gate_enable: bool = True
    camera_fov_gate_margin_deg: float = 5.0

    # Projection
    rolling_shutter_enable: bool = False
    rolling_shutter_direction: str = "top_to_bottom"
    projection_patch_size: int = 1

    # Quality gates
    projection_occlusion_epsilon_m: float = 0.0
    projection_occlusion_radius_px: int = 0
    projection_reject_depth_edges: bool = False
    projection_depth_edge_thresh: float = 0.15
    projection_depth_edge_radius_px: int = 0
    projection_confidence_min: float = 0.0
    use_invalid_mask: bool = True
    use_depth_edge_rejection: bool = True
    use_occlusion_gate: bool = True
    use_geometric_gate: bool = True

    # Geometric reliability gate (GLIM-inspired local surface cues; default off)
    projection_geometric_enable: bool = False
    geometric_k_neighbors: int = 12
    geometric_radius_m: float = 0.5
    geometric_min_neighbors: int = 5
    geometric_curvature_max: float = 0.12
    geometric_up_labels: Tuple[int, ...] = ()
    geometric_up_max_angle_deg: float = 60.0
    geometric_score_min: float = 0.0
    enable_adaptive_projection_health: bool = False
    projection_health_warn_threshold: float = 0.50
    projection_health_bad_threshold: float = 0.25
    adaptive_confidence_threshold_offset: float = 0.10
    adaptive_depth_edge_threshold_scale: float = 0.80
    adaptive_prefer_suppression_on_bad_health: bool = True

    # RT optimisations
    depth_map_subsample: int = 1       # 1=full-res, 2=half-res, 4=quarter-res depth buffer
    edge_cache_max_age_sec: float = 0.0  # 0=recompute every frame; >0=reuse edge map within window
    use_range_image_edges: str = "auto"  # "auto"=use when cloud is organized, "always", "never"

    # Overlay export (results_overlays.py format — compatible with make_gimp_layers.py)
    overlay_output_dir: str = ""       # empty = disabled
    overlay_output_stride: int = 20    # save every Nth accepted frame
    overlay_dot_radius: int = 2

    # Semantics / output
    include_unlabeled: bool = False
    colorize_labels: bool = False
    semantic_input_type: str = "labels"
    color_map: Dict = field(default_factory=dict)
    random_color_seed: int = 1
    num_labels: int = 0
    debug_project_lidar: bool = False


# ---------------------------------------------------------------------------
# LidarProjector
# ---------------------------------------------------------------------------


class LidarProjector:
    """Pure-numpy LiDAR projection, quality masking, and semantic sampling.

    All persistent buffers (depth map, edge map, RGB LUT) are owned here and
    reused across frames to eliminate per-frame malloc/GC pressure.

    Thread safety: a single instance is not thread-safe.  The ROS node owns
    one instance and calls it from a single callback thread.
    """

    def __init__(self, params: LidarProjectorParams) -> None:
        self._p = params

        # Persistent per-frame buffers — allocated on first use and reused.
        self._depth_buffer: Optional[np.ndarray] = None
        self._depth_buffer_shape: Optional[Tuple[int, int]] = None
        self._edge_buffer: Optional[np.ndarray] = None
        self._edge_buffer_shape: Optional[Tuple[int, int]] = None

        # Edge-map cache (opt 2)
        self._cached_edge_map: Optional[np.ndarray] = None
        self._cached_edge_stamp: float = -np.inf

        # Overlay export state
        self._overlay_frame_idx: int = 0

        # Per-frame context stash (set in process_frame, read in _compute_quality_mask)
        self._last_intrinsics: Optional[np.ndarray] = None
        self._current_subsample: int = 1
        self._frame_cloud_height: int = 0
        self._frame_cloud_width: int = 0
        self._frame_points_all_cam: Optional[np.ndarray] = None
        self._frame_stamp: float = 0.0

        # RGB LUT state
        self._rgb_lut: Optional[np.ndarray] = None
        self._rgb_lut_num_labels: Optional[int] = None
        self._warned_random_palette: bool = False

    def update_params(self, params: LidarProjectorParams) -> None:
        """Replace all tunable parameters atomically (called from live-tuning timer)."""
        self._p = params
        # Invalidate LUT and edge caches when params change.
        self._rgb_lut = None
        self._rgb_lut_num_labels = None
        self._cached_edge_map = None
        self._cached_edge_stamp = -np.inf

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def process_frame(
        self,
        points: np.ndarray,
        labels: Optional[np.ndarray],
        packed_img: Optional[np.ndarray],
        confidence: Optional[np.ndarray],
        projection_invalid_mask: Optional[np.ndarray],
        intrinsics: np.ndarray,
        camera_T_lidar: np.ndarray,
        target_T_lidar: np.ndarray,
        semantic_shape: Tuple[int, int],
        include_rgb: bool,
        rolling_shutter_omega_cam: Optional[np.ndarray] = None,
        rolling_shutter_readout_sec: float = 0.0,
        cloud_height: int = 0,
        cloud_width: int = 0,
        frame_stamp: float = 0.0,
        overlay_packed_img: Optional[np.ndarray] = None,
    ) -> ProjectionResult:
        """Project, sample, and quality-gate one LiDAR + semantic frame.

        Args:
            points: (N, 3) float32, already deskewed and compat-transformed.
            labels: (H, W) label array or None when semantic_input_type='rgb'.
            packed_img: (H, W) packed-uint32 RGB image or None when labels mode.
            confidence: (H, W) float32 confidence map or None.
            projection_invalid_mask: (H, W) bool invalid-pixel mask or None.
            intrinsics: (3, 3) float64 camera intrinsic matrix.
            camera_T_lidar: (4, 4) float64 extrinsic (corrected by calibration).
            target_T_lidar: (4, 4) float64 transform to output frame.
            semantic_shape: (H, W) of the semantic image (for FOV gate).
            include_rgb: whether to attach RGB to the output cloud.
            rolling_shutter_omega_cam: (3,) angular velocity in camera frame,
                pre-looked-up by the node; None disables rolling-shutter correction.
            rolling_shutter_readout_sec: total rolling-shutter readout time in
                seconds; 0 or negative disables correction.

        Returns:
            :class:`ProjectionResult` with cloud, metrics, and optional debug data.
        """
        p = self._p

        # --- FOV gate ----------------------------------------------------
        points_fov = self._process_points(points, camera_T_lidar, intrinsics, semantic_shape)

        # --- Camera-frame transform (reused by rasterizer + occlusion) ---
        points_cam_all = (
            transform_points(camera_T_lidar, points_fov)
            if points_fov.shape[0]
            else np.empty((0, 3), dtype=np.float32)
        )

        # --- Full-cloud camera-frame coords for structured LiDAR edge path ---
        # Only computed for organized clouds (height > 1); avoids overhead for Velodyne.
        p = self._p
        _use_ri = p.use_range_image_edges
        _is_organized = cloud_height > 1 and cloud_width > 0 and points.shape[0] == cloud_height * cloud_width
        _want_ri = (_use_ri == "always") or (_use_ri == "auto" and _is_organized)
        points_all_cam: Optional[np.ndarray] = (
            transform_points(camera_T_lidar, points) if _want_ri else None
        )

        # --- Debug depth colours -----------------------------------------
        debug_colors = (
            self._depth_to_debug_colors(points_cam_all[:, 2]) if p.debug_project_lidar else None
        )

        # --- Determine image shape ---------------------------------------
        if labels is not None:
            h, w = labels.shape
        else:
            h, w = packed_img.shape[:2]  # type: ignore[union-attr]

        # --- Stash per-frame context for _compute_quality_mask -----------
        self._frame_cloud_height: int = cloud_height
        self._frame_cloud_width: int = cloud_width
        self._frame_points_all_cam: Optional[np.ndarray] = points_all_cam
        self._frame_stamp: float = frame_stamp
        self._last_intrinsics: np.ndarray = intrinsics
        self._current_subsample: int = max(1, int(p.depth_map_subsample))
        self._overlay_packed_img = overlay_packed_img if overlay_packed_img is not None else packed_img

        # --- Project + sample + quality mask -----------------------------
        rgb_lut = self._get_rgb_float_lut(labels)
        cloud, rgb_values, proj_metrics, depth_map, edge_map, rs_active = (
            self._project_and_sample(
                points_fov,
                points_cam_all,
                labels,
                packed_img,
                confidence,
                projection_invalid_mask,
                intrinsics,
                camera_T_lidar,
                target_T_lidar,
                rgb_lut,
                include_rgb,
                image_shape=(h, w),
                rolling_shutter_omega_cam=rolling_shutter_omega_cam,
                rolling_shutter_readout_sec=rolling_shutter_readout_sec,
            )
        )

        return ProjectionResult(
            cloud=cloud,
            metrics=proj_metrics,
            image_shape=(h, w),
            points_fov=points_fov,
            points_cam_all=points_cam_all,
            depth_map=depth_map,
            edge_map=edge_map,
            rgb_values=rgb_values,
            debug_colors=debug_colors,
            rolling_shutter_active=rs_active,
        )

    # ------------------------------------------------------------------
    # FOV gate
    # ------------------------------------------------------------------

    def _process_points(
        self,
        points: np.ndarray,
        camera_T_lidar: np.ndarray,
        intrinsics: np.ndarray,
        semantic_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Apply max-depth clip and camera-FOV angular gate."""
        p = self._p
        if p.max_depth_m is not None and points.shape[0]:
            keep = np.linalg.norm(points, axis=1) <= float(p.max_depth_m)
            points = points[keep]
        if not (p.camera_fov_gate_enable and points.shape[0]):
            return points
        sem_h, sem_w = semantic_shape
        fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
        cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
        half_h = float(np.arctan2(max(cx, sem_w - cx), fx))
        half_v = float(np.arctan2(max(cy, sem_h - cy), fy))
        margin = float(np.deg2rad(p.camera_fov_gate_margin_deg))
        pts_cam = transform_points(camera_T_lidar, points)
        z = pts_cam[:, 2]
        in_fov = (
            (z > 0.0)
            & (np.abs(np.arctan2(pts_cam[:, 0], z)) <= half_h + margin)
            & (np.abs(np.arctan2(pts_cam[:, 1], z)) <= half_v + margin)
        )
        return points[in_fov]

    # ------------------------------------------------------------------
    # Project + sample
    # ------------------------------------------------------------------

    def _project_and_sample(
        self,
        points: np.ndarray,
        points_cam_all: np.ndarray,
        labels: Optional[np.ndarray],
        packed_img: Optional[np.ndarray],
        confidence: Optional[np.ndarray],
        projection_invalid_mask: Optional[np.ndarray],
        intrinsics: np.ndarray,
        camera_T_lidar: np.ndarray,
        target_T_lidar: np.ndarray,
        rgb_lut: Optional[np.ndarray],
        include_rgb: bool,
        image_shape: Tuple[int, int],
        rolling_shutter_omega_cam: Optional[np.ndarray],
        rolling_shutter_readout_sec: float,
    ) -> Tuple[
        SemanticPointCloud,
        Optional[np.ndarray],
        GateMetrics,
        Optional[np.ndarray],
        Optional[np.ndarray],
        bool,
    ]:
        """Project, sample, and quality-gate.  Returns (cloud, rgb_values,
        metrics, depth_map, edge_map, rolling_shutter_active).
        """
        p = self._p
        metrics = GateMetrics()
        metrics.num_points_in_front = int(np.count_nonzero(points_cam_all[:, 2] > 0.0))
        h, w = image_shape

        _proj_t0 = time.perf_counter()
        uv, inside, rs_active = self._project_points(
            points,
            intrinsics,
            camera_T_lidar,
            (w, h),
            rolling_shutter_omega_cam=rolling_shutter_omega_cam,
            rolling_shutter_readout_sec=rolling_shutter_readout_sec,
        )

        # Clamp to image bounds (project_points_to_image can return out-of-bounds u/v)
        uv_inside = uv[inside]
        u = uv_inside[:, 0].astype(int)
        v = uv_inside[:, 1].astype(int)
        in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not np.all(in_bounds):
            inside_idx = np.nonzero(inside)[0]
            inside = inside.copy()
            inside[inside_idx[~in_bounds]] = False
            u = u[in_bounds]
            v = v[in_bounds]
        metrics.runtime_projection_ms = 1000.0 * (time.perf_counter() - _proj_t0)
        metrics.num_points_projected_in_image = int(u.shape[0])

        points_selected = points[inside]

        # Pre-computed camera-frame depths for the selected subset — reused in
        # quality mask to avoid a redundant transform_points call.
        points_selected_cam = (
            points_cam_all[inside]
            if points_cam_all.shape[0] == points.shape[0]
            else None
        )

        if labels is not None:
            cloud, rgb_values, metrics, depth_map, edge_map = self._sample_labels(
                points=points,
                points_selected=points_selected,
                points_cam_all=points_cam_all,
                points_selected_cam=points_selected_cam,
                inside=inside,
                u=u,
                v=v,
                labels=labels,
                confidence=confidence,
                projection_invalid_mask=projection_invalid_mask,
                intrinsics=intrinsics,
                camera_T_lidar=camera_T_lidar,
                target_T_lidar=target_T_lidar,
                image_shape=image_shape,
                rgb_lut=rgb_lut,
                include_rgb=include_rgb,
                metrics=metrics,
            )
        else:
            cloud, rgb_values, metrics, depth_map, edge_map = self._sample_rgb(
                points=points,
                points_selected=points_selected,
                points_cam_all=points_cam_all,
                points_selected_cam=points_selected_cam,
                inside=inside,
                u=u,
                v=v,
                packed_img=packed_img,  # type: ignore[arg-type]
                confidence=confidence,
                projection_invalid_mask=projection_invalid_mask,
                intrinsics=intrinsics,
                camera_T_lidar=camera_T_lidar,
                target_T_lidar=target_T_lidar,
                image_shape=image_shape,
                include_rgb=include_rgb,
                metrics=metrics,
            )

        if include_rgb and rgb_values is None and rgb_lut is not None:
            rgb_values = rgb_lut[_labels_to_uint16(cloud.labels)]

        return cloud, rgb_values, metrics, depth_map, edge_map, rs_active

    # ------------------------------------------------------------------
    # Labels path
    # ------------------------------------------------------------------

    def _sample_labels(
        self,
        points: np.ndarray,
        points_selected: np.ndarray,
        points_cam_all: np.ndarray,
        points_selected_cam: Optional[np.ndarray],
        inside: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        labels: np.ndarray,
        confidence: Optional[np.ndarray],
        projection_invalid_mask: Optional[np.ndarray],
        intrinsics: np.ndarray,
        camera_T_lidar: np.ndarray,
        target_T_lidar: np.ndarray,
        image_shape: Tuple[int, int],
        rgb_lut: Optional[np.ndarray],
        include_rgb: bool,
        metrics: GateMetrics,
    ) -> Tuple[
        SemanticPointCloud,
        Optional[np.ndarray],
        GateMetrics,
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        p = self._p
        h, w = image_shape

        if points_selected.shape[0] == 0 and not p.include_unlabeled:
            empty_cloud = SemanticPointCloud(
                np.empty((0, 3), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                None,
            )
            return empty_cloud, None, metrics, None, None

        labels_in, conf_in = sample_projected_label_patches(
            labels, u, v, confidence=confidence, patch_size=p.projection_patch_size
        )

        _mask_t0 = time.perf_counter()
        invalid_samples = (
            sample_invalid_mask(projection_invalid_mask, u, v)
            if projection_invalid_mask is not None
            else None
        )
        metrics.runtime_mask_ms = 1000.0 * (time.perf_counter() - _mask_t0)

        quality, depth_map, edge_map = self._compute_quality_mask(
            points_all=points,
            points_selected=points_selected,
            intrinsics=intrinsics,
            camera_T_lidar=camera_T_lidar,
            image_shape=image_shape,
            u=u,
            v=v,
            point_confidence=conf_in,
            points_cam_all=points_cam_all,
            points_selected_cam=points_selected_cam,
            point_labels=labels_in,
            up_lidar=self._up_in_lidar(target_T_lidar),
        )
        keep_semantics = quality.keep.copy()

        invalid_reject = np.zeros(u.shape[0], dtype=bool)
        if invalid_samples is not None:
            metrics.num_would_hit_invalid_mask = int(np.count_nonzero(invalid_samples))
            if p.use_invalid_mask:
                invalid_reject = invalid_samples
                keep_semantics &= ~invalid_samples

        metrics.num_rejected_invalid_mask = int(np.count_nonzero(invalid_reject))
        metrics.num_rejected_confidence = int(np.count_nonzero(quality.confidence_reject))
        metrics.num_would_hit_depth_edge = int(np.count_nonzero(quality.depth_edge_reject))
        metrics.num_would_fail_occlusion = int(np.count_nonzero(quality.occlusion_reject))
        metrics.num_rejected_depth_edge = (
            metrics.num_would_hit_depth_edge
            if p.projection_reject_depth_edges and p.use_depth_edge_rejection
            else 0
        )
        metrics.num_rejected_occlusion = (
            metrics.num_would_fail_occlusion
            if p.projection_occlusion_epsilon_m > 0.0 and p.use_occlusion_gate
            else 0
        )
        metrics.num_would_hit_geometric = (
            int(np.count_nonzero(quality.geometric_reject))
            if quality.geometric_reject is not None
            else 0
        )
        metrics.num_rejected_geometric = (
            metrics.num_would_hit_geometric
            if p.projection_geometric_enable and p.use_geometric_gate
            else 0
        )
        metrics.runtime_rasterize_ms = quality.runtime_rasterize_ms
        metrics.runtime_depth_edge_ms = quality.runtime_depth_edge_ms
        metrics.runtime_occlusion_ms = quality.runtime_occlusion_ms
        metrics.runtime_geometric_ms = quality.runtime_geometric_ms
        metrics.projection_health_score = float(quality.projection_health_score)

        labels_in = labels_in.astype(np.int64, copy=False)
        if np.any(invalid_reject):
            points_selected, labels_in, conf_in, _ = filter_invalid_projection_samples(
                invalid_reject,
                points=points_selected,
                labels=labels_in,
                confidence=conf_in,
            )
            keep_semantics = keep_semantics[~invalid_reject]

        labels_in, conf_in, _ = apply_invalid_projection_samples(
            ~keep_semantics, labels=labels_in, confidence=conf_in
        )
        points_target_labeled = transform_points(target_T_lidar, points_selected)

        if p.include_unlabeled:
            unlabeled = transform_points(target_T_lidar, points[~inside])
            pts_all = np.vstack((points_target_labeled, unlabeled))
            lbl_all = np.concatenate(
                (labels_in.astype(np.int64),
                 np.full(unlabeled.shape[0], -1, dtype=np.int64))
            )
            conf_all = (
                np.concatenate(
                    (conf_in.astype(np.float32, copy=False),
                     np.zeros(unlabeled.shape[0], dtype=np.float32))
                )
                if conf_in is not None else None
            )
            cloud = SemanticPointCloud(pts_all, lbl_all, conf_all)
        else:
            cloud = SemanticPointCloud(
                points_target_labeled, labels_in.astype(np.int64), conf_in
            )

        return cloud, None, metrics, depth_map, edge_map

    # ------------------------------------------------------------------
    # RGB path
    # ------------------------------------------------------------------

    def _sample_rgb(
        self,
        points: np.ndarray,
        points_selected: np.ndarray,
        points_cam_all: np.ndarray,
        points_selected_cam: Optional[np.ndarray],
        inside: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        packed_img: np.ndarray,
        confidence: Optional[np.ndarray],
        projection_invalid_mask: Optional[np.ndarray],
        intrinsics: np.ndarray,
        camera_T_lidar: np.ndarray,
        target_T_lidar: np.ndarray,
        image_shape: Tuple[int, int],
        include_rgb: bool,
        metrics: GateMetrics,
    ) -> Tuple[
        SemanticPointCloud,
        Optional[np.ndarray],
        GateMetrics,
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        p = self._p

        rgb_values_in, conf_in = sample_projected_rgb_patches(
            packed_img, u, v, confidence=confidence, patch_size=p.projection_patch_size
        )

        _mask_t0 = time.perf_counter()
        invalid_samples = (
            sample_invalid_mask(projection_invalid_mask, u, v)
            if projection_invalid_mask is not None
            else None
        )
        metrics.runtime_mask_ms = 1000.0 * (time.perf_counter() - _mask_t0)

        quality, depth_map, edge_map = self._compute_quality_mask(
            points_all=points,
            points_selected=points_selected,
            intrinsics=intrinsics,
            camera_T_lidar=camera_T_lidar,
            image_shape=image_shape,
            u=u,
            v=v,
            point_confidence=conf_in,
            points_cam_all=points_cam_all,
            points_selected_cam=points_selected_cam,
            up_lidar=self._up_in_lidar(target_T_lidar),
        )
        keep_rgb = quality.keep.copy()

        invalid_reject = np.zeros(u.shape[0], dtype=bool)
        if invalid_samples is not None:
            metrics.num_would_hit_invalid_mask = int(np.count_nonzero(invalid_samples))
            if p.use_invalid_mask:
                invalid_reject = invalid_samples
                keep_rgb &= ~invalid_samples

        metrics.num_rejected_invalid_mask = int(np.count_nonzero(invalid_reject))
        metrics.num_rejected_confidence = int(np.count_nonzero(quality.confidence_reject))
        metrics.num_would_hit_depth_edge = int(np.count_nonzero(quality.depth_edge_reject))
        metrics.num_would_fail_occlusion = int(np.count_nonzero(quality.occlusion_reject))
        metrics.num_rejected_depth_edge = (
            metrics.num_would_hit_depth_edge
            if p.projection_reject_depth_edges and p.use_depth_edge_rejection
            else 0
        )
        metrics.num_rejected_occlusion = (
            metrics.num_would_fail_occlusion
            if p.projection_occlusion_epsilon_m > 0.0 and p.use_occlusion_gate
            else 0
        )
        metrics.num_would_hit_geometric = (
            int(np.count_nonzero(quality.geometric_reject))
            if quality.geometric_reject is not None
            else 0
        )
        metrics.num_rejected_geometric = (
            metrics.num_would_hit_geometric
            if p.projection_geometric_enable and p.use_geometric_gate
            else 0
        )
        metrics.runtime_rasterize_ms = quality.runtime_rasterize_ms
        metrics.runtime_depth_edge_ms = quality.runtime_depth_edge_ms
        metrics.runtime_occlusion_ms = quality.runtime_occlusion_ms
        metrics.runtime_geometric_ms = quality.runtime_geometric_ms

        points_selected, _, conf_in, rgb_values_in = filter_invalid_projection_samples(
            ~keep_rgb,
            points=points_selected,
            confidence=conf_in,
            rgb_values=rgb_values_in,
        )
        if not include_rgb:
            rgb_values_in = None
        elif rgb_values_in is not None:
            rgb_values_in = np.asarray(rgb_values_in, dtype=np.float32)

        points_in_t = transform_points(target_T_lidar, points_selected)
        labels_in = np.full(points_in_t.shape[0], -1, dtype=np.int64)

        if p.include_unlabeled:
            pts_out_t = transform_points(target_T_lidar, points[~inside])
            lbl_out = np.full(pts_out_t.shape[0], -1, dtype=np.int64)
            rgb_all = (
                np.concatenate(
                    (rgb_values_in, np.zeros(pts_out_t.shape[0], dtype=np.float32))
                )
                if include_rgb else None
            )
            conf_all = (
                np.concatenate(
                    (conf_in, np.zeros(pts_out_t.shape[0], dtype=np.float32))
                )
                if conf_in is not None else None
            )
            cloud = SemanticPointCloud(
                np.vstack((points_in_t, pts_out_t)),
                np.concatenate((labels_in, lbl_out)),
                conf_all,
            )
        else:
            cloud = SemanticPointCloud(points_in_t, labels_in, conf_in)
            rgb_all = rgb_values_in

        return cloud, rgb_all, metrics, depth_map, edge_map

    # ------------------------------------------------------------------
    # Geometric gate helpers
    # ------------------------------------------------------------------

    def _up_in_lidar(self, target_T_lidar: np.ndarray) -> Optional[np.ndarray]:
        """Target-frame +z expressed in LiDAR coordinates.

        Used by the semantic-normal consistency check; meaningful only when
        the target frame is roughly gravity-aligned (odom/map/base), which
        the parameter documentation states as a precondition.
        """
        if not self._p.projection_geometric_enable:
            return None
        return np.asarray(target_T_lidar, dtype=np.float64)[:3, :3].T @ np.array(
            [0.0, 0.0, 1.0]
        )

    # ------------------------------------------------------------------
    # Rolling-shutter-aware projection
    # ------------------------------------------------------------------

    def _project_points(
        self,
        points: np.ndarray,
        intrinsics: np.ndarray,
        camera_T_lidar: np.ndarray,
        image_size: Tuple[int, int],
        rolling_shutter_omega_cam: Optional[np.ndarray] = None,
        rolling_shutter_readout_sec: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Project points to image coordinates.

        Returns ``(uv, inside, rolling_shutter_active)`` where ``uv`` has
        shape ``(N, 2)`` and ``inside`` is a boolean mask of length ``N``.
        Rolling-shutter correction is applied when enabled and inputs are valid.
        """
        p = self._p
        w, h = int(image_size[0]), int(image_size[1])

        # Fast path: no rolling shutter correction.
        if (
            not p.rolling_shutter_enable
            or rolling_shutter_readout_sec <= 0.0
            or rolling_shutter_omega_cam is None
            or h <= 1
        ):
            uv, inside = project_points_to_image(
                points, intrinsics, camera_T_lidar, image_size
            )
            return uv, inside, False

        # Rolling-shutter correction: per-row time offset applied as a
        # first-order rotation in camera space.
        points_cam = transform_points(camera_T_lidar, points)
        z = points_cam[:, 2]
        in_front = z > 0
        uv = np.zeros((points_cam.shape[0], 2), dtype=float)
        uv[in_front, 0] = (
            points_cam[in_front, 0] * intrinsics[0, 0] / z[in_front] + intrinsics[0, 2]
        )
        uv[in_front, 1] = (
            points_cam[in_front, 1] * intrinsics[1, 1] / z[in_front] + intrinsics[1, 2]
        )

        row_v = uv[:, 1]
        if p.rolling_shutter_direction == "top_to_bottom":
            row_frac = row_v / float(h - 1)
        else:
            row_frac = (float(h - 1) - row_v) / float(h - 1)
        dt = np.where(np.isfinite(row_frac - 0.5), (row_frac - 0.5) * float(rolling_shutter_readout_sec), 0.0)
        cross = np.cross(rolling_shutter_omega_cam.reshape(1, 3), points_cam)
        points_cam = points_cam + dt.reshape(-1, 1) * cross

        z = points_cam[:, 2]
        in_front = z > 0
        uv = np.zeros((points_cam.shape[0], 2), dtype=float)
        uv[in_front, 0] = (
            points_cam[in_front, 0] * intrinsics[0, 0] / z[in_front] + intrinsics[0, 2]
        )
        uv[in_front, 1] = (
            points_cam[in_front, 1] * intrinsics[1, 1] / z[in_front] + intrinsics[1, 2]
        )
        inside = (
            (uv[:, 0] >= 0)
            & (uv[:, 0] < w)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < h)
            & in_front
        )
        return uv, inside, True

    # ------------------------------------------------------------------
    # Quality mask
    # ------------------------------------------------------------------

    def _compute_quality_mask(
        self,
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
    ) -> Tuple[ProjectionQualityResult, Optional[np.ndarray], Optional[np.ndarray]]:
        """Compute occlusion, depth-edge, geometric, and confidence rejection masks.

        Returns ``(quality_result, depth_map, edge_map)`` where the maps are
        ``None`` when the corresponding gates are disabled. ``point_labels``
        and ``up_lidar`` feed the optional semantic-normal consistency check
        of the geometric reliability gate.
        """
        p = self._p
        u = np.asarray(u, dtype=np.int32).reshape(-1)
        v = np.asarray(v, dtype=np.int32).reshape(-1)
        keep = np.ones(u.shape[0], dtype=bool)
        empty = np.zeros(u.shape[0], dtype=bool)
        if keep.size == 0:
            result = ProjectionQualityResult(
                keep, empty, empty, empty, None, geometric_reject=empty
            )
            return result, None, None

        runtime_rasterize_ms = 0.0
        depth_map = None
        s = max(1, int(p.depth_map_subsample))  # opt 1: subsample factor
        h_img, w_img = image_shape
        sh, sw = h_img // s, w_img // s

        if p.projection_occlusion_epsilon_m > 0.0 or p.projection_reject_depth_edges:
            _rast_t0 = time.perf_counter()

            # Opt 3: structured LiDAR range-image path avoids full-res rasterization.
            _ch = getattr(self, '_frame_cloud_height', 0)
            _cw = getattr(self, '_frame_cloud_width', 0)
            _pts_all_cam = getattr(self, '_frame_points_all_cam', None)
            _use_ri = p.use_range_image_edges
            _is_org = _ch > 1 and _cw > 0 and _pts_all_cam is not None
            _want_ri = (_use_ri == "always") or (_use_ri == "auto" and _is_org)

            if _want_ri and _pts_all_cam is not None:
                depth_map = self._range_image_depth_map(
                    _pts_all_cam, _ch, _cw, (sh, sw),
                )
            else:
                depth_map = self._rasterize_depth_map(
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
            _stamp = getattr(self, '_frame_stamp', 0.0)
            _max_age = float(p.edge_cache_max_age_sec)
            _cache_valid = (
                _max_age > 0.0
                and self._cached_edge_map is not None
                and abs(_stamp - self._cached_edge_stamp) < _max_age
                and self._cached_edge_map.shape == depth_map.shape
            )
            if _cache_valid:
                edge_map = self._cached_edge_map
            else:
                edge_map = self._depth_to_edge_map(depth_map)
                self._cached_edge_map = edge_map
                self._cached_edge_stamp = _stamp
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
            # Fold reliability into the confidence evidence only where the
            # normal is valid; unestimable normals stay neutral instead of
            # zeroing points through the confidence gate.
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

        # --- Overlay export ---
        self._maybe_save_overlays(
            u=u, v=v,
            keep=keep,
            points_selected_cam=points_selected_cam,
            image_shape=image_shape,
        )

        return result, depth_map, edge_map

    # ------------------------------------------------------------------
    # Overlay export (3-layer GIMP format)
    # ------------------------------------------------------------------

    _OVERLAY_MAX_DEPTH_M: float = 18.0  # depth range for colormap

    def _maybe_save_overlays(
        self,
        *,
        u: np.ndarray,
        v: np.ndarray,
        keep: np.ndarray,
        points_selected_cam: Optional[np.ndarray],
        image_shape: Tuple[int, int],
    ) -> None:
        p = self._p
        if not p.overlay_output_dir:
            return
        self._overlay_frame_idx += 1
        if (self._overlay_frame_idx % max(1, p.overlay_output_stride)) != 1:
            return

        try:
            import cv2
        except ImportError:
            _log.warning(
                "overlay_export",
                "cv2 is not available; overlay PNG export is disabled. Install python3-opencv in the runtime image to enable %s.",
                p.overlay_output_dir,
            )
            return

        out_dir = Path(p.overlay_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        h, w = image_shape
        tag = f"frame_{self._overlay_frame_idx:06d}"

        # 1. Base — clean camera RGB (from stashed packed_img)
        packed = getattr(self, '_overlay_packed_img', None)
        if packed is not None:
            base = packed_rgb_to_triplets(packed).astype(np.uint8)
            # When the stashed image is at a higher resolution than the
            # projection space (e.g. downsample_factor > 1), scale u,v up
            # so the dots land at the correct full-res pixel positions.
            img_h, img_w = base.shape[:2]
            if img_h != h or img_w != w:
                u = np.round(u * (img_w / w)).astype(np.int32)
                v = np.round(v * (img_h / h)).astype(np.int32)
                h, w = img_h, img_w
        else:
            base = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.imwrite(str(out_dir / f"{tag}_base.png"), cv2.cvtColor(base, cv2.COLOR_RGB2BGR))

        in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u_b, v_b = u[in_bounds], v[in_bounds]
        keep_b = keep[in_bounds]

        # 2. Projected layer — ALL in-image LiDAR points in green, RGBA transparent background
        proj_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        r = max(0, p.overlay_dot_radius)
        if r == 0:
            proj_rgba[v_b, u_b] = (0, 255, 0, 255)
        else:
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dy * dy + dx * dx <= r * r:
                        vv = np.clip(v_b + dy, 0, h - 1)
                        uu = np.clip(u_b + dx, 0, w - 1)
                        proj_rgba[vv, uu] = (0, 255, 0, 255)
        cv2.imwrite(str(out_dir / f"{tag}_projected_layer.png"),
                    cv2.cvtColor(proj_rgba, cv2.COLOR_RGBA2BGRA))

        # 3. Depth layer — jet-reversed, linear, 0-18 m, RGBA
        if points_selected_cam is not None and points_selected_cam.shape[0] == u.shape[0]:
            z_all = points_selected_cam[:, 2]
            z_b = z_all[in_bounds]
            max_d = float(self._OVERLAY_MAX_DEPTH_M)
            t = np.clip(z_b, 0.0, max_d) / max_d
            idx = np.clip(np.round(t * 255).astype(int), 0, 255)
            colors = _JET_REV_LUT[idx]  # (N, 3) RGB
            depth_rgba = np.zeros((h, w, 4), dtype=np.uint8)
            if r == 0:
                depth_rgba[v_b, u_b, :3] = colors
                depth_rgba[v_b, u_b, 3] = 255
            else:
                for i, (cu, cv_) in enumerate(zip(u_b.tolist(), v_b.tolist())):
                    c = tuple(int(x) for x in colors[i])
                    for dy in range(-r, r + 1):
                        for dx in range(-r, r + 1):
                            if dy * dy + dx * dx <= r * r:
                                depth_rgba[
                                    min(h - 1, max(0, cv_ + dy)),
                                    min(w - 1, max(0, cu + dx))
                                ] = (*c, 255)
            cv2.imwrite(str(out_dir / f"{tag}_depth_layer.png"),
                        cv2.cvtColor(depth_rgba, cv2.COLOR_RGBA2BGRA))

            # 4. Composite — base RGB with depth dots painted on top
            composite = base.copy()
            dot_mask = depth_rgba[:, :, 3] == 255
            composite[dot_mask] = depth_rgba[dot_mask, :3]
            cv2.imwrite(str(out_dir / f"{tag}_composite.png"),
                        cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

    # ------------------------------------------------------------------
    # Depth-map rasterisation (sort-reduce, persistent buffer)
    # ------------------------------------------------------------------

    def _rasterize_depth_map(
        self,
        points: np.ndarray,
        intrinsics: np.ndarray,
        camera_T_lidar: np.ndarray,
        image_shape: Tuple[int, int],
        *,
        points_cam: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fill a per-pixel min-depth buffer from projected LiDAR points.

        Uses a sort-based reduceat pattern for better cache locality than
        ``np.minimum.at``.  The buffer is pre-allocated and reused across
        frames to eliminate per-frame malloc overhead.

        Args:
            points_cam: if provided, skips the ``transform_points`` call
                (caller has already computed camera-frame coordinates).
        """
        h, w = int(image_shape[0]), int(image_shape[1])
        shape = (h, w)
        if self._depth_buffer is None or self._depth_buffer_shape != shape:
            self._depth_buffer = np.empty(shape, dtype=np.float32)
            self._depth_buffer_shape = shape
        self._depth_buffer.fill(np.inf)
        depth = self._depth_buffer

        if points_cam is None:
            points_cam = transform_points(camera_T_lidar, points)
        z = points_cam[:, 2]
        in_front = z > 0.0
        if not np.any(in_front):
            return depth

        pts = points_cam[in_front]
        z = z[in_front]
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        u = (pts[:, 0] * fx / z + cx).astype(np.int32, copy=False)
        v = (pts[:, 1] * fy / z + cy).astype(np.int32, copy=False)
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not np.any(inside):
            return depth

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
        return flat.reshape(h, w)

    # ------------------------------------------------------------------
    # Structured LiDAR: range-image depth map (opt 3)
    # ------------------------------------------------------------------

    def _range_image_depth_map(
        self,
        points_all_cam: np.ndarray,
        cloud_height: int,
        cloud_width: int,
        out_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Build a min-depth buffer from an organized point cloud's range image.

        Operates in ring/column space (O(H_lidar * W_lidar) ~= 131K for a 128-beam
        Ouster) rather than camera-image space (O(H_cam * W_cam) ~= 2.76M for MAPIR
        1440p).  Resolution-agnostic: cost does not grow with camera resolution.

        Args:
            points_all_cam: (cloud_height * cloud_width, 3) all points in camera
                frame, including those outside the FOV.
            out_shape: (sh, sw) output buffer shape — (H//s, W//s) for subsample s.
        """
        sh, sw = out_shape
        if self._depth_buffer is None or self._depth_buffer_shape != (sh, sw):
            self._depth_buffer = np.empty((sh, sw), dtype=np.float32)
            self._depth_buffer_shape = (sh, sw)
        self._depth_buffer.fill(np.inf)
        depth = self._depth_buffer

        z_all = points_all_cam[:, 2]
        in_front = z_all > 0.0
        if not np.any(in_front):
            return depth

        # Project all valid ring/col points to (subsampled) camera UV.
        intrinsics = self._last_intrinsics  # cached below in process_frame
        if intrinsics is None:
            return depth

        pts = points_all_cam[in_front]
        z = z_all[in_front]
        fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
        cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
        u = (pts[:, 0] * fx / z + cx).astype(np.int32, copy=False) // max(1, self._current_subsample)
        v = (pts[:, 1] * fy / z + cy).astype(np.int32, copy=False) // max(1, self._current_subsample)
        inside = (u >= 0) & (u < sw) & (v >= 0) & (v < sh)
        if not np.any(inside):
            return depth

        u, v, z = u[inside], v[inside], z[inside].astype(np.float32, copy=False)
        idx = v * sw + u
        sort_order = np.argsort(idx)
        idx_sorted = idx[sort_order]
        z_sorted = z[sort_order]
        seg_starts = np.concatenate(([0], np.where(np.diff(idx_sorted) != 0)[0] + 1))
        min_vals = np.minimum.reduceat(z_sorted, seg_starts)
        flat = depth.ravel()
        flat[idx_sorted[seg_starts]] = np.minimum(flat[idx_sorted[seg_starts]], min_vals)
        return flat.reshape(sh, sw)

    # ------------------------------------------------------------------
    # Sparse depth-edge map (persistent buffer)
    # ------------------------------------------------------------------

    def _depth_to_edge_map(self, depth_map: np.ndarray) -> np.ndarray:
        """Compute a normalised depth-gradient edge map.

        Works in sparse pixel coordinates (~1-2% occupancy from a single
        LiDAR scan) to avoid computing ``inf − inf = nan`` over millions of
        empty pixels at high resolutions.

        The edge value for pixel ``(r, c)`` is the maximum absolute depth
        difference with its valid 4-connected neighbours, normalised by the
        global maximum edge value in the frame.

        Note on the scatter pattern: ``np.maximum(a[idx], b, out=a[idx])``
        silently discards writes because fancy indexing returns a copy.
        The correct pattern is to compute the result first, then scatter.
        """
        depth_map = np.asarray(depth_map, dtype=np.float32)
        h, w = depth_map.shape
        if self._edge_buffer is None or self._edge_buffer_shape != (h, w):
            self._edge_buffer = np.empty((h, w), dtype=np.float32)
            self._edge_buffer_shape = (h, w)
        self._edge_buffer.fill(0.0)
        edges = self._edge_buffer

        vy, vx = np.nonzero(np.isfinite(depth_map) & (depth_map > 0.0))
        if vy.size == 0:
            return edges

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
        return edges

    # ------------------------------------------------------------------
    # RGB LUT
    # ------------------------------------------------------------------

    def _get_rgb_float_lut(
        self, labels_img: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Return the label-to-RGB float LUT, building or extending it as needed."""
        p = self._p
        if not p.colorize_labels or p.semantic_input_type != "labels":
            return None

        if p.color_map:
            if self._rgb_lut is None or self._rgb_lut_num_labels != -1:
                self._rgb_lut = _build_label_rgb_float_lut(color_map=p.color_map)
                self._rgb_lut_num_labels = -1
            return self._rgb_lut

        n: Optional[int] = int(p.num_labels) if int(p.num_labels) > 0 else None
        if n is None and labels_img is not None:
            n = self._infer_num_labels(labels_img)
        if n is None or n <= 0:
            n = 256
        if self._rgb_lut is None or self._rgb_lut_num_labels != n:
            self._rgb_lut = _build_label_rgb_float_lut(num_labels=n, seed=int(p.random_color_seed))
            self._rgb_lut_num_labels = n
            if not self._warned_random_palette:
                _log.warning(
                    "colorize_labels is true but color_map is empty; using deterministic "
                    "random palette (num_labels=%d seed=%d). Provide color_map for stable colors.",
                    n, int(p.random_color_seed),
                )
                self._warned_random_palette = True
        return self._rgb_lut

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_num_labels(labels_img: np.ndarray) -> int:
        flat = np.asarray(labels_img).reshape(-1)
        flat = flat[flat >= 0]
        return 0 if flat.size == 0 else int(flat.max()) + 1

    @staticmethod
    def _depth_to_debug_colors(depths: np.ndarray) -> Optional[np.ndarray]:
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
