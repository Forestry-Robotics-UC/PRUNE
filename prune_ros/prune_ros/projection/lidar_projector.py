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
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from prune_core.colored_pcl.sampling import (
    sample_projected_label_patches,
    sample_projected_rgb_patches,
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
from . import colorize
from .depth_rasterize import depth_to_edge_map, range_image_depth_map, rasterize_depth_map
from .quality_gates import compute_quality_mask, projection_health_from_counts
from .results_overlays import save_frame_overlays
from .types import GateMetrics, LidarProjectorParams, ProjectionQualityResult, ProjectionResult

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


_LOG = logging.getLogger(__name__)


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
            colorize.depth_to_debug_colors(points_cam_all[:, 2]) if p.debug_project_lidar else None
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
        cloud, rgb_values, proj_metrics, depth_map, edge_map, rs_active, gate_debug_colors, uv_inside = (
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
            gate_debug_colors=gate_debug_colors,
            uv_inside=uv_inside,
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
            cloud, rgb_values, metrics, depth_map, edge_map, gate_debug_colors = self._sample_labels(
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
            cloud, rgb_values, metrics, depth_map, edge_map, gate_debug_colors = self._sample_rgb(
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
            rgb_values = rgb_lut[colorize.labels_to_uint16(cloud.labels)]

        # Pixel coordinates aligned to the same in-bounds `inside` subset as
        # gate_debug_colors (u/v are already bounds-filtered above).
        uv_inside = (
            np.stack([u, v], axis=1).astype(np.float64)
            if gate_debug_colors is not None
            else None
        )
        return cloud, rgb_values, metrics, depth_map, edge_map, rs_active, gate_debug_colors, uv_inside

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
            return empty_cloud, None, metrics, None, None, None

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

        gate_debug_colors = (
            colorize.build_gate_debug_colors(
                n=u.shape[0],
                keep=quality.keep,
                invalid_reject=invalid_reject,
                depth_edge_reject=quality.depth_edge_reject,
                occlusion_reject=quality.occlusion_reject,
                geometric_reject=quality.geometric_reject,
            )
            if p.debug_project_lidar else None
        )
        return cloud, None, metrics, depth_map, edge_map, gate_debug_colors

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

        gate_debug_colors = (
            colorize.build_gate_debug_colors(
                n=u.shape[0],
                keep=quality.keep,
                invalid_reject=invalid_reject,
                depth_edge_reject=quality.depth_edge_reject,
                occlusion_reject=quality.occlusion_reject,
                geometric_reject=quality.geometric_reject,
            )
            if p.debug_project_lidar else None
        )
        return cloud, rgb_all, metrics, depth_map, edge_map, gate_debug_colors

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

        Thin adapter: the gate math lives in :func:`quality_gates.compute_quality_mask`
        as a free function; this method supplies the per-frame context and
        persistent buffers/cache that instance owns, and stores the updated
        edge-map cache back.
        """
        result, depth_map, edge_map, self._cached_edge_map, self._cached_edge_stamp = (
            compute_quality_mask(
                self._p,
                points_all=points_all,
                points_selected=points_selected,
                intrinsics=intrinsics,
                camera_T_lidar=camera_T_lidar,
                image_shape=image_shape,
                u=u,
                v=v,
                point_confidence=point_confidence,
                points_cam_all=points_cam_all,
                points_selected_cam=points_selected_cam,
                point_labels=point_labels,
                up_lidar=up_lidar,
                frame_cloud_height=self._frame_cloud_height,
                frame_cloud_width=self._frame_cloud_width,
                frame_points_all_cam=self._frame_points_all_cam,
                frame_stamp=self._frame_stamp,
                cached_edge_map=self._cached_edge_map,
                cached_edge_stamp=self._cached_edge_stamp,
                rasterize_depth_map=self._rasterize_depth_map,
                range_image_depth_map=self._range_image_depth_map,
                depth_to_edge_map=self._depth_to_edge_map,
            )
        )

        # --- Overlay export ---
        self._maybe_save_overlays(
            u=u, v=v,
            keep=result.keep,
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
            _LOG.warning(
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

        Thin adapter over :func:`depth_rasterize.rasterize_depth_map`; owns
        the persistent reuse buffer and stores it back after the call.
        """
        depth, self._depth_buffer, self._depth_buffer_shape = rasterize_depth_map(
            points, intrinsics, camera_T_lidar, image_shape,
            self._depth_buffer, self._depth_buffer_shape,
            points_cam=points_cam,
        )
        return depth

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

        Thin adapter over :func:`depth_rasterize.range_image_depth_map`; owns
        the persistent reuse buffer and stores it back after the call.
        """
        depth, self._depth_buffer, self._depth_buffer_shape = range_image_depth_map(
            points_all_cam, cloud_height, cloud_width, out_shape,
            self._depth_buffer, self._depth_buffer_shape,
            self._last_intrinsics,  # cached in process_frame
            self._current_subsample,
        )
        return depth

    # ------------------------------------------------------------------
    # Sparse depth-edge map (persistent buffer)
    # ------------------------------------------------------------------

    def _depth_to_edge_map(self, depth_map: np.ndarray) -> np.ndarray:
        """Compute a normalised depth-gradient edge map.

        Thin adapter over :func:`depth_rasterize.depth_to_edge_map`; owns
        the persistent reuse buffer and stores it back after the call.
        """
        edges, self._edge_buffer, self._edge_buffer_shape = depth_to_edge_map(
            depth_map, self._edge_buffer, self._edge_buffer_shape,
        )
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
                self._rgb_lut = colorize.build_label_rgb_float_lut(color_map=p.color_map)
                self._rgb_lut_num_labels = -1
            return self._rgb_lut

        n: Optional[int] = int(p.num_labels) if int(p.num_labels) > 0 else None
        if n is None and labels_img is not None:
            n = colorize.infer_num_labels(labels_img)
        if n is None or n <= 0:
            n = 256
        if self._rgb_lut is None or self._rgb_lut_num_labels != n:
            self._rgb_lut = colorize.build_label_rgb_float_lut(num_labels=n, seed=int(p.random_color_seed))
            self._rgb_lut_num_labels = n
            if not self._warned_random_palette:
                _LOG.warning(
                    "colorize_labels is true but color_map is empty; using deterministic "
                    "random palette (num_labels=%d seed=%d). Provide color_map for stable colors.",
                    n, int(p.random_color_seed),
                )
                self._warned_random_palette = True
        return self._rgb_lut
