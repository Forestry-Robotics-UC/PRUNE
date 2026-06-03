#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Duda Andrada
# License: GNU General Public License v3.0 (GPL-3.0)

"""Stateful feature-tracked LiDAR reprojection diagnostics.

Uses OpenCV for optical-flow tracking.  No ROS imports.
Entry point: :class:`TrackedReprojection`.

Intended primarily for offline rosbag review or focused validation runs —
heavier than the online edge-alignment score.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from entfac_fusion_ros.online_calibration import compute_semantic_edge_map

_log = logging.getLogger(__name__)


@dataclass
class TrackedReprojectionResult:
    """Output of a single :meth:`TrackedReprojection.update` call."""

    error_px: float            # median reprojection error in pixels
    overlay_img: np.ndarray    # (H, W, 3) uint8 RGB overlay for publishing
    num_tracks: int
    num_depth_edge_pixels: int


@dataclass
class TrackedReprojectionParams:
    max_corners: int = 300
    quality_level: float = 0.01
    min_distance_px: float = 8.0
    min_tracks: int = 80
    fb_thresh_px: float = 1.5
    depth_edge_thresh: float = 0.15
    min_image_edge: float = 0.05
    log_period_sec: float = 2.0


class TrackedReprojection:
    """Maintain a set of image features and track them forward with LK optical
    flow, measuring how well they align with LiDAR depth edges.

    Args:
        params: Tunable parameters.
        ensure_cv2: Zero-argument callable that returns a loaded ``cv2``
            module or raises.  The node passes ``self._ensure_cv2``; tests
            can pass a lambda.
    """

    def __init__(self, params: TrackedReprojectionParams, ensure_cv2) -> None:
        self._p = params
        self._ensure_cv2 = ensure_cv2

        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts: Optional[np.ndarray] = None   # (N, 1, 2) float32
        self._last_log_at: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        *,
        sem_img: np.ndarray,
        sem_type: str,
        depth_map: np.ndarray,
        depth_edges: np.ndarray,
    ) -> Optional[TrackedReprojectionResult]:
        """Track features and compute reprojection error against depth edges.

        Args:
            sem_img: Current semantic image (labels or RGB).
            sem_type: ``"labels"`` or ``"rgb"``.
            depth_map: LiDAR depth map from the projector (H×W float32).
            depth_edges: Normalised depth-edge map from the projector.

        Returns:
            :class:`TrackedReprojectionResult` when enough tracks are
            available, or ``None`` when the state is reset.
        """
        try:
            cv2 = self._ensure_cv2()
        except Exception:  # noqa: BLE001
            return None

        base_rgb, gray = self._to_base_and_gray(sem_img, sem_type)
        if gray.size == 0:
            return None

        # First frame or shape change: seed the tracker.
        if (
            self._prev_gray is None
            or self._prev_pts is None
            or self._prev_gray.shape != gray.shape
        ):
            self._reset(gray, cv2)
            return None

        prev_pts = np.asarray(self._prev_pts, dtype=np.float32)
        if prev_pts.size == 0:
            self._reset(gray, cv2)
            return None

        lk_kwargs = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, prev_pts, None, **lk_kwargs
        )
        if next_pts is None or status is None:
            self._reset(gray, cv2)
            return None

        back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, self._prev_gray, next_pts, None, **lk_kwargs
        )
        if back_pts is None or back_status is None:
            self._reset(gray, cv2)
            return None

        prev_xy = prev_pts.reshape(-1, 2)
        next_xy = np.asarray(next_pts, dtype=np.float32).reshape(-1, 2)
        back_xy = np.asarray(back_pts, dtype=np.float32).reshape(-1, 2)
        fb_err = np.linalg.norm(prev_xy - back_xy, axis=1)

        valid = (
            status.reshape(-1).astype(bool)
            & back_status.reshape(-1).astype(bool)
            & np.isfinite(next_xy).all(axis=1)
            & np.isfinite(fb_err)
            & (fb_err <= float(self._p.fb_thresh_px))
            & (next_xy[:, 0] >= 0.0)
            & (next_xy[:, 0] < float(gray.shape[1]))
            & (next_xy[:, 1] >= 0.0)
            & (next_xy[:, 1] < float(gray.shape[0]))
        )
        if not np.any(valid):
            self._reset(gray, cv2)
            return None

        prev_xy = prev_xy[valid]
        next_xy = next_xy[valid]
        px = np.clip(np.round(next_xy[:, 0]).astype(np.int32), 0, gray.shape[1] - 1)
        py = np.clip(np.round(next_xy[:, 1]).astype(np.int32), 0, gray.shape[0] - 1)

        sem_edges = compute_semantic_edge_map(sem_img, sem_type)
        img_edge_strength = np.asarray(sem_edges[py, px], dtype=np.float32)
        edge_keep = img_edge_strength >= float(self._p.min_image_edge)
        if np.any(edge_keep):
            prev_xy = prev_xy[edge_keep]
            next_xy = next_xy[edge_keep]
            px = px[edge_keep]
            py = py[edge_keep]

        if next_xy.shape[0] == 0:
            self._reset(gray, cv2)
            return None

        edge_mask = depth_edges >= float(self._p.depth_edge_thresh)
        if not np.any(edge_mask):
            self._reset(gray, cv2)
            return None

        dt_input = np.ones(edge_mask.shape, dtype=np.uint8)
        dt_input[edge_mask] = 0
        dist_map = cv2.distanceTransform(dt_input, cv2.DIST_L2, 3)
        reproj_err = np.asarray(dist_map[py, px], dtype=np.float32)
        error_value = float(np.median(reproj_err)) if reproj_err.size else float("nan")

        # Build overlay image for publishing.
        overlay = np.ascontiguousarray(base_rgb.copy())
        edge_tint = np.array([0, 255, 255], dtype=np.float32)
        overlay_f = overlay.astype(np.float32, copy=False)
        overlay_f[edge_mask] = 0.45 * overlay_f[edge_mask] + 0.55 * edge_tint
        overlay = np.clip(overlay_f, 0.0, 255.0).astype(np.uint8, copy=False)
        for p_prev, p_cur, err in zip(prev_xy, next_xy, reproj_err):
            color = (0, 255, 0) if err <= 1.0 else (255, 255, 0) if err <= 3.0 else (255, 0, 0)
            cv2.line(
                overlay,
                (int(round(p_prev[0])), int(round(p_prev[1]))),
                (int(round(p_cur[0])), int(round(p_cur[1]))),
                color, 1, lineType=cv2.LINE_AA,
            )
            cv2.circle(
                overlay,
                (int(round(p_cur[0])), int(round(p_cur[1]))),
                2, color, thickness=-1, lineType=cv2.LINE_AA,
            )

        # Update tracked points for next frame.
        refreshed = next_xy.reshape(-1, 1, 2).astype(np.float32)
        if refreshed.shape[0] < int(self._p.min_tracks):
            extra = self._detect_features(gray, cv2, existing=refreshed)
            if extra.size:
                refreshed = np.concatenate((refreshed, extra), axis=0)
        refreshed = refreshed[: int(self._p.max_corners)]
        self._prev_gray = gray
        self._prev_pts = refreshed

        now = time.time()
        period = float(self._p.log_period_sec)
        if period == 0.0 or (now - self._last_log_at) >= period:
            _log.info(
                "tracked_reprojection: median_err_px=%.3f tracks=%d depth_edge_px=%d",
                float(error_value if np.isfinite(error_value) else 0.0),
                int(reproj_err.size),
                int(np.count_nonzero(edge_mask)),
            )
            self._last_log_at = now

        return TrackedReprojectionResult(
            error_px=float(error_value if np.isfinite(error_value) else 0.0),
            overlay_img=overlay,
            num_tracks=int(reproj_err.size),
            num_depth_edge_pixels=int(np.count_nonzero(edge_mask)),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reset(self, gray: np.ndarray, cv2) -> None:
        gray = np.ascontiguousarray(np.asarray(gray, dtype=np.uint8))
        self._prev_gray = gray
        self._prev_pts = self._detect_features(gray, cv2)

    def _detect_features(
        self,
        gray: np.ndarray,
        cv2,
        existing: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        p = self._p
        gray = np.ascontiguousarray(np.asarray(gray, dtype=np.uint8))
        remaining = max(0, p.max_corners - (int(existing.shape[0]) if existing is not None and existing.size else 0))
        if remaining <= 0:
            return np.empty((0, 1, 2), dtype=np.float32)
        mask = np.full(gray.shape, 255, dtype=np.uint8)
        if existing is not None and existing.size:
            radius = max(2, int(round(float(p.min_distance_px))))
            for pt in np.asarray(existing, dtype=np.float32).reshape(-1, 2):
                cv2.circle(mask, (int(round(pt[0])), int(round(pt[1]))), radius, 0, -1)
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=remaining,
            qualityLevel=float(p.quality_level),
            minDistance=float(p.min_distance_px),
            mask=mask,
        )
        if pts is None:
            return np.empty((0, 1, 2), dtype=np.float32)
        return np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)

    @staticmethod
    def _to_base_and_gray(
        sem_img: np.ndarray, sem_type: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert semantic image to (base_rgb, gray) for feature tracking."""
        sem_type = (sem_type or "").strip().lower()
        if sem_type == "labels":
            labels = np.asarray(sem_img)
            if labels.ndim == 3:
                labels = labels[:, :, 0]
            labels = labels.astype(np.float32, copy=False)
            if labels.size == 0:
                return np.zeros((0, 0, 3), dtype=np.uint8), np.zeros((0, 0), dtype=np.uint8)
            flat = labels.reshape(-1)
            valid = np.isfinite(flat)
            scale = float(np.max(flat[valid])) if np.any(valid) else 0.0
            if scale <= 0.0:
                scale = 1.0
            gray = np.clip(255.0 * labels / scale, 0.0, 255.0).astype(np.uint8)
            return np.stack((gray, gray, gray), axis=-1), gray

        rgb = np.asarray(sem_img)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            h, w = rgb.shape[:2]
            return np.zeros((h, w, 3), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)
        base = np.ascontiguousarray(rgb[:, :, :3].astype(np.uint8))
        gray_f = 0.299 * base[:, :, 0] + 0.587 * base[:, :, 1] + 0.114 * base[:, :, 2]
        gray = np.clip(gray_f, 0.0, 255.0).astype(np.uint8)
        return base, gray
