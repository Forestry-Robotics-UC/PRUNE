#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Duda Andrada
# License: GNU General Public License v3.0 (GPL-3.0)

"""Lightweight online LiDAR-camera rotational misalignment estimator.

Pure numpy — no ROS imports.  Entry point: :class:`OnlineCalibration`.

Algorithm
---------
Every ``every_n_frames`` calls to :meth:`update` the class:

1. Applies the current RPY correction to obtain *corrected_camera_T_lidar*.
2. Builds semantic-edge and depth-edge maps.
3. Evaluates an edge-alignment score (normalised cross-correlation proxy).
4. When the scene is sufficiently observable, estimates per-axis gradient and
   curvature via central finite differences around the current correction and
   applies a bounded gradient-ascent update.
5. Updates the :class:`OnlineCalibrationHealth` estimator from
   ``entfac_fusion_core``.

The correction is rotation-only (no translation) by design.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from entfac_fusion_core.calibration import CalibrationHealthSnapshot, OnlineCalibrationHealth
from entfac_fusion_core.projection.lidar_projection import project_points_to_image
from entfac_fusion_core.utils.validation import require_homogeneous_transform

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helper — also used by TrackedReprojection
# ---------------------------------------------------------------------------


def compute_semantic_edge_map(sem_img: np.ndarray, sem_type: str) -> np.ndarray:
    """Compute a float32 edge map from a label image or RGB image.

    For label images: binary boundary where adjacent pixels differ.
    For RGB images: normalised gradient-magnitude map.
    """
    sem_type = (sem_type or "").strip().lower()
    if sem_type == "labels":
        labels = np.asarray(sem_img)
        if labels.ndim == 3:
            labels = labels[:, :, 0]
        if labels.ndim != 2:
            return np.zeros(labels.shape[:2], dtype=np.float32)
        edges = np.zeros_like(labels, dtype=np.float32)
        edges[:, 1:] = (labels[:, 1:] != labels[:, :-1]).astype(np.float32)
        edges[1:, :] = np.maximum(
            edges[1:, :], (labels[1:, :] != labels[:-1, :]).astype(np.float32)
        )
        return edges

    rgb = np.asarray(sem_img)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return np.zeros(rgb.shape[:2], dtype=np.float32)
    rgb = rgb[:, :, :3].astype(np.float32, copy=False)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    edges = np.zeros_like(gray, dtype=np.float32)
    edges[:, 1:] += np.abs(gray[:, 1:] - gray[:, :-1])
    edges[1:, :] += np.abs(gray[1:, :] - gray[:-1, :])
    max_val = float(np.max(edges)) if edges.size else 0.0
    if max_val > 0.0:
        edges /= max_val
    return edges


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------


@dataclass
class OnlineCalibrationParams:
    """All parameters for :class:`OnlineCalibration`."""

    every_n_frames: int = 10
    max_points: int = 8000
    step_deg: float = 0.20
    learning_rate: float = 0.25
    max_correction_deg: float = 3.0
    min_observability: float = 0.15
    min_fov_points: int = 500
    edge_threshold: float = 0.20
    min_sem_edge_density: float = 0.010
    min_depth_edge_density: float = 0.010
    log_period_sec: float = 2.0
    # OnlineCalibrationHealth construction params
    health_ema_alpha: float = 0.15
    health_std_window: int = 40
    health_std_scale: float = 0.08
    health_score_center: float = 0.25
    health_score_scale: float = 0.10


# ---------------------------------------------------------------------------
# OnlineCalibration
# ---------------------------------------------------------------------------


class OnlineCalibration:
    """Lightweight per-frame rotational extrinsic health estimator and corrector.

    Args:
        params: All tunable parameters.
        projector: A :class:`~entfac_fusion_ros.lidar_projector.LidarProjector`
            instance used to build depth and edge maps.  Passed by reference —
            the projector's persistent buffers are reused.
    """

    def __init__(self, params: OnlineCalibrationParams, projector) -> None:
        self._p = params
        self._projector = projector

        self._rpy_rad: np.ndarray = np.zeros(3, dtype=np.float64)
        self._correction_uncertainty: float = 1.0
        self._update_counter: int = 0
        self._last_snapshot: Optional[CalibrationHealthSnapshot] = None
        self._last_log_at: float = 0.0
        self._status: str = "active"

        self._health = OnlineCalibrationHealth(
            ema_alpha=float(params.health_ema_alpha),
            std_window=int(params.health_std_window),
            std_scale=float(params.health_std_scale),
            score_center=float(params.health_score_center),
            score_scale=float(params.health_score_scale),
            min_observability=float(params.min_observability),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        *,
        points: np.ndarray,
        sem_img: np.ndarray,
        sem_type: str,
        intrinsics: np.ndarray,
        camera_T_lidar: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, Optional[CalibrationHealthSnapshot]]:
        """Run one calibration step and return the corrected extrinsic.

        Returns:
            ``(corrected_camera_T_lidar, snapshot)`` where *snapshot* is
            non-None only on frames where the health estimator was updated
            (every ``every_n_frames`` calls).
        """
        p = self._p
        corrected = self._compose_corrected(camera_T_lidar)

        self._update_counter += 1
        if self._update_counter % int(p.every_n_frames) != 0:
            return corrected, None

        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            self._status = "active (no lidar points)"
            return corrected, None

        # Subsample for the alignment evaluation.
        if points.shape[0] > p.max_points:
            stride = max(1, int(np.ceil(points.shape[0] / float(p.max_points))))
            pts = points[::stride]
        else:
            pts = points

        sem_edges = compute_semantic_edge_map(sem_img, sem_type)
        score0, depth_edges = self._evaluate_alignment(
            pts, sem_edges, intrinsics, corrected, image_shape
        )

        h, w = int(image_shape[0]), int(image_shape[1])
        _, inside = project_points_to_image(pts, intrinsics, corrected, (w, h))
        in_fov = int(np.count_nonzero(inside))
        observability = self._observability(
            in_fov, int(pts.shape[0]), sem_edges, depth_edges
        )

        if in_fov >= p.min_fov_points and observability >= p.min_observability:
            delta = np.asarray(self._rpy_rad, dtype=np.float64).copy()
            step = float(np.deg2rad(p.step_deg))
            lr = float(p.learning_rate)
            max_corr = float(np.deg2rad(p.max_correction_deg))
            axis_sigma = np.full(3, max_corr, dtype=np.float64)

            for axis in range(3):
                plus, minus = delta.copy(), delta.copy()
                plus[axis] += step
                minus[axis] -= step
                sp, _ = self._evaluate_alignment(
                    pts, sem_edges, intrinsics,
                    self._compose_corrected(camera_T_lidar, plus), image_shape
                )
                sm, _ = self._evaluate_alignment(
                    pts, sem_edges, intrinsics,
                    self._compose_corrected(camera_T_lidar, minus), image_shape
                )
                grad = float((sp - sm) / (2.0 * step))
                hess = float((sp - 2.0 * score0 + sm) / (step * step))
                if np.isfinite(hess) and hess < -1e-6:
                    update = lr * grad / (abs(hess) + 1e-6)
                    axis_sigma[axis] = float(np.sqrt(1.0 / (abs(hess) + 1e-6)))
                else:
                    update = lr * grad
                    axis_sigma[axis] = max_corr
                if np.isfinite(update):
                    delta[axis] += update

            delta = np.clip(delta, -max_corr, max_corr)
            self._rpy_rad = delta
            corrected = self._compose_corrected(camera_T_lidar)
            self._correction_uncertainty = float(
                np.clip(np.mean(axis_sigma) / (max_corr + 1e-6), 0.0, 1.0)
            )
            self._status = "active"
        else:
            self._status = "active (observability-gated)"
            self._correction_uncertainty = float(
                np.clip(self._correction_uncertainty + 0.05, 0.0, 1.0)
            )

        snapshot = self._health.update(
            score_raw=float(score0),
            observability=float(observability),
            correction_rpy_rad=self._rpy_rad,
            correction_uncertainty=float(self._correction_uncertainty),
        )
        self._last_snapshot = snapshot

        now = time.time()
        period = float(p.log_period_sec)
        if period == 0.0 or (now - self._last_log_at) >= period:
            _log.info(
                "online_calibration: health=%.3f uncertainty=%.3f score=%.3f "
                "score_ema=%.3f obs=%.3f corr_deg=[%.3f %.3f %.3f] in_fov=%d/%d status=%s",
                float(snapshot.health), float(snapshot.uncertainty),
                float(snapshot.score_raw), float(snapshot.score_ema),
                float(snapshot.observability),
                float(snapshot.correction_roll_deg),
                float(snapshot.correction_pitch_deg),
                float(snapshot.correction_yaw_deg),
                in_fov, int(pts.shape[0]), self._status,
            )
            self._last_log_at = now

        return corrected, snapshot

    def get_corrected_camera_T_lidar(self, base: np.ndarray) -> np.ndarray:
        """Return the current RPY correction applied to *base*."""
        return self._compose_corrected(base)

    @property
    def last_snapshot(self) -> Optional[CalibrationHealthSnapshot]:
        return self._last_snapshot

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compose_corrected(
        self,
        camera_T_lidar: np.ndarray,
        rpy_rad: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if rpy_rad is None:
            rpy_rad = self._rpy_rad
        rpy_rad = np.asarray(rpy_rad, dtype=np.float64).reshape(3)
        delta = np.eye(4, dtype=np.float64)
        delta[:3, :3] = _rotation_from_rpy(
            float(rpy_rad[0]), float(rpy_rad[1]), float(rpy_rad[2])
        )
        return require_homogeneous_transform(
            delta @ np.asarray(camera_T_lidar, dtype=np.float64)
        )

    def _evaluate_alignment(
        self,
        points: np.ndarray,
        sem_edges: np.ndarray,
        intrinsics: np.ndarray,
        camera_T_lidar: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> Tuple[float, np.ndarray]:
        depth_map = self._projector._rasterize_depth_map(
            points, intrinsics, camera_T_lidar, image_shape
        )
        depth_edges = self._projector._depth_to_edge_map(depth_map)
        score = _edge_alignment_score(sem_edges, depth_edges)
        return float(score), depth_edges

    def _observability(
        self,
        in_fov: int,
        total: int,
        sem_edges: np.ndarray,
        depth_edges: np.ndarray,
    ) -> float:
        p = self._p
        if total <= 0:
            return 0.0
        fov_ratio = float(np.clip(in_fov / float(total), 0.0, 1.0))
        thr = float(p.edge_threshold)
        sem_density = float(np.mean(np.asarray(sem_edges, dtype=np.float32) >= thr))
        dep_density = float(np.mean(np.asarray(depth_edges, dtype=np.float32) >= thr))
        sem_term = float(np.clip(sem_density / float(p.min_sem_edge_density), 0.0, 1.0))
        dep_term = float(np.clip(dep_density / float(p.min_depth_edge_density), 0.0, 1.0))
        return float(np.clip((fov_ratio * sem_term * dep_term) ** (1.0 / 3.0), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Module-level pure helpers
# ---------------------------------------------------------------------------


def _edge_alignment_score(sem: np.ndarray, dep: np.ndarray) -> float:
    sem = np.asarray(sem, dtype=np.float32)
    dep = np.asarray(dep, dtype=np.float32)
    if sem.shape != dep.shape or sem.size == 0:
        return 0.0
    denom = float(np.sqrt(np.sum(sem * sem) * np.sum(dep * dep))) + 1e-6
    return float(np.sum(sem * dep) / denom)


def _rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx
