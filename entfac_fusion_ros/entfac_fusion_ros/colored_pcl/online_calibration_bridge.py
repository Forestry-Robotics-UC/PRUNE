"""Online calibration bridge for colored PCL."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from entfac_fusion_ros.online_calibration import OnlineCalibration, OnlineCalibrationParams


class OnlineCalibrationBridge:
    def __init__(self, node: Any):
        self._node = node
        self._calibration: Optional[OnlineCalibration] = None

    def build(self, projector) -> Optional[OnlineCalibration]:
        if not self._node.online_calibration_enable:
            self._calibration = None
            return None
        params = OnlineCalibrationParams(
            every_n_frames=int(self._node.online_calibration_every_n_frames),
            max_points=int(self._node.online_calibration_max_points),
            step_deg=float(self._node.online_calibration_step_deg),
            learning_rate=float(self._node.online_calibration_learning_rate),
            max_correction_deg=float(self._node.online_calibration_max_correction_deg),
            min_observability=float(self._node.online_calibration_min_observability),
            min_fov_points=int(self._node.online_calibration_min_fov_points),
            edge_threshold=float(self._node.online_calibration_edge_threshold),
            min_sem_edge_density=float(self._node.online_calibration_min_sem_edge_density),
            min_depth_edge_density=float(self._node.online_calibration_min_depth_edge_density),
            log_period_sec=float(self._node.online_calibration_log_period_sec),
            health_ema_alpha=float(self._node.online_calibration_health_ema_alpha),
            health_std_window=int(self._node.online_calibration_health_std_window),
            health_std_scale=float(self._node.online_calibration_health_std_scale),
            health_score_center=float(self._node.online_calibration_health_score_center),
            health_score_scale=float(self._node.online_calibration_health_score_scale),
        )
        self._calibration = OnlineCalibration(params, projector)
        return self._calibration

    @property
    def status(self) -> str:
        if self._calibration is None:
            return self._node._online_calibration_status
        return self._calibration._status

    @property
    def correction_rpy_rad(self) -> np.ndarray:
        if self._calibration is None:
            return np.asarray(self._node._online_calibration_rpy_rad, dtype=np.float64)
        return np.asarray(self._calibration._rpy_rad, dtype=np.float64)

    def update(
        self,
        *,
        points: np.ndarray,
        sem_img,
        sem_type: str,
        intrinsics: np.ndarray,
        camera_T_lidar: np.ndarray,
        image_shape: Tuple[int, int],
    ):
        if self._calibration is None or sem_img is None:
            return camera_T_lidar, None
        corrected, snapshot = self._calibration.update(
            points=points,
            sem_img=sem_img,
            sem_type=sem_type,
            intrinsics=intrinsics,
            camera_T_lidar=camera_T_lidar,
            image_shape=image_shape,
        )
        self._node._online_calibration_status = self._calibration._status
        self._node._online_calibration_rpy_rad = np.asarray(self._calibration._rpy_rad, dtype=np.float64)
        return corrected, snapshot
