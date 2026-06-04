"""Initialization helpers for colored PCL node startup assembly."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo

from entfac_fusion_core.utils.validation import require_homogeneous_transform
from entfac_fusion_ros.prune.config import (
    load_calibration_config,
    load_color_config,
    load_debug_config,
    load_experiment_config,
    load_ply_config,
    load_projection_config,
    load_sync_config,
)
from entfac_fusion_ros.tf_utils import format_matrix


class NodeInitializer:
    def __init__(self, node: Any):
        self._node = node

    def load_runtime_config(self) -> None:
        node = self._node
        sync_config = load_sync_config(node)
        node._apply_loaded_config(sync_config)
        color_config = load_color_config(node)
        node._apply_loaded_config(color_config)
        projection_config = load_projection_config(node)
        node._apply_loaded_config(projection_config)
        debug_config = load_debug_config(node)
        node._apply_loaded_config(debug_config)
        experiment_config = load_experiment_config(node)
        node._apply_loaded_config(experiment_config)
        calibration_config = load_calibration_config(node)
        node._apply_loaded_config(calibration_config)
        node._online_calibration_requested = bool(node.online_calibration_enable)
        node._online_calibration_status = (
            "requested" if node._online_calibration_requested else "disabled"
        )
        node._undistort_requested = bool(node.undistort_semantic)
        node._undistort_status = "requested" if node._undistort_requested else "disabled"
        node._rolling_shutter_requested = bool(node.rolling_shutter_enable)
        node._rolling_shutter_status = (
            "requested" if node._rolling_shutter_requested else "disabled"
        )
        node._lidar_deskew_requested = bool(node.lidar_deskew_enable)
        node._lidar_deskew_status = (
            "requested" if node._lidar_deskew_requested else "disabled"
        )
        node._apply_loaded_config(load_ply_config(node))

    def configure_compat_transforms(self) -> None:
        node = self._node
        if node._compat_declared_lidar_T_points is not None:
            node._compat_lidar_points_status = "active (custom matrix)"
        elif node.compat_ouster_sensor_frame:
            compat = np.eye(4, dtype=float)
            compat[0, 0] = -1.0
            compat[1, 1] = -1.0
            compat[2, 3] = -0.038195
            node._compat_declared_lidar_T_points = require_homogeneous_transform(compat)
            node._compat_lidar_points_status = "active (ouster sensor->lidar)"
        else:
            node._compat_lidar_points_status = "disabled"

    def load_camera_info(self) -> None:
        node = self._node
        node._camera_info_source = ""
        if node.camera_info_txt:
            (
                intrinsics,
                frame_id_from_file,
                distortion,
                distortion_model,
                camera_info_size,
                resolved_path,
            ) = node._load_camera_info_txt(node.camera_info_txt)
            node.intrinsics = intrinsics
            node.intrinsics_raw = node.intrinsics.copy()
            node._camera_distortion = distortion
            node._camera_distortion_model = distortion_model
            node._camera_info_size = camera_info_size
            node.camera_frame = frame_id_from_file or node.camera_frame_param
            if not node.camera_frame:
                raise ValueError(
                    "~camera_info_txt requires a camera frame. Add frame_id/camera_frame in the file or set ~camera_frame."
                )
            node._camera_info_source = f"txt:{resolved_path}"
            if node.camera_info_topic:
                node._log.info(
                    "__init__",
                    "Using camera intrinsics from ~camera_info_txt=%s (topic ~camera_info=%s is ignored for intrinsics).",
                    resolved_path,
                    node.camera_info_topic,
                )
            node._log.info(
                "__init__",
                "Camera intrinsics loaded from file: frame_id=%s size=%dx%d",
                node.camera_frame,
                int(node._camera_info_size[1]),
                int(node._camera_info_size[0]),
            )
            node._log.debug("__init__", "Intrinsics K (file)=\n%s", format_matrix(node.intrinsics))
            return

        node._log.debug("__init__", "Waiting for CameraInfo on topic=%s", node.camera_info_topic)
        cam_info = None
        wait_start = time.time()
        next_warn = wait_start + 5.0
        while cam_info is None and not rospy.is_shutdown():
            cam_info = node._bootstrap.wait_for_msg(
                node.camera_info_topic,
                CameraInfo,
                timeout=1.0,
                warn_on_timeout=False,
            )
            if cam_info is None and time.time() >= next_warn:
                node._log.warn(
                    "__init__",
                    "Waiting for CameraInfo on %s (check topic name and bag/driver)",
                    node.camera_info_topic,
                )
                next_warn = time.time() + 5.0
        if cam_info is None:
            raise rospy.ROSInterruptException("Shutdown while waiting for CameraInfo")
        node.intrinsics = np.asarray(cam_info.K, dtype=float).reshape(3, 3)
        node.camera_frame = cam_info.header.frame_id
        node._log.debug(
            "__init__",
            "CameraInfo received: frame_id=%s stamp=%.6f",
            node.camera_frame,
            cam_info.header.stamp.to_sec(),
        )
        node._log.debug("__init__", "Intrinsics K=\n%s", format_matrix(node.intrinsics))
        node.intrinsics_raw = node.intrinsics.copy()
        node._camera_distortion = np.asarray(cam_info.D, dtype=float) if cam_info.D else None
        node._camera_distortion_model = (cam_info.distortion_model or "").strip().lower()
        node._camera_info_size = (int(cam_info.height), int(cam_info.width))
        node._camera_info_source = f"topic:{node.camera_info_topic}"
