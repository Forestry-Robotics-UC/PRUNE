"""Initialization helpers for prune node startup assembly."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo

from prune_core.utils.validation import require_homogeneous_transform
from .bootstrap import StartupBootstrap
from prune_ros.config import (
    load_calibration_config,
    load_color_config,
    load_debug_config,
    load_experiment_config,
    load_ply_config,
    load_projection_config,
    load_sync_config,
)
from .runtime_setup import RuntimeSetup
from .startup_reporting import StartupReporter
from ..pipelines.tf_resolver import TransformResolver
from ..runtime.status import StatusReporter
from ..runtime.tf_utils import format_matrix


class NodeInitializer:
    def __init__(self, node: Any):
        self._node = node

    def load_initial_params(self) -> None:
        node = self._node
        node.debug = node._get_param_bool(
            "~debug",
            False,
            "Enable debug parameter report at startup (and DEBUG logs if set via launch arg).",
        )
        node.core_debug = node._get_param_bool(
            "~core_debug",
            False,
            "Enable prune_core DEBUG logs (can be noisy at 10–30 Hz).",
        )
        node.mode = node._get_param_str(
            "~mode",
            "",
            "Force fusion mode ('depth' or 'lidar'); empty string enables auto-detect.",
            allow_empty=True,
        ).lower()
        node.target_frame = node._get_param_str(
            "~target_frame",
            "base_link",
            "Output frame for published semantic point cloud.",
        )
        node.semantic_topic = node._get_param_str(
            "~semantic_topic",
            "/semantic/labels",
            "Semantic label image topic (sensor_msgs/Image).",
        )
        node.include_unlabeled = node._get_param_bool(
            "~include_unlabeled",
            False,
            "If true, keep points outside the camera FoV as unlabeled samples instead of dropping them.",
        )
        node.perception_invalid_label = node._get_param_int(
            "~perception_invalid_label",
            65535,
            "Label value from Perception indicating invalid/low-confidence pixels; mapped to -1 (unlabeled) before fusion. The perception stack uses 65535 by default.",
        )
        node.semantic_input_type = node._get_param_str(
            "~semantic_input_type",
            "labels",
            "Semantic image representation: 'labels' (single-channel label IDs) or 'rgb' (3-channel colors used directly for output coloring).",
        )
        semantic_type_raw = (node.semantic_input_type or "").strip().lower()
        if semantic_type_raw in ("labels", "label", "label_ids"):
            node.semantic_input_type = "labels"
        elif semantic_type_raw in (
            "rgb",
            "color",
            "color_image",
            "original_color",
            "colored",
            "palette",
            "color_segmentation",
            "colored_segmentation",
        ):
            node.semantic_input_type = "rgb"
        else:
            raise ValueError(
                "Invalid ~semantic_input_type. Expected 'labels' or 'rgb', got: "
                f"{node.semantic_input_type!r}"
            )
        node._param_meta["~semantic_input_type"]["value"] = node.semantic_input_type
        node.undistort_semantic = node._get_param_bool(
            "~undistort_semantic",
            False,
            "If true, undistort semantic images using CameraInfo distortion before projection (lidar mode only).",
        )
        node.undistort_alpha = node._get_param_float(
            "~undistort_alpha",
            0.0,
            "Undistort balance/alpha in [0,1]; 0=crop to valid pixels, 1=keep all pixels.",
        )
        if not (0.0 <= node.undistort_alpha <= 1.0):
            raise ValueError("~undistort_alpha must be in [0, 1]")
        node.rolling_shutter_enable = node._get_param_bool(
            "~rolling_shutter_enable",
            False,
            "Apply rotation-only rolling shutter correction using IMU.",
        )
        node.rolling_shutter_readout_sec = node._get_param_float(
            "~rolling_shutter_readout_sec",
            0.0,
            "Rolling shutter total readout time in seconds (0 disables).",
        )
        if node.rolling_shutter_readout_sec < 0.0:
            raise ValueError("~rolling_shutter_readout_sec must be >= 0")
        node.rolling_shutter_direction = node._get_param_str(
            "~rolling_shutter_direction",
            "top_to_bottom",
            "Rolling shutter readout direction: top_to_bottom or bottom_to_top.",
        ).strip().lower()
        if node.rolling_shutter_direction not in ("top_to_bottom", "bottom_to_top"):
            raise ValueError("~rolling_shutter_direction must be top_to_bottom or bottom_to_top")
        node.imu_topic = node._get_param_str(
            "~imu_topic",
            "",
            "IMU topic used for rolling shutter correction (sensor_msgs/Imu).",
            allow_empty=True,
        )
        node.imu_frame = node._get_param_str(
            "~imu_frame",
            "",
            "Optional IMU frame override for rolling shutter correction.",
            allow_empty=True,
        )
        node.imu_cache_size = node._get_param_int(
            "~imu_cache_size",
            2000,
            "IMU cache size for rolling shutter correction.",
        )
        if node.imu_cache_size < 10:
            raise ValueError("~imu_cache_size must be >= 10")
        node.imu_cache_max_dt_sec = node._get_param_float(
            "~imu_cache_max_dt_sec",
            0.02,
            "Max allowed dt (seconds) between semantic frame and IMU for correction.",
        )
        if node.imu_cache_max_dt_sec < 0.0:
            raise ValueError("~imu_cache_max_dt_sec must be >= 0")
        node.camera_metadata_topic = node._get_param_str(
            "~camera_metadata_topic",
            "",
            "Camera metadata topic for rolling shutter readout (realsense2_camera_msgs/Metadata).",
            allow_empty=True,
        )
        node.metadata_readout_key = node._get_param_int(
            "~metadata_readout_key",
            -1,
            "Metadata key for readout time; set -1 to disable metadata readout.",
        )
        node.metadata_readout_scale = node._get_param_float(
            "~metadata_readout_scale",
            1e-6,
            "Scale applied to metadata value to convert to seconds (e.g., use 1e-6 for usec).",
        )
        if node.metadata_readout_scale <= 0.0:
            raise ValueError("~metadata_readout_scale must be > 0")
        node.metadata_max_dt_sec = node._get_param_float(
            "~metadata_max_dt_sec",
            0.1,
            "Max allowed dt (seconds) between metadata and semantic frame for readout.",
        )
        if node.metadata_max_dt_sec < 0.0:
            raise ValueError("~metadata_max_dt_sec must be >= 0")
        node.lidar_deskew_enable = node._get_param_bool(
            "~lidar_deskew_enable",
            False,
            "Enable LiDAR deskew using per-point time + IMU.",
        )
        node.lidar_deskew_mode = node._get_param_str(
            "~lidar_deskew_mode",
            "rotation",
            "Deskew mode: rotation, translation, or both.",
        ).strip().lower()
        if node.lidar_deskew_mode not in ("rotation", "translation", "both"):
            raise ValueError("~lidar_deskew_mode must be rotation, translation, or both")
        node.lidar_time_field = node._get_param_str(
            "~lidar_time_field",
            "t",
            "PointCloud2 field name for per-point time (default: t).",
        ).strip()
        node.lidar_time_scale = node._get_param_float(
            "~lidar_time_scale",
            1e-9,
            "Scale factor to convert per-point time to seconds (e.g., ns -> 1e-9).",
        )
        if node.lidar_time_scale <= 0.0:
            raise ValueError("~lidar_time_scale must be > 0")
        node.lidar_deskew_ref = node._get_param_str(
            "~lidar_deskew_ref",
            "start",
            "Deskew reference time: start or mid (scan start recommended).",
        ).strip().lower()
        if node.lidar_deskew_ref not in ("start", "mid"):
            raise ValueError("~lidar_deskew_ref must be start or mid")
        node.lidar_deskew_imu_samples = node._get_param_int(
            "~lidar_deskew_imu_samples",
            1,
            "Number of IMU samples used across each scan for LiDAR deskew (1 keeps the lightweight single-sample model; values >1 better handle fast motion).",
        )
        if node.lidar_deskew_imu_samples < 1:
            raise ValueError("~lidar_deskew_imu_samples must be >= 1")
        node.lidar_imu_topic = node._get_param_str(
            "~lidar_imu_topic",
            "",
            "IMU topic used for LiDAR deskew (sensor_msgs/Imu).",
            allow_empty=True,
        )
        node.lidar_imu_frame = node._get_param_str(
            "~lidar_imu_frame",
            "",
            "Optional IMU frame override for LiDAR deskew.",
            allow_empty=True,
        )
        node.lidar_imu_cache_size = node._get_param_int(
            "~lidar_imu_cache_size",
            2000,
            "IMU cache size for LiDAR deskew.",
        )
        if node.lidar_imu_cache_size < 10:
            raise ValueError("~lidar_imu_cache_size must be >= 10")
        node.lidar_imu_cache_max_dt_sec = node._get_param_float(
            "~lidar_imu_cache_max_dt_sec",
            0.02,
            "Max allowed dt (seconds) between LiDAR scan time and IMU for deskew.",
        )
        if node.lidar_imu_cache_max_dt_sec < 0.0:
            raise ValueError("~lidar_imu_cache_max_dt_sec must be >= 0")
        node.lidar_imu_accel_is_gravity_compensated = node._get_param_bool(
            "~lidar_imu_accel_gravity_compensated",
            True,
            "If true, IMU linear_acceleration is gravity-compensated (recommended).",
        )
        node.compat_ouster_sensor_frame = node._get_param_bool(
            "~compat_ouster_sensor_frame",
            False,
            "Legacy-bag compatibility: treat incoming Ouster PointCloud2 XYZ as sensor-frame points mislabeled as the LiDAR frame and convert them back into the declared LiDAR frame before deskew/projection.",
        )
        node._compat_declared_lidar_T_points = node._get_matrix_param(
            "~compat_declared_lidar_T_points",
            "Optional static 4x4 row-major matrix mapping incoming point-data coordinates into the declared LiDAR frame. Applied before deskew/projection. Overrides the built-in ~compat_ouster_sensor_frame transform when provided.",
        )
        self.configure_compat_transforms()
        node.conf_topic = node._get_param_str(
            "~confidence_topic",
            None,
            "Optional confidence image topic aligned with semantic labels (sensor_msgs/Image).",
        )
        node.camera_info_topic = node._get_param_str(
            "~camera_info",
            None,
            "CameraInfo topic providing intrinsics and camera frame_id (sensor_msgs/CameraInfo).",
        )
        node.camera_info_txt = node._get_param_str(
            "~camera_info_txt",
            "",
            "Optional path to a camera calibration text file. When set, intrinsics are loaded from file and ~camera_info topic is optional.",
            allow_empty=True,
        )
        node.camera_frame_param = node._get_param_str(
            "~camera_frame",
            "",
            "Optional camera frame override used when ~camera_info_txt does not include frame_id.",
            allow_empty=True,
        )
        if not node.camera_info_topic and not node.camera_info_txt:
            raise ValueError("Either ~camera_info or ~camera_info_txt is required")
        node.depth_input_topic = node._get_param_str(
            "~depth_input_topic",
            None,
            "Geometry input topic: depth (sensor_msgs/Image) or LiDAR (sensor_msgs/PointCloud2). The node auto-detects which message type is published and selects the fusion mode.",
        )
        if not node.depth_input_topic:
            raise ValueError(
                "~depth_input_topic is required (sensor_msgs/Image depth or sensor_msgs/PointCloud2 LiDAR)"
            )

    def load_runtime_support_params(self) -> None:
        node = self._node
        node.debug_projected_topic = "/debug/lidar_projection"
        node.debug_lidar_depth_topic = "/debug/lidar_depth"
        node.debug_lidar_edge_topic = "/debug/lidar_edge"
        node.debug_reprojection_heatmap_topic = "/debug/reprojection_heatmap"
        node.debug_alignment_score_topic = "/debug/alignment_score"
        node.debug_tracked_reprojection_topic = "/debug/tracked_reprojection"
        node.debug_tracked_reprojection_error_topic = "/debug/tracked_reprojection_error_px"
        node.debug_fov_points_topic = "/debug/lidar_points_in_fov"
        node.enable_profiling = node._get_param_bool(
            "~enable_profiling",
            False,
            "If true, print a short cProfile summary per callback (future C++/numba profiling hook).",
        )
        node.depth_scale = node._get_param_float(
            "~depth_scale",
            0.0,
            "Scale factor to convert depth values to meters (0=auto: 16UC1/mono16 treated as mm -> 0.001; 32FC1 treated as meters -> 1.0).",
        )
        if node.depth_scale < 0.0:
            raise ValueError("~depth_scale must be >= 0")
        node.max_depth_m = node._get_param_float(
            "~max_depth_m",
            0.0,
            "Optional maximum depth in meters (<=0 disables).",
        )
        if node.max_depth_m <= 0.0:
            node.max_depth_m = None
        node.camera_fov_gate_enable = node._get_param_bool(
            "~camera_fov_gate_enable",
            True,
            "Drop LiDAR points outside the camera FoV before projection. Reduces processed point count from 360-deg LiDAR to ~18% on a typical 70-deg camera, giving ~5x speedup on downstream projection/sampling steps.",
        )
        node.camera_fov_gate_margin_deg = node._get_param_float(
            "~camera_fov_gate_margin_deg",
            5.0,
            "Angular margin in degrees added to each side of the camera FoV gate to avoid hard cutoffs at image edges.",
        )
        if node.camera_fov_gate_margin_deg < 0.0:
            raise ValueError("~camera_fov_gate_margin_deg must be >= 0")
        node.filter_invalid_depth = node._get_param_bool(
            "~filter_invalid_depth",
            True,
            "If true, treat common uint16 depth sentinels (0, 65535) as invalid before scaling.",
        )
        node.static_target_T_depth = node._get_matrix_param(
            "~static_target_T_depth",
            "Optional static 4x4 row-major matrix: depth_frame -> target_frame. Overrides TF.",
        )
        node.static_camera_T_lidar = node._get_matrix_param(
            "~static_camera_T_lidar",
            "Optional static 4x4 row-major matrix: lidar_frame -> camera_frame. Overrides TF.",
        )
        node.static_target_T_lidar = node._get_matrix_param(
            "~static_target_T_lidar",
            "Optional static 4x4 row-major matrix: lidar_frame -> target_frame. Overrides TF.",
        )
        status_period_raw = node._get_param(
            "~status_period",
            "",
            "Seconds between periodic status table prints. Empty=auto (1s when debug=true, else disabled). Set to 0 to disable explicitly.",
            allow_empty=True,
        )
        if status_period_raw in ("", None):
            node.status_period = 1.0 if node.debug else 0.0
        else:
            node.status_period = float(status_period_raw)
        if node.status_period < 0.0:
            raise ValueError("~status_period must be >= 0")
        node._param_meta["~status_period"]["value"] = float(node.status_period)
        node._status = StatusReporter(period_sec=float(node.status_period))

    def setup_startup_helpers(self) -> None:
        node = self._node
        node._tf_resolver = TransformResolver(node, node._log)
        node.tf_buffer = node._tf_resolver.tf_buffer
        node.tf_listener = node._tf_resolver.tf_listener
        node._bootstrap = StartupBootstrap(node, node._log)
        node._runtime_setup = RuntimeSetup(node)
        node._startup_reporting = StartupReporter(node)

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
