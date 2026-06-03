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
#   ROS node: converts semantic image + geometry (depth or LiDAR) into semantic PointCloud2.

"""ROS wrapper that converts semantic + geometry into semantic PointCloud2.

This node is part of ENTFAC Sensor Fusion and is explicitly *stateless*:
each callback processes one frame and publishes a semantic measurement for a
separate mapping layer to accumulate over time.

ROS Interface (v1.0)
-------------------

Subscriptions
^^^^^^^^^^^^^

Required:
  - ``~semantic_topic`` (``sensor_msgs/Image``):
    - ``~semantic_input_type=labels``: single-channel label IDs (e.g. ``mono8``,
      ``16UC1``, ``32SC1``).
    - ``~semantic_input_type=rgb``: 3/4-channel colors (e.g. ``rgb8``, ``bgr8``,
      ``rgba8``, ``bgra8``). Colors are passed through to the output when
      ``~colorize_labels`` is enabled.
  - Geometry input:
    - Preferred: ``~depth_input_topic`` (auto-detected):
      - ``sensor_msgs/Image`` → depth mode
      - ``sensor_msgs/PointCloud2`` → lidar mode

Optional:
  - Camera intrinsics source:
    - ``~camera_info`` (``sensor_msgs/CameraInfo``): topic providing intrinsics
      ``K`` and camera frame ID.
    - ``~camera_info_txt`` (``str``): calibration file path. Supports
      ``K: [k00,...,k22]`` / ``camera_matrix.data: [...]`` or keyed
      ``fx/fy/cx/cy`` fields.
    - ``~camera_frame`` (``str``): fallback frame ID used with
      ``~camera_info_txt`` when the file does not include one.
  - ``~confidence_topic`` (``sensor_msgs/Image``): confidence/probability aligned
    with semantic labels (single-channel numeric).
  - ``~projection_invalid_mask_topic`` (``sensor_msgs/Image``): optional
    single-channel invalid mask aligned with ``~semantic_topic``. Invalid pixels
    reject transferred labels/RGB and zero confidence.

Publications
^^^^^^^^^^^^

  - ``semantic_pointcloud`` (``sensor_msgs/PointCloud2``): semantic measurement
    in ``~target_frame`` with fields:
    - ``x, y, z`` (float32)
    - ``label`` (uint16; ``65535`` means unknown/unlabeled)
    - ``confidence`` (float32, optional)
    - ``rgb`` (float32 packed RGB, optional; only when ``~colorize_labels:=true``)

TF / Extrinsics
^^^^^^^^^^^^^^^

Depth mode requires a transform from the depth frame to ``~target_frame``:
  - Static parameter: ``~static_target_T_depth`` (16-element row-major 4x4)
  - Otherwise: TF lookup ``target_frame <- depth_frame`` (resolved once).

LiDAR mode requires:
  - ``~static_camera_T_lidar`` or TF lookup ``camera_frame <- lidar_frame``
  - ``~static_target_T_lidar`` or TF lookup ``target_frame <- lidar_frame``

Services
^^^^^^^^

  - ``~save_ply`` (``std_srvs/Trigger``): write the last published cloud to PLY.
  - ``~set_ply_recording`` (``std_srvs/SetBool``): enable/disable continuous PLY
    recording (written asynchronously).

Failure Behavior
^^^^^^^^^^^^^^^^

  - Invalid configuration raises ``ValueError`` during initialization.
  - Missing TF/extrinsics at runtime logs a warning and skips publishing until
    resolved.
  - Shape/dtype mismatches raise ``ValueError`` to fail fast.
"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import rospy

try:
    from dynamic_reconfigure.server import Server as DynamicReconfigureServer
except Exception:  # noqa: BLE001
    DynamicReconfigureServer = None

try:
    from scipy import ndimage as _scipy_ndimage
except ImportError:
    _scipy_ndimage = None

# Ensure core package is importable when running from a monorepo source tree.
# In a proper catkin workspace, this is handled by PYTHONPATH via devel/setup.bash.
_THIS = Path(__file__).resolve()
for parent in _THIS.parents:
    cand = parent / "entfac_fusion_core" / "src"
    if (cand / "entfac_fusion_core").is_dir() and str(cand) not in sys.path:
        sys.path.insert(0, str(cand))
        break

from entfac_fusion_ros.colored_pcl_params import coerce_bool as _coerce_bool
from entfac_fusion_ros.colored_pcl.camera_model import CameraModel
from entfac_fusion_ros.colored_pcl.bootstrap import StartupBootstrap
from entfac_fusion_ros.colored_pcl.frame_inputs import FrameInputPreparer
from entfac_fusion_ros.colored_pcl.initializer import NodeInitializer
from entfac_fusion_ros.colored_pcl.param_reader import ParamReader
from entfac_fusion_ros.colored_pcl.ply_service import PlyRecordingService
from entfac_fusion_ros.colored_pcl.runtime_setup import RuntimeSetup
from entfac_fusion_ros.colored_pcl.online_calibration_bridge import OnlineCalibrationBridge
from entfac_fusion_ros.colored_pcl.semantic_inputs import SemanticInputParser
from entfac_fusion_ros.colored_pcl.startup_reporting import StartupReporter
from entfac_fusion_ros.colored_pcl.sync_policy import StampPolicy
from entfac_fusion_ros.colored_pcl.tf_resolver import TransformResolver
from entfac_fusion_ros.colored_pcl.tracked_reprojection_runtime import TrackedReprojectionRuntime
from entfac_fusion_ros.colored_pcl_startup import (
    log_correction_statuses as _log_correction_statuses_helper,
    log_param_report as _log_param_report_helper,
    log_startup_transforms as _log_startup_transforms_helper,
    render_startup_table as _render_startup_table_helper,
)
from entfac_fusion_ros.logging_ros import NodeLogger, configure_core_logging
from entfac_fusion_ros.status import StatusReporter

try:
    from entfac_fusion_ros.cfg import ColoredPclTuningConfig
except Exception:  # noqa: BLE001
    ColoredPclTuningConfig = None


def _rosargv_bool(name: str, default: bool = False) -> bool:
    prefix = f"_{name}:="
    for arg in sys.argv:
        if arg.startswith(prefix):
            return _coerce_bool(arg[len(prefix) :])
    return default


@dataclass
class _ProjectionQualityResult:
    keep: np.ndarray
    confidence_reject: np.ndarray
    depth_edge_reject: np.ndarray
    occlusion_reject: np.ndarray
    depth_edge_map: Optional[np.ndarray]
    runtime_rasterize_ms: float = 0.0
    runtime_depth_edge_ms: float = 0.0
    runtime_occlusion_ms: float = 0.0


@dataclass
class _ProjectionMetrics:
    num_points_in_front: int = 0
    num_points_projected_in_image: int = 0
    num_rejected_invalid_mask: int = 0
    num_rejected_confidence: int = 0
    num_rejected_depth_edge: int = 0
    num_rejected_occlusion: int = 0
    num_rejected_other: int = 0
    num_would_hit_invalid_mask: int = 0
    num_would_hit_depth_edge: int = 0
    num_would_fail_occlusion: int = 0
    runtime_projection_ms: float = 0.0
    runtime_mask_ms: float = 0.0
    runtime_rasterize_ms: float = 0.0
    runtime_depth_edge_ms: float = 0.0
    runtime_occlusion_ms: float = 0.0
    runtime_publish_ms: float = 0.0


class ColoredPclNode:
    """ROS node bridging topics to the numpy fusion core."""

    def __init__(self):
        self._param_meta: Dict[str, Dict[str, Any]] = {}
        self._node_name = rospy.get_name().lstrip("/")
        self._log = NodeLogger(self._node_name)
        self._params = ParamReader(self)
        self._record_param = self._params.record_param
        self._get_param = self._params.get_param
        self._get_param_str = self._params.get_param_str
        self._get_param_bool = self._params.get_param_bool
        self._get_param_int = self._params.get_param_int
        self._get_param_float = self._params.get_param_float
        self._get_matrix_param = self._params.get_matrix_param
        self._get_color_map = self._params.get_color_map
        self._apply_loaded_config = self._params.apply_loaded_config
        self._load_camera_info_txt = self._params.load_camera_info_txt

        self.debug = self._get_param_bool(
            "~debug",
            False,
            "Enable debug parameter report at startup (and DEBUG logs if set via launch arg).",
        )
        self.core_debug = self._get_param_bool(
            "~core_debug",
            False,
            "Enable entfac_fusion_core DEBUG logs (can be noisy at 10–30 Hz).",
        )
        configure_core_logging(self._node_name, debug=self.core_debug)

        # 'depth' uses aligned depth image; 'lidar' projects LiDAR points into the image.
        self.mode = self._get_param_str(
            "~mode",
            "",
            "Force fusion mode ('depth' or 'lidar'); empty string enables auto-detect.",
            allow_empty=True,
        ).lower()

        self.target_frame = self._get_param_str(
            "~target_frame",
            "base_link",
            "Output frame for published semantic point cloud.",
        )
        self.semantic_topic = self._get_param_str(
            "~semantic_topic",
            "/semantic/labels",
            "Semantic label image topic (sensor_msgs/Image).",
        )
        self.semantic_input_type = self._get_param_str(
            "~semantic_input_type",
            "labels",
            "Semantic image representation: 'labels' (single-channel label IDs) or 'rgb' (3-channel colors used directly for output coloring).",
        )
        semantic_type_raw = (self.semantic_input_type or "").strip().lower()
        if semantic_type_raw in ("labels", "label", "label_ids"):
            self.semantic_input_type = "labels"
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
            self.semantic_input_type = "rgb"
        else:
            raise ValueError(
                "Invalid ~semantic_input_type. Expected 'labels' or 'rgb', got: "
                f"{self.semantic_input_type!r}"
            )
        self._param_meta["~semantic_input_type"]["value"] = self.semantic_input_type
        self.undistort_semantic = self._get_param_bool(
            "~undistort_semantic",
            False,
            "If true, undistort semantic images using CameraInfo distortion before projection (lidar mode only).",
        )
        self.undistort_alpha = self._get_param_float(
            "~undistort_alpha",
            0.0,
            "Undistort balance/alpha in [0,1]; 0=crop to valid pixels, 1=keep all pixels.",
        )
        if not (0.0 <= self.undistort_alpha <= 1.0):
            raise ValueError("~undistort_alpha must be in [0, 1]")
        self.rolling_shutter_enable = self._get_param_bool(
            "~rolling_shutter_enable",
            False,
            "Apply rotation-only rolling shutter correction using IMU.",
        )
        self.rolling_shutter_readout_sec = self._get_param_float(
            "~rolling_shutter_readout_sec",
            0.0,
            "Rolling shutter total readout time in seconds (0 disables).",
        )
        if self.rolling_shutter_readout_sec < 0.0:
            raise ValueError("~rolling_shutter_readout_sec must be >= 0")
        self.rolling_shutter_direction = self._get_param_str(
            "~rolling_shutter_direction",
            "top_to_bottom",
            "Rolling shutter readout direction: top_to_bottom or bottom_to_top.",
        ).strip().lower()
        if self.rolling_shutter_direction not in ("top_to_bottom", "bottom_to_top"):
            raise ValueError(
                "~rolling_shutter_direction must be top_to_bottom or bottom_to_top"
            )
        self.imu_topic = self._get_param_str(
            "~imu_topic",
            "",
            "IMU topic used for rolling shutter correction (sensor_msgs/Imu).",
            allow_empty=True,
        )
        self.imu_frame = self._get_param_str(
            "~imu_frame",
            "",
            "Optional IMU frame override for rolling shutter correction.",
            allow_empty=True,
        )
        self.imu_cache_size = self._get_param_int(
            "~imu_cache_size",
            2000,
            "IMU cache size for rolling shutter correction.",
        )
        if self.imu_cache_size < 10:
            raise ValueError("~imu_cache_size must be >= 10")
        self.imu_cache_max_dt_sec = self._get_param_float(
            "~imu_cache_max_dt_sec",
            0.02,
            "Max allowed dt (seconds) between semantic frame and IMU for correction.",
        )
        if self.imu_cache_max_dt_sec < 0.0:
            raise ValueError("~imu_cache_max_dt_sec must be >= 0")
        self.camera_metadata_topic = self._get_param_str(
            "~camera_metadata_topic",
            "",
            "Camera metadata topic for rolling shutter readout (realsense2_camera_msgs/Metadata).",
            allow_empty=True,
        )
        self.metadata_readout_key = self._get_param_int(
            "~metadata_readout_key",
            -1,
            "Metadata key for readout time; set -1 to disable metadata readout.",
        )
        self.metadata_readout_scale = self._get_param_float(
            "~metadata_readout_scale",
            1e-6,
            "Scale applied to metadata value to convert to seconds (e.g., use 1e-6 for usec).",
        )
        if self.metadata_readout_scale <= 0.0:
            raise ValueError("~metadata_readout_scale must be > 0")
        self.metadata_max_dt_sec = self._get_param_float(
            "~metadata_max_dt_sec",
            0.1,
            "Max allowed dt (seconds) between metadata and semantic frame for readout.",
        )
        if self.metadata_max_dt_sec < 0.0:
            raise ValueError("~metadata_max_dt_sec must be >= 0")
        self.lidar_deskew_enable = self._get_param_bool(
            "~lidar_deskew_enable",
            False,
            "Enable LiDAR deskew using per-point time + IMU.",
        )
        self.lidar_deskew_mode = self._get_param_str(
            "~lidar_deskew_mode",
            "rotation",
            "Deskew mode: rotation, translation, or both.",
        ).strip().lower()
        if self.lidar_deskew_mode not in ("rotation", "translation", "both"):
            raise ValueError("~lidar_deskew_mode must be rotation, translation, or both")
        self.lidar_time_field = self._get_param_str(
            "~lidar_time_field",
            "t",
            "PointCloud2 field name for per-point time (default: t).",
        ).strip()
        self.lidar_time_scale = self._get_param_float(
            "~lidar_time_scale",
            1e-9,
            "Scale factor to convert per-point time to seconds (e.g., ns -> 1e-9).",
        )
        if self.lidar_time_scale <= 0.0:
            raise ValueError("~lidar_time_scale must be > 0")
        self.lidar_deskew_ref = self._get_param_str(
            "~lidar_deskew_ref",
            "start",
            "Deskew reference time: start or mid (scan start recommended).",
        ).strip().lower()
        if self.lidar_deskew_ref not in ("start", "mid"):
            raise ValueError("~lidar_deskew_ref must be start or mid")
        self.lidar_deskew_imu_samples = self._get_param_int(
            "~lidar_deskew_imu_samples",
            1,
            "Number of IMU samples used across each scan for LiDAR deskew (1 keeps the lightweight single-sample model; values >1 better handle fast motion).",
        )
        if self.lidar_deskew_imu_samples < 1:
            raise ValueError("~lidar_deskew_imu_samples must be >= 1")
        self.lidar_imu_topic = self._get_param_str(
            "~lidar_imu_topic",
            "",
            "IMU topic used for LiDAR deskew (sensor_msgs/Imu).",
            allow_empty=True,
        )
        self.lidar_imu_frame = self._get_param_str(
            "~lidar_imu_frame",
            "",
            "Optional IMU frame override for LiDAR deskew.",
            allow_empty=True,
        )
        self.lidar_imu_cache_size = self._get_param_int(
            "~lidar_imu_cache_size",
            2000,
            "IMU cache size for LiDAR deskew.",
        )
        if self.lidar_imu_cache_size < 10:
            raise ValueError("~lidar_imu_cache_size must be >= 10")
        self.lidar_imu_cache_max_dt_sec = self._get_param_float(
            "~lidar_imu_cache_max_dt_sec",
            0.02,
            "Max allowed dt (seconds) between LiDAR scan time and IMU for deskew.",
        )
        if self.lidar_imu_cache_max_dt_sec < 0.0:
            raise ValueError("~lidar_imu_cache_max_dt_sec must be >= 0")
        self.lidar_imu_accel_is_gravity_compensated = self._get_param_bool(
            "~lidar_imu_accel_gravity_compensated",
            True,
            "If true, IMU linear_acceleration is gravity-compensated (recommended).",
        )
        self.compat_ouster_sensor_frame = self._get_param_bool(
            "~compat_ouster_sensor_frame",
            False,
            "Legacy-bag compatibility: treat incoming Ouster PointCloud2 XYZ as sensor-frame points mislabeled as the LiDAR frame and convert them back into the declared LiDAR frame before deskew/projection.",
        )
        self._compat_declared_lidar_T_points = self._get_matrix_param(
            "~compat_declared_lidar_T_points",
            "Optional static 4x4 row-major matrix mapping incoming point-data coordinates into the declared LiDAR frame. Applied before deskew/projection. Overrides the built-in ~compat_ouster_sensor_frame transform when provided.",
        )
        self._initializer = NodeInitializer(self)
        self._initializer.configure_compat_transforms()
        self.conf_topic = self._get_param_str(
            "~confidence_topic",
            None,
            "Optional confidence image topic aligned with semantic labels (sensor_msgs/Image).",
        )
        self.camera_info_topic = self._get_param_str(
            "~camera_info",
            None,
            "CameraInfo topic providing intrinsics and camera frame_id (sensor_msgs/CameraInfo).",
        )
        self.camera_info_txt = self._get_param_str(
            "~camera_info_txt",
            "",
            "Optional path to a camera calibration text file. When set, intrinsics are loaded from file and ~camera_info topic is optional.",
            allow_empty=True,
        )
        self.camera_frame_param = self._get_param_str(
            "~camera_frame",
            "",
            "Optional camera frame override used when ~camera_info_txt does not include frame_id.",
            allow_empty=True,
        )
        if not self.camera_info_topic and not self.camera_info_txt:
            raise ValueError("Either ~camera_info or ~camera_info_txt is required")

        self.depth_input_topic = self._get_param_str(
            "~depth_input_topic",
            None,
            "Geometry input topic: depth (sensor_msgs/Image) or LiDAR (sensor_msgs/PointCloud2). The node auto-detects which message type is published and selects the fusion mode.",
        )
        if not self.depth_input_topic:
            raise ValueError(
                "~depth_input_topic is required (sensor_msgs/Image depth or sensor_msgs/PointCloud2 LiDAR)"
            )

        self._initializer.load_runtime_config()
        # Fixed debug projection settings to avoid extra parameters.
        self.debug_projected_topic = "/debug/lidar_projection"
        self.debug_lidar_depth_topic = "/debug/lidar_depth"
        self.debug_lidar_edge_topic = "/debug/lidar_edge"
        self.debug_reprojection_heatmap_topic = "/debug/reprojection_heatmap"
        self.debug_alignment_score_topic = "/debug/alignment_score"
        self.debug_tracked_reprojection_topic = "/debug/tracked_reprojection"
        self.debug_tracked_reprojection_error_topic = "/debug/tracked_reprojection_error_px"
        self.debug_fov_points_topic = "/debug/lidar_points_in_fov"
        self.debug_calibration_health_topic = "/debug/calibration_health"
        self.debug_calibration_uncertainty_topic = "/debug/calibration_uncertainty"
        self.enable_profiling = self._get_param_bool(
            "~enable_profiling",
            False,
            "If true, print a short cProfile summary per callback (future C++/numba profiling hook).",
        )
        self.depth_scale = self._get_param_float(
            "~depth_scale",
            0.0,
            "Scale factor to convert depth values to meters (0=auto: 16UC1/mono16 treated as mm -> 0.001; 32FC1 treated as meters -> 1.0).",
        )
        if self.depth_scale < 0.0:
            raise ValueError("~depth_scale must be >= 0")
        self.max_depth_m = self._get_param_float(
            "~max_depth_m",
            0.0,
            "Optional maximum depth in meters (<=0 disables).",
        )
        if self.max_depth_m <= 0.0:
            self.max_depth_m = None
        self.camera_fov_gate_enable = self._get_param_bool(
            "~camera_fov_gate_enable",
            True,
            "Drop LiDAR points outside the camera FoV before projection. "
            "Reduces processed point count from 360-deg LiDAR to ~18% on a typical 70-deg camera, "
            "giving ~5x speedup on downstream projection/sampling steps.",
        )
        self.camera_fov_gate_margin_deg = self._get_param_float(
            "~camera_fov_gate_margin_deg",
            5.0,
            "Angular margin in degrees added to each side of the camera FoV gate "
            "to avoid hard cutoffs at image edges.",
        )
        if self.camera_fov_gate_margin_deg < 0.0:
            raise ValueError("~camera_fov_gate_margin_deg must be >= 0")
        self.filter_invalid_depth = self._get_param_bool(
            "~filter_invalid_depth",
            True,
            "If true, treat common uint16 depth sentinels (0, 65535) as invalid before scaling.",
        )

        self.static_target_T_depth = self._get_matrix_param(
            "~static_target_T_depth",
            "Optional static 4x4 row-major matrix: depth_frame -> target_frame. Overrides TF.",
        )
        self.static_camera_T_lidar = self._get_matrix_param(
            "~static_camera_T_lidar",
            "Optional static 4x4 row-major matrix: lidar_frame -> camera_frame. Overrides TF.",
        )
        self.static_target_T_lidar = self._get_matrix_param(
            "~static_target_T_lidar",
            "Optional static 4x4 row-major matrix: lidar_frame -> target_frame. Overrides TF.",
        )

        status_period_raw = self._get_param(
            "~status_period",
            "",
            "Seconds between periodic status table prints. Empty=auto (1s when debug=true, else disabled). Set to 0 to disable explicitly.",
            allow_empty=True,
        )
        if status_period_raw in ("", None):
            self.status_period = 1.0 if self.debug else 0.0
        else:
            self.status_period = float(status_period_raw)
        if self.status_period < 0.0:
            raise ValueError("~status_period must be >= 0")
        self._param_meta["~status_period"]["value"] = float(self.status_period)
        self._status = StatusReporter(period_sec=float(self.status_period))

        self._tf_resolver = TransformResolver(self, self._log)
        self.tf_buffer = self._tf_resolver.tf_buffer
        self.tf_listener = self._tf_resolver.tf_listener
        self._bootstrap = StartupBootstrap(self, self._log)
        self._runtime_setup = RuntimeSetup(self)
        self._startup_reporting = StartupReporter(self)

        self._initializer.load_camera_info()

        self._output_topic = rospy.resolve_name("semantic_pointcloud")
        self.target_T_depth = None
        self.camera_T_lidar = None
        self.target_T_lidar = None
        self._depth_frame = ""
        self._lidar_frame = ""

        self._mode_source = "forced" if self.mode in ("depth", "lidar") else "auto"
        self._mode_detail = "forced via ~mode" if self._mode_source == "forced" else ""
        if self.mode not in ("depth", "lidar"):
            self.mode = self._bootstrap.detect_mode()
        self._online_calibration_rpy_rad = np.zeros(3, dtype=np.float64)
        if self.online_calibration_enable and self.mode != "lidar":
            self._log.warn(
                "__init__",
                "online_calibration_enable=true requires lidar mode; disabling because mode=%s",
                self.mode,
            )
            self.online_calibration_enable = False
            self._online_calibration_status = f"disabled (mode={self.mode})"
        elif self.online_calibration_enable:
            self._online_calibration_status = "active"
        else:
            self._online_calibration_status = "disabled"
        self._stamp_policy = StampPolicy(
            self,
            SyncConfig(
                sync_slop_sec=self.sync_slop_sec,
                pair_max_dt_sec=self.pair_max_dt_sec,
                semantic_time_offset_sec=self.semantic_time_offset_sec,
                sync_queue_size=self.sync_queue_size,
                cloud_time_offset_sec=self.cloud_time_offset_sec,
                cloud_stamp_source=self.cloud_stamp_source,
                stamp_debug_log_period_sec=self.stamp_debug_log_period_sec,
            ),
            self._log,
        )
        self._stamp_policy.resolve_cloud_stamp_source()
        self._camera_model = CameraModel(self, self._log)
        self._camera_model.load()
        self._ply_service = PlyRecordingService(self, self._log)
        self._tracked_runtime = TrackedReprojectionRuntime(self)
        self._calibration_bridge = OnlineCalibrationBridge(self)
        self._semantic_parser = SemanticInputParser(
            self,
            ColorConfig(
                colorize_labels=self.colorize_labels,
                color_map=dict(self.color_map) if self.color_map else {},
                random_color_seed=int(self.random_color_seed),
                num_labels=int(self.num_labels),
                semantic_color_quantization_step=int(self.semantic_color_quantization_step),
            ),
            ProjectionConfig(
                projection_patch_size=int(self.projection_patch_size),
                projection_confidence_min=float(self.projection_confidence_min),
                projection_invalid_mask_topic=str(self.projection_invalid_mask_topic),
                projection_invalid_mask_value=int(self.projection_invalid_mask_value),
                projection_invalid_mask_dilate_px=int(self.projection_invalid_mask_dilate_px),
                projection_occlusion_epsilon_m=float(self.projection_occlusion_epsilon_m),
                projection_occlusion_radius_px=int(self.projection_occlusion_radius_px),
                projection_reject_depth_edges=bool(self.projection_reject_depth_edges),
                projection_depth_edge_thresh=float(self.projection_depth_edge_thresh),
                projection_depth_edge_radius_px=int(self.projection_depth_edge_radius_px),
                downsample_factor=int(self.downsample_factor),
            ),
            self._log,
        )
        self._frame_inputs = FrameInputPreparer(self)
        self._runtime_setup.initialize_runtime_state()

        self._runtime_setup.validate_mode_dependent_flags()

        self._runtime_setup.setup_metrics_and_ply()
        self._runtime_setup.setup_projector_and_buffers()
        self._runtime_setup.setup_subsystems()
        self._runtime_setup.setup_ros_runtime(
            DynamicReconfigureServer,
            ColoredPclTuningConfig,
        )
        self._log.info("__init__", "\n%s", self._startup_reporting.render_startup_table())
        self._startup_reporting.log_correction_statuses()
        self._startup_reporting.log_startup_transforms()
        self._log.debug(
            "__init__",
            "Runtime: target_frame=%s camera_frame=%s semantic_input_type=%s colorize_labels=%s include_unlabeled=%s downsample=%d",
            self.target_frame,
            self.camera_frame,
            self.semantic_input_type,
            bool(self.colorize_labels),
            bool(self.include_unlabeled),
            int(self.downsample_factor),
        )
        if self.debug:
            self._startup_reporting.log_param_report()

    def _metadata_callback(self, msg) -> None:
        try:
            key = int(msg.key)
            value = int(msg.value)
        except Exception:  # noqa: BLE001
            return
        self._metadata_latest[key] = (msg.header.stamp, value)

    def _ensure_cv2(self, context: str) -> bool:
        if self._cv2 is not None:
            return True
        try:
            import cv2  # type: ignore
        except Exception as exc:  # noqa: BLE001
            self._log.warn(
                context,
                "OpenCV not available (%s); disabling the requested image-space diagnostic.",
                exc,
            )
            return False
        self._cv2 = cv2
        return True

    def _maybe_init_undistort(self) -> None:
        self._camera_model._maybe_init_undistort()

    def _undistort_array(self, data: np.ndarray, *, interpolation: str) -> np.ndarray:
        return self._camera_model.undistort_array(data, interpolation=interpolation)

    def _lookup_transform(self, target_frame, source_frame, stamp):
        return self._tf_resolver.lookup(target_frame, source_frame, stamp)

    def _lookup_transform_with_stamp(self, target_frame, source_frame, stamp):
        return self._tf_resolver.lookup_with_stamp(target_frame, source_frame, stamp)

    @staticmethod
    def _scale_intrinsics(intrinsics, factor):
        scaled = intrinsics.copy()
        scaled[0, 0] /= factor
        scaled[1, 1] /= factor
        scaled[0, 2] /= factor
        scaled[1, 2] /= factor
        return scaled

    @contextmanager
    def _maybe_profile(self, label):
        if not self.enable_profiling:
            yield
            return
        prof = cProfile.Profile()
        prof.enable()
        try:
            yield
        finally:
            prof.disable()
            s = io.StringIO()
            ps = pstats.Stats(prof, stream=s).sort_stats("tottime")
            ps.print_stats(10)
            self._log.info("_profile", "%s profile:\n%s", label, s.getvalue())

    def _depth_callback(self, sem_msg, depth_msg, conf_msg=None, invalid_mask_msg=None):
        with self._maybe_profile("depth_callback"):
            self._depth_pipeline.process(sem_msg, depth_msg, conf_msg, invalid_mask_msg)

    def _lidar_callback(self, sem_msg, lidar_msg, conf_msg=None, invalid_mask_msg=None):
        with self._maybe_profile("lidar_callback"):
            self._lidar_pipeline.process(sem_msg, lidar_msg, conf_msg, invalid_mask_msg)


def main():
    import threading
    log_level = rospy.DEBUG if _rosargv_bool("debug", False) else rospy.INFO
    rospy.init_node("colored_pcl_node", log_level=log_level)
    ColoredPclNode()
    num_threads = int(rospy.get_param("~spin_threads", 1))
    if num_threads > 1:
        threads = [threading.Thread(target=rospy.spin) for _ in range(num_threads - 1)]
        for t in threads:
            t.daemon = True
            t.start()
    rospy.spin()


if __name__ == "__main__":
    main()
