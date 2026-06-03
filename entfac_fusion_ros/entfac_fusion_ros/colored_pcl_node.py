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
import json
import pstats
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import rospy
import tf2_ros
from message_filters import ApproximateTimeSynchronizer, Cache, Subscriber
from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud2
from std_msgs.msg import Float32
from std_srvs.srv import SetBool, SetBoolResponse, Trigger, TriggerResponse

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

from entfac_fusion_core.colored_pcl import fuse_lidar_semantics
from entfac_fusion_core.calibration import CalibrationHealthSnapshot
from entfac_fusion_core.types import (
    PointObservation,
    SemanticPointCloud,
)
from entfac_fusion_core.utils.masks import (
    apply_invalid_projection_samples,
    invalid_image_to_mask,
    sample_invalid_mask,
)
from entfac_fusion_core.utils.validation import require_homogeneous_transform

from entfac_fusion_ros.conversions import (
    image_to_numpy,
    rgb_to_packed_u32,
)
from entfac_fusion_ros.experiment_metrics import FrameMetrics, MetricsCsvLogger
from entfac_fusion_ros.lidar_projector import (
    LidarProjector,
    LidarProjectorParams,
    ProjectionMetrics,
    ProjectionResult,
)
from entfac_fusion_ros.online_calibration import OnlineCalibration
from entfac_fusion_ros.tracked_reprojection import TrackedReprojection
from entfac_fusion_ros.debug_publisher import DebugPublisher, DebugPublisherParams
from entfac_fusion_ros.colored_pcl_params import (
    coerce_bool as _coerce_bool,
    get_color_map as _get_color_map_helper,
    get_matrix_param as _get_matrix_param_helper,
    get_param as _get_param_helper,
    get_param_bool as _get_param_bool_helper,
    get_param_float as _get_param_float_helper,
    get_param_int as _get_param_int_helper,
    get_param_str as _get_param_str_helper,
    load_camera_info_txt as _load_camera_info_txt_helper,
    record_param as _record_param_helper,
)
from entfac_fusion_ros.colored_pcl.config import (
    load_calibration_config,
    load_color_config,
    load_debug_config,
    load_experiment_config,
    load_ply_config,
    load_projection_config,
    load_sync_config,
)
from entfac_fusion_ros.colored_pcl.results import LastPcl
from entfac_fusion_ros.colored_pcl.camera_model import CameraModel
from entfac_fusion_ros.colored_pcl.ros_io import ColoredPclRosIo
from entfac_fusion_ros.colored_pcl.ply_service import PlyRecordingService
from entfac_fusion_ros.colored_pcl.diagnostics import DiagnosticsOrchestrator
from entfac_fusion_ros.colored_pcl.depth_pipeline import DepthFusionPipeline
from entfac_fusion_ros.colored_pcl.online_calibration_bridge import OnlineCalibrationBridge
from entfac_fusion_ros.colored_pcl.semantic_inputs import SemanticInputParser
from entfac_fusion_ros.colored_pcl.sync_policy import StampPolicy
from entfac_fusion_ros.colored_pcl.tf_resolver import TransformResolver
from entfac_fusion_ros.colored_pcl.lidar_pipeline import LidarFusionPipeline
from entfac_fusion_ros.colored_pcl.tracked_reprojection_runtime import TrackedReprojectionRuntime
from entfac_fusion_ros.colored_pcl_startup import (
    log_correction_statuses as _log_correction_statuses_helper,
    log_param_report as _log_param_report_helper,
    log_startup_transforms as _log_startup_transforms_helper,
    render_startup_table as _render_startup_table_helper,
)
from entfac_fusion_ros.logging_ros import NodeLogger, configure_core_logging
from entfac_fusion_ros.pc2 import labels_to_uint16
from entfac_fusion_ros.status import StatusReporter
from entfac_fusion_ros.tf_utils import format_matrix

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
        if self._compat_declared_lidar_T_points is not None:
            self._compat_lidar_points_status = "active (custom matrix)"
        elif self.compat_ouster_sensor_frame:
            # Default Ouster sensor->lidar compatibility transform for legacy bags
            # where XYZLut output was published as os_lidar. The 38.195 mm z-offset
            # comes from the recorded metadata.lidar_to_sensor_transform.
            compat = np.eye(4, dtype=float)
            compat[0, 0] = -1.0
            compat[1, 1] = -1.0
            compat[2, 3] = -0.038195
            self._compat_declared_lidar_T_points = require_homogeneous_transform(compat)
            self._compat_lidar_points_status = "active (ouster sensor->lidar)"
        else:
            self._compat_lidar_points_status = "disabled"
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

        sync_config = load_sync_config(self)
        self._apply_loaded_config(sync_config)
        color_config = load_color_config(self)
        self._apply_loaded_config(color_config)
        projection_config = load_projection_config(self)
        self._apply_loaded_config(projection_config)
        debug_config = load_debug_config(self)
        self._apply_loaded_config(debug_config)
        experiment_config = load_experiment_config(self)
        self._apply_loaded_config(experiment_config)
        calibration_config = load_calibration_config(self)
        self._apply_loaded_config(calibration_config)
        self._online_calibration_requested = bool(self.online_calibration_enable)
        self._online_calibration_status = (
            "requested" if self._online_calibration_requested else "disabled"
        )
        self._undistort_requested = bool(self.undistort_semantic)
        self._undistort_status = (
            "requested" if self._undistort_requested else "disabled"
        )
        self._rolling_shutter_requested = bool(self.rolling_shutter_enable)
        self._rolling_shutter_status = (
            "requested" if self._rolling_shutter_requested else "disabled"
        )
        self._lidar_deskew_requested = bool(self.lidar_deskew_enable)
        self._lidar_deskew_status = (
            "requested" if self._lidar_deskew_requested else "disabled"
        )
        self._apply_loaded_config(load_ply_config(self))
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

        self._camera_info_source = ""
        if self.camera_info_txt:
            (
                intrinsics,
                frame_id_from_file,
                distortion,
                distortion_model,
                camera_info_size,
                resolved_path,
            ) = self._load_camera_info_txt(self.camera_info_txt)
            self.intrinsics = intrinsics
            self.intrinsics_raw = self.intrinsics.copy()
            self._camera_distortion = distortion
            self._camera_distortion_model = distortion_model
            self._camera_info_size = camera_info_size
            self.camera_frame = frame_id_from_file or self.camera_frame_param
            if not self.camera_frame:
                raise ValueError(
                    "~camera_info_txt requires a camera frame. Add frame_id/camera_frame in the file or set ~camera_frame."
                )
            self._camera_info_source = f"txt:{resolved_path}"
            if self.camera_info_topic:
                self._log.info(
                    "__init__",
                    "Using camera intrinsics from ~camera_info_txt=%s (topic ~camera_info=%s is ignored for intrinsics).",
                    resolved_path,
                    self.camera_info_topic,
                )
            self._log.info(
                "__init__",
                "Camera intrinsics loaded from file: frame_id=%s size=%dx%d",
                self.camera_frame,
                int(self._camera_info_size[1]),
                int(self._camera_info_size[0]),
            )
            self._log.debug(
                "__init__",
                "Intrinsics K (file)=\n%s",
                format_matrix(self.intrinsics),
            )
        else:
            self._log.debug(
                "__init__", "Waiting for CameraInfo on topic=%s", self.camera_info_topic
            )
            cam_info = None
            wait_start = time.time()
            next_warn = wait_start + 5.0
            while cam_info is None and not rospy.is_shutdown():
                cam_info = self._wait_for_msg(
                    self.camera_info_topic,
                    CameraInfo,
                    timeout=1.0,
                    warn_on_timeout=False,
                )
                if cam_info is None and time.time() >= next_warn:
                    self._log.warn(
                        "__init__",
                        "Waiting for CameraInfo on %s (check topic name and bag/driver)",
                        self.camera_info_topic,
                    )
                    next_warn = time.time() + 5.0
            if cam_info is None:
                raise rospy.ROSInterruptException("Shutdown while waiting for CameraInfo")
            self.intrinsics = np.asarray(cam_info.K, dtype=float).reshape(3, 3)
            self.camera_frame = cam_info.header.frame_id
            self._log.debug(
                "__init__",
                "CameraInfo received: frame_id=%s stamp=%.6f",
                self.camera_frame,
                cam_info.header.stamp.to_sec(),
            )
            self._log.debug("__init__", "Intrinsics K=\n%s", format_matrix(self.intrinsics))
            self.intrinsics_raw = self.intrinsics.copy()
            self._camera_distortion = (
                np.asarray(cam_info.D, dtype=float) if cam_info.D else None
            )
            self._camera_distortion_model = (
                (cam_info.distortion_model or "").strip().lower()
            )
            self._camera_info_size = (int(cam_info.height), int(cam_info.width))
            self._camera_info_source = f"topic:{self.camera_info_topic}"

        self._output_topic = rospy.resolve_name("semantic_pointcloud")
        self.target_T_depth = None
        self.camera_T_lidar = None
        self.target_T_lidar = None
        self._depth_frame = ""
        self._lidar_frame = ""

        self._mode_source = "forced" if self.mode in ("depth", "lidar") else "auto"
        self._mode_detail = "forced via ~mode" if self._mode_source == "forced" else ""
        if self.mode not in ("depth", "lidar"):
            self.mode = self._detect_mode()
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
        self._tf_cache = self._tf_resolver.tf_cache
        self._prime_transforms()
        self._undistort_map1 = getattr(self._camera_model, "_undistort_map1", None)
        self._undistort_map2 = getattr(self._camera_model, "_undistort_map2", None)
        self._undistort_active = bool(getattr(self._camera_model, "_undistort_active", False))
        self._cv2 = getattr(self._camera_model, "_cv2", None)
        self._imu_cache = None
        self._imu_sub = None
        self._imu_to_camera_R = None
        self._metadata_latest = {}
        self._metadata_sub = None
        self._lidar_imu_cache = None
        self._lidar_imu_sub = None
        self._lidar_imu_to_lidar_R = None
        self._debug_callback_seq = 0
        self._rolling_shutter_log_at = 0.0
        self._rolling_shutter_warn_at = 0.0
        self._lidar_deskew_log_at = 0.0
        self._lidar_deskew_warn_at = 0.0
        self._lidar_deskew_missing_time_warn_at = 0.0
        self._live_param_refresh_period_sec = 0.5
        self._live_param_last_refresh_at = 0.0
        self._dynamic_reconfigure_server = None
        self._dynamic_reconfigure_initialized = False

        if self.tracked_reprojection_enable:
            if self.mode != "lidar":
                self._log.warn(
                    "__init__",
                    "tracked_reprojection_enable=true requires lidar mode; disabling because mode=%s",
                    self.mode,
                )
                self.tracked_reprojection_enable = False
            else:
                self._log.warn(
                    "__init__",
                    "Tracked reprojection diagnostics are stateful and CPU-heavier than the online path; use them primarily for offline bag review or focused validation runs.",
                )

        self._rgb_lut = None
        self._rgb_lut_num_labels = None
        self._warned_random_palette = False
        self._warned_rgb_color_map = False
        self._logged_depth_scaling = False
        self._logged_depth_summary = False
        self._logged_lidar_summary = False
        self._stamp_debug_last_log_at = 0.0

        self._ply_writer = self._ply_service._writer
        self._ply_recording = False
        self._ply_queue_warned_at = 0.0
        self._ply_seq = 0
        self._last_pcl: Optional[LastPcl] = None
        self._results_frame_index = 0
        self._metrics_logger: Optional[MetricsCsvLogger] = None
        self._results_run_dir: Optional[Path] = None
        if self.enable_metrics_csv:
            bag_name = self.experiment_bag_name or "unknown_bag"
            variant_name = self.experiment_variant_name or "default"
            root = Path(self.results_dir or (Path(self.debug_output_dir).parent / "results"))
            self._results_run_dir = root / bag_name / variant_name
            self._results_run_dir.mkdir(parents=True, exist_ok=True)
            if self.enable_metrics_csv:
                self._metrics_logger = MetricsCsvLogger(
                    self._results_run_dir / "metrics_per_frame.csv"
                )
                rospy.on_shutdown(self._close_metrics_logger)

        self._ply_service.setup()

        # Persistent buffers for LiDAR rasterization — kept here for the
        # _publish_range_view_debug and online-calibration paths that call
        # _rasterize_lidar_depth_map / _depth_map_to_edge_map directly.
        # The projector owns its own copies; these are legacy for those paths.
        self._depth_buffer: Optional[np.ndarray] = None
        self._depth_buffer_shape: Optional[Tuple[int, int]] = None
        self._edge_buffer: Optional[np.ndarray] = None
        self._edge_buffer_shape: Optional[Tuple[int, int]] = None

        # Pure-numpy projection engine — no ROS dependency.
        self._projector = LidarProjector(self._build_projector_params())

        # Optional subsystems — None when disabled so hot path has zero branch cost.
        self._calibration: Optional[OnlineCalibration] = self._calibration_bridge.build(self._projector)
        self._tracked_repr: Optional[TrackedReprojection] = self._tracked_runtime.build()
        self._debug_pub: Optional[DebugPublisher] = (
            DebugPublisher(
                self._build_debug_pub_params(),
                node_name=self._node_name,
                lidar_frame=self._lidar_frame,
                target_frame=self.target_frame,
            )
            if any([
                self.debug_project_lidar, self.debug_range_view,
                self.debug_publish_fov_points, self.tracked_reprojection_enable,
                self.online_calibration_enable,
            ]) else None
        )
        self._diagnostics = DiagnosticsOrchestrator(self, self._debug_pub)
        self._depth_pipeline = DepthFusionPipeline(self)
        self._lidar_pipeline = LidarFusionPipeline(self)

        self._ros_io = ColoredPclRosIo(self)
        self._ros_io.setup_publishers()
        self._setup_dynamic_reconfigure()
        self._ros_io.setup_services()
        self._ros_io.register_subscribers()
        self._log.info("__init__", "\n%s", self._render_startup_table())
        self._log_correction_statuses()
        self._log_startup_transforms()
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
            self._log_param_report()

    # ----------------------------
    # Param helpers (with meta)
    # ----------------------------

    def _record_param(self, name, value, source, description):
        return _record_param_helper(self, name, value, source, description)

    def _get_param(self, name, default, description, *, allow_empty=False):
        return _get_param_helper(
            self, name, default, description, allow_empty=allow_empty
        )

    def _get_param_str(self, name, default, description, *, allow_empty=False):
        return _get_param_str_helper(
            self, name, default, description, allow_empty=allow_empty
        )

    def _get_param_bool(self, name, default, description):
        return _get_param_bool_helper(self, name, default, description)

    def _get_param_int(self, name, default, description):
        return _get_param_int_helper(self, name, default, description)

    def _get_param_float(self, name, default, description):
        return _get_param_float_helper(self, name, default, description)

    def _get_matrix_param(self, name, description):
        return _get_matrix_param_helper(self, name, description)

    def _get_color_map(self, name, description):
        return _get_color_map_helper(self, name, description)

    def _apply_loaded_config(self, config: Any) -> None:
        for field in fields(config):
            setattr(self, field.name, getattr(config, field.name))

    def _load_camera_info_txt(
        self, txt_path: str
    ) -> Tuple[np.ndarray, str, Optional[np.ndarray], str, Tuple[int, int], str]:
        return _load_camera_info_txt_helper(self, txt_path)

    def _build_projector_params(self) -> LidarProjectorParams:
        return LidarProjectorParams(
            max_depth_m=self.max_depth_m,
            camera_fov_gate_enable=bool(self.camera_fov_gate_enable),
            camera_fov_gate_margin_deg=float(self.camera_fov_gate_margin_deg),
            rolling_shutter_enable=bool(self.rolling_shutter_enable),
            rolling_shutter_direction=str(self.rolling_shutter_direction),
            projection_patch_size=int(self.projection_patch_size),
            projection_occlusion_epsilon_m=float(self.projection_occlusion_epsilon_m),
            projection_occlusion_radius_px=int(self.projection_occlusion_radius_px),
            projection_reject_depth_edges=bool(self.projection_reject_depth_edges),
            projection_depth_edge_thresh=float(self.projection_depth_edge_thresh),
            projection_depth_edge_radius_px=int(self.projection_depth_edge_radius_px),
            projection_confidence_min=float(self.projection_confidence_min),
            use_invalid_mask=bool(self.use_invalid_mask),
            use_depth_edge_rejection=bool(self.use_depth_edge_rejection),
            use_occlusion_gate=bool(self.use_occlusion_gate),
            include_unlabeled=bool(self.include_unlabeled),
            colorize_labels=bool(self.colorize_labels),
            semantic_input_type=str(self.semantic_input_type),
            color_map=dict(self.color_map) if self.color_map else {},
            random_color_seed=int(self.random_color_seed),
            num_labels=int(self.num_labels),
            debug_project_lidar=bool(self.debug_project_lidar),
        )

    def _build_debug_pub_params(self) -> DebugPublisherParams:
        return DebugPublisherParams(
            debug_project_lidar=bool(self.debug_project_lidar),
            debug_project_lidar_stride=int(self.debug_project_lidar_stride),
            debug_project_lidar_radius=int(self.debug_project_lidar_radius),
            debug_project_lidar_outline_only=bool(self.debug_project_lidar_outline_only),
            debug_range_view=bool(self.debug_range_view),
            debug_publish_fov_points=bool(self.debug_publish_fov_points),
            tracked_reprojection_enable=bool(self.tracked_reprojection_enable),
            online_calibration_enable=bool(self.online_calibration_enable),
            debug_output_dir=str(self.debug_output_dir),
            debug_output_stride=int(self.debug_output_stride),
        )

    # ----------------------------
    # Startup report
    # ----------------------------

    def _log_startup_transforms(self):
        _log_startup_transforms_helper(self)

    def _log_param_report(self):
        _log_param_report_helper(self)

    def _log_correction_statuses(self) -> None:
        _log_correction_statuses_helper(self)

    # ----------------------------
    # Subscriptions / TF
    # ----------------------------

    def _register_subscribers(self):
        semantic_sub = Subscriber(self.semantic_topic, Image, queue_size=self.sync_queue_size)
        conf_sub = (
            Subscriber(self.conf_topic, Image, queue_size=self.sync_queue_size)
            if self.conf_topic
            else None
        )
        invalid_mask_sub = (
            Subscriber(
                self.projection_invalid_mask_topic,
                Image,
                queue_size=self.sync_queue_size,
            )
            if self.projection_invalid_mask_topic
            else None
        )
        self._configure_rolling_shutter_subscribers()
        self._configure_lidar_deskew_subscribers()
        if self.mode == "depth":
            depth_sub = Subscriber(self.depth_input_topic, Image)
            subs = [semantic_sub, depth_sub]
            if conf_sub is not None:
                subs.append(conf_sub)
            if invalid_mask_sub is not None:
                subs.append(invalid_mask_sub)
            sync = ApproximateTimeSynchronizer(
                subs, queue_size=self.sync_queue_size, slop=self.sync_slop_sec
            )
            sync.registerCallback(self._build_depth_sync_callback(conf_sub, invalid_mask_sub))
            self._sync = sync
        else:
            lidar_sub = Subscriber(self.depth_input_topic, PointCloud2)
            subs = [semantic_sub, lidar_sub]
            if conf_sub is not None:
                subs.append(conf_sub)
            if invalid_mask_sub is not None:
                subs.append(invalid_mask_sub)
            sync = ApproximateTimeSynchronizer(
                subs, queue_size=self.sync_queue_size, slop=self.sync_slop_sec
            )
            sync.registerCallback(self._build_lidar_sync_callback(conf_sub, invalid_mask_sub))
            self._sync = sync

        self._log.debug(
            "_register_subscribers",
            "Registering subscribers (mode=%s): semantic=%s depth=%s lidar=%s confidence=%s invalid_mask=%s",
            self.mode,
            self.semantic_topic,
            self.depth_input_topic,
            self.depth_input_topic,
            self.conf_topic,
            self.projection_invalid_mask_topic or "",
        )

    def _configure_rolling_shutter_subscribers(self) -> None:
        self._rolling_shutter_status = (
            "disabled" if not self._rolling_shutter_requested else "requested"
        )
        if not self.rolling_shutter_enable:
            return
        if not self.imu_topic:
            self._rolling_shutter_status = "disabled (missing ~imu_topic)"
            self._log.warn(
                "_register_subscribers",
                "rolling_shutter_enable is true but ~imu_topic is empty; disabling rolling shutter.",
            )
            self.rolling_shutter_enable = False
            return

        self._imu_sub = Subscriber(self.imu_topic, Imu, queue_size=2000)
        self._imu_cache = Cache(self._imu_sub, self.imu_cache_size)
        if self.rolling_shutter_readout_sec > 0.0:
            self._rolling_shutter_status = "armed (fixed readout)"
        elif self.camera_metadata_topic and self.metadata_readout_key >= 0:
            self._rolling_shutter_status = "armed (metadata readout)"
        else:
            self._rolling_shutter_status = "idle (readout=0 and metadata disabled)"

        if self.camera_metadata_topic and self.metadata_readout_key >= 0:
            try:
                from realsense2_camera_msgs.msg import Metadata  # type: ignore
            except Exception as exc:  # noqa: BLE001
                self._log.warn(
                    "_register_subscribers",
                    "Cannot import realsense2_camera_msgs/Metadata (%s); metadata readout disabled.",
                    exc,
                )
            else:
                self._metadata_sub = rospy.Subscriber(
                    self.camera_metadata_topic,
                    Metadata,
                    self._metadata_callback,
                    queue_size=2000,
                )

    def _configure_lidar_deskew_subscribers(self) -> None:
        self._lidar_deskew_status = (
            "disabled" if not self._lidar_deskew_requested else "requested"
        )
        if not self.lidar_deskew_enable:
            return
        if not self.lidar_imu_topic:
            self._lidar_deskew_status = "disabled (missing ~lidar_imu_topic)"
            self._log.warn(
                "_register_subscribers",
                "lidar_deskew_enable is true but ~lidar_imu_topic is empty; disabling deskew.",
            )
            self.lidar_deskew_enable = False
            return

        self._lidar_imu_sub = Subscriber(self.lidar_imu_topic, Imu, queue_size=2000)
        self._lidar_imu_cache = Cache(self._lidar_imu_sub, self.lidar_imu_cache_size)
        self._lidar_deskew_status = "armed"

    def _build_depth_sync_callback(self, conf_sub, invalid_mask_sub):
        if conf_sub is not None and invalid_mask_sub is not None:
            return self._depth_callback
        if conf_sub is not None:
            return lambda sem, depth, conf: self._depth_callback(sem, depth, conf, None)
        if invalid_mask_sub is not None:
            return lambda sem, depth, invalid_mask: self._depth_callback(
                sem, depth, None, invalid_mask
            )
        return self._depth_callback

    def _build_lidar_sync_callback(self, conf_sub, invalid_mask_sub):
        if conf_sub is not None and invalid_mask_sub is not None:
            return self._lidar_callback
        if conf_sub is not None:
            return lambda sem, lidar, conf: self._lidar_callback(sem, lidar, conf, None)
        if invalid_mask_sub is not None:
            return lambda sem, lidar, invalid_mask: self._lidar_callback(
                sem, lidar, None, invalid_mask
            )
        return self._lidar_callback

    def _setup_dynamic_reconfigure(self) -> None:
        if DynamicReconfigureServer is None or ColoredPclTuningConfig is None:
            missing = []
            if DynamicReconfigureServer is None:
                missing.append("dynamic_reconfigure.server")
            if ColoredPclTuningConfig is None:
                missing.append("entfac_fusion_ros.cfg.ColoredPclTuningConfig")
            self._log.warn(
                "_setup_dynamic_reconfigure",
                "rqt_reconfigure support is unavailable because %s could not be imported. Build the catkin workspace so the generated dynamic_reconfigure modules exist.",
                ", ".join(missing),
            )
            return

        self._dynamic_reconfigure_server = DynamicReconfigureServer(
            ColoredPclTuningConfig, self._dynamic_reconfigure_callback
        )
        self._log.info(
            "_setup_dynamic_reconfigure",
            "rqt_reconfigure is ready on %s for live projection/debug tuning.",
            rospy.get_name(),
        )

    def _apply_tuning_params(self, get_value, log_source: str = "") -> bool:
        """Apply tuning params from get_value callable. Returns True if any changed.

        get_value(attr, default, validator) should return value if valid, else raise.
        """
        changes = []

        def _update(attr: str, default, validator) -> None:
            try:
                value = get_value(attr, default)
                if not validator(value):
                    return
            except Exception:  # noqa: BLE001
                return
            current = getattr(self, attr)
            if current != value:
                setattr(self, attr, value)
                changes.append(f"{attr}={value}")

        _update("projection_patch_size", self.projection_patch_size, lambda v: v >= 1 and (v % 2) == 1)
        _update("projection_confidence_min", self.projection_confidence_min, lambda v: 0.0 <= v <= 1.0)
        _update("projection_occlusion_epsilon_m", self.projection_occlusion_epsilon_m, lambda v: v >= 0.0)
        _update("projection_occlusion_radius_px", self.projection_occlusion_radius_px, lambda v: v >= 0)
        _update("projection_reject_depth_edges", self.projection_reject_depth_edges, lambda v: isinstance(v, bool))
        _update("projection_depth_edge_thresh", self.projection_depth_edge_thresh, lambda v: 0.0 <= v <= 1.0)
        _update("projection_depth_edge_radius_px", self.projection_depth_edge_radius_px, lambda v: v >= 0)
        _update("debug_project_lidar", self.debug_project_lidar, lambda v: isinstance(v, bool))
        _update("debug_project_lidar_stride", self.debug_project_lidar_stride, lambda v: v >= 1)
        _update("debug_project_lidar_radius", self.debug_project_lidar_radius, lambda v: v >= 0)
        _update("debug_project_lidar_outline_only", self.debug_project_lidar_outline_only, lambda v: isinstance(v, bool))
        _update("tracked_reprojection_fb_thresh_px", self.tracked_reprojection_fb_thresh_px, lambda v: v > 0.0)
        _update("tracked_reprojection_depth_edge_thresh", self.tracked_reprojection_depth_edge_thresh, lambda v: 0.0 <= v <= 1.0)
        _update("tracked_reprojection_min_image_edge", self.tracked_reprojection_min_image_edge, lambda v: 0.0 <= v <= 1.0)
        _update("tracked_reprojection_min_tracks", self.tracked_reprojection_min_tracks, lambda v: v >= 10)

        if self.debug_project_lidar and self._debug_proj_pub is None:
            self._debug_proj_pub = rospy.Publisher(
                self.debug_projected_topic, Image, queue_size=1
            )

        if changes:
            self._projector.update_params(self._build_projector_params())
            if self._debug_pub is not None:
                self._debug_pub.update_params(self._build_debug_pub_params())
            if log_source:
                self._log.info(log_source, "Live tuning update: %s", ", ".join(changes))
        return bool(changes)

    def _dynamic_reconfigure_callback(self, config, _level):
        initialized = self._dynamic_reconfigure_initialized

        patch_size = int(config["projection_patch_size"])
        if patch_size < 1:
            patch_size = 1
        if (patch_size % 2) == 0:
            patch_size = patch_size + 1 if patch_size < 9 else patch_size - 1
        config["projection_patch_size"] = patch_size

        def get_from_config(attr: str, default):
            key = attr
            if key in config:
                return config[key]
            raise KeyError(key)

        self._apply_tuning_params(get_from_config, "_dynamic_reconfigure_callback" if initialized else "")
        self._dynamic_reconfigure_initialized = True
        return config

    def _get_live_param_float(self, name: str, fallback: float) -> float:
        try:
            return float(rospy.get_param(name, fallback))
        except Exception:  # noqa: BLE001
            return fallback

    def _get_live_param_int(self, name: str, fallback: int) -> int:
        try:
            return int(rospy.get_param(name, fallback))
        except Exception:  # noqa: BLE001
            return fallback

    def _get_live_param_bool(self, name: str, fallback: bool) -> bool:
        try:
            return _coerce_bool(rospy.get_param(name, fallback))
        except Exception:  # noqa: BLE001
            return fallback

    def _maybe_refresh_live_tuning_params(self) -> None:
        now = time.time()
        if (
            self._live_param_last_refresh_at > 0.0
            and (now - self._live_param_last_refresh_at) < self._live_param_refresh_period_sec
        ):
            return
        self._live_param_last_refresh_at = now

        def get_from_rospy(attr: str, default):
            if attr == "projection_patch_size":
                return self._get_live_param_int(f"~{attr}", default)
            elif attr in {"projection_confidence_min", "projection_occlusion_epsilon_m", "projection_depth_edge_thresh", "tracked_reprojection_fb_thresh_px", "tracked_reprojection_depth_edge_thresh", "tracked_reprojection_min_image_edge"}:
                return self._get_live_param_float(f"~{attr}", default)
            elif attr in {"projection_occlusion_radius_px", "projection_depth_edge_radius_px", "debug_project_lidar_stride", "debug_project_lidar_radius", "tracked_reprojection_min_tracks"}:
                return self._get_live_param_int(f"~{attr}", default)
            else:
                return self._get_live_param_bool(f"~{attr}", default)

        self._apply_tuning_params(get_from_rospy, "_maybe_refresh_live_tuning_params")

    def _parse_semantic_inputs(
        self, sem_msg, conf_msg, invalid_mask_msg, callback_name: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        parsed = self._semantic_parser.parse(sem_msg, conf_msg, invalid_mask_msg, callback_name)
        return (
            parsed.labels,
            parsed.packed_rgb,
            parsed.confidence,
            parsed.projection_invalid_mask,
        )

    def _prepare_frame_inputs(
        self,
        sem_msg,
        conf_msg,
        invalid_mask_msg,
        callback_name: str,
    ) -> Tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        bool,
        np.ndarray,
        Tuple[int, int],
        str,
        np.ndarray,
    ]:
        include_rgb = bool(self.colorize_labels) if self.semantic_input_type == "labels" else True
        labels, packed_img, confidence, projection_invalid_mask, rgb_lut = self._parse_semantic_inputs(
            sem_msg, conf_msg, invalid_mask_msg, callback_name
        )
        semantic_debug_type = "labels" if labels is not None else "rgb"
        semantic_debug_img = labels if labels is not None else packed_img
        if semantic_debug_img is None:
            raise ValueError("semantic input could not be prepared")

        if self.downsample_factor > 1:
            f = self.downsample_factor
            if labels is not None:
                labels = labels[::f, ::f]
            else:
                packed_img = packed_img[::f, ::f]
            if confidence is not None:
                confidence = confidence[::f, ::f]
            if projection_invalid_mask is not None:
                projection_invalid_mask = projection_invalid_mask[::f, ::f]
            intrinsics = self._scale_intrinsics(self.intrinsics, f)
        else:
            intrinsics = self.intrinsics

        semantic_shape = (
            labels.shape if labels is not None else packed_img.shape[:2]
        )
        semantic_debug_img = labels if labels is not None else packed_img
        if semantic_debug_img is None:
            raise ValueError("semantic input could not be prepared")
        return (
            labels,
            packed_img,
            confidence,
            projection_invalid_mask,
            rgb_lut,
            include_rgb,
            intrinsics,
            semantic_shape,
            semantic_debug_type,
            semantic_debug_img,
        )

    def _write_lidar_metrics(
        self,
        *,
        frame_index: int,
        sem_msg,
        lidar_msg,
        pair_dt_sec: float,
        pair_accepted: int,
        drop_reason: str,
        num_input_points: int,
        projection_metrics: ProjectionMetrics,
        num_output_points: int,
        runtime_total_ms: float,
        runtime_publish_ms: float,
    ) -> None:
        if self._metrics_logger is None:
            return
        projected = int(projection_metrics.num_points_projected_in_image)
        known_rejected = (
            int(projection_metrics.num_rejected_invalid_mask)
            + int(projection_metrics.num_rejected_confidence)
            + int(projection_metrics.num_rejected_depth_edge)
            + int(projection_metrics.num_rejected_occlusion)
        )
        num_rejected_other = max(0, projected - int(num_output_points) - known_rejected)
        output_retention_ratio = float(num_output_points) / max(projected, 1)
        self._metrics_logger.write(
            FrameMetrics(
                bag_name=self.experiment_bag_name or "unknown_bag",
                variant_name=self.experiment_variant_name or "default",
                frame_index=int(frame_index),
                stamp_semantic=float(sem_msg.header.stamp.to_sec()),
                stamp_cloud=float(lidar_msg.header.stamp.to_sec()),
                pair_dt_sec=float(pair_dt_sec),
                pair_accepted=int(pair_accepted),
                drop_reason=drop_reason,
                num_input_points=int(num_input_points),
                num_points_in_front=int(projection_metrics.num_points_in_front),
                num_points_projected_in_image=projected,
                num_rejected_invalid_mask=int(projection_metrics.num_rejected_invalid_mask),
                num_rejected_confidence=int(projection_metrics.num_rejected_confidence),
                num_rejected_depth_edge=int(projection_metrics.num_rejected_depth_edge),
                num_rejected_occlusion=int(projection_metrics.num_rejected_occlusion),
                num_rejected_other=num_rejected_other,
                num_output_points=int(num_output_points),
                output_retention_ratio=output_retention_ratio,
                runtime_total_ms=float(runtime_total_ms),
                runtime_projection_ms=float(projection_metrics.runtime_projection_ms),
                runtime_mask_ms=float(projection_metrics.runtime_mask_ms),
                runtime_rasterize_ms=float(projection_metrics.runtime_rasterize_ms),
                runtime_depth_edge_ms=float(projection_metrics.runtime_depth_edge_ms),
                runtime_occlusion_ms=float(projection_metrics.runtime_occlusion_ms),
                runtime_publish_ms=float(runtime_publish_ms),
                num_would_hit_invalid_mask=int(projection_metrics.num_would_hit_invalid_mask),
                would_hit_invalid_mask_ratio=float(projection_metrics.num_would_hit_invalid_mask) / max(projected, 1),
                num_would_hit_depth_edge=int(projection_metrics.num_would_hit_depth_edge),
                would_hit_depth_edge_ratio=float(projection_metrics.num_would_hit_depth_edge) / max(projected, 1),
                num_would_fail_occlusion=int(projection_metrics.num_would_fail_occlusion),
                would_fail_occlusion_ratio=float(projection_metrics.num_would_fail_occlusion) / max(projected, 1),
            )
        )

    def _close_metrics_logger(self) -> None:
        if self._metrics_logger is not None:
            self._metrics_logger.close()
            self._metrics_logger = None

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

    def _wait_for_msg(self, topic, msg_type, timeout=2.0, warn_on_timeout=True):
        try:
            self._log.debug(
                "_wait_for_msg",
                "Waiting for %s on topic=%s (timeout=%.2fs)",
                msg_type.__name__,
                topic,
                float(timeout),
            )
            msg = rospy.wait_for_message(topic, msg_type, timeout=timeout)
            self._log.debug(
                "_wait_for_msg",
                "Received %s on topic=%s (stamp=%.6f frame_id=%s)",
                msg_type.__name__,
                topic,
                msg.header.stamp.to_sec() if hasattr(msg, "header") else 0.0,
                getattr(getattr(msg, "header", None), "frame_id", ""),
            )
            return msg
        except rospy.ROSException as exc:
            if warn_on_timeout:
                self._log.warn(
                    "_wait_for_msg", "Timeout waiting for %s: %s", topic, exc
                )
            else:
                self._log.debug(
                    "_wait_for_msg", "Timeout waiting for %s: %s", topic, exc
                )
            return None

    def _wait_for_topic_type(self, topic, timeout=2.0, warn_on_timeout=True):
        start = time.time()
        while not rospy.is_shutdown():
            try:
                published = rospy.get_published_topics(namespace="/")
                for name, type_str in published:
                    if name == topic:
                        return type_str
                if (time.time() - start) > float(timeout):
                    raise rospy.ROSException("timeout exceeded while waiting for topic type")
                rospy.sleep(0.05)
            except rospy.ROSException as exc:
                if warn_on_timeout:
                    self._log.warn(
                        "_wait_for_topic_type",
                        "Timeout waiting for topic type on %s: %s",
                        topic,
                        exc,
                    )
                else:
                    self._log.debug(
                        "_wait_for_topic_type",
                        "Timeout waiting for topic type on %s: %s",
                        topic,
                        exc,
                    )
                return None

    def _detect_mode(self):
        """Detect mode based on configured topics' message types."""
        type_str = None
        wait_start = time.time()
        next_warn = wait_start + 5.0
        while type_str is None and not rospy.is_shutdown():
            type_str = self._wait_for_topic_type(
                self.depth_input_topic, timeout=1.0, warn_on_timeout=False
            )
            if type_str is None and time.time() >= next_warn:
                self._log.warn(
                    "_detect_mode",
                    "Waiting for %s to appear to auto-detect mode (expected sensor_msgs/Image or sensor_msgs/PointCloud2)",
                    self.depth_input_topic,
                )
                next_warn = time.time() + 5.0

        if type_str == "sensor_msgs/Image":
            self._mode_source = "auto"
            self._mode_detail = (
                f"auto via ~depth_input_topic={self.depth_input_topic} ({type_str})"
            )
            return "depth"
        if type_str == "sensor_msgs/PointCloud2":
            self._mode_source = "auto"
            self._mode_detail = (
                f"auto via ~depth_input_topic={self.depth_input_topic} ({type_str})"
            )
            return "lidar"
        if type_str == "sensor_msgs/CompressedImage":
            raise ValueError(
                "~depth_input_topic is sensor_msgs/CompressedImage; republish to raw Image via image_transport or set use_republish:=true in launch"
            )
        if type_str:
            raise ValueError(
                f"Unsupported ~depth_input_topic message type: {type_str} (expected sensor_msgs/Image or sensor_msgs/PointCloud2)"
            )
        raise ValueError(
            f"Unable to determine message type for ~depth_input_topic={self.depth_input_topic}"
        )

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

    def _prime_transforms(self):
        if self.mode == "depth":
            if self.static_target_T_depth is not None:
                self.target_T_depth = self.static_target_T_depth
                self._depth_frame = "<depth_frame>"
                return
            msg = self._wait_for_msg(self.depth_input_topic, Image, timeout=5.0)
            if msg is None:
                self._log.debug(
                    "_prime_transforms",
                    "No depth message available at init; will lookup depth->target on first callback",
                )
                return
            depth_frame = msg.header.frame_id
            self._depth_frame = depth_frame or ""
            if depth_frame:
                mat = self._lookup_transform(
                    self.target_frame, depth_frame, rospy.Time(0)
                )
                if mat is not None:
                    self.target_T_depth = mat
            return

        # LiDAR mode
        if self.static_camera_T_lidar is not None:
            self.camera_T_lidar = self.static_camera_T_lidar
        if self.static_target_T_lidar is not None:
            self.target_T_lidar = self.static_target_T_lidar
        if self.camera_T_lidar is not None and self.target_T_lidar is not None:
            return

        lidar_msg = self._wait_for_msg(
            self.depth_input_topic, PointCloud2, timeout=5.0
        )
        if lidar_msg is None:
            self._log.debug(
                "_prime_transforms",
                "No LiDAR message available at init; will lookup transforms on first callback",
            )
            return
        lidar_frame = lidar_msg.header.frame_id
        self._lidar_frame = lidar_frame or ""
        if not lidar_frame:
            self._log.warn("_prime_transforms", "LiDAR message has empty frame_id")
            return
        if self.camera_T_lidar is None:
            mat = self._lookup_transform(
                self.camera_frame, lidar_frame, rospy.Time(0)
            )
            if mat is not None:
                self.camera_T_lidar = mat
        if self.target_T_lidar is None:
            mat = self._lookup_transform(
                self.target_frame, lidar_frame, rospy.Time(0)
            )
            if mat is not None:
                self.target_T_lidar = mat

    # ----------------------------
    # Semantic parsing and coloring
    # ----------------------------








    def _get_rgb_float_lut(self, labels_img: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Delegate to LidarProjector (owns the LUT cache and warned-flag)."""
        return self._projector._get_rgb_float_lut(labels_img)

    # ----------------------------
    # PLY services
    # ----------------------------

    # ----------------------------
    # Callbacks
    # ----------------------------

    def _maybe_emit_status(self, *, points: int, callback_sec: float) -> None:
        self._diagnostics.emit_status(points=int(points), callback_sec=float(callback_sec))

    def _render_startup_table(self) -> str:
        return _render_startup_table_helper(
            self,
            rospy.resolve_name("~save_ply"),
            rospy.resolve_name("~set_ply_recording"),
        )

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
