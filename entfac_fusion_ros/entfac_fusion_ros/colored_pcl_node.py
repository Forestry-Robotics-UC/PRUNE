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
from dataclasses import dataclass
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

from entfac_fusion_core.colored_pcl import (
    fuse_depth_semantics,
    fuse_lidar_semantics,
)
from entfac_fusion_core.colored_pcl.fusion import (
    sample_projected_label_patches,
    sample_projected_rgb_patches,
)
from entfac_fusion_core.calibration import (
    CalibrationHealthSnapshot,
    OnlineCalibrationHealth,
)
from entfac_fusion_core.projection.depth import depth_to_points
from entfac_fusion_core.projection.lidar_projection import project_points_to_image
from entfac_fusion_core.transforms.se3 import transform_points
from entfac_fusion_core.types import (
    DepthObservation,
    PointObservation,
    SemanticObservation,
    SemanticPointCloud,
)
from entfac_fusion_core.utils.masks import (
    apply_invalid_projection_samples,
    filter_invalid_projection_samples,
    invalid_image_to_mask,
    sample_invalid_mask,
)
from entfac_fusion_core.utils.semantics import packed_rgb_to_triplets
from entfac_fusion_core.utils.validation import (
    flatten_masked,
    require_homogeneous_transform,
)

from entfac_fusion_ros.conversions import (
    image_to_numpy,
    pointcloud2_to_xyz,
    pointcloud2_to_xyz_t,
    rgb_to_packed_u32,
)
from entfac_fusion_ros.experiment_metrics import FrameMetrics, MetricsCsvLogger
from entfac_fusion_ros.lidar_projector import (
    LidarProjector,
    LidarProjectorParams,
    ProjectionMetrics,
    ProjectionResult,
)
from entfac_fusion_ros.online_calibration import OnlineCalibration, OnlineCalibrationParams
from entfac_fusion_ros.tracked_reprojection import TrackedReprojection, TrackedReprojectionParams
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
from entfac_fusion_ros.colored_pcl_startup import (
    log_correction_statuses as _log_correction_statuses_helper,
    log_param_report as _log_param_report_helper,
    log_startup_transforms as _log_startup_transforms_helper,
    render_startup_table as _render_startup_table_helper,
)
from entfac_fusion_ros.logging_ros import NodeLogger, configure_core_logging
from entfac_fusion_ros.pc2 import (
    build_label_rgb_float_lut,
    labels_to_uint16,
    semantic_pointcloud_to_msg,
)
from entfac_fusion_ros.ply import PlyJob, PlyWriterThread
from entfac_fusion_ros.status import StatusReporter, render_status_table
from entfac_fusion_ros.tf_utils import format_matrix, transform_stamped_to_matrix
from entfac_fusion_ros.imu_cache import interpolate_imu_msg

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
class _LastPcl:
    stamp: rospy.Time
    points_xyz: np.ndarray
    labels: np.ndarray
    confidence: Optional[np.ndarray]
    rgb_packed_float: Optional[np.ndarray]


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

        self.include_unlabeled = self._get_param_bool(
            "~include_unlabeled_pts",
            False,
            "If true, keep points outside the camera FOV (label=-1).",
        )
        self.perception_invalid_label = self._get_param_int(
            "~perception_invalid_label",
            65535,
            "Label value from Perception indicating invalid/low-confidence pixels; mapped to -1 (unlabeled) before fusion. ENTFAC-Perception uses 65535 by default.",
        )
        self.colorize_labels = self._get_param_bool(
            "~colorize_labels",
            False,
            "If true, publish an extra PointCloud2 field 'rgb' (label palette in 'labels' mode; passthrough colors in 'rgb' mode).",
        )
        self.color_map = self._get_color_map(
            "~color_map",
            "Optional dict {label_id: [r,g,b]} used to colorize labels when ~semantic_input_type='labels'. YAML keys must be quoted (e.g. \"0\": [0,0,0]).",
        )
        self.random_color_seed = self._get_param_int(
            "~random_color_seed",
            1,
            "Seed for deterministic random label palette when ~colorize_labels is true and ~color_map is empty.",
        )
        self.num_labels = self._get_param_int(
            "~num_labels",
            0,
            "Optional number of label IDs (0=auto from first label image). Used only when ~semantic_input_type='labels' and ~colorize_labels is true with empty ~color_map.",
        )
        self.semantic_color_quantization_step = self._get_param_int(
            "~semantic_color_quantization_step",
            1,
            "Quantize RGB/BGR semantic images to nearest multiple of this step before packing for the PointCloud2 rgb field (helps with JPEG artifacts).",
        )
        if self.semantic_color_quantization_step < 1:
            raise ValueError("~semantic_color_quantization_step must be >= 1")
        self.projection_patch_size = self._get_param_int(
            "~projection_patch_size",
            1,
            "Odd patch size for robust LiDAR-to-image sampling (1=center pixel, 3=3x3, 5=5x5).",
        )
        if self.projection_patch_size < 1 or self.projection_patch_size % 2 == 0:
            raise ValueError("~projection_patch_size must be an odd integer >= 1")
        self.projection_confidence_min = self._get_param_float(
            "~projection_confidence_min",
            0.0,
            "Minimum patch confidence required to trust transferred image color/label (0 disables).",
        )
        if not 0.0 <= self.projection_confidence_min <= 1.0:
            raise ValueError("~projection_confidence_min must be in [0, 1]")
        self.projection_invalid_mask_topic = self._get_param_str(
            "~projection_invalid_mask_topic",
            "",
            "Optional single-channel invalid-mask image topic aligned with ~semantic_topic; pixels equal to ~projection_invalid_mask_value reject transferred labels/RGB.",
            allow_empty=True,
        )
        self.projection_invalid_mask_value = self._get_param_int(
            "~projection_invalid_mask_value",
            255,
            "Pixel value in ~projection_invalid_mask_topic that marks invalid/rejected samples.",
        )
        if not 0 <= self.projection_invalid_mask_value <= 65535:
            raise ValueError("~projection_invalid_mask_value must be in [0, 65535]")
        self.projection_invalid_mask_dilate_px = self._get_param_int(
            "~projection_invalid_mask_dilate_px",
            0,
            "Optional dilation radius in pixels applied to the invalid mask before projection sampling.",
        )
        if self.projection_invalid_mask_dilate_px < 0:
            raise ValueError("~projection_invalid_mask_dilate_px must be >= 0")
        self.projection_occlusion_epsilon_m = self._get_param_float(
            "~projection_occlusion_epsilon_m",
            0.0,
            "Allow image transfer only when the point depth is within this margin of the nearest LiDAR depth at that pixel (meters, 0 disables).",
        )
        if self.projection_occlusion_epsilon_m < 0.0:
            raise ValueError("~projection_occlusion_epsilon_m must be >= 0")
        self.projection_occlusion_radius_px = self._get_param_int(
            "~projection_occlusion_radius_px",
            0,
            "Pixel radius for local min-depth occlusion gating (0 uses only the exact projected pixel).",
        )
        if self.projection_occlusion_radius_px < 0:
            raise ValueError("~projection_occlusion_radius_px must be >= 0")
        self.projection_reject_depth_edges = self._get_param_bool(
            "~projection_reject_depth_edges",
            False,
            "If true, reject color/label transfer for projected points that land on strong LiDAR depth discontinuities.",
        )
        self.projection_depth_edge_thresh = self._get_param_float(
            "~projection_depth_edge_thresh",
            0.15,
            "Normalized depth-edge threshold used when ~projection_reject_depth_edges is enabled.",
        )
        if not 0.0 <= self.projection_depth_edge_thresh <= 1.0:
            raise ValueError("~projection_depth_edge_thresh must be in [0, 1]")
        self.projection_depth_edge_radius_px = self._get_param_int(
            "~projection_depth_edge_radius_px",
            0,
            "Pixel radius used to dilate the LiDAR depth-edge reject mask (helps suppress sky bleed near thin objects).",
        )
        if self.projection_depth_edge_radius_px < 0:
            raise ValueError("~projection_depth_edge_radius_px must be >= 0")
        self.use_invalid_mask = self._get_param_bool(
            "~use_invalid_mask",
            True,
            "Experiment switch: if false, invalid-mask samples are counted but not rejected.",
        )
        self.use_depth_edge_rejection = self._get_param_bool(
            "~use_depth_edge_rejection",
            True,
            "Experiment switch: if false, depth-edge samples are counted but not rejected.",
        )
        self.use_occlusion_gate = self._get_param_bool(
            "~use_occlusion_gate",
            True,
            "Experiment switch: if false, occlusion-risk samples are counted but not rejected.",
        )
        self.experiment_variant_name = self._get_param_str(
            "~experiment_variant_name",
            "",
            "Experiment variant name written to metrics_per_frame.csv.",
            allow_empty=True,
        )
        self.experiment_bag_name = self._get_param_str(
            "~bag_name",
            "",
            "Bag/run name written to metrics_per_frame.csv.",
            allow_empty=True,
        )
        self.results_dir = self._get_param_str(
            "~results_dir",
            "",
            "Root directory for experiment metrics and overlay outputs.",
            allow_empty=True,
        )
        self.enable_metrics_csv = self._get_param_bool(
            "~enable_metrics_csv",
            False,
            "Write per-frame experiment metrics to results/<bag>/<variant>/metrics_per_frame.csv.",
        )
        self.downsample_factor = self._get_param_int(
            "~downsample_factor",
            1,
            "Integer >=1 stride used to subsample images for CPU/ARM targets.",
        )
        if self.downsample_factor < 1:
            raise ValueError("~downsample_factor must be >= 1")
        self.sync_slop_sec = self._get_param_float(
            "~sync_slop_sec",
            0.1,
            "ApproximateTimeSynchronizer slop in seconds for semantic/depth or semantic/lidar pairing.",
        )
        if self.sync_slop_sec < 0.0:
            raise ValueError("~sync_slop_sec must be >= 0")
        self.pair_max_dt_sec = self._get_param_float(
            "~pair_max_dt_sec",
            0.03,
            "Hard max allowed |Δt| (seconds) between semantic and geometry; <=0 disables.",
        )
        if self.pair_max_dt_sec < 0.0:
            raise ValueError("~pair_max_dt_sec must be >= 0")
        self.semantic_time_offset_sec = self._get_param_float(
            "~semantic_time_offset_sec",
            0.0,
            "Signed offset (seconds) applied to semantic timestamps for pairing and timestamp selection (negative shifts semantic earlier).",
        )
        self.sync_queue_size = self._get_param_int(
            "~sync_queue_size",
            5,
            "ApproximateTimeSynchronizer queue size for semantic/depth or semantic/lidar pairing.",
        )
        if self.sync_queue_size < 1:
            raise ValueError("~sync_queue_size must be >= 1")
        self.cloud_time_offset_sec = self._get_param_float(
            "~cloud_time_offset_sec",
            0.0,
            "Signed offset (seconds) added to published cloud timestamps (negative shifts earlier).",
        )
        self.cloud_stamp_source = self._get_param_str(
            "~cloud_stamp_source",
            "",
            "Timestamp source for published PointCloud2: auto, semantic, depth, lidar, latest, earliest, midpoint.",
            allow_empty=True,
        )
        self.cloud_stamp_source = (self.cloud_stamp_source or "").strip().lower()
        self.stamp_debug_log_period_sec = self._get_param_float(
            "~stamp_debug_log_period_sec",
            2.0,
            "Minimum period (seconds) between timestamp/offset debug logs; set 0 to log every callback when debug=true.",
        )
        if self.stamp_debug_log_period_sec < 0.0:
            raise ValueError("~stamp_debug_log_period_sec must be >= 0")
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
        self.debug_project_lidar = self._get_param_bool(
            "~debug_project_lidar",
            False,
            "If true (lidar mode), publish a debug image with projected lidar points overlaid.",
        )
        self.debug_project_lidar_stride = self._get_param_int(
            "~debug_project_lidar_stride",
            5,
            "Subsample factor for projected LiDAR debug overlay (1 draws every projected point).",
        )
        if self.debug_project_lidar_stride < 1:
            raise ValueError("~debug_project_lidar_stride must be >= 1")
        self.debug_project_lidar_radius = self._get_param_int(
            "~debug_project_lidar_radius",
            0,
            "Marker radius in pixels for the projected LiDAR debug overlay (0 draws single pixels).",
        )
        if self.debug_project_lidar_radius < 0:
            raise ValueError("~debug_project_lidar_radius must be >= 0")
        self.debug_project_lidar_outline_only = self._get_param_bool(
            "~debug_project_lidar_outline_only",
            False,
            "If true, draw projected LiDAR markers as outlines so the RGB image stays visible underneath.",
        )
        self.debug_range_view = self._get_param_bool(
            "~debug_range_view",
            False,
            "If true (lidar mode), publish LiDAR depth/edge images, a reprojection heatmap, and an alignment score.",
        )
        self.debug_output_dir = self._get_param_str(
            "~debug_output_dir",
            "",
            "Directory where sampled debug overlays are written (empty uses <entfac_fusion_ros>/output/debug).",
            allow_empty=True,
        )
        if not self.debug_output_dir:
            try:
                import rospkg  # lazy import

                pkg_path = rospkg.RosPack().get_path("entfac_fusion_ros")
                self.debug_output_dir = str(Path(pkg_path) / "output" / "debug")
                self._param_meta["~debug_output_dir"]["value"] = self.debug_output_dir
            except Exception as exc:  # noqa: BLE001
                fallback = Path.home() / ".ros" / "entfac_fusion_ros" / "debug"
                self.debug_output_dir = str(fallback)
                self._param_meta["~debug_output_dir"]["value"] = self.debug_output_dir
                self._log.warn(
                    "__init__",
                    "Unable to resolve package path for default ~debug_output_dir (%s); using %s",
                    exc,
                    self.debug_output_dir,
                )
        Path(self.debug_output_dir).mkdir(parents=True, exist_ok=True)
        self.debug_output_stride = self._get_param_int(
            "~debug_output_stride",
            20,
            "Save every Nth debug callback per stream (1 saves every frame).",
        )
        if self.debug_output_stride < 1:
            raise ValueError("~debug_output_stride must be >= 1")
        self.tracked_reprojection_enable = self._get_param_bool(
            "~tracked_reprojection_enable",
            False,
            "Enable stateful feature-tracked LiDAR reprojection diagnostics. This is heavier than the online edge score and is intended mainly for offline rosbag review.",
        )
        self.tracked_reprojection_max_corners = self._get_param_int(
            "~tracked_reprojection_max_corners",
            300,
            "Maximum number of tracked image features used by the tracked reprojection diagnostic.",
        )
        if self.tracked_reprojection_max_corners < 20:
            raise ValueError("~tracked_reprojection_max_corners must be >= 20")
        self.tracked_reprojection_quality_level = self._get_param_float(
            "~tracked_reprojection_quality_level",
            0.01,
            "Shi-Tomasi quality level for tracked reprojection feature detection.",
        )
        if not 0.0 < self.tracked_reprojection_quality_level <= 1.0:
            raise ValueError("~tracked_reprojection_quality_level must be in (0, 1]")
        self.tracked_reprojection_min_distance_px = self._get_param_float(
            "~tracked_reprojection_min_distance_px",
            8.0,
            "Minimum pixel spacing between tracked reprojection features.",
        )
        if self.tracked_reprojection_min_distance_px <= 0.0:
            raise ValueError("~tracked_reprojection_min_distance_px must be > 0")
        self.tracked_reprojection_min_tracks = self._get_param_int(
            "~tracked_reprojection_min_tracks",
            80,
            "Minimum number of active tracks to maintain before replenishing features.",
        )
        if self.tracked_reprojection_min_tracks < 10:
            raise ValueError("~tracked_reprojection_min_tracks must be >= 10")
        self.tracked_reprojection_fb_thresh_px = self._get_param_float(
            "~tracked_reprojection_fb_thresh_px",
            1.5,
            "Forward-backward optical-flow consistency threshold in pixels.",
        )
        if self.tracked_reprojection_fb_thresh_px <= 0.0:
            raise ValueError("~tracked_reprojection_fb_thresh_px must be > 0")
        self.tracked_reprojection_depth_edge_thresh = self._get_param_float(
            "~tracked_reprojection_depth_edge_thresh",
            0.15,
            "Normalized LiDAR depth-edge threshold used to convert the projected depth map into an edge target for tracked reprojection.",
        )
        if not 0.0 <= self.tracked_reprojection_depth_edge_thresh <= 1.0:
            raise ValueError("~tracked_reprojection_depth_edge_thresh must be in [0, 1]")
        self.tracked_reprojection_min_image_edge = self._get_param_float(
            "~tracked_reprojection_min_image_edge",
            0.05,
            "Minimum image-edge strength required for a tracked feature to contribute to the reprojection error metric.",
        )
        if not 0.0 <= self.tracked_reprojection_min_image_edge <= 1.0:
            raise ValueError("~tracked_reprojection_min_image_edge must be in [0, 1]")
        self.tracked_reprojection_log_period_sec = self._get_param_float(
            "~tracked_reprojection_log_period_sec",
            2.0,
            "Minimum seconds between tracked reprojection status logs.",
        )
        if self.tracked_reprojection_log_period_sec < 0.0:
            raise ValueError("~tracked_reprojection_log_period_sec must be >= 0")
        self.debug_publish_fov_points = self._get_param_bool(
            "~debug_publish_fov_points",
            False,
            "If true (lidar mode), publish only the LiDAR points that passed the camera FOV test as a debug PointCloud2 in the LiDAR frame.",
        )
        self.online_calibration_enable = self._get_param_bool(
            "~online_calibration_enable",
            False,
            "Enable lightweight online LiDAR-camera misalignment estimation with health/uncertainty and small projection correction (classical, no neural models).",
        )
        self.online_calibration_every_n_frames = self._get_param_int(
            "~online_calibration_every_n_frames",
            10,
            "Run online calibration update every N lidar callbacks (>=1).",
        )
        if self.online_calibration_every_n_frames < 1:
            raise ValueError("~online_calibration_every_n_frames must be >= 1")
        self.online_calibration_max_points = self._get_param_int(
            "~online_calibration_max_points",
            8000,
            "Max number of LiDAR points used by online calibration updates (uniform stride subsampling above this).",
        )
        if self.online_calibration_max_points < 200:
            raise ValueError("~online_calibration_max_points must be >= 200")
        self.online_calibration_edge_threshold = self._get_param_float(
            "~online_calibration_edge_threshold",
            0.20,
            "Edge threshold in [0,1] used for observability density checks on semantic/depth edge maps.",
        )
        if not 0.0 <= self.online_calibration_edge_threshold <= 1.0:
            raise ValueError("~online_calibration_edge_threshold must be in [0, 1]")
        self.online_calibration_step_deg = self._get_param_float(
            "~online_calibration_step_deg",
            0.20,
            "Finite-difference perturbation step in degrees for rotational misalignment estimation.",
        )
        if self.online_calibration_step_deg <= 0.0:
            raise ValueError("~online_calibration_step_deg must be > 0")
        self.online_calibration_learning_rate = self._get_param_float(
            "~online_calibration_learning_rate",
            0.25,
            "Update gain for online rotational correction (smaller is more conservative).",
        )
        if self.online_calibration_learning_rate <= 0.0:
            raise ValueError("~online_calibration_learning_rate must be > 0")
        self.online_calibration_max_correction_deg = self._get_param_float(
            "~online_calibration_max_correction_deg",
            3.0,
            "Clamp for each correction angle component (roll/pitch/yaw) in degrees.",
        )
        if self.online_calibration_max_correction_deg <= 0.0:
            raise ValueError("~online_calibration_max_correction_deg must be > 0")
        self.online_calibration_min_observability = self._get_param_float(
            "~online_calibration_min_observability",
            0.15,
            "Minimum observability required before correction updates are applied.",
        )
        if not 0.0 <= self.online_calibration_min_observability <= 1.0:
            raise ValueError("~online_calibration_min_observability must be in [0, 1]")
        self.online_calibration_min_fov_points = self._get_param_int(
            "~online_calibration_min_fov_points",
            500,
            "Minimum in-FOV LiDAR points required by the online calibration update.",
        )
        if self.online_calibration_min_fov_points < 1:
            raise ValueError("~online_calibration_min_fov_points must be >= 1")
        self.online_calibration_min_sem_edge_density = self._get_param_float(
            "~online_calibration_min_sem_edge_density",
            0.010,
            "Minimum semantic edge density expected for well-observable frames.",
        )
        if self.online_calibration_min_sem_edge_density <= 0.0:
            raise ValueError("~online_calibration_min_sem_edge_density must be > 0")
        self.online_calibration_min_depth_edge_density = self._get_param_float(
            "~online_calibration_min_depth_edge_density",
            0.010,
            "Minimum LiDAR depth-edge density expected for well-observable frames.",
        )
        if self.online_calibration_min_depth_edge_density <= 0.0:
            raise ValueError("~online_calibration_min_depth_edge_density must be > 0")
        self.online_calibration_health_ema_alpha = self._get_param_float(
            "~online_calibration_health_ema_alpha",
            0.15,
            "EMA alpha for calibration health score smoothing.",
        )
        if not 0.0 < self.online_calibration_health_ema_alpha <= 1.0:
            raise ValueError("~online_calibration_health_ema_alpha must be in (0, 1]")
        self.online_calibration_health_std_window = self._get_param_int(
            "~online_calibration_health_std_window",
            40,
            "Sliding window size used to estimate alignment-score stability.",
        )
        if self.online_calibration_health_std_window < 2:
            raise ValueError("~online_calibration_health_std_window must be >= 2")
        self.online_calibration_health_std_scale = self._get_param_float(
            "~online_calibration_health_std_scale",
            0.08,
            "Scale that maps alignment-score std into stability confidence.",
        )
        if self.online_calibration_health_std_scale <= 0.0:
            raise ValueError("~online_calibration_health_std_scale must be > 0")
        self.online_calibration_health_score_center = self._get_param_float(
            "~online_calibration_health_score_center",
            0.25,
            "Alignment-score midpoint used by the health logistic transfer.",
        )
        self.online_calibration_health_score_scale = self._get_param_float(
            "~online_calibration_health_score_scale",
            0.10,
            "Alignment-score scale used by the health logistic transfer.",
        )
        if self.online_calibration_health_score_scale <= 0.0:
            raise ValueError("~online_calibration_health_score_scale must be > 0")
        self.online_calibration_log_period_sec = self._get_param_float(
            "~online_calibration_log_period_sec",
            2.0,
            "Minimum seconds between online calibration status logs.",
        )
        if self.online_calibration_log_period_sec < 0.0:
            raise ValueError("~online_calibration_log_period_sec must be >= 0")
        self._online_calibration_requested = bool(self.online_calibration_enable)
        self._online_calibration_status = (
            "requested" if self._online_calibration_requested else "disabled"
        )

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

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

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
        self.pcl_pub = rospy.Publisher("semantic_pointcloud", PointCloud2, queue_size=1)
        self._debug_proj_pub = None
        if self.debug_project_lidar:
            self._debug_proj_pub = rospy.Publisher(
                self.debug_projected_topic, Image, queue_size=1
            )
        self._debug_depth_pub = None
        self._debug_edge_pub = None
        self._debug_heatmap_pub = None
        self._debug_score_pub = None
        self._debug_tracked_reprojection_pub = None
        self._debug_tracked_reprojection_error_pub = None
        self._debug_fov_points_pub = None
        if self.debug_range_view:
            self._debug_depth_pub = rospy.Publisher(
                self.debug_lidar_depth_topic, Image, queue_size=1
            )
            self._debug_edge_pub = rospy.Publisher(
                self.debug_lidar_edge_topic, Image, queue_size=1
            )
            self._debug_heatmap_pub = rospy.Publisher(
                self.debug_reprojection_heatmap_topic, Image, queue_size=1
            )
            self._debug_score_pub = rospy.Publisher(
                self.debug_alignment_score_topic, Float32, queue_size=1
            )
        if self.tracked_reprojection_enable:
            self._debug_tracked_reprojection_pub = rospy.Publisher(
                self.debug_tracked_reprojection_topic, Image, queue_size=1
            )
            self._debug_tracked_reprojection_error_pub = rospy.Publisher(
                self.debug_tracked_reprojection_error_topic, Float32, queue_size=1
            )
        if self.debug_publish_fov_points:
            self._debug_fov_points_pub = rospy.Publisher(
                self.debug_fov_points_topic, PointCloud2, queue_size=1
            )
        self._debug_calibration_health_pub = None
        self._debug_calibration_uncertainty_pub = None
        if self.online_calibration_enable:
            self._debug_calibration_health_pub = rospy.Publisher(
                self.debug_calibration_health_topic, Float32, queue_size=1
            )
            self._debug_calibration_uncertainty_pub = rospy.Publisher(
                self.debug_calibration_uncertainty_topic, Float32, queue_size=1
            )

        self.target_T_depth = None
        self.camera_T_lidar = None
        self.target_T_lidar = None
        self._depth_frame = ""
        self._lidar_frame = ""

        self._mode_source = "forced" if self.mode in ("depth", "lidar") else "auto"
        self._mode_detail = "forced via ~mode" if self._mode_source == "forced" else ""
        if self.mode not in ("depth", "lidar"):
            self.mode = self._detect_mode()
        self._online_calibration_health = OnlineCalibrationHealth(
            ema_alpha=float(self.online_calibration_health_ema_alpha),
            std_window=int(self.online_calibration_health_std_window),
            std_scale=float(self.online_calibration_health_std_scale),
            score_center=float(self.online_calibration_health_score_center),
            score_scale=float(self.online_calibration_health_score_scale),
            min_observability=float(self.online_calibration_min_observability),
        )
        self._online_calibration_rpy_rad = np.zeros(3, dtype=np.float64)
        self._online_calibration_correction_uncertainty = 1.0
        self._online_calibration_update_counter = 0
        self._online_calibration_last_snapshot: Optional[CalibrationHealthSnapshot] = None
        self._online_calibration_last_log_at = 0.0
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
        self._resolve_cloud_stamp_source()
        self._tf_cache: Dict[Tuple[str, str], Tuple[np.ndarray, rospy.Time]] = {}
        self._prime_transforms()
        self._undistort_map1 = None
        self._undistort_map2 = None
        self._undistort_active = False
        self._cv2 = None
        self._maybe_init_undistort()
        self._imu_cache = None
        self._imu_sub = None
        self._imu_to_camera_R = None
        self._metadata_latest = {}
        self._metadata_sub = None
        self._lidar_imu_cache = None
        self._lidar_imu_sub = None
        self._lidar_imu_to_lidar_R = None
        self._tracked_reprojection_prev_gray = None
        self._tracked_reprojection_prev_pts = None
        self._tracked_reprojection_last_log_at = 0.0
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

        self._ply_writer = PlyWriterThread(queue_size=2)
        self._ply_recording = False
        self._ply_queue_warned_at = 0.0
        self._ply_seq = 0
        self._last_pcl: Optional[_LastPcl] = None
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
        self._calibration: Optional[OnlineCalibration] = (
            OnlineCalibration(self._build_calibration_params(), self._projector)
            if self.online_calibration_enable else None
        )
        self._tracked_repr: Optional[TrackedReprojection] = (
            TrackedReprojection(self._build_tracked_repr_params(), self._ensure_cv2)
            if self.tracked_reprojection_enable else None
        )
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

        self.ply_output_dir = self._get_param_str(
            "~ply_output_dir",
            "",
            "Directory where PLY files are written (empty uses <entfac_fusion_ros>/output/ply).",
            allow_empty=True,
        )
        if not self.ply_output_dir:
            try:
                import rospkg  # lazy import

                pkg_path = rospkg.RosPack().get_path("entfac_fusion_ros")
                self.ply_output_dir = str(Path(pkg_path) / "output" / "ply")
                self._param_meta["~ply_output_dir"]["value"] = self.ply_output_dir
            except Exception as exc:  # noqa: BLE001
                fallback = Path.home() / ".ros" / "entfac_fusion_ros" / "ply"
                self.ply_output_dir = str(fallback)
                self._param_meta["~ply_output_dir"]["value"] = self.ply_output_dir
                self._log.warn(
                    "__init__",
                    "Unable to resolve package path for default ~ply_output_dir (%s); using %s",
                    exc,
                    self.ply_output_dir,
                )
        Path(self.ply_output_dir).mkdir(parents=True, exist_ok=True)
        self.ply_recording_enable = self._get_param_bool(
            "~ply_recording_enable",
            False,
            "If true, automatically enable PLY recording at startup (can also be toggled via ~set_ply_recording service).",
        )
        self._ply_recording = self.ply_recording_enable
        if self._ply_recording:
            self._ply_writer.start()
            self._log.info(
                "__init__",
                "PLY recording enabled at startup (output_dir=%s)",
                self.ply_output_dir,
            )

        self.ply_target_frame = self._get_param_str(
            "~ply_target_frame",
            "",
            "Optional TF frame to transform PLY output to (ply_target_frame <- target_frame). Empty means use target_frame.",
            allow_empty=True,
        )
        self.ply_tf_use_latest = self._get_param_bool(
            "~ply_tf_use_latest",
            False,
            "When true, fall back to the latest TF for PLY export if exact-time lookup fails.",
        )
        self.ply_tf_tolerance_sec = self._get_param_float(
            "~ply_tf_tolerance_sec",
            0.02,
            "Max allowed time difference (seconds) when using latest TF for PLY export.",
        )
        if self.ply_tf_tolerance_sec < 0.0:
            raise ValueError("~ply_tf_tolerance_sec must be >= 0")

        self._srv_set_record = rospy.Service(
            "~set_ply_recording", SetBool, self._srv_set_ply_recording
        )
        self._srv_save_ply = rospy.Service("~save_ply", Trigger, self._srv_save_ply)

        self._setup_dynamic_reconfigure()
        self._register_subscribers()
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

    def _build_calibration_params(self) -> OnlineCalibrationParams:
        return OnlineCalibrationParams(
            every_n_frames=int(self.online_calibration_every_n_frames),
            max_points=int(self.online_calibration_max_points),
            step_deg=float(self.online_calibration_step_deg),
            learning_rate=float(self.online_calibration_learning_rate),
            max_correction_deg=float(self.online_calibration_max_correction_deg),
            min_observability=float(self.online_calibration_min_observability),
            min_fov_points=int(self.online_calibration_min_fov_points),
            edge_threshold=float(self.online_calibration_edge_threshold),
            min_sem_edge_density=float(self.online_calibration_min_sem_edge_density),
            min_depth_edge_density=float(self.online_calibration_min_depth_edge_density),
            log_period_sec=float(self.online_calibration_log_period_sec),
            health_ema_alpha=float(self.online_calibration_health_ema_alpha),
            health_std_window=int(self.online_calibration_health_std_window),
            health_std_scale=float(self.online_calibration_health_std_scale),
            health_score_center=float(self.online_calibration_health_score_center),
            health_score_scale=float(self.online_calibration_health_score_scale),
        )

    def _build_tracked_repr_params(self) -> TrackedReprojectionParams:
        return TrackedReprojectionParams(
            max_corners=int(self.tracked_reprojection_max_corners),
            quality_level=float(self.tracked_reprojection_quality_level),
            min_distance_px=float(self.tracked_reprojection_min_distance_px),
            min_tracks=int(self.tracked_reprojection_min_tracks),
            fb_thresh_px=float(self.tracked_reprojection_fb_thresh_px),
            depth_edge_thresh=float(self.tracked_reprojection_depth_edge_thresh),
            min_image_edge=float(self.tracked_reprojection_min_image_edge),
            log_period_sec=float(self.tracked_reprojection_log_period_sec),
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
        self._rolling_shutter_status = (
            "disabled" if not self._rolling_shutter_requested else "requested"
        )
        if self.rolling_shutter_enable:
            if not self.imu_topic:
                self._rolling_shutter_status = "disabled (missing ~imu_topic)"
                self._log.warn(
                    "_register_subscribers",
                    "rolling_shutter_enable is true but ~imu_topic is empty; disabling rolling shutter.",
                )
                self.rolling_shutter_enable = False
            else:
                self._imu_sub = Subscriber(self.imu_topic, Imu, queue_size=2000)
                self._imu_cache = Cache(self._imu_sub, self.imu_cache_size)
                if self.rolling_shutter_readout_sec > 0.0:
                    self._rolling_shutter_status = "armed (fixed readout)"
                elif self.camera_metadata_topic and self.metadata_readout_key >= 0:
                    self._rolling_shutter_status = "armed (metadata readout)"
                else:
                    self._rolling_shutter_status = (
                        "idle (readout=0 and metadata disabled)"
                    )
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
        self._lidar_deskew_status = (
            "disabled" if not self._lidar_deskew_requested else "requested"
        )
        if self.lidar_deskew_enable:
            if not self.lidar_imu_topic:
                self._lidar_deskew_status = "disabled (missing ~lidar_imu_topic)"
                self._log.warn(
                    "_register_subscribers",
                    "lidar_deskew_enable is true but ~lidar_imu_topic is empty; disabling deskew.",
                )
                self.lidar_deskew_enable = False
            else:
                self._lidar_imu_sub = Subscriber(
                    self.lidar_imu_topic, Imu, queue_size=2000
                )
                self._lidar_imu_cache = Cache(
                    self._lidar_imu_sub, self.lidar_imu_cache_size
                )
                self._lidar_deskew_status = "armed"
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
            if conf_sub is not None and invalid_mask_sub is not None:
                sync.registerCallback(self._depth_callback)
            elif conf_sub is not None:
                sync.registerCallback(
                    lambda sem, depth, conf: self._depth_callback(
                        sem, depth, conf, None
                    )
                )
            elif invalid_mask_sub is not None:
                sync.registerCallback(
                    lambda sem, depth, invalid_mask: self._depth_callback(
                        sem, depth, None, invalid_mask
                    )
                )
            else:
                sync.registerCallback(self._depth_callback)
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
            if conf_sub is not None and invalid_mask_sub is not None:
                sync.registerCallback(self._lidar_callback)
            elif conf_sub is not None:
                sync.registerCallback(
                    lambda sem, lidar, conf: self._lidar_callback(
                        sem, lidar, conf, None
                    )
                )
            elif invalid_mask_sub is not None:
                sync.registerCallback(
                    lambda sem, lidar, invalid_mask: self._lidar_callback(
                        sem, lidar, None, invalid_mask
                    )
                )
            else:
                sync.registerCallback(self._lidar_callback)
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
        """Parse semantic inputs (labels or RGB), confidence, and invalid mask.

        Returns: (labels, packed_img, confidence, projection_invalid_mask)
        One of labels or packed_img will be None depending on semantic_input_type.
        """
        # For rgb input type, always include rgb field; colorize_labels only applies to label mode
        include_rgb = bool(self.colorize_labels) if self.semantic_input_type == "labels" else True
        rgb_lut = None

        if self.semantic_input_type == "labels":
            labels = self._parse_semantic_labels(sem_msg)
            invalid_from_perception = labels == self.perception_invalid_label
            if np.any(invalid_from_perception):
                labels = labels.copy().astype(np.int64)
                labels[invalid_from_perception] = -1
            if include_rgb:
                rgb_lut = self._get_rgb_float_lut(labels)
            packed_img = None
        else:
            packed_img = self._parse_semantic_rgb_packed(sem_msg)
            labels = None
            if include_rgb and self.color_map and not self._warned_rgb_color_map:
                self._log.warn(
                    callback_name,
                    "~color_map is ignored when semantic_input_type=rgb (colors are passed through)",
                )
                self._warned_rgb_color_map = True

        confidence = image_to_numpy(conf_msg).astype(float) if conf_msg else None
        if confidence is not None and self._undistort_active:
            confidence = self._undistort_array(confidence, interpolation="linear")
        semantic_shape = labels.shape if labels is not None else packed_img.shape
        projection_invalid_mask = self._parse_projection_invalid_mask(
            invalid_mask_msg,
            semantic_shape,
        )
        return labels, packed_img, confidence, projection_invalid_mask, rgb_lut

    def _lidar_validate_inputs(self, sem_msg, lidar_msg) -> Optional[Tuple[rospy.Time, rospy.Time]]:
        """Validate timestamps and pairing. Returns (chosen_stamp, stamp) or None if invalid."""
        sem_stamp = sem_msg.header.stamp
        lidar_stamp = lidar_msg.header.stamp
        sem_pair_stamp = self._apply_stamp_offset(sem_stamp, self.semantic_time_offset_sec)
        if self.pair_max_dt_sec > 0.0:
            dt = abs((sem_pair_stamp - lidar_stamp).to_sec())
            if dt > self.pair_max_dt_sec:
                self._log.warn(
                    "_lidar_callback",
                    "Dropping pair: |Δt|=%.6fs > %.6fs",
                    dt,
                    float(self.pair_max_dt_sec),
                )
                return None
        chosen_stamp = self._choose_cloud_stamp(sem_pair_stamp, lidar_stamp, "lidar")
        stamp = self._apply_cloud_time_offset(chosen_stamp)
        self._log_stamp_debug(
            "_lidar_callback", sem_stamp, lidar_stamp, "lidar", chosen_stamp, stamp,
            sem_pair_stamp=sem_pair_stamp,
        )
        return chosen_stamp, stamp

    def _compute_pair_dt_sec(self, sem_msg, lidar_msg) -> float:
        sem_stamp = sem_msg.header.stamp
        lidar_stamp = lidar_msg.header.stamp
        sem_pair_stamp = self._apply_stamp_offset(sem_stamp, self.semantic_time_offset_sec)
        return abs((sem_pair_stamp - lidar_stamp).to_sec())

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

    def _lidar_lookup_transforms(self, lidar_frame: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Lookup and cache lidar->camera and lidar->target transforms. Returns (camera_T_lidar, target_T_lidar) or None."""
        camera_T_lidar = self.camera_T_lidar
        target_T_lidar = self.target_T_lidar
        if camera_T_lidar is None or target_T_lidar is None:
            if lidar_frame:
                if camera_T_lidar is None:
                    self._log.debug(
                        "_lidar_callback",
                        "Priming lidar->camera transform on first callback (%s -> %s)",
                        lidar_frame,
                        self.camera_frame,
                    )
                    camera_T_lidar = self._lookup_transform(
                        self.camera_frame, lidar_frame, rospy.Time(0)
                    )
                    if camera_T_lidar is not None:
                        self.camera_T_lidar = camera_T_lidar
                if target_T_lidar is None:
                    self._log.debug(
                        "_lidar_callback",
                        "Priming lidar->target transform on first callback (%s -> %s)",
                        lidar_frame,
                        self.target_frame,
                    )
                    target_T_lidar = self._lookup_transform(
                        self.target_frame, lidar_frame, rospy.Time(0)
                    )
                    if target_T_lidar is not None:
                        self.target_T_lidar = target_T_lidar
        if camera_T_lidar is None or target_T_lidar is None:
            self._log.warn("_lidar_callback", "No lidar transforms available")
            return None
        return camera_T_lidar, target_T_lidar

    def _lidar_assemble_and_publish(
        self,
        pcl: SemanticPointCloud,
        debug_colors: Optional[np.ndarray],
        image_shape: Tuple[int, int],
        points: np.ndarray,
        semantic_debug_img: Optional[np.ndarray],
        semantic_debug_type: str,
        intrinsics: np.ndarray,
        corrected_camera_T_lidar: np.ndarray,
        include_rgb: bool,
        rgb_values: Optional[np.ndarray],
        rgb_lut: Optional[np.ndarray],
        stamp: rospy.Time,
        sem_msg,
        dt: float,
        depth_map: Optional[np.ndarray] = None,
        edge_map: Optional[np.ndarray] = None,
    ) -> None:
        """Assemble cloud, publish, save PLY, and emit status."""
        h, w = image_shape

        if self._debug_pub is not None:
            self._debug_pub.tick()

        if (
            self._debug_pub is not None
            and self.debug_range_view
            and semantic_debug_img is not None
            and pcl.points_xyz.shape[0]
        ):
            _dmap = depth_map if depth_map is not None else self._projector._rasterize_depth_map(
                points, intrinsics, corrected_camera_T_lidar, (h, w)
            )
            _emap = edge_map if edge_map is not None else self._projector._depth_to_edge_map(_dmap)
            self._debug_pub.publish_range_view(
                depth_map=_dmap,
                edge_map=_emap,
                sem_img=semantic_debug_img,
                sem_type=semantic_debug_type,
                u=np.arange(w, dtype=np.int32),
                v=np.arange(h, dtype=np.int32),
                point_confidence=None,
                header=sem_msg.header,
            )

        if self._tracked_repr is not None and semantic_debug_img is not None:
            _dmap = depth_map if depth_map is not None else self._projector._rasterize_depth_map(
                points, intrinsics, corrected_camera_T_lidar, (h, w)
            )
            _emap = edge_map if edge_map is not None else self._projector._depth_to_edge_map(_dmap)
            _tr_result = self._tracked_repr.update(
                sem_img=semantic_debug_img,
                sem_type=semantic_debug_type,
                depth_map=_dmap,
                depth_edges=_emap,
            )
            if _tr_result is not None and self._debug_pub is not None:
                self._debug_pub.publish_tracked_reprojection(
                    _tr_result.overlay_img, _tr_result.error_px, sem_msg.header
                )

        # Log summary on first callback
        if self.debug and not self._logged_lidar_summary:
            if pcl.points_xyz.shape[0]:
                mins = pcl.points_xyz.min(axis=0)
                maxs = pcl.points_xyz.max(axis=0)
                self._log.info(
                    "_lidar_callback",
                    "LiDAR PCL bbox in %s: x=[%.3f, %.3f] y=[%.3f, %.3f] z=[%.3f, %.3f] (input_points=%d)",
                    self.target_frame,
                    float(mins[0]),
                    float(maxs[0]),
                    float(mins[1]),
                    float(maxs[1]),
                    float(mins[2]),
                    float(maxs[2]),
                    int(points.shape[0]),
                )
            else:
                self._log.warn(
                    "_lidar_callback",
                    "LiDAR fusion produced empty point cloud (check intrinsics, transforms, and image alignment)",
                )
            self._logged_lidar_summary = True

        # Convert and publish message
        pcl_msg = semantic_pointcloud_to_msg(
            pcl,
            self.target_frame,
            stamp,
            colorize_labels=include_rgb,
            rgb_lut=rgb_lut,
            rgb_values=rgb_values,
        )
        self.pcl_pub.publish(pcl_msg)

        # Save to PLY and emit status
        self._last_pcl = _LastPcl(
            stamp=stamp,
            points_xyz=pcl.points_xyz,
            labels=pcl.labels,
            confidence=pcl.confidence,
            rgb_packed_float=rgb_values if include_rgb else None,
        )
        if self._ply_recording:
            self._enqueue_ply(self._last_pcl)

        self._maybe_emit_status(points=int(pcl.points_xyz.shape[0]), callback_sec=dt)

    def _metadata_callback(self, msg) -> None:
        try:
            key = int(msg.key)
            value = int(msg.value)
        except Exception:  # noqa: BLE001
            return
        self._metadata_latest[key] = (msg.header.stamp, value)

    def _get_readout_sec(self, stamp: rospy.Time) -> float:
        if not self.rolling_shutter_enable:
            return 0.0
        if self.metadata_readout_key >= 0 and self._metadata_latest:
            entry = self._metadata_latest.get(int(self.metadata_readout_key))
            if entry is not None:
                meta_stamp, value = entry
                if self.metadata_max_dt_sec > 0.0:
                    dt = abs((meta_stamp - stamp).to_sec())
                    if dt > self.metadata_max_dt_sec:
                        return float(self.rolling_shutter_readout_sec)
                return float(value) * float(self.metadata_readout_scale)
        return float(self.rolling_shutter_readout_sec)

    def _interpolate_imu_msg(
        self, before: Imu, after: Imu, stamp: rospy.Time
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Interpolate IMU message (omega + accel). Delegates to standalone function."""
        return interpolate_imu_msg(before, after, stamp)

    def _lookup_imu_omega(self, stamp: rospy.Time) -> Optional[np.ndarray]:
        if self._imu_cache is None:
            return None
        before = self._imu_cache.getElemBeforeTime(stamp)
        after = self._imu_cache.getElemAfterTime(stamp)
        if before is None and after is None:
            return None
        omega, _, best_dt = self._interpolate_imu_msg(before, after, stamp)
        if omega is None:
            return None
        if self.imu_cache_max_dt_sec > 0.0:
            if best_dt > self.imu_cache_max_dt_sec:
                return None
        imu_frame = self.imu_frame or (after.header.frame_id if before is None else before.header.frame_id)
        if not imu_frame:
            return None
        if self._imu_to_camera_R is None:
            mat = self._lookup_transform(self.camera_frame, imu_frame, rospy.Time(0))
            if mat is None:
                return None
            self._imu_to_camera_R = mat[:3, :3]
        omega_cam = self._imu_to_camera_R @ omega
        return omega_cam

    def _lookup_lidar_imu(self, stamp: rospy.Time) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._lidar_imu_cache is None:
            return None
        before = self._lidar_imu_cache.getElemBeforeTime(stamp)
        after = self._lidar_imu_cache.getElemAfterTime(stamp)
        if before is None and after is None:
            return None
        omega, accel, best_dt = self._interpolate_imu_msg(before, after, stamp)
        if omega is None or accel is None:
            return None
        if self.lidar_imu_cache_max_dt_sec > 0.0:
            if best_dt > self.lidar_imu_cache_max_dt_sec:
                return None
        imu_frame = self.lidar_imu_frame or (before.header.frame_id if before is not None else after.header.frame_id)
        if not imu_frame:
            return None
        if self._lidar_imu_to_lidar_R is None:
            if not self._lidar_frame:
                return None
            mat = self._lookup_transform(self._lidar_frame, imu_frame, rospy.Time(0))
            if mat is None:
                return None
            self._lidar_imu_to_lidar_R = mat[:3, :3]
        omega_lidar = self._lidar_imu_to_lidar_R @ omega
        accel_lidar = self._lidar_imu_to_lidar_R @ accel
        return omega_lidar, accel_lidar

    def _deskew_lidar_points(
        self,
        points: np.ndarray,
        t_raw: np.ndarray,
        scan_stamp: rospy.Time,
    ) -> np.ndarray:
        if not self.lidar_deskew_enable:
            return points
        if t_raw is None or t_raw.size == 0:
            self._lidar_deskew_status = (
                f"armed (missing point time field '{self.lidar_time_field}')"
            )
            now = time.time()
            if now - self._lidar_deskew_missing_time_warn_at > 2.0:
                self._log.warn(
                    "_deskew_lidar_points",
                    "Deskew enabled but point cloud has no usable '%s' field; skipping deskew.",
                    self.lidar_time_field,
                )
                self._lidar_deskew_missing_time_warn_at = now
            return points
        dt = t_raw.astype(np.float64) * float(self.lidar_time_scale)
        scan_span = float(np.nanmax(dt)) if dt.size else 0.0
        ref_offset = 0.5 * scan_span if self.lidar_deskew_ref == "mid" else 0.0
        rel_dt = dt - ref_offset
        ref_stamp = scan_stamp + rospy.Duration.from_sec(float(ref_offset))
        sample_count = min(max(1, int(self.lidar_deskew_imu_samples)), 32)
        omega_eval = None
        accel_eval = None
        if sample_count <= 1 or scan_span <= 1e-9:
            ref_imu = self._lookup_lidar_imu(ref_stamp)
            if ref_imu is None:
                self._lidar_deskew_status = "armed (waiting for IMU)"
                now = time.time()
                if now - self._lidar_deskew_warn_at > 2.0:
                    self._log.warn(
                        "_deskew_lidar_points",
                        "Deskew enabled but IMU lookup failed near the selected reference time; skipping deskew.",
                    )
                    self._lidar_deskew_warn_at = now
                return points
            omega_ref, accel_ref = ref_imu
            omega_eval = np.repeat(
                np.asarray(omega_ref, dtype=np.float64).reshape(1, 3),
                points.shape[0],
                axis=0,
            )
            accel_eval = np.repeat(
                np.asarray(accel_ref, dtype=np.float64).reshape(1, 3),
                points.shape[0],
                axis=0,
            )
        else:
            sample_offsets = np.linspace(0.0, scan_span, sample_count, dtype=np.float64)
            omega_samples = []
            accel_samples = []
            dense_sampling_ok = True
            for offset in sample_offsets:
                imu_sample = self._lookup_lidar_imu(
                    scan_stamp + rospy.Duration.from_sec(float(offset))
                )
                if imu_sample is None:
                    dense_sampling_ok = False
                    self._lidar_deskew_status = "armed (waiting for dense IMU support)"
                    now = time.time()
                    if now - self._lidar_deskew_warn_at > 2.0:
                        self._log.warn(
                            "_deskew_lidar_points",
                            "Deskew requested %d IMU samples across the scan but lookup failed at %.6fs; falling back to the single-sample model.",
                            sample_count,
                            float(offset),
                        )
                        self._lidar_deskew_warn_at = now
                    break
                omega_i, accel_i = imu_sample
                omega_samples.append(np.asarray(omega_i, dtype=np.float64).reshape(3))
                accel_samples.append(np.asarray(accel_i, dtype=np.float64).reshape(3))
            if dense_sampling_ok:
                omega_samples = np.asarray(omega_samples, dtype=np.float64)
                accel_samples = np.asarray(accel_samples, dtype=np.float64)
                sample_rel_dt = sample_offsets - ref_offset
                if sample_rel_dt.size <= 1:
                    omega_eval = np.repeat(omega_samples[:1], points.shape[0], axis=0)
                    accel_eval = np.repeat(accel_samples[:1], points.shape[0], axis=0)
                else:
                    boundaries = 0.5 * (sample_rel_dt[:-1] + sample_rel_dt[1:])
                    sample_idx = np.searchsorted(boundaries, rel_dt, side="right")
                    sample_idx = np.clip(sample_idx, 0, omega_samples.shape[0] - 1)
                    omega_eval = omega_samples[sample_idx]
                    accel_eval = accel_samples[sample_idx]
            else:
                ref_imu = self._lookup_lidar_imu(ref_stamp)
                if ref_imu is None:
                    self._lidar_deskew_status = "armed (waiting for IMU)"
                    return points
                omega_ref, accel_ref = ref_imu
                omega_eval = np.repeat(
                    np.asarray(omega_ref, dtype=np.float64).reshape(1, 3),
                    points.shape[0],
                    axis=0,
                )
                accel_eval = np.repeat(
                    np.asarray(accel_ref, dtype=np.float64).reshape(1, 3),
                    points.shape[0],
                    axis=0,
                )
        now = time.time()
        self._lidar_deskew_status = "active"
        if now - self._lidar_deskew_log_at > 2.0:
            self._log.info(
                "_deskew_lidar_points",
                "Deskew active: mode=%s ref=%s dt_max=%.6f imu_samples=%d",
                self.lidar_deskew_mode,
                self.lidar_deskew_ref,
                scan_span,
                sample_count,
            )
            self._lidar_deskew_log_at = now
        if self.lidar_deskew_mode in ("rotation", "both"):
            cross = np.cross(omega_eval, points)
            points = points - rel_dt.reshape(-1, 1) * cross
        if self.lidar_deskew_mode in ("translation", "both"):
            if not self.lidar_imu_accel_is_gravity_compensated:
                self._log.warn(
                    "_deskew_lidar_points",
                    "Translation deskew assumes gravity-compensated IMU accel; set ~lidar_imu_accel_gravity_compensated=true if already corrected.",
                )
            disp = 0.5 * accel_eval * (rel_dt.reshape(-1, 1) ** 2)
            points = points - disp
        return points

    def _apply_lidar_points_compat(self, points: np.ndarray) -> np.ndarray:
        mat = self._compat_declared_lidar_T_points
        if mat is None:
            return points
        return transform_points(mat, points)

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
        if not self.undistort_semantic:
            self._undistort_status = "disabled"
            return
        if self.mode != "lidar":
            self._undistort_status = f"disabled (mode={self.mode})"
            self._log.warn(
                "_maybe_init_undistort",
                "undistort_semantic enabled but mode=%s; undistort is only applied in lidar mode.",
                self.mode,
            )
            return
        if self._camera_distortion is None or not self._camera_distortion.size:
            self._undistort_status = "disabled (CameraInfo has no distortion)"
            self._log.warn(
                "_maybe_init_undistort",
                "CameraInfo has no distortion coefficients; skipping undistort.",
            )
            return
        if np.allclose(self._camera_distortion, 0.0):
            self._undistort_status = "disabled (zero distortion coefficients)"
            self._log.info(
                "_maybe_init_undistort",
                "CameraInfo distortion coefficients are zero; skipping undistort.",
            )
            return
        if not self._ensure_cv2("_maybe_init_undistort"):
            self._undistort_status = "disabled (OpenCV unavailable)"
            return
        cv2 = self._cv2
        h, w = self._camera_info_size
        if h <= 0 or w <= 0:
            self._undistort_status = "disabled (invalid CameraInfo image size)"
            self._log.warn(
                "_maybe_init_undistort",
                "CameraInfo image size invalid (%d x %d); skipping undistort.",
                w,
                h,
            )
            return
        model = self._camera_distortion_model
        k_mat = np.asarray(self.intrinsics_raw, dtype=float)
        d_vec = np.asarray(self._camera_distortion, dtype=float).reshape(-1)

        if model in ("plumb_bob", "rational_polynomial", ""):
            new_k, _ = cv2.getOptimalNewCameraMatrix(
                k_mat, d_vec, (w, h), float(self.undistort_alpha)
            )
            map1, map2 = cv2.initUndistortRectifyMap(
                k_mat, d_vec, None, new_k, (w, h), cv2.CV_32FC1
            )
        elif model == "equidistant":
            new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                k_mat, d_vec, (w, h), np.eye(3), balance=float(self.undistort_alpha)
            )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                k_mat, d_vec, np.eye(3), new_k, (w, h), cv2.CV_32FC1
            )
        else:
            self._undistort_status = f"disabled (unsupported model {model})"
            self._log.warn(
                "_maybe_init_undistort",
                "Unsupported distortion_model=%s; skipping undistort.",
                model,
            )
            return

        self._undistort_map1 = map1
        self._undistort_map2 = map2
        self._undistort_active = True
        self._undistort_status = "active"
        self.intrinsics = np.asarray(new_k, dtype=float)
        self._log.info(
            "_maybe_init_undistort",
            "Undistort enabled (model=%s alpha=%.2f); intrinsics updated.",
            model,
            float(self.undistort_alpha),
        )

    def _undistort_array(self, data: np.ndarray, *, interpolation: str) -> np.ndarray:
        if not self._undistort_active or self._cv2 is None:
            return data
        h, w = self._camera_info_size
        if data.shape[0] != h or data.shape[1] != w:
            self._log.warn(
                "_undistort_array",
                "Semantic image size %s does not match CameraInfo %dx%d; skipping undistort.",
                data.shape,
                w,
                h,
            )
            return data
        if interpolation == "nearest":
            interp = self._cv2.INTER_NEAREST
        else:
            interp = self._cv2.INTER_LINEAR
        orig_dtype = data.dtype
        if data.dtype not in (np.uint8, np.uint16, np.float32):
            work = data.astype(np.float32)
        else:
            work = data
        remapped = self._cv2.remap(
            work, self._undistort_map1, self._undistort_map2, interp
        )
        if remapped.dtype != orig_dtype:
            remapped = remapped.astype(orig_dtype)
        return remapped

    def _lookup_transform(self, target_frame, source_frame, stamp):
        # Check cache first for static transforms (using stamp=0).
        cache_key = (target_frame, source_frame)
        if stamp == rospy.Time(0) and cache_key in self._tf_cache:
            cached_mat, _ = self._tf_cache[cache_key]
            self._log.debug(
                "_lookup_transform",
                "TF cache hit %s -> %s",
                source_frame,
                target_frame,
            )
            return cached_mat
        
        try:
            # Use shorter timeout (0.1s) to fail fast on missing transforms.
            tf_msg = self.tf_buffer.lookup_transform(
                target_frame, source_frame, stamp, rospy.Duration(0.1)
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            self._log.warn(
                "_lookup_transform",
                "TF lookup failed (%s -> %s): %s",
                source_frame,
                target_frame,
                exc,
            )
            return None
        try:
            mat = transform_stamped_to_matrix(tf_msg)
        except ValueError as exc:
            self._log.warn(
                "_lookup_transform",
                "Rejected TF (%s -> %s): %s",
                source_frame,
                target_frame,
                exc,
            )
            return None
        
        # Cache static transforms (stamp=0) for future reuse.
        if stamp == rospy.Time(0):
            self._tf_cache[cache_key] = (mat, tf_msg.header.stamp)
        
        self._log.debug(
            "_lookup_transform",
            "TF %s -> %s:\n%s",
            source_frame,
            target_frame,
            format_matrix(mat),
        )
        return mat

    def _lookup_transform_with_stamp(self, target_frame, source_frame, stamp):
        # Check cache first for static transforms (using stamp=0).
        cache_key = (target_frame, source_frame)
        if stamp == rospy.Time(0) and cache_key in self._tf_cache:
            cached_mat, cached_stamp = self._tf_cache[cache_key]
            self._log.debug(
                "_lookup_transform",
                "TF cache hit %s -> %s",
                source_frame,
                target_frame,
            )
            return cached_mat, cached_stamp
        
        try:
            # Use shorter timeout (0.1s) to fail fast on missing transforms.
            tf_msg = self.tf_buffer.lookup_transform(
                target_frame, source_frame, stamp, rospy.Duration(0.1)
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            self._log.warn(
                "_lookup_transform",
                "TF lookup failed (%s -> %s): %s",
                source_frame,
                target_frame,
                exc,
            )
            return None, None
        try:
            mat = transform_stamped_to_matrix(tf_msg)
        except ValueError as exc:
            self._log.warn(
                "_lookup_transform",
                "Rejected TF (%s -> %s): %s",
                source_frame,
                target_frame,
                exc,
            )
            return None, None
        
        # Cache static transforms (stamp=0) for future reuse.
        if stamp == rospy.Time(0):
            self._tf_cache[cache_key] = (mat, tf_msg.header.stamp)
        
        self._log.debug(
            "_lookup_transform",
            "TF %s -> %s:\n%s",
            source_frame,
            target_frame,
            format_matrix(mat),
        )
        return mat, tf_msg.header.stamp

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

    def _apply_cloud_time_offset(self, stamp: rospy.Time) -> rospy.Time:
        if self.cloud_time_offset_sec == 0.0 or stamp == rospy.Time():
            return stamp
        shifted = stamp + rospy.Duration(self.cloud_time_offset_sec)
        if shifted.to_sec() < 0.0:
            return rospy.Time(0)
        return shifted

    def _resolve_cloud_stamp_source(self) -> None:
        raw = (self.cloud_stamp_source or "").strip().lower()
        if not raw or raw == "auto":
            resolved = "latest" if self.mode == "depth" else "semantic"
        else:
            aliases = {
                "sem": "semantic",
                "labels": "semantic",
                "label": "semantic",
                "max": "latest",
                "latest": "latest",
                "min": "earliest",
                "earliest": "earliest",
                "avg": "midpoint",
                "average": "midpoint",
                "mid": "midpoint",
                "middle": "midpoint",
                "midpoint": "midpoint",
                "depth": "depth",
                "lidar": "lidar",
            }
            resolved = aliases.get(raw)
            if resolved is None:
                self._log.warn(
                    "_resolve_cloud_stamp_source",
                    "Unknown ~cloud_stamp_source=%r; falling back to auto.",
                    raw,
                )
                resolved = "latest" if self.mode == "depth" else "semantic"

        valid = {
            "depth": {"semantic", "depth", "latest", "earliest", "midpoint"},
            "lidar": {"semantic", "lidar", "latest", "earliest", "midpoint"},
        }
        if resolved not in valid.get(self.mode, set()):
            self._log.warn(
                "_resolve_cloud_stamp_source",
                "Invalid ~cloud_stamp_source=%s for mode=%s; falling back to auto.",
                resolved,
                self.mode,
            )
            resolved = "latest" if self.mode == "depth" else "semantic"

        self.cloud_stamp_source = resolved
        if "~cloud_stamp_source" in self._param_meta:
            self._param_meta["~cloud_stamp_source"]["value"] = resolved

    def _choose_cloud_stamp(
        self,
        sem_stamp: rospy.Time,
        other_stamp: rospy.Time,
        other_label: str,
    ) -> rospy.Time:
        if sem_stamp == rospy.Time():
            return other_stamp
        if other_stamp == rospy.Time():
            return sem_stamp

        source = self.cloud_stamp_source
        if source == "semantic":
            return sem_stamp
        if source == other_label:
            return other_stamp
        if source == "latest":
            return sem_stamp if sem_stamp > other_stamp else other_stamp
        if source == "earliest":
            return sem_stamp if sem_stamp < other_stamp else other_stamp
        if source == "midpoint":
            mid_sec = 0.5 * (sem_stamp.to_sec() + other_stamp.to_sec())
            return rospy.Time.from_sec(mid_sec)

        return sem_stamp if sem_stamp > other_stamp else other_stamp

    def _apply_stamp_offset(self, stamp: rospy.Time, offset_sec: float) -> rospy.Time:
        if stamp == rospy.Time() or offset_sec == 0.0:
            return stamp
        return stamp + rospy.Duration(offset_sec)

    def _log_stamp_debug(
        self,
        context: str,
        sem_stamp: rospy.Time,
        other_stamp: rospy.Time,
        other_label: str,
        chosen_stamp: rospy.Time,
        shifted_stamp: rospy.Time,
        sem_pair_stamp: Optional[rospy.Time] = None,
    ) -> None:
        if not self.debug:
            return
        now = time.time()
        period = float(self.stamp_debug_log_period_sec)
        if period > 0.0 and (now - self._stamp_debug_last_log_at) < period:
            return
        sem_pair_stamp = sem_stamp if sem_pair_stamp is None else sem_pair_stamp
        raw_dt_sec = (sem_stamp - other_stamp).to_sec()
        pair_dt_sec = (sem_pair_stamp - other_stamp).to_sec()
        self._log.debug(
            context,
            "stamps: semantic=%.9f semantic_pair=%.9f %s=%.9f raw_dt=%.9f pair_dt=%.9f chosen=%.9f shifted=%.9f source=%s semantic_offset=%.6f cloud_offset=%.6f",
            sem_stamp.to_sec(),
            sem_pair_stamp.to_sec(),
            other_label,
            other_stamp.to_sec(),
            raw_dt_sec,
            pair_dt_sec,
            chosen_stamp.to_sec(),
            shifted_stamp.to_sec(),
            self.cloud_stamp_source,
            float(self.semantic_time_offset_sec),
            float(self.cloud_time_offset_sec),
        )
        self._stamp_debug_last_log_at = now

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

    def _parse_semantic_labels(self, msg):
        data = image_to_numpy(msg)
        if self._undistort_active:
            data = self._undistort_array(data, interpolation="nearest")
        if data.ndim == 3:
            # Accept bgr8/rgb8 label images where R=G=B=class_id (e.g. ICNF segmentation)
            data = data[..., 0]
        if data.ndim != 2:
            raise ValueError(
                "semantic_input_type=labels requires a single-channel label image (e.g., mono8/16UC1/32SC1). "
                f"Got encoding={msg.encoding} shape={data.shape}."
            )
        return data

    def _parse_semantic_rgb_packed(self, msg):
        data = image_to_numpy(msg)
        if self._undistort_active:
            data = self._undistort_array(data, interpolation="linear")
        if data.ndim != 3:
            raise ValueError(
                "semantic_input_type=rgb requires a 3/4-channel image (rgb8/bgr8/rgba8/bgra8). "
                f"Got encoding={msg.encoding} shape={data.shape}."
            )
        return rgb_to_packed_u32(
            data,
            msg.encoding,
            quantize_step=int(self.semantic_color_quantization_step),
        )

    def _parse_projection_invalid_mask(
        self,
        msg,
        expected_shape: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        if msg is None:
            return None
        data = image_to_numpy(msg)
        if self._undistort_active:
            data = self._undistort_array(data, interpolation="nearest")
        invalid = invalid_image_to_mask(
            data,
            invalid_value=int(self.projection_invalid_mask_value),
            dilate_px=int(self.projection_invalid_mask_dilate_px),
        )
        if invalid.shape != tuple(expected_shape):
            raise ValueError(
                "projection invalid mask shape "
                f"{invalid.shape} must match semantic image shape {tuple(expected_shape)}"
            )
        return invalid




















    def _get_rgb_float_lut(self, labels_img: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Delegate to LidarProjector (owns the LUT cache and warned-flag)."""
        return self._projector._get_rgb_float_lut(labels_img)

    # ----------------------------
    # PLY services
    # ----------------------------

    def _srv_set_ply_recording(self, req: SetBool.Request) -> SetBoolResponse:
        enable = bool(req.data)
        self._ply_recording = enable
        if enable:
            self._ply_writer.start()
            self._log.info(
                "_srv_set_ply_recording",
                "PLY recording enabled (output_dir=%s)",
                self.ply_output_dir,
            )
        else:
            self._log.info("_srv_set_ply_recording", "PLY recording disabled")
        return SetBoolResponse(success=True, message=str(enable))

    def _next_ply_path(self, stamp: rospy.Time) -> Path:
        if hasattr(stamp, "to_nsec"):
            t_ns = int(stamp.to_nsec())
        else:
            t_ns = int(stamp.to_sec() * 1e9)
        self._ply_seq += 1
        name = f"colored_pcl_{t_ns}_{self._ply_seq:06d}.ply"
        return Path(self.ply_output_dir) / name

    def _enqueue_ply(self, last: _LastPcl) -> bool:
        points_xyz = last.points_xyz
        if self.ply_target_frame and self.ply_target_frame != self.target_frame:
            mat = self._lookup_transform(
                self.ply_target_frame, self.target_frame, last.stamp
            )
            if mat is None and self.ply_tf_use_latest:
                mat, tf_stamp = self._lookup_transform_with_stamp(
                    self.ply_target_frame, self.target_frame, rospy.Time(0)
                )
                if mat is None:
                    self._log.warn(
                        "_enqueue_ply",
                        "PLY transform unavailable (%s -> %s); skipping write",
                        self.target_frame,
                        self.ply_target_frame,
                    )
                    return False
                delta = abs((tf_stamp - last.stamp).to_sec())
                if delta > self.ply_tf_tolerance_sec:
                    self._log.warn(
                        "_enqueue_ply",
                        "PLY latest TF too far from cloud stamp (dt=%.6fs tol=%.6fs); skipping write",
                        delta,
                        self.ply_tf_tolerance_sec,
                    )
                    return False
            elif mat is None:
                self._log.warn(
                    "_enqueue_ply",
                    "PLY transform unavailable (%s -> %s); skipping write",
                    self.target_frame,
                    self.ply_target_frame,
                )
                return False
            points_xyz = transform_points(mat, points_xyz)
        self._ply_writer.start()
        job = PlyJob(
            path=self._next_ply_path(last.stamp),
            points_xyz=points_xyz,
            labels=last.labels,
            confidence=last.confidence,
            rgb_packed_float=last.rgb_packed_float,
        )
        ok = self._ply_writer.enqueue(job)
        if not ok:
            now = time.time()
            if now - self._ply_queue_warned_at > 1.0:
                self._log.warn(
                    "_enqueue_ply",
                    "PLY writer queue is full; dropping frames. Consider lowering publish rate or increasing queue_size.",
                )
                self._ply_queue_warned_at = now
        return ok

    def _srv_save_ply(self, req: Trigger.Request) -> TriggerResponse:
        if self._last_pcl is None:
            return TriggerResponse(success=False, message="No point cloud published yet")
        ok = self._enqueue_ply(self._last_pcl)
        if ok:
            return TriggerResponse(success=True, message="enqueued")
        return TriggerResponse(success=False, message="enqueue failed")

    # ----------------------------
    # Callbacks
    # ----------------------------

    def _maybe_emit_status(self, *, points: int, callback_sec: float) -> None:
        snap = self._status.record(points=int(points), callback_sec=float(callback_sec))
        if snap is None:
            return
        table = render_status_table(
            node_name=self._node_name,
            mode=self.mode,
            semantic_input_type=self.semantic_input_type,
            target_frame=self.target_frame,
            output_topic=self._output_topic,
            points_last=snap.points_last,
            pub_hz=snap.pub_hz,
            avg_points=snap.avg_points,
            avg_callback_ms=snap.avg_callback_ms,
        )
        self._log.debug("status", "\n%s", table)

    def _render_startup_table(self) -> str:
        return _render_startup_table_helper(
            self,
            rospy.resolve_name("~save_ply"),
            rospy.resolve_name("~set_ply_recording"),
        )

    def _depth_validate_inputs(self, sem_msg, depth_msg) -> Optional[Tuple[rospy.Time, rospy.Time]]:
        """Validate timestamps and pairing. Returns (chosen_stamp, stamp) or None if invalid."""
        sem_stamp = sem_msg.header.stamp
        depth_stamp = depth_msg.header.stamp
        sem_pair_stamp = self._apply_stamp_offset(sem_stamp, self.semantic_time_offset_sec)
        if self.pair_max_dt_sec > 0.0:
            dt = abs((sem_pair_stamp - depth_stamp).to_sec())
            if dt > self.pair_max_dt_sec:
                self._log.warn(
                    "_depth_callback",
                    "Dropping pair: |Δt|=%.6fs > %.6fs",
                    dt,
                    float(self.pair_max_dt_sec),
                )
                return None
        chosen_stamp = self._choose_cloud_stamp(sem_pair_stamp, depth_stamp, "depth")
        stamp = self._apply_cloud_time_offset(chosen_stamp)
        self._log_stamp_debug(
            "_depth_callback", sem_stamp, depth_stamp, "depth", chosen_stamp, stamp,
            sem_pair_stamp=sem_pair_stamp,
        )
        return chosen_stamp, stamp

    def _depth_lookup_transforms(self) -> Optional[np.ndarray]:
        """Lookup and cache depth->target transform. Returns target_T_depth or None."""
        target_T_depth = self.target_T_depth
        if target_T_depth is None:
            depth_frame = getattr(self, '_current_depth_frame', None)
            if depth_frame:
                self._log.debug(
                    "_depth_callback",
                    "Priming depth->target transform on first callback (%s -> %s)",
                    depth_frame,
                    self.target_frame,
                )
                target_T_depth = self._lookup_transform(
                    self.target_frame, depth_frame, rospy.Time(0)
                )
                if target_T_depth is not None:
                    self.target_T_depth = target_T_depth
        if target_T_depth is None:
            self._log.warn("_depth_callback", "No depth->target transform available")
            return None
        return target_T_depth

    def _depth_unproject_and_fuse(
        self,
        depth: np.ndarray,
        labels: Optional[np.ndarray],
        packed_img: Optional[np.ndarray],
        confidence: Optional[np.ndarray],
        projection_invalid_mask: Optional[np.ndarray],
        intrinsics: np.ndarray,
        target_T_depth: np.ndarray,
        rgb_lut: Optional[np.ndarray],
        include_rgb: bool,
    ) -> Tuple[SemanticPointCloud, Optional[np.ndarray]]:
        """Unproject depth to 3D and fuse with semantics. Returns (pcl, rgb_values)."""
        rgb_values = None
        if labels is not None:
            if projection_invalid_mask is not None:
                labels = labels.copy()
                labels[projection_invalid_mask] = -1
                if confidence is not None:
                    confidence = confidence.copy()
                    confidence[projection_invalid_mask] = 0.0
            semantic_obs = SemanticObservation(labels=labels, confidence=confidence)
            depth_obs = DepthObservation(depth=depth)
            pcl = fuse_depth_semantics(
                semantic_obs,
                depth_obs,
                intrinsics,
                target_T_depth,
                include_unlabeled=self.include_unlabeled,
                max_depth_m=self.max_depth_m,
            )
        else:
            points_cam, valid_mask = depth_to_points(
                depth, intrinsics, max_depth_m=self.max_depth_m
            )
            if points_cam.shape[0] == 0:
                pcl = SemanticPointCloud(
                    np.empty((0, 3)),
                    np.empty((0,), dtype=np.int64),
                    None,
                )
            else:
                labels_all = np.full(points_cam.shape[0], -1, dtype=np.int64)
                conf_flat = (
                    flatten_masked(confidence, valid_mask)
                    if confidence is not None
                    else None
                )
                invalid_flat = (
                    flatten_masked(projection_invalid_mask, valid_mask)
                    if projection_invalid_mask is not None
                    else None
                )
                if conf_flat is not None and invalid_flat is not None:
                    conf_flat = conf_flat.astype(np.float32, copy=True)
                    conf_flat[invalid_flat] = 0.0
                points_target = transform_points(target_T_depth, points_cam)
                pcl = SemanticPointCloud(points_target, labels_all, conf_flat)
                if include_rgb:
                    colors_packed = packed_img[valid_mask].reshape(-1).astype(
                        np.uint32, copy=True
                    )
                    if invalid_flat is not None:
                        points_target, labels_all, conf_flat, colors_packed = (
                            filter_invalid_projection_samples(
                                invalid_flat,
                                points=points_target,
                                labels=labels_all,
                                confidence=conf_flat,
                                rgb_values=colors_packed,
                            )
                        )
                        pcl = SemanticPointCloud(points_target, labels_all, conf_flat)
                    rgb_values = colors_packed.astype("<u4", copy=False).view("<f4")

        if include_rgb and rgb_values is None and rgb_lut is not None:
            rgb_values = rgb_lut[labels_to_uint16(pcl.labels)]
            rgb_lut = None

        return pcl, rgb_values

    def _depth_assemble_and_publish(
        self,
        pcl: SemanticPointCloud,
        include_rgb: bool,
        rgb_values: Optional[np.ndarray],
        rgb_lut: Optional[np.ndarray],
        stamp: rospy.Time,
        dt: float,
    ) -> None:
        """Assemble cloud, publish, save PLY, and emit status."""
        pcl_msg = semantic_pointcloud_to_msg(
            pcl,
            self.target_frame,
            stamp,
            colorize_labels=include_rgb,
            rgb_lut=rgb_lut,
            rgb_values=rgb_values,
        )
        self.pcl_pub.publish(pcl_msg)

        self._last_pcl = _LastPcl(
            stamp=stamp,
            points_xyz=pcl.points_xyz,
            labels=pcl.labels,
            confidence=pcl.confidence,
            rgb_packed_float=rgb_values if include_rgb else None,
        )
        if self._ply_recording:
            self._enqueue_ply(self._last_pcl)

        self._maybe_emit_status(points=int(pcl.points_xyz.shape[0]), callback_sec=dt)

    def _depth_callback(self, sem_msg, depth_msg, conf_msg=None, invalid_mask_msg=None):
        t0 = time.perf_counter()
        with self._maybe_profile("depth_callback"):
            self._maybe_refresh_live_tuning_params()

            # PHASE 1: Validate inputs and timestamps
            result = self._depth_validate_inputs(sem_msg, depth_msg)
            if result is None:
                return
            chosen_stamp, stamp = result

            # PHASE 2: Lookup transforms
            self._current_depth_frame = depth_msg.header.frame_id
            target_T_depth = self._depth_lookup_transforms()
            if target_T_depth is None:
                return

            # PHASE 3: Parse semantic inputs and prepare depth
            include_rgb = bool(self.colorize_labels) if self.semantic_input_type == "labels" else True
            labels, packed_img, confidence, projection_invalid_mask, rgb_lut = (
                self._parse_semantic_inputs(sem_msg, conf_msg, invalid_mask_msg, "_depth_callback")
            )

            depth_raw = image_to_numpy(depth_msg)
            depth_enc = depth_msg.encoding.lower()
            invalid_raw = None
            if self.filter_invalid_depth and depth_enc in ("16uc1", "mono16", "16sc1"):
                invalid_raw = (depth_raw == 0) | (
                    depth_raw == np.iinfo(np.uint16).max
                )

            if self.downsample_factor > 1:
                f = self.downsample_factor
                if labels is not None:
                    labels = labels[::f, ::f]
                else:
                    packed_img = packed_img[::f, ::f]
                depth_raw = depth_raw[::f, ::f]
                if invalid_raw is not None:
                    invalid_raw = invalid_raw[::f, ::f]
                if confidence is not None:
                    confidence = confidence[::f, ::f]
                if projection_invalid_mask is not None:
                    projection_invalid_mask = projection_invalid_mask[::f, ::f]
                intrinsics = self._scale_intrinsics(self.intrinsics, f)
            else:
                intrinsics = self.intrinsics

            # PHASE 4: Process depth (scaling and validation)
            scale = float(self.depth_scale)
            if scale == 0.0:
                if depth_enc in ("16uc1", "16sc1", "mono16"):
                    scale = 0.001
                else:
                    scale = 1.0
            if self.debug and not self._logged_depth_scaling:
                self._log.info(
                    "_depth_callback",
                    "Depth scaling: encoding=%s scale=%.6f (depth_scale_param=%.6f)",
                    depth_msg.encoding,
                    float(scale),
                    float(self.depth_scale),
                )
                self._logged_depth_scaling = True

            depth = depth_raw.astype(np.float32, copy=False) * float(scale)
            if invalid_raw is not None:
                depth[invalid_raw] = 0.0
            if confidence is not None:
                confidence = confidence.astype(np.float32, copy=False)

            if self.debug and not self._logged_depth_summary:
                valid = np.isfinite(depth) & (depth > 0)
                valid_count = int(np.count_nonzero(valid))
                if valid_count:
                    dmin = float(depth[valid].min())
                    dmax = float(depth[valid].max())
                else:
                    dmin, dmax = float("nan"), float("nan")
                self._log.info(
                    "_depth_callback",
                    "Depth inputs: semantic_shape=%s depth_shape=%s depth_encoding=%s valid_depth=%d min=%.3f max=%.3f downsample=%d",
                    labels.shape if labels is not None else packed_img.shape,
                    depth.shape,
                    depth_msg.encoding,
                    valid_count,
                    dmin,
                    dmax,
                    int(self.downsample_factor),
                )
                self._logged_depth_summary = True

            # PHASE 5: Unproject to 3D and fuse with semantics
            pcl, rgb_values = self._depth_unproject_and_fuse(
                depth=depth,
                labels=labels,
                packed_img=packed_img,
                confidence=confidence,
                projection_invalid_mask=projection_invalid_mask,
                intrinsics=intrinsics,
                target_T_depth=target_T_depth,
                rgb_lut=rgb_lut,
                include_rgb=include_rgb,
            )

            # PHASE 6: Output assembly & publishing
            self._depth_assemble_and_publish(
                pcl=pcl,
                include_rgb=include_rgb,
                rgb_values=rgb_values,
                rgb_lut=rgb_lut,
                stamp=stamp,
                dt=time.perf_counter() - t0,
            )

    def _lidar_callback(self, sem_msg, lidar_msg, conf_msg=None, invalid_mask_msg=None):
        t0 = time.perf_counter()
        frame_index = int(self._results_frame_index)
        self._results_frame_index += 1
        pair_dt_sec = self._compute_pair_dt_sec(sem_msg, lidar_msg)
        with self._maybe_profile("lidar_callback"):
            self._maybe_refresh_live_tuning_params()

            # PHASE 1: Validate inputs and timestamps
            result = self._lidar_validate_inputs(sem_msg, lidar_msg)
            if result is None:
                self._write_lidar_metrics(
                    frame_index=frame_index,
                    sem_msg=sem_msg,
                    lidar_msg=lidar_msg,
                    pair_dt_sec=pair_dt_sec,
                    pair_accepted=0,
                    drop_reason="pair_dt_too_large",
                    num_input_points=0,
                    projection_metrics=ProjectionMetrics(),
                    num_output_points=0,
                    runtime_total_ms=1000.0 * (time.perf_counter() - t0),
                    runtime_publish_ms=0.0,
                )
                return
            chosen_stamp, stamp = result

            # PHASE 2: Lookup transforms
            transforms = self._lidar_lookup_transforms(lidar_msg.header.frame_id)
            if transforms is None:
                self._write_lidar_metrics(
                    frame_index=frame_index,
                    sem_msg=sem_msg,
                    lidar_msg=lidar_msg,
                    pair_dt_sec=pair_dt_sec,
                    pair_accepted=0,
                    drop_reason="missing_tf",
                    num_input_points=0,
                    projection_metrics=ProjectionMetrics(),
                    num_output_points=0,
                    runtime_total_ms=1000.0 * (time.perf_counter() - t0),
                    runtime_publish_ms=0.0,
                )
                return
            camera_T_lidar, target_T_lidar = transforms

            # PHASE 3: Parse semantic inputs and prepare intrinsics
            include_rgb = bool(self.colorize_labels) if self.semantic_input_type == "labels" else True
            labels, packed_img, confidence, projection_invalid_mask, rgb_lut = (
                self._parse_semantic_inputs(sem_msg, conf_msg, invalid_mask_msg, "_lidar_callback")
            )
            semantic_shape = labels.shape if labels is not None else packed_img.shape[:2]
            semantic_debug_type = "labels" if labels is not None else "rgb"
            semantic_debug_img = labels if labels is not None else packed_img
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

            # PHASE 4: Parse and process LiDAR points
            lidar_stamp = sem_msg.header.stamp
            if lidar_msg.header.frame_id and not self._lidar_frame:
                self._lidar_frame = lidar_msg.header.frame_id
            if self.lidar_deskew_enable:
                points, t_raw = pointcloud2_to_xyz_t(
                    lidar_msg, time_field=self.lidar_time_field
                )
                points = self._apply_lidar_points_compat(points)
                points = self._deskew_lidar_points(points, t_raw, lidar_stamp)
            else:
                points = pointcloud2_to_xyz(lidar_msg)
                points = self._apply_lidar_points_compat(points)
            num_input_points = int(points.shape[0])
            # Online calibration update (runs before projection to get corrected extrinsic)
            corrected_camera_T_lidar = camera_T_lidar
            if self._calibration is not None and semantic_debug_img is not None:
                sem_h, sem_w = semantic_debug_img.shape[:2]
                corrected_camera_T_lidar, _calib_snapshot = self._calibration.update(
                    points=points,
                    sem_img=semantic_debug_img,
                    sem_type=semantic_debug_type,
                    intrinsics=intrinsics,
                    camera_T_lidar=camera_T_lidar,
                    image_shape=(int(sem_h), int(sem_w)),
                )
                if _calib_snapshot is not None and self._debug_pub is not None:
                    self._debug_pub.publish_calibration_health(_calib_snapshot)
                self._online_calibration_status = self._calibration._status

            # PHASE 5: Projection & Sampling (delegated to LidarProjector)
            rolling_shutter_readout_sec = (
                self._get_readout_sec(sem_msg.header.stamp)
                if self.rolling_shutter_enable else 0.0
            )
            rolling_shutter_omega_cam = (
                self._lookup_imu_omega(sem_msg.header.stamp)
                if rolling_shutter_readout_sec > 0.0 else None
            )
            _proj_result: ProjectionResult = self._projector.process_frame(
                points=points,
                labels=labels,
                packed_img=packed_img,
                confidence=confidence,
                projection_invalid_mask=projection_invalid_mask,
                intrinsics=intrinsics,
                camera_T_lidar=corrected_camera_T_lidar,
                target_T_lidar=target_T_lidar,
                semantic_shape=semantic_shape,
                include_rgb=include_rgb,
                rolling_shutter_omega_cam=rolling_shutter_omega_cam,
                rolling_shutter_readout_sec=rolling_shutter_readout_sec,
            )
            if _proj_result.rolling_shutter_active:
                self._rolling_shutter_status = "active"
            elif self.rolling_shutter_enable:
                self._rolling_shutter_status = (
                    "idle (readout<=0)" if rolling_shutter_readout_sec <= 0.0
                    else "armed (waiting for IMU)"
                )
            pcl = _proj_result.cloud
            debug_colors = _proj_result.debug_colors
            image_shape = _proj_result.image_shape
            rgb_values = _proj_result.rgb_values
            projection_metrics = _proj_result.metrics
            points = _proj_result.points_fov  # FOV-gated; passed to assemble/debug

            # Optional per-frame debug publishes.
            if self._debug_pub is not None:
                if self.debug_project_lidar:
                    _base_rgb = (
                        (np.stack([(labels.astype(np.int32) % 256).astype(np.uint8)] * 3, axis=-1))
                        if labels is not None
                        else packed_rgb_to_triplets(packed_img)
                    )
                    _uv, _ = project_points_to_image(
                        points, intrinsics, corrected_camera_T_lidar,
                        (image_shape[1], image_shape[0])
                    )
                    self._debug_pub.publish_lidar_projection(
                        _base_rgb, image_shape, _uv, sem_msg.header,
                        colors_u8=debug_colors,
                    )
                if self.debug_publish_fov_points and points.shape[0]:
                    self._debug_pub.publish_fov_points(
                        points, lidar_msg.header.frame_id, lidar_msg.header.stamp
                    )

            # PHASE 6: Output assembly & publishing
            _publish_t0 = time.perf_counter()
            self._lidar_assemble_and_publish(
                pcl=pcl,
                debug_colors=debug_colors,
                image_shape=image_shape,
                points=points,
                semantic_debug_img=semantic_debug_img,
                semantic_debug_type=semantic_debug_type,
                intrinsics=intrinsics,
                corrected_camera_T_lidar=corrected_camera_T_lidar,
                include_rgb=include_rgb,
                rgb_values=rgb_values,
                rgb_lut=rgb_lut,
                stamp=stamp,
                sem_msg=sem_msg,
                dt=time.perf_counter() - t0,
                depth_map=_proj_result.depth_map,
                edge_map=_proj_result.edge_map,
            )
            runtime_publish_ms = 1000.0 * (time.perf_counter() - _publish_t0)
            projection_metrics.runtime_publish_ms = runtime_publish_ms
            self._write_lidar_metrics(
                frame_index=frame_index,
                sem_msg=sem_msg,
                lidar_msg=lidar_msg,
                pair_dt_sec=pair_dt_sec,
                pair_accepted=1,
                drop_reason="none",
                num_input_points=num_input_points,
                projection_metrics=projection_metrics,
                num_output_points=int(pcl.points_xyz.shape[0]),
                runtime_total_ms=1000.0 * (time.perf_counter() - t0),
                runtime_publish_ms=runtime_publish_ms,
            )

        self._debug_callback_seq += 1


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
