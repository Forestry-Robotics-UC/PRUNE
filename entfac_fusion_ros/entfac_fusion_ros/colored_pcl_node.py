#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Derived from Semantic SLAM
#
# Original Author:
#   Xuan Zhang
#
# Subsequent Contributions:
#   David Russell
#
# Modified by:
#   Duda Andrada (ENTFAC Sensor Fusion)
#
# Original project:
#   https://github.com/floatlazer/semantic_slam
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
  - ``~camera_info`` (``sensor_msgs/CameraInfo``): provides intrinsics ``K`` and
    the camera frame ID (used for LiDAR projection mode).
  - Geometry input:
    - Preferred: ``~depth_input_topic`` (auto-detected):
      - ``sensor_msgs/Image`` → depth mode
      - ``sensor_msgs/PointCloud2`` → lidar mode

Optional:
  - ``~confidence_topic`` (``sensor_msgs/Image``): confidence/probability aligned
    with semantic labels (single-channel numeric).

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
from std_srvs.srv import SetBool, SetBoolResponse, Trigger, TriggerResponse

# Ensure core package is importable when running from a monorepo source tree.
# In a proper catkin workspace, this is handled by PYTHONPATH via devel/setup.bash.
_THIS = Path(__file__).resolve()
for parent in _THIS.parents:
    cand = parent / "entfac_fusion_core" / "src"
    if (cand / "entfac_fusion_core").is_dir() and str(cand) not in sys.path:
        sys.path.insert(0, str(cand))
        break

from entfac_fusion_core.projection.depth import depth_to_points
from entfac_fusion_core.projection.lidar_projection import project_points_to_image
from entfac_fusion_core.colored_pcl import (
    fuse_depth_semantics,
    fuse_lidar_semantics,
)
from entfac_fusion_core.transforms.se3 import transform_points
from entfac_fusion_core.types import (
    DepthObservation,
    PointObservation,
    SemanticObservation,
    SemanticPointCloud,
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
from entfac_fusion_ros.logging_ros import NodeLogger, configure_core_logging
from entfac_fusion_ros.pc2 import (
    build_label_rgb_float_lut,
    labels_to_uint16,
    semantic_pointcloud_to_msg,
)
from entfac_fusion_ros.ply import PlyJob, PlyWriterThread
from entfac_fusion_ros.status import StatusReporter, render_kv_table, render_status_table
from entfac_fusion_ros.tf_utils import format_matrix, transform_stamped_to_matrix


def _coerce_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "on")
    return bool(val)


def _rosargv_bool(name: str, default: bool = False) -> bool:
    prefix = f"_{name}:="
    for arg in sys.argv:
        if arg.startswith(prefix):
            return _coerce_bool(arg[len(prefix) :])
    return default


def _rosargv_has_private_param(name: str) -> bool:
    if not isinstance(name, str):
        return False
    key = name
    if key.startswith("~"):
        key = key[1:]
    if "/" in key:
        key = key.rsplit("/", 1)[-1]
    prefix = f"_{key}:="
    return any(arg.startswith(prefix) for arg in sys.argv)


@dataclass
class _LastPcl:
    stamp: rospy.Time
    points_xyz: np.ndarray
    labels: np.ndarray
    confidence: Optional[np.ndarray]
    rgb_packed_float: Optional[np.ndarray]


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
        if not self.camera_info_topic:
            raise ValueError("~camera_info is required")

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
        # Fixed debug projection settings to avoid extra parameters.
        self.debug_projected_topic = "/debug/lidar_projection"
        self.debug_projected_colorize = "depth"
        self.debug_projected_depth_min = 0.0
        self.debug_projected_depth_max = 0.0
        self.debug_projected_stride = 5

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
        self._camera_distortion = np.asarray(cam_info.D, dtype=float) if cam_info.D else None
        self._camera_distortion_model = (cam_info.distortion_model or "").strip().lower()
        self._camera_info_size = (int(cam_info.height), int(cam_info.width))

        self._output_topic = rospy.resolve_name("semantic_pointcloud")
        self.pcl_pub = rospy.Publisher("semantic_pointcloud", PointCloud2, queue_size=1)
        self._debug_proj_pub = None
        if self.debug_project_lidar:
            self._debug_proj_pub = rospy.Publisher(
                self.debug_projected_topic, Image, queue_size=1
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
        self._resolve_cloud_stamp_source()
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
        self._lidar_deskew_log_at = 0.0
        self._lidar_deskew_warn_at = 0.0

        self._rgb_lut = None
        self._rgb_lut_num_labels = None
        self._warned_random_palette = False
        self._warned_rgb_color_map = False
        self._logged_depth_scaling = False
        self._logged_depth_summary = False
        self._logged_lidar_summary = False

        self._ply_writer = PlyWriterThread(queue_size=2)
        self._ply_recording = False
        self._ply_queue_warned_at = 0.0
        self._ply_seq = 0
        self._last_pcl: Optional[_LastPcl] = None

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

        self._register_subscribers()
        self._log.info("__init__", "\n%s", self._render_startup_table())
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
        self._param_meta[name] = {
            "value": value,
            "source": source,
            "description": description,
        }

    def _get_param(self, name, default, description, *, allow_empty=False):
        has = rospy.has_param(name)
        if has:
            value = rospy.get_param(name)
            source = "param_server"
        else:
            value = default
            source = "default"
        if _rosargv_has_private_param(name):
            source = "cli"
        if not allow_empty and value in ("", None):
            value = default
        self._record_param(name, value, source, description)
        return value

    def _get_param_str(self, name, default, description, *, allow_empty=False):
        raw = self._get_param(name, default, description, allow_empty=allow_empty)
        if raw is None:
            val = None
        else:
            val = str(raw)
        self._param_meta[name]["value"] = val
        return val

    def _get_param_bool(self, name, default, description):
        raw = self._get_param(name, default, description, allow_empty=True)
        val = _coerce_bool(raw)
        self._param_meta[name]["value"] = val
        return val

    def _get_param_int(self, name, default, description):
        raw = self._get_param(name, default, description)
        val = int(raw)
        self._param_meta[name]["value"] = val
        return val

    def _get_param_float(self, name, default, description):
        raw = self._get_param(name, default, description)
        val = float(raw)
        self._param_meta[name]["value"] = val
        return val

    def _get_matrix_param(self, name, description):
        has = rospy.has_param(name)
        raw = rospy.get_param(name, [])
        source = "param_server" if has else "default"
        mat = None
        if isinstance(raw, list) and len(raw) == 16:
            try:
                mat = require_homogeneous_transform(
                    np.asarray(raw, dtype=float).reshape(4, 4)
                )
            except ValueError as exc:
                self._log.warn("_get_matrix_param", "%s rejected: %s", name, exc)
        elif raw not in (None, [], {}):
            self._log.warn(
                "_get_matrix_param",
                "%s expected 16-element list (row-major 4x4), got: %r",
                name,
                raw,
            )
        self._record_param(name, mat, source, description)
        return mat

    def _get_color_map(self, name, description):
        has = rospy.has_param(name)
        raw = rospy.get_param(name, {})
        source = "param_server" if has else "default"
        color_map = None
        if isinstance(raw, dict):
            parsed = {}
            for k, v in raw.items():
                try:
                    key_int = int(k)
                    if isinstance(v, (list, tuple)) and len(v) == 3:
                        parsed[key_int] = [int(v[0]), int(v[1]), int(v[2])]
                except Exception:  # noqa: BLE001
                    continue
            color_map = parsed if parsed else None
        self._record_param(name, color_map, source, description)
        return color_map

    # ----------------------------
    # Startup report
    # ----------------------------

    def _log_param_report(self):
        self._log.info("_log_param_report", "colored_pcl_node debug report:")
        self._log.info(
            "_log_param_report",
            "  source=cli means passed as _param:=...; source=param_server means set via YAML/launch; source=default means unset.",
        )
        for name in sorted(self._param_meta.keys()):
            meta = self._param_meta[name]
            val = meta["value"]
            if isinstance(val, np.ndarray):
                val_str = format_matrix(val)
            else:
                val_str = repr(val)
            self._log.info(
                "_log_param_report",
                "  %s=%s (%s) - %s",
                name,
                val_str,
                meta["source"],
                meta["description"],
            )
        self._log.info("_log_param_report", "derived mode=%s", self.mode)
        self._log.info("_log_param_report", "derived camera_frame=%s", self.camera_frame)
        self._log.info(
            "_log_param_report", "derived intrinsics=%s", format_matrix(self.intrinsics)
        )
        self._log.info(
            "_log_param_report",
            "active subscriptions: semantic=%s depth_input=%s confidence=%s",
            self.semantic_topic,
            self.depth_input_topic,
            self.conf_topic,
        )
        self._log.info(
            "_log_param_report",
            "primed transforms: target_T_depth=%s camera_T_lidar=%s target_T_lidar=%s",
            self.target_T_depth is not None,
            self.camera_T_lidar is not None,
            self.target_T_lidar is not None,
        )
        if self.target_T_depth is not None:
            self._log.info(
                "_log_param_report",
                "target_T_depth=\n%s",
                format_matrix(self.target_T_depth),
            )
        if self.camera_T_lidar is not None:
            self._log.info(
                "_log_param_report",
                "camera_T_lidar=\n%s",
                format_matrix(self.camera_T_lidar),
            )
        if self.target_T_lidar is not None:
            self._log.info(
                "_log_param_report",
                "target_T_lidar=\n%s",
                format_matrix(self.target_T_lidar),
            )

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
        if self.rolling_shutter_enable:
            if not self.imu_topic:
                self._log.warn(
                    "_register_subscribers",
                    "rolling_shutter_enable is true but ~imu_topic is empty; disabling rolling shutter.",
                )
                self.rolling_shutter_enable = False
            else:
                self._imu_sub = Subscriber(self.imu_topic, Imu, queue_size=2000)
                self._imu_cache = Cache(self._imu_sub, self.imu_cache_size)
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
        if self.lidar_deskew_enable:
            if not self.lidar_imu_topic:
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
        if self.mode == "depth":
            depth_sub = Subscriber(self.depth_input_topic, Image)
            subs = [semantic_sub, depth_sub]
            if conf_sub is not None:
                subs.append(conf_sub)
            sync = ApproximateTimeSynchronizer(
                subs, queue_size=self.sync_queue_size, slop=self.sync_slop_sec
            )
            sync.registerCallback(self._depth_callback)
            self._sync = sync
        else:
            lidar_sub = Subscriber(self.depth_input_topic, PointCloud2)
            subs = [semantic_sub, lidar_sub]
            if conf_sub is not None:
                subs.append(conf_sub)
            sync = ApproximateTimeSynchronizer(
                subs, queue_size=self.sync_queue_size, slop=self.sync_slop_sec
            )
            sync.registerCallback(self._lidar_callback)
            self._sync = sync

        self._log.debug(
            "_register_subscribers",
            "Registering subscribers (mode=%s): semantic=%s depth=%s lidar=%s confidence=%s",
            self.mode,
            self.semantic_topic,
            self.depth_input_topic,
            self.depth_input_topic,
            self.conf_topic,
        )

    def _get_live_param_float(self, name: str, fallback: float) -> float:
        try:
            return float(rospy.get_param(name, fallback))
        except Exception:  # noqa: BLE001
            return fallback

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

    def _lookup_imu_omega(self, stamp: rospy.Time) -> Optional[np.ndarray]:
        if self._imu_cache is None:
            return None
        before = self._imu_cache.getElemBeforeTime(stamp)
        after = self._imu_cache.getElemAfterTime(stamp)
        if before is None and after is None:
            return None
        if before is None:
            chosen = after
        elif after is None:
            chosen = before
        else:
            dt = (after.header.stamp - before.header.stamp).to_sec()
            if dt > 0.0:
                alpha = (stamp - before.header.stamp).to_sec() / dt
                alpha = float(np.clip(alpha, 0.0, 1.0))
                omega_b = np.array(
                    [
                        before.angular_velocity.x,
                        before.angular_velocity.y,
                        before.angular_velocity.z,
                    ],
                    dtype=float,
                )
                omega_a = np.array(
                    [
                        after.angular_velocity.x,
                        after.angular_velocity.y,
                        after.angular_velocity.z,
                    ],
                    dtype=float,
                )
                omega = (1.0 - alpha) * omega_b + alpha * omega_a
                chosen = None
            else:
                chosen = before
                omega = None
        if chosen is not None:
            omega = np.array(
                [
                    chosen.angular_velocity.x,
                    chosen.angular_velocity.y,
                    chosen.angular_velocity.z,
                ],
                dtype=float,
            )
        if self.imu_cache_max_dt_sec > 0.0:
            dt = abs((stamp - (chosen.header.stamp if chosen is not None else before.header.stamp)).to_sec())
            if dt > self.imu_cache_max_dt_sec:
                return None

        imu_frame = self.imu_frame or (chosen.header.frame_id if chosen is not None else before.header.frame_id)
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
        if before is None:
            chosen = after
            omega = np.array(
                [
                    chosen.angular_velocity.x,
                    chosen.angular_velocity.y,
                    chosen.angular_velocity.z,
                ],
                dtype=float,
            )
            accel = np.array(
                [
                    chosen.linear_acceleration.x,
                    chosen.linear_acceleration.y,
                    chosen.linear_acceleration.z,
                ],
                dtype=float,
            )
        elif after is None:
            chosen = before
            omega = np.array(
                [
                    chosen.angular_velocity.x,
                    chosen.angular_velocity.y,
                    chosen.angular_velocity.z,
                ],
                dtype=float,
            )
            accel = np.array(
                [
                    chosen.linear_acceleration.x,
                    chosen.linear_acceleration.y,
                    chosen.linear_acceleration.z,
                ],
                dtype=float,
            )
        else:
            dt = (after.header.stamp - before.header.stamp).to_sec()
            if dt > 0.0:
                alpha = (stamp - before.header.stamp).to_sec() / dt
                alpha = float(np.clip(alpha, 0.0, 1.0))
                omega_b = np.array(
                    [
                        before.angular_velocity.x,
                        before.angular_velocity.y,
                        before.angular_velocity.z,
                    ],
                    dtype=float,
                )
                omega_a = np.array(
                    [
                        after.angular_velocity.x,
                        after.angular_velocity.y,
                        after.angular_velocity.z,
                    ],
                    dtype=float,
                )
                accel_b = np.array(
                    [
                        before.linear_acceleration.x,
                        before.linear_acceleration.y,
                        before.linear_acceleration.z,
                    ],
                    dtype=float,
                )
                accel_a = np.array(
                    [
                        after.linear_acceleration.x,
                        after.linear_acceleration.y,
                        after.linear_acceleration.z,
                    ],
                    dtype=float,
                )
                omega = (1.0 - alpha) * omega_b + alpha * omega_a
                accel = (1.0 - alpha) * accel_b + alpha * accel_a
                chosen = before
            else:
                chosen = before
                omega = np.array(
                    [
                        chosen.angular_velocity.x,
                        chosen.angular_velocity.y,
                        chosen.angular_velocity.z,
                    ],
                    dtype=float,
                )
                accel = np.array(
                    [
                        chosen.linear_acceleration.x,
                        chosen.linear_acceleration.y,
                        chosen.linear_acceleration.z,
                    ],
                    dtype=float,
                )
        if self.lidar_imu_cache_max_dt_sec > 0.0:
            dt = abs((stamp - chosen.header.stamp).to_sec())
            if dt > self.lidar_imu_cache_max_dt_sec:
                return None
        imu_frame = self.lidar_imu_frame or chosen.header.frame_id
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

    def _project_lidar_points(
        self,
        points: np.ndarray,
        intrinsics: np.ndarray,
        camera_T_lidar: np.ndarray,
        image_size,
        stamp: rospy.Time,
    ):
        if not self.rolling_shutter_enable:
            return project_points_to_image(points, intrinsics, camera_T_lidar, image_size)
        readout_sec = self._get_readout_sec(stamp)
        if readout_sec <= 0.0:
            return project_points_to_image(points, intrinsics, camera_T_lidar, image_size)
        omega_cam = self._lookup_imu_omega(stamp)
        if omega_cam is None:
            return project_points_to_image(points, intrinsics, camera_T_lidar, image_size)

        w, h = int(image_size[0]), int(image_size[1])
        points_cam = transform_points(camera_T_lidar, points)
        z = points_cam[:, 2]
        in_front = z > 0
        uv = np.zeros((points_cam.shape[0], 2), dtype=float)
        uv[in_front, 0] = (points_cam[in_front, 0] * intrinsics[0, 0] / z[in_front]) + intrinsics[0, 2]
        uv[in_front, 1] = (points_cam[in_front, 1] * intrinsics[1, 1] / z[in_front]) + intrinsics[1, 2]

        if h <= 1:
            return project_points_to_image(points, intrinsics, camera_T_lidar, image_size)
        v = uv[:, 1]
        if self.rolling_shutter_direction == "top_to_bottom":
            row_frac = v / float(h - 1)
        else:
            row_frac = (float(h - 1) - v) / float(h - 1)
        dt = (row_frac - 0.5) * float(readout_sec)
        dt = np.where(np.isfinite(dt), dt, 0.0)
        cross = np.cross(omega_cam.reshape(1, 3), points_cam)
        points_cam = points_cam + dt.reshape(-1, 1) * cross
        z = points_cam[:, 2]
        in_front = z > 0
        uv = np.zeros((points_cam.shape[0], 2), dtype=float)
        uv[in_front, 0] = (points_cam[in_front, 0] * intrinsics[0, 0] / z[in_front]) + intrinsics[0, 2]
        uv[in_front, 1] = (points_cam[in_front, 1] * intrinsics[1, 1] / z[in_front]) + intrinsics[1, 2]
        inside = (
            (uv[:, 0] >= 0)
            & (uv[:, 0] < w)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < h)
            & in_front
        )
        return uv, inside

    def _deskew_lidar_points(
        self,
        points: np.ndarray,
        t_raw: np.ndarray,
        scan_stamp: rospy.Time,
    ) -> np.ndarray:
        if not self.lidar_deskew_enable:
            return points
        if t_raw is None or t_raw.size == 0:
            return points
        imu = self._lookup_lidar_imu(scan_stamp)
        if imu is None:
            now = time.time()
            if now - self._lidar_deskew_warn_at > 2.0:
                self._log.warn(
                    "_deskew_lidar_points",
                    "Deskew enabled but IMU lookup failed; skipping deskew.",
                )
                self._lidar_deskew_warn_at = now
            return points
        omega, accel = imu
        dt = t_raw.astype(np.float64) * float(self.lidar_time_scale)
        if self.lidar_deskew_ref == "mid":
            dt = dt - 0.5 * float(np.nanmax(dt))
        now = time.time()
        if now - self._lidar_deskew_log_at > 2.0:
            self._log.info(
                "_deskew_lidar_points",
                "Deskew active: mode=%s ref=%s dt_max=%.6f",
                self.lidar_deskew_mode,
                self.lidar_deskew_ref,
                float(np.nanmax(dt)) if dt.size else 0.0,
            )
            self._lidar_deskew_log_at = now
        if self.lidar_deskew_mode in ("rotation", "both"):
            cross = np.cross(omega.reshape(1, 3), points)
            points = points - dt.reshape(-1, 1) * cross
        if self.lidar_deskew_mode in ("translation", "both"):
            if not self.lidar_imu_accel_is_gravity_compensated:
                self._log.warn(
                    "_deskew_lidar_points",
                    "Translation deskew assumes gravity-compensated IMU accel; set ~lidar_imu_accel_gravity_compensated=true if already corrected.",
                )
            disp = 0.5 * accel.reshape(1, 3) * (dt.reshape(-1, 1) ** 2)
            points = points - disp
        return points

    def _maybe_init_undistort(self) -> None:
        if not self.undistort_semantic:
            return
        if self.mode != "lidar":
            self._log.warn(
                "_maybe_init_undistort",
                "undistort_semantic enabled but mode=%s; undistort is only applied in lidar mode.",
                self.mode,
            )
            return
        if self._camera_distortion is None or not self._camera_distortion.size:
            self._log.warn(
                "_maybe_init_undistort",
                "CameraInfo has no distortion coefficients; skipping undistort.",
            )
            return
        if np.allclose(self._camera_distortion, 0.0):
            self._log.info(
                "_maybe_init_undistort",
                "CameraInfo distortion coefficients are zero; skipping undistort.",
            )
            return
        try:
            import cv2  # type: ignore
        except Exception as exc:  # noqa: BLE001
            self._log.warn(
                "_maybe_init_undistort",
                "OpenCV not available (%s); cannot undistort semantic images.",
                exc,
            )
            return
        h, w = self._camera_info_size
        if h <= 0 or w <= 0:
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
            self._log.warn(
                "_maybe_init_undistort",
                "Unsupported distortion_model=%s; skipping undistort.",
                model,
            )
            return

        self._cv2 = cv2
        self._undistort_map1 = map1
        self._undistort_map2 = map2
        self._undistort_active = True
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
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                target_frame, source_frame, stamp, rospy.Duration(1.0)
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
        self._log.debug(
            "_lookup_transform",
            "TF %s -> %s:\n%s",
            source_frame,
            target_frame,
            format_matrix(mat),
        )
        return mat

    def _lookup_transform_with_stamp(self, target_frame, source_frame, stamp):
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                target_frame, source_frame, stamp, rospy.Duration(1.0)
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

    def _log_stamp_debug(
        self,
        context: str,
        sem_stamp: rospy.Time,
        other_stamp: rospy.Time,
        other_label: str,
        chosen_stamp: rospy.Time,
        shifted_stamp: rospy.Time,
    ) -> None:
        if not self.debug:
            return
        dt_sec = (sem_stamp - other_stamp).to_sec()
        self._log.debug(
            context,
            "stamps: semantic=%.9f %s=%.9f dt=%.9f chosen=%.9f shifted=%.9f source=%s offset=%.6f",
            sem_stamp.to_sec(),
            other_label,
            other_stamp.to_sec(),
            dt_sec,
            chosen_stamp.to_sec(),
            shifted_stamp.to_sec(),
            self.cloud_stamp_source,
            float(self.cloud_time_offset_sec),
        )

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

    def _publish_lidar_projection_debug(
        self, base_rgb, image_shape, uv, header, colors_u8=None
    ):
        if self._debug_proj_pub is None:
            return
        if image_shape is None or uv is None or uv.size == 0:
            return
        h, w = image_shape
        if base_rgb is None:
            base_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        img = np.ascontiguousarray(base_rgb.copy())
        uv_int = np.round(uv).astype(np.int32, copy=False)
        u = uv_int[:, 0]
        v = uv_int[:, 1]
        in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u = u[in_bounds]
        v = v[in_bounds]
        if colors_u8 is not None:
            colors_u8 = np.asarray(colors_u8, dtype=np.uint8)
            colors_u8 = colors_u8[in_bounds]
        stride = int(self.debug_projected_stride)
        if stride > 1 and u.size:
            u = u[::stride]
            v = v[::stride]
            if colors_u8 is not None:
                colors_u8 = colors_u8[::stride]
        if u.size:
            if colors_u8 is None:
                img[v, u] = (255, 0, 0)
            else:
                img[v, u] = colors_u8
        msg = Image()
        msg.header = header
        msg.height = h
        msg.width = w
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = w * 3
        msg.data = img.tobytes()
        self._debug_proj_pub.publish(msg)

    def _infer_num_labels(self, labels_img: np.ndarray) -> int:
        labels_img = np.asarray(labels_img)
        flat = labels_img.reshape(-1)
        flat = flat[flat >= 0]
        if flat.size == 0:
            return 0
        return int(flat.max()) + 1

    def _depth_to_debug_colors(self, depths: np.ndarray) -> Optional[np.ndarray]:
        if depths is None or depths.size == 0:
            return None
        depths = np.asarray(depths, dtype=np.float32).reshape(-1)
        valid = np.isfinite(depths) & (depths > 0)
        if not np.any(valid):
            return None
        dmin = float(self.debug_projected_depth_min)
        if dmin <= 0:
            dmin = float(np.nanmin(depths[valid]))
        dmax = float(self.debug_projected_depth_max)
        if dmax <= 0:
            if self.max_depth_m is not None and self.max_depth_m > dmin:
                dmax = float(self.max_depth_m)
            else:
                dmax = float(np.nanpercentile(depths[valid], 95))
        if dmax <= dmin:
            dmax = dmin + 1e-3
        t = (depths - dmin) / (dmax - dmin)
        t = np.clip(t, 0.0, 1.0)
        r = (t * 255.0).astype(np.uint8, copy=False)
        g = np.zeros_like(r, dtype=np.uint8)
        b = ((1.0 - t) * 255.0).astype(np.uint8, copy=False)
        return np.stack((r, g, b), axis=-1)

    def _get_rgb_float_lut(self, labels_img: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        if not self.colorize_labels:
            return None
        if self.semantic_input_type != "labels":
            return None

        if self.color_map is not None:
            if self._rgb_lut is None or self._rgb_lut_num_labels != -1:
                self._rgb_lut = build_label_rgb_float_lut(color_map=self.color_map)
                self._rgb_lut_num_labels = -1
                self._log.debug(
                    "_get_rgb_float_lut",
                    "Built label->rgb LUT from color_map (entries=%d)",
                    len(self.color_map),
                )
            return self._rgb_lut

        # No color_map: deterministic random palette based on number of labels.
        n = int(self.num_labels) if int(self.num_labels) > 0 else None
        if n is None and labels_img is not None:
            n = self._infer_num_labels(labels_img)
        if n is None or n <= 0:
            n = 256
        if self._rgb_lut is None or self._rgb_lut_num_labels != int(n):
            self._rgb_lut = build_label_rgb_float_lut(
                num_labels=int(n), seed=int(self.random_color_seed)
            )
            self._rgb_lut_num_labels = int(n)
            if not self._warned_random_palette:
                self._log.warn(
                    "_get_rgb_float_lut",
                    "colorize_labels is true but ~color_map is empty; using deterministic random palette (num_labels=%d seed=%d). Provide ~color_map for stable colors.",
                    int(n),
                    int(self.random_color_seed),
                )
                self._warned_random_palette = True
        return self._rgb_lut

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
        services = (
            f"save_ply={rospy.resolve_name('~save_ply')} "
            f"set_ply_recording={rospy.resolve_name('~set_ply_recording')}"
        )
        ply_target = self.ply_target_frame or "-"
        ply = (
            f"recording={self._ply_recording} dir={self.ply_output_dir} "
            f"target_frame={ply_target} use_latest={self.ply_tf_use_latest} "
            f"tol={self.ply_tf_tolerance_sec:.6f}"
        )
        help_text = "\n".join(
            [
                f"rosservice call {rospy.resolve_name('~save_ply')} \"{{}}\"",
                f"rosservice call {rospy.resolve_name('~set_ply_recording')} \"data: true\"",
            ]
        )

        rows = [
            ("node", self._node_name),
            ("mode", f"{self.mode} ({self._mode_source})"),
            ("mode_detail", self._mode_detail or "-"),
            ("target_frame", self.target_frame),
            ("camera_frame", self.camera_frame),
            ("output", self._output_topic),
            ("semantic_topic", self.semantic_topic),
            ("semantic_type", self.semantic_input_type),
            ("confidence_topic", self.conf_topic or "-"),
            ("camera_info", self.camera_info_topic),
            ("depth_input_topic", self.depth_input_topic or "-"),
            ("downsample", str(int(self.downsample_factor))),
            ("sync_slop_sec", f"{self.sync_slop_sec:.6f}"),
            ("pair_max_dt_sec", f"{self.pair_max_dt_sec:.6f}"),
            ("sync_queue_size", str(int(self.sync_queue_size))),
            ("undistort_semantic", str(bool(self.undistort_semantic))),
            ("undistort_alpha", f"{self.undistort_alpha:.2f}"),
            ("rolling_shutter_enable", str(bool(self.rolling_shutter_enable))),
            ("rolling_shutter_readout_sec", f"{self.rolling_shutter_readout_sec:.6f}"),
            ("rolling_shutter_direction", self.rolling_shutter_direction),
            ("imu_topic", self.imu_topic or "-"),
            ("imu_frame", self.imu_frame or "-"),
            ("camera_metadata_topic", self.camera_metadata_topic or "-"),
            ("metadata_readout_key", str(int(self.metadata_readout_key))),
            ("metadata_readout_scale", f"{self.metadata_readout_scale:.6g}"),
            ("metadata_max_dt_sec", f"{self.metadata_max_dt_sec:.3f}"),
            ("lidar_deskew_enable", str(bool(self.lidar_deskew_enable))),
            ("lidar_deskew_mode", self.lidar_deskew_mode),
            ("lidar_deskew_ref", self.lidar_deskew_ref),
            ("lidar_time_field", self.lidar_time_field),
            ("lidar_time_scale", f"{self.lidar_time_scale:.3g}"),
            ("lidar_imu_topic", self.lidar_imu_topic or "-"),
            ("lidar_imu_frame", self.lidar_imu_frame or "-"),
            ("lidar_imu_cache_size", str(int(self.lidar_imu_cache_size))),
            ("lidar_imu_cache_max_dt_sec", f"{self.lidar_imu_cache_max_dt_sec:.3f}"),
            (
                "lidar_imu_accel_gravity_compensated",
                str(bool(self.lidar_imu_accel_is_gravity_compensated)),
            ),
            ("max_depth_m", f"{self.max_depth_m:.3f}" if self.max_depth_m is not None else "-"),
            ("cloud_time_offset_sec", f"{self.cloud_time_offset_sec:.6f}"),
            ("cloud_stamp_source", self.cloud_stamp_source),
            ("colorize_labels", str(bool(self.colorize_labels))),
            ("ply", ply),
            ("services", services),
            ("help", help_text),
        ]

        def _fmt_tf(name: str, mat: Optional[np.ndarray], src: str, src_frame: str, dst_frame: str):
            label = f"{src_frame or '<frame>'} -> {dst_frame}"
            if mat is None:
                return (name, f"{label} ({src}: pending)")
            return (name, f"{label} ({src})\n{format_matrix(mat)}")

        if self.mode == "depth":
            src = "static_target_T_depth" if self.static_target_T_depth is not None else "tf2"
            rows.append(
                _fmt_tf(
                    "target_T_depth",
                    self.target_T_depth,
                    src,
                    self._depth_frame,
                    self.target_frame,
                )
            )
        else:
            src_cam = "static_camera_T_lidar" if self.static_camera_T_lidar is not None else "tf2"
            src_tgt = "static_target_T_lidar" if self.static_target_T_lidar is not None else "tf2"
            rows.append(
                _fmt_tf(
                    "camera_T_lidar",
                    self.camera_T_lidar,
                    src_cam,
                    self._lidar_frame,
                    self.camera_frame,
                )
            )
            rows.append(
                _fmt_tf(
                    "target_T_lidar",
                    self.target_T_lidar,
                    src_tgt,
                    self._lidar_frame,
                    self.target_frame,
                )
            )

        return render_kv_table(rows)

    def _passes_pair_gate(self, stamp_a: rospy.Time, stamp_b: rospy.Time, ctx: str) -> bool:
        if self.pair_max_dt_sec <= 0.0:
            return True
        dt = abs((stamp_a - stamp_b).to_sec())
        if dt <= self.pair_max_dt_sec:
            return True
        self._log.warn(
            ctx,
            "Dropping pair: |Δt|=%.6fs > %.6fs",
            dt,
            float(self.pair_max_dt_sec),
        )
        return False

    def _depth_callback(self, sem_msg, depth_msg, conf_msg=None):
        t0 = time.perf_counter()
        with self._maybe_profile("depth_callback"):
            sem_stamp = sem_msg.header.stamp
            depth_stamp = depth_msg.header.stamp
            if not self._passes_pair_gate(sem_stamp, depth_stamp, "_depth_callback"):
                return
            chosen_stamp = self._choose_cloud_stamp(sem_stamp, depth_stamp, "depth")
            stamp = self._apply_cloud_time_offset(chosen_stamp)
            self._log_stamp_debug(
                "_depth_callback",
                sem_stamp,
                depth_stamp,
                "depth",
                chosen_stamp,
                stamp,
            )
            target_T_depth = self.target_T_depth
            if target_T_depth is None:
                depth_frame = depth_msg.header.frame_id
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
                return

            include_rgb = bool(self.colorize_labels)
            rgb_values = None
            rgb_lut = None

            if self.semantic_input_type == "labels":
                labels = self._parse_semantic_labels(sem_msg)
                if include_rgb:
                    rgb_lut = self._get_rgb_float_lut(labels)
            else:
                packed_img = self._parse_semantic_rgb_packed(sem_msg)
                labels = None
                if include_rgb and self.color_map and not self._warned_rgb_color_map:
                    self._log.warn(
                        "_depth_callback",
                        "~color_map is ignored when semantic_input_type=rgb (colors are passed through)",
                    )
                    self._warned_rgb_color_map = True

            depth_raw = image_to_numpy(depth_msg)
            depth_enc = depth_msg.encoding.lower()
            invalid_raw = None
            if self.filter_invalid_depth and depth_enc in ("16uc1", "mono16", "16sc1"):
                invalid_raw = (depth_raw == 0) | (
                    depth_raw == np.iinfo(np.uint16).max
                )
            confidence = (
                image_to_numpy(conf_msg).astype(np.float32, copy=False)
                if conf_msg
                else None
            )
            if confidence is not None and self._undistort_active:
                confidence = self._undistort_array(confidence, interpolation="linear")

            if self.downsample_factor > 1:
                f = self.downsample_factor
                if labels is not None:
                    labels = labels[::f, ::f]
                else:
                    packed_img = packed_img[::f, ::f]
                depth_raw = depth_raw[::f, ::f]
                if confidence is not None:
                    confidence = confidence[::f, ::f]
                intrinsics = self._scale_intrinsics(self.intrinsics, f)
            else:
                intrinsics = self.intrinsics

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

            if self.semantic_input_type == "labels":
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
                    points_target = transform_points(target_T_depth, points_cam)
                    pcl = SemanticPointCloud(points_target, labels_all, conf_flat)
                    if include_rgb:
                        colors_packed = packed_img[valid_mask].reshape(-1).astype(
                            np.uint32, copy=False
                        )
                        rgb_values = colors_packed.astype("<u4", copy=False).view("<f4")

            if include_rgb and rgb_values is None and rgb_lut is not None:
                rgb_values = rgb_lut[labels_to_uint16(pcl.labels)]
                rgb_lut = None

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

        dt = time.perf_counter() - t0
        self._maybe_emit_status(points=int(pcl.points_xyz.shape[0]), callback_sec=dt)

    def _lidar_callback(self, sem_msg, lidar_msg, conf_msg=None):
        t0 = time.perf_counter()
        with self._maybe_profile("lidar_callback"):
            sem_stamp = sem_msg.header.stamp
            lidar_stamp = lidar_msg.header.stamp
            if not self._passes_pair_gate(sem_stamp, lidar_stamp, "_lidar_callback"):
                return
            chosen_stamp = self._choose_cloud_stamp(sem_stamp, lidar_stamp, "lidar")
            stamp = self._apply_cloud_time_offset(chosen_stamp)
            self._log_stamp_debug(
                "_lidar_callback",
                sem_stamp,
                lidar_stamp,
                "lidar",
                chosen_stamp,
                stamp,
            )
            camera_T_lidar = self.camera_T_lidar
            target_T_lidar = self.target_T_lidar
            if camera_T_lidar is None or target_T_lidar is None:
                lidar_frame = lidar_msg.header.frame_id
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
                return

            include_rgb = bool(self.colorize_labels)
            rgb_values = None
            rgb_lut = None

            if self.semantic_input_type == "labels":
                labels = self._parse_semantic_labels(sem_msg)
                if include_rgb:
                    rgb_lut = self._get_rgb_float_lut(labels)
            else:
                packed_img = self._parse_semantic_rgb_packed(sem_msg)
                labels = None
                if include_rgb and self.color_map and not self._warned_rgb_color_map:
                    self._log.warn(
                        "_lidar_callback",
                        "~color_map is ignored when semantic_input_type=rgb (colors are passed through)",
                    )
                    self._warned_rgb_color_map = True

            confidence = (
                image_to_numpy(conf_msg).astype(np.float32, copy=False)
                if conf_msg
                else None
            )
            if self.downsample_factor > 1:
                f = self.downsample_factor
                if labels is not None:
                    labels = labels[::f, ::f]
                else:
                    packed_img = packed_img[::f, ::f]
                if confidence is not None:
                    confidence = confidence[::f, ::f]
                intrinsics = self._scale_intrinsics(self.intrinsics, f)
            else:
                intrinsics = self.intrinsics

            if self.lidar_deskew_enable:
                points, t_raw = pointcloud2_to_xyz_t(
                    lidar_msg, time_field=self.lidar_time_field
                )
                points = self._deskew_lidar_points(points, t_raw, lidar_stamp)
            else:
                points = pointcloud2_to_xyz(lidar_msg)
            debug_colors = None
            if self.debug_project_lidar and self.debug_projected_colorize == "depth":
                points_cam_dbg = transform_points(camera_T_lidar, points)
                debug_colors = self._depth_to_debug_colors(points_cam_dbg[:, 2])

            if self.semantic_input_type == "labels":
                h, w = labels.shape
                uv, inside = self._project_lidar_points(
                    points, intrinsics, camera_T_lidar, (w, h), sem_msg.header.stamp
                )
                if self.debug_project_lidar and labels is not None:
                    base = (labels.astype(np.int32) % 256).astype(np.uint8)
                    base_rgb = np.stack((base, base, base), axis=-1)
                    self._publish_lidar_projection_debug(
                        base_rgb,
                        (h, w),
                        uv,
                        sem_msg.header,
                        colors_u8=debug_colors,
                    )
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

                labeled_points = points[inside]
                if labeled_points.shape[0] == 0 and not self.include_unlabeled:
                    pcl = SemanticPointCloud(
                        np.empty((0, 3)), np.empty((0,), dtype=np.int64), None
                    )
                else:
                    labels_in = labels[v, u]
                    if confidence is not None:
                        conf_in = confidence[v, u].astype(np.float32, copy=False)
                    else:
                        conf_in = None
                    points_target_labeled = transform_points(
                        target_T_lidar, labeled_points
                    )
                    if self.include_unlabeled:
                        unlabeled_points = points[~inside]
                        points_all = np.vstack(
                            (
                                points_target_labeled,
                                transform_points(target_T_lidar, unlabeled_points),
                            )
                        )
                        labels_all = np.concatenate(
                            (
                                labels_in.astype(np.int64),
                                np.full(
                                    unlabeled_points.shape[0],
                                    -1,
                                    dtype=np.int64,
                                ),
                            )
                        )
                        if conf_in is not None:
                            conf_all = np.concatenate(
                                (
                                    conf_in.astype(np.float32, copy=False),
                                    np.zeros(
                                        unlabeled_points.shape[0], dtype=np.float32
                                    ),
                                )
                            )
                        else:
                            conf_all = None
                        pcl = SemanticPointCloud(points_all, labels_all, conf_all)
                    else:
                        pcl = SemanticPointCloud(
                            points_target_labeled,
                            labels_in.astype(np.int64),
                            conf_in,
                        )
            else:
                h, w = packed_img.shape
                uv, inside = self._project_lidar_points(
                    points, intrinsics, camera_T_lidar, (w, h), sem_msg.header.stamp
                )
                if self.debug_project_lidar:
                    base_rgb = packed_rgb_to_triplets(packed_img)
                    self._publish_lidar_projection_debug(
                        base_rgb,
                        (h, w),
                        uv,
                        sem_msg.header,
                        colors_u8=debug_colors,
                    )
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

                points_in = points[inside]
                if include_rgb:
                    colors_packed = packed_img[v, u].astype(np.uint32, copy=False)
                    rgb_values_in = colors_packed.astype("<u4", copy=False).view("<f4")
                else:
                    rgb_values_in = None
                conf_in = (
                    confidence[v, u].astype(np.float32, copy=False)
                    if confidence is not None
                    else None
                )
                points_in_t = transform_points(target_T_lidar, points_in)
                labels_in = np.full(points_in_t.shape[0], -1, dtype=np.int64)

                if self.include_unlabeled:
                    points_out = points[~inside]
                    points_out_t = transform_points(target_T_lidar, points_out)
                    labels_out = np.full(points_out_t.shape[0], -1, dtype=np.int64)
                    if include_rgb:
                        rgb_out = np.zeros(points_out_t.shape[0], dtype=np.float32)
                        rgb_values = np.concatenate((rgb_values_in, rgb_out))
                    else:
                        rgb_values = None
                    if conf_in is not None:
                        conf_out = np.zeros(points_out_t.shape[0], dtype=np.float32)
                        conf_all = np.concatenate((conf_in, conf_out))
                    else:
                        conf_all = None
                    points_all = np.vstack((points_in_t, points_out_t))
                    labels_all = np.concatenate((labels_in, labels_out))
                else:
                    points_all = points_in_t
                    labels_all = labels_in
                    rgb_values = rgb_values_in
                    conf_all = conf_in

                pcl = SemanticPointCloud(points_all, labels_all, conf_all)

            if self.max_depth_m is not None and pcl.points_xyz.shape[0]:
                ranges = np.linalg.norm(pcl.points_xyz, axis=1)
                keep = ranges <= float(self.max_depth_m)
                if not np.all(keep):
                    pcl = SemanticPointCloud(
                        pcl.points_xyz[keep],
                        pcl.labels[keep],
                        pcl.confidence[keep] if pcl.confidence is not None else None,
                    )
                    if rgb_values is not None:
                        rgb_values = rgb_values[keep]

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

            if include_rgb and rgb_values is None and rgb_lut is not None:
                rgb_values = rgb_lut[labels_to_uint16(pcl.labels)]
                rgb_lut = None

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

        dt = time.perf_counter() - t0
        self._maybe_emit_status(points=int(pcl.points_xyz.shape[0]), callback_sec=dt)


def main():
    log_level = rospy.DEBUG if _rosargv_bool("debug", False) else rospy.INFO
    rospy.init_node("colored_pcl_node", log_level=log_level)
    ColoredPclNode()
    rospy.spin()


if __name__ == "__main__":
    main()
