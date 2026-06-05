"""Runtime assembly helpers for prune node."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from prune_ros.startup import runtime_builders
from prune_ros.pipelines import DepthFusionPipeline
from prune_ros.diagnostics import DiagnosticsOrchestrator
from prune_ros.pipelines import LidarFusionPipeline
from prune_ros.runtime import LiveTuningController
from prune_ros.diagnostics import MetricsReporter
from prune_ros.pipelines import PruneRosIo
from prune_ros.projection import LidarProjector


class RuntimeSetup:
    def __init__(self, node: Any):
        self._node = node

    def initialize_runtime_state(self) -> None:
        node = self._node
        node._tf_cache = node._tf_resolver.tf_cache
        node._bootstrap.prime_transforms()
        node._undistort_map1 = getattr(node._camera_model, "_undistort_map1", None)
        node._undistort_map2 = getattr(node._camera_model, "_undistort_map2", None)
        node._undistort_active = bool(getattr(node._camera_model, "_undistort_active", False))
        node._cv2 = getattr(node._camera_model, "_cv2", None)
        node._imu_cache = None
        node._imu_sub = None
        node._imu_to_camera_R = None
        node._metadata_latest = {}
        node._metadata_sub = None
        node._lidar_imu_cache = None
        node._lidar_imu_sub = None
        node._lidar_imu_to_lidar_R = None
        node._debug_callback_seq = 0
        node._rolling_shutter_log_at = 0.0
        node._rolling_shutter_warn_at = 0.0
        node._lidar_deskew_log_at = 0.0
        node._lidar_deskew_warn_at = 0.0
        node._lidar_deskew_missing_time_warn_at = 0.0
        node._live_param_refresh_period_sec = 0.5
        node._live_param_last_refresh_at = 0.0
        node._dynamic_reconfigure_server = None
        node._dynamic_reconfigure_initialized = False
        node._rgb_lut = None
        node._rgb_lut_num_labels = None
        node._warned_random_palette = False
        node._warned_rgb_color_map = False
        node._logged_depth_scaling = False
        node._logged_depth_summary = False
        node._logged_lidar_summary = False
        node._stamp_debug_last_log_at = 0.0
        node._ply_writer = node._ply_service._writer
        node._ply_recording = False
        node._ply_queue_warned_at = 0.0
        node._ply_seq = 0
        node._last_pcl = None
        node._results_frame_index = 0
        node._metrics_logger = None
        node._results_run_dir = None

    def validate_mode_dependent_flags(self) -> None:
        node = self._node
        if node.tracked_reprojection_enable:
            if node.mode != "lidar":
                node._log.warn(
                    "__init__",
                    "tracked_reprojection_enable=true requires lidar mode; disabling because mode=%s",
                    node.mode,
                )
                node.tracked_reprojection_enable = False
            else:
                node._log.warn(
                    "__init__",
                    "Tracked reprojection diagnostics are stateful and CPU-heavier than the online path; use them primarily for offline bag review or focused validation runs.",
                )

    def setup_metrics_and_ply(self) -> None:
        node = self._node
        node._metrics_reporter = MetricsReporter(node)
        node._metrics_reporter.setup()
        node._ply_service.setup()

    def setup_projector_and_buffers(self) -> None:
        node = self._node
        node._depth_buffer = None
        node._depth_buffer_shape = None
        node._edge_buffer = None
        node._edge_buffer_shape = None
        node._runtime_builders = runtime_builders
        node._projector = LidarProjector(node._runtime_builders.build_projector_params(node))

    def setup_subsystems(self) -> None:
        node = self._node
        node._calibration = node._calibration_bridge.build(node._projector)
        node._tracked_repr = node._tracked_runtime.build()
        node._debug_pub = node._runtime_builders.build_debug_publisher(node)
        node._diagnostics = DiagnosticsOrchestrator(node, node._debug_pub)
        node._depth_pipeline = DepthFusionPipeline(node)
        node._lidar_pipeline = LidarFusionPipeline(node)

    def setup_ros_runtime(self, reconfigure_server_cls: Any, reconfigure_config_cls: Any) -> None:
        node = self._node
        node._ros_io = PruneRosIo(node)
        node._live_tuning = LiveTuningController(
            node,
            node._log,
            reconfigure_server_cls,
            reconfigure_config_cls,
        )
        node._ros_io.setup_publishers()
        node._live_tuning.setup_dynamic_reconfigure()
        node._ros_io.register_subscribers()
