"""Typed startup assembly helpers for prune node."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import rospy

from prune_ros.pipelines import CameraModel
from prune_ros.config import ColorConfig, GateConfig, SyncConfig
from prune_ros.pipelines import FrameInputPreparer
from prune_ros.pipelines import OnlineCalibrationBridge
from prune_ros.pipelines import PlyRecordingService
from prune_ros.pipelines import SemanticInputParser
from prune_ros.pipelines import StampPolicy
from prune_ros.pipelines import TrackedReprojectionRuntime


@dataclass
class StartupComponents:
    stamp_policy: StampPolicy
    camera_model: CameraModel
    ply_service: PlyRecordingService
    tracked_runtime: TrackedReprojectionRuntime
    calibration_bridge: OnlineCalibrationBridge
    semantic_parser: SemanticInputParser
    frame_inputs: FrameInputPreparer


class PruneStartupBuilder:
    def __init__(self, node: Any):
        self._node = node

    def prepare_runtime_state(self) -> None:
        node = self._node
        node._output_topic = rospy.resolve_name("semantic_pointcloud")
        node.target_T_depth = None
        node.camera_T_lidar = None
        node.target_T_lidar = None
        node._depth_frame = ""
        node._lidar_frame = ""
        node._mode_source = "forced" if node.mode in ("depth", "lidar") else "auto"
        node._mode_detail = "forced via ~mode" if node._mode_source == "forced" else ""
        if node.mode not in ("depth", "lidar"):
            node.mode = node._bootstrap.detect_mode()

    def finalize_mode_status(self) -> None:
        node = self._node
        node._online_calibration_rpy_rad = np.zeros(3, dtype=np.float64)
        if node.online_calibration_enable and node.mode != "lidar":
            node._log.warn(
                "__init__",
                "online_calibration_enable=true requires lidar mode; disabling because mode=%s",
                node.mode,
            )
            node.online_calibration_enable = False
            node._online_calibration_status = f"disabled (mode={node.mode})"
        elif node.online_calibration_enable:
            node._online_calibration_status = "active"
        else:
            node._online_calibration_status = "disabled"

    def build_components(self) -> StartupComponents:
        node = self._node
        stamp_policy = StampPolicy(
            node,
            SyncConfig(
                sync_slop_sec=node.sync_slop_sec,
                pair_max_dt_sec=node.pair_max_dt_sec,
                semantic_time_offset_sec=node.semantic_time_offset_sec,
                sync_queue_size=node.sync_queue_size,
                cloud_time_offset_sec=node.cloud_time_offset_sec,
                cloud_stamp_source=node.cloud_stamp_source,
                stamp_debug_log_period_sec=node.stamp_debug_log_period_sec,
            ),
            node._log,
        )
        stamp_policy.resolve_cloud_stamp_source()
        camera_model = CameraModel(node, node._log)
        camera_model.load()
        return StartupComponents(
            stamp_policy=stamp_policy,
            camera_model=camera_model,
            ply_service=PlyRecordingService(node, node._log),
            tracked_runtime=TrackedReprojectionRuntime(node),
            calibration_bridge=OnlineCalibrationBridge(node),
            semantic_parser=SemanticInputParser(
                node,
                ColorConfig(
                    colorize_labels=node.colorize_labels,
                    color_map=dict(node.color_map) if node.color_map else {},
                    random_color_seed=int(node.random_color_seed),
                    num_labels=int(node.num_labels),
                    semantic_color_quantization_step=int(node.semantic_color_quantization_step),
                ),
                GateConfig(
                    projection_patch_size=int(node.projection_patch_size),
                    projection_confidence_min=float(node.projection_confidence_min),
                    projection_invalid_mask_topic=str(node.projection_invalid_mask_topic),
                    projection_invalid_mask_value=int(node.projection_invalid_mask_value),
                    projection_invalid_mask_dilate_px=int(node.projection_invalid_mask_dilate_px),
                    projection_occlusion_epsilon_m=float(node.projection_occlusion_epsilon_m),
                    projection_occlusion_radius_px=int(node.projection_occlusion_radius_px),
                    projection_reject_depth_edges=bool(node.projection_reject_depth_edges),
                    projection_depth_edge_thresh=float(node.projection_depth_edge_thresh),
                    projection_depth_edge_radius_px=int(node.projection_depth_edge_radius_px),
                    downsample_factor=int(node.downsample_factor),
                    depth_map_subsample=int(node.depth_map_subsample),
                    edge_cache_max_age_sec=float(node.edge_cache_max_age_sec),
                    use_range_image_edges=str(node.use_range_image_edges),
                    overlay_output_dir=str(node.overlay_output_dir),
                    overlay_output_stride=int(node.overlay_output_stride),
                    overlay_dot_radius=int(node.overlay_dot_radius),
                ),
                node._log,
            ),
            frame_inputs=FrameInputPreparer(node),
        )
