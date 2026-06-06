"""Typed startup assembly helpers for prune node."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import rospy

from ..pipelines.camera_model import CameraModel
from prune_ros.config import ColorConfig, SyncConfig
from ..config.config_gate import load_gate_config
from ..pipelines.frame_inputs import FrameInputPreparer
from ..pipelines.ply_service import PlyRecordingService
from ..pipelines.semantic_inputs import SemanticInputParser
from ..pipelines.sync_policy import StampPolicy
from ..pipelines.tracked_reprojection_runtime import TrackedReprojectionRuntime


@dataclass
class StartupComponents:
    stamp_policy: StampPolicy
    camera_model: CameraModel
    ply_service: PlyRecordingService
    tracked_runtime: TrackedReprojectionRuntime
    semantic_parser: SemanticInputParser
    frame_inputs: FrameInputPreparer

    def apply_to(self, node: Any) -> None:
        node._stamp_policy = self.stamp_policy
        node._camera_model = self.camera_model
        node._ply_service = self.ply_service
        node._tracked_runtime = self.tracked_runtime
        node._semantic_parser = self.semantic_parser
        node._frame_inputs = self.frame_inputs


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
        return

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
        node._undistort_status = camera_model._undistort_status
        return StartupComponents(
            stamp_policy=stamp_policy,
            camera_model=camera_model,
            ply_service=PlyRecordingService(node, node._log),
            tracked_runtime=TrackedReprojectionRuntime(node),
            semantic_parser=SemanticInputParser(
                node,
                ColorConfig(
                    colorize_labels=node.colorize_labels,
                    color_map=dict(node.color_map) if node.color_map else {},
                    random_color_seed=int(node.random_color_seed),
                    num_labels=int(node.num_labels),
                    semantic_color_quantization_step=int(node.semantic_color_quantization_step),
                ),
                load_gate_config(node),
                node._log,
            ),
            frame_inputs=FrameInputPreparer(node),
        )
