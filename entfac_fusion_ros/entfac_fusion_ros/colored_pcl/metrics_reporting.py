"""Metrics and startup-report helpers for colored PCL."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import rospy

from entfac_fusion_ros.experiment_metrics import FrameMetrics, MetricsCsvLogger
from entfac_fusion_ros.lidar_projector import ProjectionMetrics


class MetricsReporter:
    def __init__(self, node: Any):
        self._node = node

    def setup(self) -> None:
        self._node._results_frame_index = 0
        self._node._metrics_logger = None
        self._node._results_run_dir = None
        if not self._node.enable_metrics_csv:
            return
        bag_name = self._node.experiment_bag_name or "unknown_bag"
        variant_name = self._node.experiment_variant_name or "default"
        root = Path(self._node.results_dir or (Path(self._node.debug_output_dir).parent / "results"))
        self._node._results_run_dir = root / bag_name / variant_name
        self._node._results_run_dir.mkdir(parents=True, exist_ok=True)
        self._node._metrics_logger = MetricsCsvLogger(
            self._node._results_run_dir / "metrics_per_frame.csv"
        )
        rospy.on_shutdown(self.close)

    def write_lidar_metrics(
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
        if self._node._metrics_logger is None:
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
        self._node._metrics_logger.write(
            FrameMetrics(
                bag_name=self._node.experiment_bag_name or "unknown_bag",
                variant_name=self._node.experiment_variant_name or "default",
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

    def close(self) -> None:
        if self._node._metrics_logger is not None:
            self._node._metrics_logger.close()
            self._node._metrics_logger = None
