"""Metrics and startup-report helpers for prune."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import rospy

from .experiment_metrics import FrameMetrics, MetricsCsvLogger
from ..projection.lidar_projector import GateMetrics, projection_health_from_counts


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
        projection_metrics: GateMetrics,
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
            + int(projection_metrics.num_rejected_geometric)
        )
        num_rejected_other = max(0, projected - int(num_output_points) - known_rejected)
        output_retention_ratio = float(num_output_points) / max(projected, 1)
        input_points = max(int(num_input_points), 1)
        in_front = int(projection_metrics.num_points_in_front)
        projection_projected_ratio = float(projected) / float(input_points)
        projection_in_front_ratio = float(in_front) / float(input_points)
        projection_in_image_ratio = float(projected) / float(max(in_front, 1))
        projection_invalid_mask_hit_ratio = float(projection_metrics.num_would_hit_invalid_mask) / float(max(projected, 1))
        projection_confidence_rejection_ratio = float(projection_metrics.num_rejected_confidence) / float(max(projected, 1))
        projection_depth_edge_rejection_ratio = float(projection_metrics.num_rejected_depth_edge) / float(max(projected, 1))
        projection_occlusion_rejection_ratio = float(projection_metrics.num_rejected_occlusion) / float(max(projected, 1))
        projection_geometric_rejection_ratio = float(projection_metrics.num_rejected_geometric) / float(max(projected, 1))
        num_projection_rejected = known_rejected + num_rejected_other
        num_projection_accepted = int(num_output_points)
        num_projection_suppressed = max(0, projected - num_projection_accepted)
        projection_health_score = projection_health_from_counts(
            total_points=int(num_input_points),
            in_front_points=in_front,
            projected_points=projected,
            rejection_ratio=max(
                projection_invalid_mask_hit_ratio,
                projection_confidence_rejection_ratio,
                projection_depth_edge_rejection_ratio,
                projection_occlusion_rejection_ratio,
                projection_geometric_rejection_ratio,
            ),
        )
        if projection_metrics.projection_health_score > 0.0:
            projection_health_score = float(projection_metrics.projection_health_score)
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
                projection_projected_ratio=projection_projected_ratio,
                projection_in_front_ratio=projection_in_front_ratio,
                projection_in_image_ratio=projection_in_image_ratio,
                projection_invalid_mask_hit_ratio=projection_invalid_mask_hit_ratio,
                projection_confidence_rejection_ratio=projection_confidence_rejection_ratio,
                projection_depth_edge_rejection_ratio=projection_depth_edge_rejection_ratio,
                projection_occlusion_rejection_ratio=projection_occlusion_rejection_ratio,
                projection_health_score=projection_health_score,
                num_projection_suppressed=num_projection_suppressed,
                num_projection_rejected=num_projection_rejected,
                num_projection_accepted=num_projection_accepted,
                num_rejected_geometric=int(projection_metrics.num_rejected_geometric),
                num_would_hit_geometric=int(projection_metrics.num_would_hit_geometric),
                would_hit_geometric_ratio=float(projection_metrics.num_would_hit_geometric) / max(projected, 1),
                projection_geometric_rejection_ratio=projection_geometric_rejection_ratio,
                runtime_geometric_ms=float(projection_metrics.runtime_geometric_ms),
            )
        )

    def close(self) -> None:
        if self._node._metrics_logger is not None:
            self._node._metrics_logger.close()
            self._node._metrics_logger = None
