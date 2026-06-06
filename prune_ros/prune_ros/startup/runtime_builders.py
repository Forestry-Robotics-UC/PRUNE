"""Runtime builder helpers for prune."""

from __future__ import annotations

from typing import Any

from ..runtime.debug_publisher import DebugPublisher, DebugPublisherParams
from ..projection.lidar_projector import LidarProjectorParams


def build_projector_params(node: Any) -> LidarProjectorParams:
    return LidarProjectorParams(
        max_depth_m=node.max_depth_m,
        camera_fov_gate_enable=bool(node.camera_fov_gate_enable),
        camera_fov_gate_margin_deg=float(node.camera_fov_gate_margin_deg),
        rolling_shutter_enable=bool(node.rolling_shutter_enable),
        rolling_shutter_direction=str(node.rolling_shutter_direction),
        projection_patch_size=int(node.projection_patch_size),
        projection_occlusion_epsilon_m=float(node.projection_occlusion_epsilon_m),
        projection_occlusion_radius_px=int(node.projection_occlusion_radius_px),
        projection_reject_depth_edges=bool(node.projection_reject_depth_edges),
        projection_depth_edge_thresh=float(node.projection_depth_edge_thresh),
        projection_depth_edge_radius_px=int(node.projection_depth_edge_radius_px),
        projection_confidence_min=float(node.projection_confidence_min),
        use_invalid_mask=bool(node.use_invalid_mask),
        use_depth_edge_rejection=bool(node.use_depth_edge_rejection),
        use_occlusion_gate=bool(node.use_occlusion_gate),
        include_unlabeled=bool(node.include_unlabeled),
        colorize_labels=bool(node.colorize_labels),
        semantic_input_type=str(node.semantic_input_type),
        color_map=dict(node.color_map) if node.color_map else {},
        random_color_seed=int(node.random_color_seed),
        num_labels=int(node.num_labels),
        debug_project_lidar=bool(node.debug_project_lidar),
        depth_map_subsample=int(node.depth_map_subsample),
        edge_cache_max_age_sec=float(node.edge_cache_max_age_sec),
        use_range_image_edges=str(node.use_range_image_edges),
        overlay_output_dir=str(node.overlay_output_dir),
        overlay_output_stride=int(node.overlay_output_stride),
        overlay_dot_radius=int(node.overlay_dot_radius),
    )


def build_debug_pub_params(node: Any) -> DebugPublisherParams:
    return DebugPublisherParams(
        debug_project_lidar=bool(node.debug_project_lidar),
        debug_project_lidar_stride=int(node.debug_project_lidar_stride),
        debug_project_lidar_radius=int(node.debug_project_lidar_radius),
        debug_project_lidar_outline_only=bool(node.debug_project_lidar_outline_only),
        debug_range_view=bool(node.debug_range_view),
        debug_publish_fov_points=bool(node.debug_publish_fov_points),
        tracked_reprojection_enable=bool(node.tracked_reprojection_enable),
        debug_output_dir=str(node.debug_output_dir),
        debug_output_stride=int(node.debug_output_stride),
    )


def build_debug_publisher(node: Any) -> DebugPublisher | None:
    if not any(
        [
            node.debug_project_lidar,
            node.debug_range_view,
            node.debug_publish_fov_points,
            node.tracked_reprojection_enable,
        ]
    ):
        return None
    return DebugPublisher(
        build_debug_pub_params(node),
        node_name=node._node_name,
        lidar_frame=node._lidar_frame,
        target_frame=node.target_frame,
    )
