"""Evidence gate configuration for prune."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config_helpers import get_float, get_int


@dataclass
class GateConfig:
    projection_patch_size: int
    projection_confidence_min: float
    projection_invalid_mask_topic: str
    projection_invalid_mask_value: int
    projection_invalid_mask_dilate_px: int
    projection_occlusion_epsilon_m: float
    projection_occlusion_radius_px: int
    projection_reject_depth_edges: bool
    projection_depth_edge_thresh: float
    projection_depth_edge_radius_px: int
    enable_adaptive_projection_health: bool
    projection_health_warn_threshold: float
    projection_health_bad_threshold: float
    adaptive_confidence_threshold_offset: float
    adaptive_depth_edge_threshold_scale: float
    adaptive_prefer_suppression_on_bad_health: bool
    downsample_factor: int
    depth_map_subsample: int
    edge_cache_max_age_sec: float
    use_range_image_edges: str
    overlay_output_dir: str
    overlay_output_stride: int
    overlay_dot_radius: int


def load_gate_config(node: Any) -> GateConfig:
    projection_patch_size = get_int(node, '~projection_patch_size', 1, 'Odd patch size for robust LiDAR-to-image sampling (1=center pixel, 3=3x3, 5=5x5).', min_value=1)
    if projection_patch_size % 2 == 0:
        raise ValueError('~projection_patch_size must be an odd integer >= 1')
    projection_confidence_min = get_float(node, '~projection_confidence_min', 0.0, 'Minimum patch confidence required to trust transferred image color/label (0 disables).', min_value=0.0, max_value=1.0)
    adaptive_depth_edge_threshold_scale = get_float(node, '~adaptive_depth_edge_threshold_scale', 0.80, 'Scale applied to depth-edge threshold when adaptive projection health is enabled and health is bad; values below 1 reject more edge-near points.', min_value=0.000001)
    return GateConfig(
        projection_patch_size=projection_patch_size,
        projection_confidence_min=projection_confidence_min,
        projection_invalid_mask_topic=node._get_param_str('~projection_invalid_mask_topic', '', 'Optional single-channel invalid-mask image topic aligned with ~semantic_topic; pixels equal to ~projection_invalid_mask_value reject transferred labels/RGB.', allow_empty=True),
        projection_invalid_mask_value=get_int(node, '~projection_invalid_mask_value', 255, 'Pixel value in ~projection_invalid_mask_topic that marks invalid/rejected samples.', min_value=0, max_value=65535),
        projection_invalid_mask_dilate_px=get_int(node, '~projection_invalid_mask_dilate_px', 0, 'Optional dilation radius in pixels applied to the invalid mask before projection sampling.', min_value=0),
        projection_occlusion_epsilon_m=get_float(node, '~projection_occlusion_epsilon_m', 0.0, 'Allow image transfer only when the point depth is within this margin of the nearest LiDAR depth at that pixel (meters, 0 disables).', min_value=0.0),
        projection_occlusion_radius_px=get_int(node, '~projection_occlusion_radius_px', 0, 'Optional pixel radius used when evaluating the LiDAR depth support map for occlusion rejection.', min_value=0),
        projection_reject_depth_edges=node._get_param_bool('~projection_reject_depth_edges', False, 'Reject image transfer near strong LiDAR depth discontinuities.'),
        projection_depth_edge_thresh=get_float(node, '~projection_depth_edge_thresh', 0.15, 'Depth discontinuity threshold in meters/pixel for the depth-edge gate.', min_value=0.0),
        projection_depth_edge_radius_px=get_int(node, '~projection_depth_edge_radius_px', 0, 'Optional radius used to expand depth-edge rejection neighborhoods.', min_value=0),
        enable_adaptive_projection_health=node._get_param_bool('~enable_adaptive_projection_health', False, 'If true, make PRUNE projection gates more conservative when projection-health diagnostics are poor.'),
        projection_health_warn_threshold=get_float(node, '~projection_health_warn_threshold', 0.50, 'Projection-health score below this value is reported as warning quality.', min_value=0.0, max_value=1.0),
        projection_health_bad_threshold=get_float(node, '~projection_health_bad_threshold', 0.25, 'Projection-health score below this value is treated as bad for optional adaptive gates.', min_value=0.0, max_value=1.0),
        adaptive_confidence_threshold_offset=get_float(node, '~adaptive_confidence_threshold_offset', 0.10, 'Offset added to the confidence threshold when adaptive projection health is enabled and health is bad.', min_value=0.0, max_value=1.0),
        adaptive_depth_edge_threshold_scale=adaptive_depth_edge_threshold_scale,
        adaptive_prefer_suppression_on_bad_health=node._get_param_bool('~adaptive_prefer_suppression_on_bad_health', True, 'When adaptive projection health is enabled and health is bad, prefer suppressing labels over deleting geometry where the active PRUNE path supports suppression.'),
        downsample_factor=get_int(node, '~downsample_factor', 1, 'Uniform integer downsample applied to semantic/depth images before fusion.', min_value=1),
        depth_map_subsample=get_int(node, '~depth_map_subsample', 1, 'Depth buffer resolution divisor (1=full, 2=half, 4=quarter). Reduces rasterization and edge-map cost.', min_value=1),
        edge_cache_max_age_sec=get_float(node, '~edge_cache_max_age_sec', 0.0, 'Reuse the edge map for frames within this time window (seconds). 0 disables caching.', min_value=0.0),
        use_range_image_edges=node._get_param_str('~use_range_image_edges', 'auto', 'Edge-map strategy: "auto" uses range-image path for organized clouds, "always", or "never".'),
        overlay_output_dir=node._get_param_str('~overlay_output_dir', '', 'Directory for 3-layer GIMP overlays (base, accepted, depth). Empty = disabled.', allow_empty=True),
        overlay_output_stride=get_int(node, '~overlay_output_stride', 20, 'Save overlay every Nth accepted frame.', min_value=1),
        overlay_dot_radius=get_int(node, '~overlay_dot_radius', 2, 'Dot radius in pixels for overlay layers.', min_value=0),
    )
