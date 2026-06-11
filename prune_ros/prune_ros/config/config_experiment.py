"""Experiment and calibration parameter loaders for prune."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config_helpers import get_float, get_int


@dataclass
class ExperimentConfig:
    use_invalid_mask: bool
    use_depth_edge_rejection: bool
    use_occlusion_gate: bool
    use_geometric_gate: bool
    experiment_variant_name: str
    experiment_bag_name: str
    results_dir: str
    enable_metrics_csv: bool


@dataclass
class CalibrationConfig:
    tracked_reprojection_enable: bool
    tracked_reprojection_max_corners: int
    tracked_reprojection_quality_level: float
    tracked_reprojection_min_distance_px: float
    tracked_reprojection_min_tracks: int
    tracked_reprojection_fb_thresh_px: float
    tracked_reprojection_depth_edge_thresh: float
    tracked_reprojection_min_image_edge: float
    tracked_reprojection_log_period_sec: float


def load_experiment_config(node: Any) -> ExperimentConfig:
    return ExperimentConfig(
        use_invalid_mask=node._get_param_bool('~use_invalid_mask', True, 'Enable G1 invalid-mask evidence gate in experiment metrics and projector runtime.'),
        use_depth_edge_rejection=node._get_param_bool('~use_depth_edge_rejection', True, 'Enable G2 depth-edge evidence gate in experiment metrics and projector runtime.'),
        use_occlusion_gate=node._get_param_bool('~use_occlusion_gate', True, 'Enable G3 occlusion evidence gate in experiment metrics and projector runtime.'),
        use_geometric_gate=node._get_param_bool('~use_geometric_gate', True, 'Enable G5 geometric-reliability evidence gate in experiment metrics and projector runtime (active only when ~projection_geometric_enable is true).'),
        experiment_variant_name=node._get_param_str('~experiment_variant_name', '', 'Optional experiment label recorded in metrics CSV outputs.', allow_empty=True),
        experiment_bag_name=node._get_param_str('~bag_name', '', 'Optional bag identifier recorded in metrics CSV outputs.', allow_empty=True),
        results_dir=node._get_param_str('~results_dir', '', 'Optional output directory for metrics/debug result bundles.', allow_empty=True),
        enable_metrics_csv=node._get_param_bool('~enable_metrics_csv', False, 'Write per-frame metrics CSV for LiDAR experiments.'),
    )


def load_calibration_config(node: Any) -> CalibrationConfig:
    return CalibrationConfig(
        tracked_reprojection_enable=node._get_param_bool('~tracked_reprojection_enable', False, 'Enable tracked reprojection diagnostics.'),
        tracked_reprojection_max_corners=get_int(node, '~tracked_reprojection_max_corners', 200, 'Maximum number of tracked reprojection corners.', min_value=1),
        tracked_reprojection_quality_level=get_float(node, '~tracked_reprojection_quality_level', 0.01, 'Feature quality threshold for tracked reprojection.', min_value=0.0),
        tracked_reprojection_min_distance_px=get_float(node, '~tracked_reprojection_min_distance_px', 7.0, 'Minimum feature spacing for tracked reprojection.', min_value=0.0),
        tracked_reprojection_min_tracks=get_int(node, '~tracked_reprojection_min_tracks', 20, 'Minimum active tracks before the tracked reprojection diagnostic resets.', min_value=10),
        tracked_reprojection_fb_thresh_px=get_float(node, '~tracked_reprojection_fb_thresh_px', 1.0, 'Forward-backward pixel error threshold for tracked reprojection.', min_value=0.0),
        tracked_reprojection_depth_edge_thresh=get_float(node, '~tracked_reprojection_depth_edge_thresh', 0.15, 'Depth-edge threshold reused by tracked reprojection diagnostics.', min_value=0.0),
        tracked_reprojection_min_image_edge=get_float(node, '~tracked_reprojection_min_image_edge', 0.05, 'Minimum normalized image-edge strength in [0,1] required for a tracked feature to contribute to the reprojection error metric.', min_value=0.0, max_value=1.0),
        tracked_reprojection_log_period_sec=get_float(node, '~tracked_reprojection_log_period_sec', 2.0, 'Minimum seconds between tracked reprojection log messages.', min_value=0.0),
    )
