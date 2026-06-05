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
    online_calibration_enable: bool
    online_calibration_every_n_frames: int
    online_calibration_max_points: int
    online_calibration_edge_threshold: float
    online_calibration_step_deg: float
    online_calibration_learning_rate: float
    online_calibration_max_correction_deg: float
    online_calibration_min_observability: float
    online_calibration_min_fov_points: int
    online_calibration_min_sem_edge_density: float
    online_calibration_min_depth_edge_density: float
    online_calibration_health_ema_alpha: float
    online_calibration_health_std_window: int
    online_calibration_health_std_scale: float
    online_calibration_health_score_center: float
    online_calibration_health_score_scale: float
    online_calibration_log_period_sec: float


def load_experiment_config(node: Any) -> ExperimentConfig:
    return ExperimentConfig(
        use_invalid_mask=node._get_param_bool('~use_invalid_mask', True, 'Enable G1 invalid-mask evidence gate in experiment metrics and projector runtime.'),
        use_depth_edge_rejection=node._get_param_bool('~use_depth_edge_rejection', True, 'Enable G2 depth-edge evidence gate in experiment metrics and projector runtime.'),
        use_occlusion_gate=node._get_param_bool('~use_occlusion_gate', True, 'Enable G3 occlusion evidence gate in experiment metrics and projector runtime.'),
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
        tracked_reprojection_min_image_edge=get_float(node, '~tracked_reprojection_min_image_edge', 3.0, 'Minimum image-edge margin for tracked reprojection features.', min_value=0.0),
        tracked_reprojection_log_period_sec=get_float(node, '~tracked_reprojection_log_period_sec', 2.0, 'Minimum seconds between tracked reprojection log messages.', min_value=0.0),
        online_calibration_enable=node._get_param_bool('~online_calibration_enable', False, 'Enable online LiDAR-camera calibration updates.'),
        online_calibration_every_n_frames=get_int(node, '~online_calibration_every_n_frames', 5, 'Evaluate online calibration every N accepted LiDAR frames.', min_value=1),
        online_calibration_max_points=get_int(node, '~online_calibration_max_points', 2000, 'Maximum LiDAR points used per online calibration update.', min_value=1),
        online_calibration_edge_threshold=get_float(node, '~online_calibration_edge_threshold', 0.15, 'Edge threshold used by online calibration alignment scoring.', min_value=0.0),
        online_calibration_step_deg=get_float(node, '~online_calibration_step_deg', 0.1, 'Small-angle search step for online calibration updates (degrees).', min_value=0.0),
        online_calibration_learning_rate=get_float(node, '~online_calibration_learning_rate', 0.5, 'Online calibration correction learning rate.', min_value=0.0),
        online_calibration_max_correction_deg=get_float(node, '~online_calibration_max_correction_deg', 3.0, 'Maximum absolute online calibration correction per axis (degrees).', min_value=0.0),
        online_calibration_min_observability=get_float(node, '~online_calibration_min_observability', 0.1, 'Minimum observability score required to update online calibration.', min_value=0.0),
        online_calibration_min_fov_points=get_int(node, '~online_calibration_min_fov_points', 200, 'Minimum in-FoV LiDAR points required for online calibration.', min_value=1),
        online_calibration_min_sem_edge_density=get_float(node, '~online_calibration_min_sem_edge_density', 0.002, 'Minimum semantic edge density required for online calibration.', min_value=0.0),
        online_calibration_min_depth_edge_density=get_float(node, '~online_calibration_min_depth_edge_density', 0.002, 'Minimum depth-edge density required for online calibration.', min_value=0.0),
        online_calibration_health_ema_alpha=get_float(node, '~online_calibration_health_ema_alpha', 0.1, 'EMA alpha for online calibration health smoothing.', min_value=0.0, max_value=1.0),
        online_calibration_health_std_window=get_int(node, '~online_calibration_health_std_window', 30, 'Window size for online calibration health variability.', min_value=2),
        online_calibration_health_std_scale=get_float(node, '~online_calibration_health_std_scale', 10.0, 'Scale factor for online calibration health variability score.', min_value=0.0),
        online_calibration_health_score_center=get_float(node, '~online_calibration_health_score_center', 0.5, 'Score center for online calibration health normalization.'),
        online_calibration_health_score_scale=get_float(node, '~online_calibration_health_score_scale', 10.0, 'Score scale for online calibration health normalization.', min_value=0.0),
        online_calibration_log_period_sec=get_float(node, '~online_calibration_log_period_sec', 2.0, 'Minimum seconds between online calibration log messages.', min_value=0.0),
    )
