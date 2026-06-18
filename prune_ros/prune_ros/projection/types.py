#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: GNU General Public License v3.0 (GPL-3.0)
# Repository: PRUNE

"""Domain dataclasses for LiDAR projection: inputs, outputs, and parameters.

No ROS imports; these are plain data carriers shared between
:mod:`lidar_projector` and :mod:`quality_gates`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from prune_core.types import SemanticPointCloud

# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ProjectionQualityResult:
    """Per-point quality gate results for one frame."""

    keep: np.ndarray
    confidence_reject: np.ndarray
    depth_edge_reject: np.ndarray
    occlusion_reject: np.ndarray
    depth_edge_map: Optional[np.ndarray]
    runtime_rasterize_ms: float = 0.0
    runtime_depth_edge_ms: float = 0.0
    runtime_occlusion_ms: float = 0.0
    projection_health_score: float = 0.0
    geometric_reject: Optional[np.ndarray] = None
    runtime_geometric_ms: float = 0.0
    # Per-point enrichment computed by the geometric gate; available for
    # future Tier 2 (per-point reliability state) export to ENTFAC-Mapping.
    surface_normals: Optional[np.ndarray] = None
    geometric_reliability: Optional[np.ndarray] = None


@dataclass
class GateMetrics:
    """Counters and runtimes for one projection frame."""

    num_points_in_front: int = 0
    num_points_projected_in_image: int = 0
    num_rejected_invalid_mask: int = 0
    num_rejected_confidence: int = 0
    num_rejected_depth_edge: int = 0
    num_rejected_occlusion: int = 0
    num_rejected_geometric: int = 0
    num_rejected_other: int = 0
    num_would_hit_invalid_mask: int = 0
    num_would_hit_depth_edge: int = 0
    num_would_fail_occlusion: int = 0
    num_would_hit_geometric: int = 0
    runtime_projection_ms: float = 0.0
    runtime_mask_ms: float = 0.0
    runtime_rasterize_ms: float = 0.0
    runtime_depth_edge_ms: float = 0.0
    runtime_occlusion_ms: float = 0.0
    runtime_geometric_ms: float = 0.0
    runtime_publish_ms: float = 0.0
    projection_health_score: float = 0.0


@dataclass
class ProjectionResult:
    """All outputs of a single :meth:`LidarProjector.process_frame` call."""

    cloud: SemanticPointCloud
    metrics: GateMetrics
    image_shape: Tuple[int, int]
    # FOV-gated points in their original frames (for calibration and debug)
    points_fov: np.ndarray          # LiDAR frame, shape (M, 3)
    points_cam_all: np.ndarray      # camera frame, shape (M, 3)
    # Optional outputs populated when corresponding gates are enabled
    depth_map: Optional[np.ndarray] = None
    edge_map: Optional[np.ndarray] = None
    # Optional outputs for debug publishers
    rgb_values: Optional[np.ndarray] = None
    debug_colors: Optional[np.ndarray] = None
    rolling_shutter_active: bool = False
    # Gate-status overlay colors and matching pixel coordinates, both aligned
    # to the `inside` (in-image) subset; populated only when
    # debug_project_lidar is enabled. Consumers must not re-align these
    # against points_fov.
    gate_debug_colors: Optional[np.ndarray] = None
    uv_inside: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Parameter dataclass
# ---------------------------------------------------------------------------


@dataclass
class LidarProjectorParams:
    """All parameters consumed by :class:`LidarProjector`.

    Build from node parameters; pass to the constructor or :meth:`update_params`.
    """

    # FOV gate
    max_depth_m: Optional[float] = None
    camera_fov_gate_enable: bool = True
    camera_fov_gate_margin_deg: float = 5.0

    # Projection
    rolling_shutter_enable: bool = False
    rolling_shutter_direction: str = "top_to_bottom"
    projection_patch_size: int = 1

    # Quality gates
    projection_occlusion_epsilon_m: float = 0.0
    projection_occlusion_radius_px: int = 0
    projection_reject_depth_edges: bool = False
    projection_depth_edge_thresh: float = 0.15
    projection_depth_edge_radius_px: int = 0
    projection_confidence_min: float = 0.0
    use_invalid_mask: bool = True
    use_depth_edge_rejection: bool = True
    use_occlusion_gate: bool = True
    use_geometric_gate: bool = True

    # Geometric reliability gate (GLIM-inspired local surface cues; default off)
    projection_geometric_enable: bool = False
    geometric_k_neighbors: int = 12
    geometric_radius_m: float = 0.5
    geometric_min_neighbors: int = 5
    geometric_curvature_max: float = 0.12
    geometric_up_labels: Tuple[int, ...] = ()
    geometric_up_max_angle_deg: float = 60.0
    geometric_score_min: float = 0.0
    geometric_fold_into_confidence: bool = False
    enable_adaptive_projection_health: bool = False
    projection_health_warn_threshold: float = 0.50
    projection_health_bad_threshold: float = 0.25
    adaptive_confidence_threshold_offset: float = 0.10
    adaptive_depth_edge_threshold_scale: float = 0.80
    adaptive_prefer_suppression_on_bad_health: bool = True

    # RT optimisations
    depth_map_subsample: int = 1       # 1=full-res, 2=half-res, 4=quarter-res depth buffer
    edge_cache_max_age_sec: float = 0.0  # 0=recompute every frame; >0=reuse edge map within window
    use_range_image_edges: str = "auto"  # "auto"=use when cloud is organized, "always", "never"

    # Overlay export (results_overlays.py format — compatible with make_gimp_layers.py)
    overlay_output_dir: str = ""       # empty = disabled
    overlay_output_stride: int = 20    # save every Nth accepted frame
    overlay_dot_radius: int = 2

    # Semantics / output
    include_unlabeled: bool = False
    colorize_labels: bool = False
    semantic_input_type: str = "labels"
    color_map: Dict = field(default_factory=dict)
    random_color_seed: int = 1
    num_labels: int = 0
    debug_project_lidar: bool = False
