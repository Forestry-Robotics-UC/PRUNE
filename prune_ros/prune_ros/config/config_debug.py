"""Debug-related prune parameter loaders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config_helpers import get_int, resolve_default_output_dir


@dataclass
class DebugConfig:
    debug_project_lidar: bool
    debug_project_lidar_stride: int
    debug_project_lidar_radius: int
    debug_project_lidar_outline_only: bool
    debug_range_view: bool
    debug_output_dir: str
    debug_output_stride: int
    debug_publish_fov_points: bool


def load_debug_config(node: Any) -> DebugConfig:
    output_dir_default = resolve_default_output_dir(node)
    config = DebugConfig(
        debug_project_lidar=node._get_param_bool('~debug_project_lidar', False, 'Publish LiDAR projection overlay debug images.'),
        debug_project_lidar_stride=get_int(node, '~debug_project_lidar_stride', 1, 'Subsample stride used when rendering LiDAR projection overlays.', min_value=1),
        debug_project_lidar_radius=get_int(node, '~debug_project_lidar_radius', 1, 'Point radius in pixels for LiDAR projection overlay rendering.', min_value=0),
        debug_project_lidar_outline_only=node._get_param_bool('~debug_project_lidar_outline_only', False, 'Render LiDAR projection overlay as outlines only.'),
        debug_range_view=node._get_param_bool('~debug_range_view', False, 'Publish depth/edge/range-view debug images.'),
        debug_output_dir=node._get_param_str('~debug_output_dir', output_dir_default, 'Directory for saved debug images and sidecar outputs.'),
        debug_output_stride=get_int(node, '~debug_output_stride', 1, 'Save every Nth debug frame to disk.', min_value=1),
        debug_publish_fov_points=node._get_param_bool('~debug_publish_fov_points', False, 'Publish LiDAR points that survive the camera FoV gate.'),
    )
    Path(config.debug_output_dir).mkdir(parents=True, exist_ok=True)
    return config
