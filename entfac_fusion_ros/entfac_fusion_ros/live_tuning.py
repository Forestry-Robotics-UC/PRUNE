#!/usr/bin/env python3
"""Live Parameter Tuning for colored_pcl_node.

Provides unified interface for both dynamic_reconfigure and runtime parameter updates.
Consolidates all parameter validation and application logic to prevent divergence
between the two update paths.
"""

from typing import Callable, Any


# Tuning parameter definition: (attr_name, type_hint, validator_fn)
# Used by both dynamic_reconfigure callback and live parameter refresh
TUNING_PARAMS = [
    ("projection_patch_size", int, lambda v: v >= 1 and (v % 2) == 1),
    ("projection_confidence_min", float, lambda v: 0.0 <= v <= 1.0),
    ("projection_occlusion_epsilon_m", float, lambda v: v >= 0.0),
    ("projection_occlusion_radius_px", int, lambda v: v >= 0),
    ("projection_reject_depth_edges", bool, lambda v: isinstance(v, bool)),
    ("projection_depth_edge_thresh", float, lambda v: 0.0 <= v <= 1.0),
    ("projection_depth_edge_radius_px", int, lambda v: v >= 0),
    ("debug_project_lidar", bool, lambda v: isinstance(v, bool)),
    ("debug_project_lidar_stride", int, lambda v: v >= 1),
    ("debug_project_lidar_radius", int, lambda v: v >= 0),
    ("debug_project_lidar_outline_only", bool, lambda v: isinstance(v, bool)),
    ("tracked_reprojection_fb_thresh_px", float, lambda v: v > 0.0),
    ("tracked_reprojection_depth_edge_thresh", float, lambda v: 0.0 <= v <= 1.0),
    ("tracked_reprojection_min_image_edge", float, lambda v: 0.0 <= v <= 1.0),
    ("tracked_reprojection_min_tracks", int, lambda v: v >= 10),
]


def apply_tuning_params(
    node_instance: Any,
    get_value: Callable[[str, Any], Any],
    log_fn: Callable[[str], None] = None,
) -> bool:
    """Apply tuning parameters to node instance.

    Args:
        node_instance: The colored_pcl_node instance to update
        get_value: Callable(attr_name, default) that returns validated value or raises
        log_fn: Optional logging callback for change notifications

    Returns:
        True if any parameters changed, False otherwise
    """
    changes = []

    for attr_name, type_hint, validator in TUNING_PARAMS:
        try:
            default = getattr(node_instance, attr_name)
            value = get_value(attr_name, default)
            if not validator(value):
                continue
        except Exception:
            continue

        current = getattr(node_instance, attr_name)
        if current != value:
            setattr(node_instance, attr_name, value)
            changes.append(f"{attr_name}={value}")

    if changes and log_fn:
        log_fn("Live tuning update: " + ", ".join(changes))

    return bool(changes)
