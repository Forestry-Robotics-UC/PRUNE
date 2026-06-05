"""Projection and gating helpers for PRUNE."""

from importlib import import_module

_EXPORTS = {
    "query_neighborhood_reduce": (".gate_utils", "query_neighborhood_reduce"),
    "reduce_image_neighborhood": (".gate_utils", "reduce_image_neighborhood"),
    "GateMetrics": (".lidar_projector", "GateMetrics"),
    "LidarProjector": (".lidar_projector", "LidarProjector"),
    "LidarProjectorParams": (".lidar_projector", "LidarProjectorParams"),
    "has_overlay_projection_samples": (".results_overlays", "has_overlay_projection_samples"),
    "draw_projection_points": (".results_overlays", "draw_projection_points"),
    "blend_mask": (".results_overlays", "blend_mask"),
    "save_frame_overlays": (".results_overlays", "save_frame_overlays"),
    "write_selected_frames_manifest": (".results_overlays", "write_selected_frames_manifest"),
    "TrackedReprojection": (".tracked_reprojection", "TrackedReprojection"),
    "TrackedReprojectionParams": (".tracked_reprojection", "TrackedReprojectionParams"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + __all__)
