"""Live tuning and dynamic reconfigure support for prune node."""

from __future__ import annotations

import time
from typing import Any, Callable

import rospy
from sensor_msgs.msg import Image

from prune_ros.config import coerce_bool as _coerce_bool


# Shared tuning parameter definition: (attr_name, type_hint, validator_fn)
TUNING_PARAMS = [
    ("projection_patch_size", int, lambda v: v >= 1 and (v % 2) == 1),
    ("projection_confidence_min", float, lambda v: 0.0 <= v <= 1.0),
    ("projection_occlusion_epsilon_m", float, lambda v: v >= 0.0),
    ("projection_occlusion_radius_px", int, lambda v: v >= 0),
    ("projection_reject_depth_edges", bool, lambda v: isinstance(v, bool)),
    ("projection_depth_edge_thresh", float, lambda v: 0.0 <= v <= 1.0),
    ("projection_depth_edge_radius_px", int, lambda v: v >= 0),
    ("use_invalid_mask", bool, lambda v: isinstance(v, bool)),
    ("projection_invalid_mask_dilate_px", int, lambda v: v >= 0),
    ("use_depth_edge_rejection", bool, lambda v: isinstance(v, bool)),
    ("use_occlusion_gate", bool, lambda v: isinstance(v, bool)),
    ("use_geometric_gate", bool, lambda v: isinstance(v, bool)),
    ("projection_geometric_enable", bool, lambda v: isinstance(v, bool)),
    ("geometric_curvature_max", float, lambda v: 0.0 <= v <= 1.0),
    ("geometric_score_min", float, lambda v: 0.0 <= v <= 1.0),
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
    log_fn: Callable[[str], None] | None = None,
) -> bool:
    """Apply validated live-tuning values onto *node_instance*."""
    changes = []

    for attr_name, _type_hint, validator in TUNING_PARAMS:
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

    if changes and log_fn is not None:
        log_fn("Live tuning update: " + ", ".join(changes))

    return bool(changes)


class LiveTuningController:
    def __init__(self, node: Any, logger: Any, reconfigure_server_cls: Any, reconfigure_config_cls: Any):
        self._node = node
        self._log = logger
        self._server_cls = reconfigure_server_cls
        self._config_cls = reconfigure_config_cls

    def setup_dynamic_reconfigure(self) -> None:
        if self._server_cls is None or self._config_cls is None:
            missing = []
            if self._server_cls is None:
                missing.append("dynamic_reconfigure.server")
            if self._config_cls is None:
                missing.append("prune_ros.cfg.PruneTuningConfig")
            self._log.warn(
                "_setup_dynamic_reconfigure",
                "rqt_reconfigure support is unavailable because %s could not be imported. Build the catkin workspace so the generated dynamic_reconfigure modules exist.",
                ", ".join(missing),
            )
            return

        self._node._dynamic_reconfigure_server = self._server_cls(
            self._config_cls, self.dynamic_reconfigure_callback
        )
        self._log.info(
            "_setup_dynamic_reconfigure",
            "rqt_reconfigure is ready on %s for live projection/debug tuning.",
            rospy.get_name(),
        )

    def apply_tuning_params(self, get_value, log_source: str = "") -> bool:
        changes_logged = []

        def log_fn(message: str) -> None:
            changes_logged.append(message)

        tracked_attrs = [
            name for name, _, _ in TUNING_PARAMS if name.startswith("tracked_reprojection_")
        ]
        tracked_before = {
            name: getattr(self._node, name, None) for name in tracked_attrs
        }

        changed = apply_tuning_params(self._node, get_value, log_fn if log_source else None)

        if self._node.debug_project_lidar and self._node._debug_proj_pub is None:
            self._node._debug_proj_pub = rospy.Publisher(
                self._node.debug_projected_topic, Image, queue_size=1
            )

        if changed:
            self._node._projector.update_params(self._node._runtime_builders.build_projector_params(self._node))
            # The tracker snapshots its params at build time, so live changes
            # to tracked_reprojection_* only take effect through a rebuild.
            # Rebuilding resets track state, so do it only when one of those
            # params actually changed and the tracker is active.
            if self._node._tracked_repr is not None and any(
                getattr(self._node, name, None) != tracked_before[name]
                for name in tracked_attrs
            ):
                self._node._tracked_repr = self._node._tracked_runtime.build()
            if self._node._debug_pub is not None:
                self._node._debug_pub.update_params(self._node._runtime_builders.build_debug_pub_params(self._node))
            if log_source and changes_logged:
                self._log.info(log_source, "%s", changes_logged[0])
        return changed

    def dynamic_reconfigure_callback(self, config, _level):
        initialized = self._node._dynamic_reconfigure_initialized
        patch_size = int(config["projection_patch_size"])
        if patch_size < 1:
            patch_size = 1
        if (patch_size % 2) == 0:
            patch_size = patch_size + 1 if patch_size < 9 else patch_size - 1
        config["projection_patch_size"] = patch_size

        def get_from_config(attr: str, default):
            if attr in config:
                return config[attr]
            raise KeyError(attr)

        self.apply_tuning_params(
            get_from_config,
            "_dynamic_reconfigure_callback" if initialized else "",
        )
        self._node._dynamic_reconfigure_initialized = True
        return config

    def maybe_refresh_params(self) -> None:
        now = time.time()
        if (
            self._node._live_param_last_refresh_at > 0.0
            and (now - self._node._live_param_last_refresh_at) < self._node._live_param_refresh_period_sec
        ):
            return
        self._node._live_param_last_refresh_at = now

        def get_from_rospy(attr: str, default):
            if attr == "projection_patch_size":
                return int(rospy.get_param(f"~{attr}", default))
            if attr in {
                "projection_confidence_min",
                "projection_occlusion_epsilon_m",
                "projection_depth_edge_thresh",
                "geometric_curvature_max",
                "geometric_score_min",
                "tracked_reprojection_fb_thresh_px",
                "tracked_reprojection_depth_edge_thresh",
                "tracked_reprojection_min_image_edge",
            }:
                return float(rospy.get_param(f"~{attr}", default))
            if attr in {
                "projection_occlusion_radius_px",
                "projection_depth_edge_radius_px",
                "projection_invalid_mask_dilate_px",
                "debug_project_lidar_stride",
                "debug_project_lidar_radius",
                "tracked_reprojection_min_tracks",
            }:
                return int(rospy.get_param(f"~{attr}", default))
            return _coerce_bool(rospy.get_param(f"~{attr}", default))

        self.apply_tuning_params(get_from_rospy, "_maybe_refresh_live_tuning_params")
