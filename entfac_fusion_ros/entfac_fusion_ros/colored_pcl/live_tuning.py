"""Live tuning and dynamic reconfigure support for colored PCL."""

from __future__ import annotations

import time
from typing import Any

import rospy
from sensor_msgs.msg import Image

from entfac_fusion_ros.colored_pcl_params import coerce_bool as _coerce_bool


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
                missing.append("entfac_fusion_ros.cfg.ColoredPclTuningConfig")
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
        changes = []

        def update(attr: str, default, validator) -> None:
            try:
                value = get_value(attr, default)
                if not validator(value):
                    return
            except Exception:  # noqa: BLE001
                return
            current = getattr(self._node, attr)
            if current != value:
                setattr(self._node, attr, value)
                changes.append(f"{attr}={value}")

        update("projection_patch_size", self._node.projection_patch_size, lambda v: v >= 1 and (v % 2) == 1)
        update("projection_confidence_min", self._node.projection_confidence_min, lambda v: 0.0 <= v <= 1.0)
        update("projection_occlusion_epsilon_m", self._node.projection_occlusion_epsilon_m, lambda v: v >= 0.0)
        update("projection_occlusion_radius_px", self._node.projection_occlusion_radius_px, lambda v: v >= 0)
        update("projection_reject_depth_edges", self._node.projection_reject_depth_edges, lambda v: isinstance(v, bool))
        update("projection_depth_edge_thresh", self._node.projection_depth_edge_thresh, lambda v: 0.0 <= v <= 1.0)
        update("projection_depth_edge_radius_px", self._node.projection_depth_edge_radius_px, lambda v: v >= 0)
        update("debug_project_lidar", self._node.debug_project_lidar, lambda v: isinstance(v, bool))
        update("debug_project_lidar_stride", self._node.debug_project_lidar_stride, lambda v: v >= 1)
        update("debug_project_lidar_radius", self._node.debug_project_lidar_radius, lambda v: v >= 0)
        update("debug_project_lidar_outline_only", self._node.debug_project_lidar_outline_only, lambda v: isinstance(v, bool))
        update("tracked_reprojection_fb_thresh_px", self._node.tracked_reprojection_fb_thresh_px, lambda v: v > 0.0)
        update("tracked_reprojection_depth_edge_thresh", self._node.tracked_reprojection_depth_edge_thresh, lambda v: 0.0 <= v <= 1.0)
        update("tracked_reprojection_min_image_edge", self._node.tracked_reprojection_min_image_edge, lambda v: 0.0 <= v <= 1.0)
        update("tracked_reprojection_min_tracks", self._node.tracked_reprojection_min_tracks, lambda v: v >= 10)

        if self._node.debug_project_lidar and self._node._debug_proj_pub is None:
            self._node._debug_proj_pub = rospy.Publisher(
                self._node.debug_projected_topic, Image, queue_size=1
            )

        if changes:
            self._node._projector.update_params(self._node._build_projector_params())
            if self._node._debug_pub is not None:
                self._node._debug_pub.update_params(self._node._build_debug_pub_params())
            if log_source:
                self._log.info(log_source, "Live tuning update: %s", ", ".join(changes))
        return bool(changes)

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
                "tracked_reprojection_fb_thresh_px",
                "tracked_reprojection_depth_edge_thresh",
                "tracked_reprojection_min_image_edge",
            }:
                return float(rospy.get_param(f"~{attr}", default))
            if attr in {
                "projection_occlusion_radius_px",
                "projection_depth_edge_radius_px",
                "debug_project_lidar_stride",
                "debug_project_lidar_radius",
                "tracked_reprojection_min_tracks",
            }:
                return int(rospy.get_param(f"~{attr}", default))
            return _coerce_bool(rospy.get_param(f"~{attr}", default))

        self.apply_tuning_params(get_from_rospy, "_maybe_refresh_live_tuning_params")
