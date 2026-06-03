"""Startup and status reporting helpers for colored PCL."""

from __future__ import annotations

from typing import Any

import rospy

from entfac_fusion_ros.colored_pcl_startup import (
    log_correction_statuses as _log_correction_statuses_helper,
    log_param_report as _log_param_report_helper,
    log_startup_transforms as _log_startup_transforms_helper,
    render_startup_table as _render_startup_table_helper,
)


class StartupReporter:
    def __init__(self, node: Any):
        self._node = node

    def log_startup_transforms(self) -> None:
        _log_startup_transforms_helper(self._node)

    def log_param_report(self) -> None:
        _log_param_report_helper(self._node)

    def log_correction_statuses(self) -> None:
        _log_correction_statuses_helper(self._node)

    def render_startup_table(self) -> str:
        return _render_startup_table_helper(
            self._node,
            rospy.resolve_name("~save_ply"),
            rospy.resolve_name("~set_ply_recording"),
        )

    def emit_status(self, *, points: int, callback_sec: float) -> None:
        self._node._diagnostics.emit_status(
            points=int(points),
            callback_sec=float(callback_sec),
        )
