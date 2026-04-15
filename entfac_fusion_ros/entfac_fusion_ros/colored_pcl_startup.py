"""Startup-report helpers for the ENTFAC colored point-cloud ROS node."""

from __future__ import annotations

from typing import Optional

import numpy as np
import rospy

from entfac_fusion_ros.status import render_kv_table
from entfac_fusion_ros.tf_utils import format_matrix


def log_startup_transforms(node) -> None:
    """Emit a compact info-level report of transforms active at startup."""
    if node.mode == "depth":
        src = "static_target_T_depth" if node.static_target_T_depth is not None else "tf2"
        label = f"{node._depth_frame or '<depth_frame>'} -> {node.target_frame}"
        if node.target_T_depth is None:
            node._log.info(
                "_log_startup_transforms",
                "startup transform target_T_depth [%s] (%s): pending",
                label,
                src,
            )
        else:
            node._log.info(
                "_log_startup_transforms",
                "startup transform target_T_depth [%s] (%s):\n%s",
                label,
                src,
                format_matrix(node.target_T_depth),
            )
        return

    src_cam = "static_camera_T_lidar" if node.static_camera_T_lidar is not None else "tf2"
    src_tgt = "static_target_T_lidar" if node.static_target_T_lidar is not None else "tf2"
    lidar_label = node._lidar_frame or "<lidar_frame>"
    cam_label = f"{lidar_label} -> {node.camera_frame}"
    tgt_label = f"{lidar_label} -> {node.target_frame}"

    if node.camera_T_lidar is None:
        node._log.info(
            "_log_startup_transforms",
            "startup transform camera_T_lidar [%s] (%s): pending",
            cam_label,
            src_cam,
        )
    else:
        node._log.info(
            "_log_startup_transforms",
            "startup transform camera_T_lidar [%s] (%s):\n%s",
            cam_label,
            src_cam,
            format_matrix(node.camera_T_lidar),
        )

    if node.target_T_lidar is None:
        node._log.info(
            "_log_startup_transforms",
            "startup transform target_T_lidar [%s] (%s): pending",
            tgt_label,
            src_tgt,
        )
    else:
        node._log.info(
            "_log_startup_transforms",
            "startup transform target_T_lidar [%s] (%s):\n%s",
            tgt_label,
            src_tgt,
            format_matrix(node.target_T_lidar),
        )


def log_param_report(node) -> None:
    """Emit the verbose parameter report used in debug mode."""
    node._log.info("_log_param_report", "colored_pcl_node debug report:")
    node._log.info(
        "_log_param_report",
        "  source=cli means passed as _param:=...; source=param_server means set via YAML/launch; source=default means unset.",
    )
    for name in sorted(node._param_meta.keys()):
        meta = node._param_meta[name]
        val = meta["value"]
        val_str = format_matrix(val) if isinstance(val, np.ndarray) else repr(val)
        node._log.info(
            "_log_param_report",
            "  %s=%s (%s) - %s",
            name,
            val_str,
            meta["source"],
            meta["description"],
        )
    node._log.info("_log_param_report", "derived mode=%s", node.mode)
    node._log.info("_log_param_report", "derived camera_frame=%s", node.camera_frame)
    node._log.info(
        "_log_param_report", "derived intrinsics=%s", format_matrix(node.intrinsics)
    )
    node._log.info(
        "_log_param_report",
        "active subscriptions: semantic=%s depth_input=%s confidence=%s",
        node.semantic_topic,
        node.depth_input_topic,
        node.conf_topic,
    )
    node._log.info(
        "_log_param_report",
        "primed transforms: target_T_depth=%s camera_T_lidar=%s target_T_lidar=%s",
        node.target_T_depth is not None,
        node.camera_T_lidar is not None,
        node.target_T_lidar is not None,
    )
    if node.target_T_depth is not None:
        node._log.info(
            "_log_param_report",
            "target_T_depth=\n%s",
            format_matrix(node.target_T_depth),
        )
    if node.camera_T_lidar is not None:
        node._log.info(
            "_log_param_report",
            "camera_T_lidar=\n%s",
            format_matrix(node.camera_T_lidar),
        )
    if node.target_T_lidar is not None:
        node._log.info(
            "_log_param_report",
            "target_T_lidar=\n%s",
            format_matrix(node.target_T_lidar),
        )


def log_correction_statuses(node) -> None:
    """Emit the startup summary of correction/compatibility modes."""
    node._log.info(
        "_log_correction_statuses",
        "correction status: undistort=%s rolling_shutter=%s lidar_deskew=%s lidar_points_compat=%s online_calibration=%s",
        node._undistort_status,
        node._rolling_shutter_status,
        node._lidar_deskew_status,
        node._compat_lidar_points_status,
        node._online_calibration_status,
    )


def render_startup_table(node) -> str:
    """Render the startup summary table shown after subscriber registration."""
    services = (
        f"save_ply={rospy.resolve_name('~save_ply')} "
        f"set_ply_recording={rospy.resolve_name('~set_ply_recording')}"
    )
    ply_target = node.ply_target_frame or "-"
    ply = (
        f"recording={node._ply_recording} dir={node.ply_output_dir} "
        f"target_frame={ply_target} use_latest={node.ply_tf_use_latest} "
        f"tol={node.ply_tf_tolerance_sec:.6f}"
    )
    help_text = "\n".join(
        [
            f"rosservice call {rospy.resolve_name('~save_ply')} \"{{}}\"",
            f"rosservice call {rospy.resolve_name('~set_ply_recording')} \"data: true\"",
        ]
    )

    rows = [
        ("node", node._node_name),
        ("mode", f"{node.mode} ({node._mode_source})"),
        ("mode_detail", node._mode_detail or "-"),
        ("target_frame", node.target_frame),
        ("camera_frame", node.camera_frame),
        ("output", node._output_topic),
        ("semantic_topic", node.semantic_topic),
        ("semantic_type", node.semantic_input_type),
        ("confidence_topic", node.conf_topic or "-"),
        ("camera_info_source", node._camera_info_source or "-"),
        ("camera_info", node.camera_info_topic or "-"),
        ("camera_info_txt", node.camera_info_txt or "-"),
        ("depth_input_topic", node.depth_input_topic or "-"),
        ("downsample", str(int(node.downsample_factor))),
        ("projection_patch_size", str(int(node.projection_patch_size))),
        ("projection_confidence_min", f"{node.projection_confidence_min:.3f}"),
        (
            "projection_occlusion_epsilon_m",
            f"{node.projection_occlusion_epsilon_m:.3f}",
        ),
        (
            "projection_occlusion_radius_px",
            str(int(node.projection_occlusion_radius_px)),
        ),
        (
            "projection_reject_depth_edges",
            str(bool(node.projection_reject_depth_edges)),
        ),
        (
            "projection_depth_edge_thresh",
            f"{node.projection_depth_edge_thresh:.3f}",
        ),
        (
            "projection_depth_edge_radius_px",
            str(int(node.projection_depth_edge_radius_px)),
        ),
        ("sync_slop_sec", f"{node.sync_slop_sec:.6f}"),
        ("pair_max_dt_sec", f"{node.pair_max_dt_sec:.6f}"),
        ("semantic_time_offset_sec", f"{node.semantic_time_offset_sec:.6f}"),
        ("stamp_debug_log_period_sec", f"{node.stamp_debug_log_period_sec:.3f}"),
        ("sync_queue_size", str(int(node.sync_queue_size))),
        ("undistort_semantic", str(bool(node.undistort_semantic))),
        ("undistort_status", node._undistort_status),
        ("undistort_alpha", f"{node.undistort_alpha:.2f}"),
        ("debug_project_lidar", str(bool(node.debug_project_lidar))),
        ("debug_project_lidar_stride", str(int(node.debug_project_lidar_stride))),
        ("debug_project_lidar_radius", str(int(node.debug_project_lidar_radius))),
        (
            "debug_project_lidar_outline_only",
            str(bool(node.debug_project_lidar_outline_only)),
        ),
        ("debug_range_view", str(bool(node.debug_range_view))),
        ("debug_publish_fov_points", str(bool(node.debug_publish_fov_points))),
        ("online_calibration_enable", str(bool(node.online_calibration_enable))),
        ("online_calibration_status", node._online_calibration_status),
        (
            "online_calibration_every_n_frames",
            str(int(node.online_calibration_every_n_frames)),
        ),
        (
            "online_calibration_max_points",
            str(int(node.online_calibration_max_points)),
        ),
        (
            "online_calibration_step_deg",
            f"{node.online_calibration_step_deg:.3f}",
        ),
        (
            "online_calibration_learning_rate",
            f"{node.online_calibration_learning_rate:.3f}",
        ),
        (
            "online_calibration_max_correction_deg",
            f"{node.online_calibration_max_correction_deg:.3f}",
        ),
        (
            "online_calibration_min_observability",
            f"{node.online_calibration_min_observability:.3f}",
        ),
        (
            "online_calibration_rpy_deg",
            "[%.3f %.3f %.3f]"
            % tuple(np.degrees(node._online_calibration_rpy_rad).tolist()),
        ),
        ("rolling_shutter_enable", str(bool(node.rolling_shutter_enable))),
        ("rolling_shutter_status", node._rolling_shutter_status),
        ("rolling_shutter_readout_sec", f"{node.rolling_shutter_readout_sec:.6f}"),
        ("rolling_shutter_direction", node.rolling_shutter_direction),
        ("imu_topic", node.imu_topic or "-"),
        ("imu_frame", node.imu_frame or "-"),
        ("camera_metadata_topic", node.camera_metadata_topic or "-"),
        ("metadata_readout_key", str(int(node.metadata_readout_key))),
        ("metadata_readout_scale", f"{node.metadata_readout_scale:.6g}"),
        ("metadata_max_dt_sec", f"{node.metadata_max_dt_sec:.3f}"),
        ("lidar_deskew_enable", str(bool(node.lidar_deskew_enable))),
        ("lidar_deskew_status", node._lidar_deskew_status),
        ("lidar_deskew_mode", node.lidar_deskew_mode),
        ("lidar_deskew_ref", node.lidar_deskew_ref),
        ("lidar_time_field", node.lidar_time_field),
        ("lidar_time_scale", f"{node.lidar_time_scale:.3g}"),
        ("lidar_imu_topic", node.lidar_imu_topic or "-"),
        ("lidar_imu_frame", node.lidar_imu_frame or "-"),
        ("lidar_imu_cache_size", str(int(node.lidar_imu_cache_size))),
        ("lidar_imu_cache_max_dt_sec", f"{node.lidar_imu_cache_max_dt_sec:.3f}"),
        (
            "lidar_imu_accel_gravity_compensated",
            str(bool(node.lidar_imu_accel_is_gravity_compensated)),
        ),
        ("compat_ouster_sensor_frame", str(bool(node.compat_ouster_sensor_frame))),
        ("compat_lidar_points_status", node._compat_lidar_points_status),
        (
            "max_depth_m",
            f"{node.max_depth_m:.3f}" if node.max_depth_m is not None else "-",
        ),
        ("cloud_time_offset_sec", f"{node.cloud_time_offset_sec:.6f}"),
        ("cloud_stamp_source", node.cloud_stamp_source),
        ("colorize_labels", str(bool(node.colorize_labels))),
        ("ply", ply),
        ("services", services),
        ("help", help_text),
    ]

    def _fmt_tf(
        name: str,
        mat: Optional[np.ndarray],
        src: str,
        src_frame: str,
        dst_frame: str,
    ):
        label = f"{src_frame or '<frame>'} -> {dst_frame}"
        if mat is None:
            return (name, f"{label} ({src}: pending)")
        return (name, f"{label} ({src})\n{format_matrix(mat)}")

    if node.mode == "depth":
        src = "static_target_T_depth" if node.static_target_T_depth is not None else "tf2"
        rows.append(
            _fmt_tf(
                "target_T_depth",
                node.target_T_depth,
                src,
                node._depth_frame,
                node.target_frame,
            )
        )
    else:
        src_cam = "static_camera_T_lidar" if node.static_camera_T_lidar is not None else "tf2"
        src_tgt = "static_target_T_lidar" if node.static_target_T_lidar is not None else "tf2"
        rows.append(
            _fmt_tf(
                "camera_T_lidar",
                node.camera_T_lidar,
                src_cam,
                node._lidar_frame,
                node.camera_frame,
            )
        )
        rows.append(
            _fmt_tf(
                "target_T_lidar",
                node.target_T_lidar,
                src_tgt,
                node._lidar_frame,
                node.target_frame,
            )
        )

    return render_kv_table(rows)
