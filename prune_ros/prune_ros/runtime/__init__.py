"""Runtime ROS helpers for PRUNE."""

from importlib import import_module

_EXPORTS = {
    "image_to_numpy": (".conversions", "image_to_numpy"),
    "pointcloud2_to_xyz": (".conversions", "pointcloud2_to_xyz"),
    "pointcloud2_to_xyz_t": (".conversions", "pointcloud2_to_xyz_t"),
    "rgb_to_packed_u32": (".conversions", "rgb_to_packed_u32"),
    "DebugPublisher": (".debug_publisher", "DebugPublisher"),
    "DebugPublisherParams": (".debug_publisher", "DebugPublisherParams"),
    "interpolate_imu_msg": (".imu_cache", "interpolate_imu_msg"),
    "LiveTuningController": (".live_tuning", "LiveTuningController"),
    "apply_tuning_params": (".live_tuning", "apply_tuning_params"),
    "TUNING_PARAMS": (".live_tuning", "TUNING_PARAMS"),
    "NodeLogger": (".logging_ros", "NodeLogger"),
    "configure_core_logging": (".logging_ros", "configure_core_logging"),
    "labels_to_uint16": (".pc2", "labels_to_uint16"),
    "build_label_rgb_float_lut": (".pc2", "build_label_rgb_float_lut"),
    "semantic_pointcloud_to_msg": (".pc2", "semantic_pointcloud_to_msg"),
    "PlyJob": (".ply", "PlyJob"),
    "PlyWriterThread": (".ply", "PlyWriterThread"),
    "quaternion_multiply": (".pose_to_tf_math", "quaternion_multiply"),
    "normalize_quaternion": (".pose_to_tf_math", "normalize_quaternion"),
    "apply_yaw_offset_deg": (".pose_to_tf_math", "apply_yaw_offset_deg"),
    "StatusReporter": (".status", "StatusReporter"),
    "render_kv_table": (".status", "render_kv_table"),
    "render_status_table": (".status", "render_status_table"),
    "transform_stamped_to_matrix": (".tf_utils", "transform_stamped_to_matrix"),
    "format_matrix": (".tf_utils", "format_matrix"),
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
