"""Grouped ROS parameter loaders for the colored point-cloud node."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class SyncConfig:
    sync_slop_sec: float
    pair_max_dt_sec: float
    semantic_time_offset_sec: float
    sync_queue_size: int
    cloud_time_offset_sec: float
    cloud_stamp_source: str
    stamp_debug_log_period_sec: float


@dataclass
class ColorConfig:
    colorize_labels: bool
    color_map: dict
    random_color_seed: int
    num_labels: int
    semantic_color_quantization_step: int


@dataclass
class ProjectionConfig:
    projection_patch_size: int
    projection_confidence_min: float
    projection_invalid_mask_topic: str
    projection_invalid_mask_value: int
    projection_invalid_mask_dilate_px: int
    projection_occlusion_epsilon_m: float
    projection_occlusion_radius_px: int
    projection_reject_depth_edges: bool
    projection_depth_edge_thresh: float
    projection_depth_edge_radius_px: int
    downsample_factor: int


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


@dataclass
class PlyConfig:
    ply_output_dir: str
    ply_recording_enable: bool
    ply_target_frame: str
    ply_tf_use_latest: bool
    ply_tf_tolerance_sec: float


def load_sync_config(node: Any) -> SyncConfig:
    cloud_stamp_source = node._get_param_str(
        "~cloud_stamp_source",
        "",
        "Timestamp source for published PointCloud2: auto, semantic, depth, lidar, latest, earliest, midpoint.",
        allow_empty=True,
    )
    return SyncConfig(
        sync_slop_sec=_get_float(
            node,
            "~sync_slop_sec",
            0.1,
            "ApproximateTimeSynchronizer slop in seconds for semantic/depth or semantic/lidar pairing.",
            min_value=0.0,
        ),
        pair_max_dt_sec=_get_float(
            node,
            "~pair_max_dt_sec",
            0.03,
            "Hard max allowed |Δt| (seconds) between semantic and geometry; <=0 disables.",
            min_value=0.0,
        ),
        semantic_time_offset_sec=node._get_param_float(
            "~semantic_time_offset_sec",
            0.0,
            "Signed offset (seconds) applied to semantic timestamps for pairing and timestamp selection (negative shifts semantic earlier).",
        ),
        sync_queue_size=_get_int(
            node,
            "~sync_queue_size",
            5,
            "ApproximateTimeSynchronizer queue size for semantic/depth or semantic/lidar pairing.",
            min_value=1,
        ),
        cloud_time_offset_sec=node._get_param_float(
            "~cloud_time_offset_sec",
            0.0,
            "Signed offset (seconds) added to published cloud timestamps (negative shifts earlier).",
        ),
        cloud_stamp_source=(cloud_stamp_source or "").strip().lower(),
        stamp_debug_log_period_sec=_get_float(
            node,
            "~stamp_debug_log_period_sec",
            2.0,
            "Minimum period (seconds) between timestamp/offset debug logs; set 0 to log every callback when debug=true.",
            min_value=0.0,
        ),
    )


def load_color_config(node: Any) -> ColorConfig:
    return ColorConfig(
        colorize_labels=node._get_param_bool(
            "~colorize_labels",
            False,
            "If true, publish an extra PointCloud2 field 'rgb' (label palette in 'labels' mode; passthrough colors in 'rgb' mode).",
        ),
        color_map=node._get_color_map(
            "~color_map",
            "Optional dict {label_id: [r,g,b]} used to colorize labels when ~semantic_input_type='labels'. YAML keys must be quoted (e.g. \"0\": [0,0,0]).",
        ),
        random_color_seed=node._get_param_int(
            "~random_color_seed",
            1,
            "Seed for deterministic random label palette when ~colorize_labels is true and ~color_map is empty.",
        ),
        num_labels=node._get_param_int(
            "~num_labels",
            0,
            "Optional number of label IDs (0=auto from first label image). Used only when ~semantic_input_type='labels' and ~colorize_labels is true with empty ~color_map.",
        ),
        semantic_color_quantization_step=_get_int(
            node,
            "~semantic_color_quantization_step",
            1,
            "Quantize RGB/BGR semantic images to nearest multiple of this step before packing for the PointCloud2 rgb field (helps with JPEG artifacts).",
            min_value=1,
        ),
    )


def load_projection_config(node: Any) -> ProjectionConfig:
    projection_patch_size = _get_int(
        node,
        "~projection_patch_size",
        1,
        "Odd patch size for robust LiDAR-to-image sampling (1=center pixel, 3=3x3, 5=5x5).",
        min_value=1,
    )
    if projection_patch_size % 2 == 0:
        raise ValueError("~projection_patch_size must be an odd integer >= 1")
    projection_confidence_min = _get_float(
        node,
        "~projection_confidence_min",
        0.0,
        "Minimum patch confidence required to trust transferred image color/label (0 disables).",
        min_value=0.0,
        max_value=1.0,
    )
    return ProjectionConfig(
        projection_patch_size=projection_patch_size,
        projection_confidence_min=projection_confidence_min,
        projection_invalid_mask_topic=node._get_param_str(
            "~projection_invalid_mask_topic",
            "",
            "Optional single-channel invalid-mask image topic aligned with ~semantic_topic; pixels equal to ~projection_invalid_mask_value reject transferred labels/RGB.",
            allow_empty=True,
        ),
        projection_invalid_mask_value=_get_int(
            node,
            "~projection_invalid_mask_value",
            255,
            "Pixel value in ~projection_invalid_mask_topic that marks invalid/rejected samples.",
            min_value=0,
            max_value=65535,
        ),
        projection_invalid_mask_dilate_px=_get_int(
            node,
            "~projection_invalid_mask_dilate_px",
            0,
            "Optional dilation radius in pixels applied to the invalid mask before projection sampling.",
            min_value=0,
        ),
        projection_occlusion_epsilon_m=_get_float(
            node,
            "~projection_occlusion_epsilon_m",
            0.0,
            "Allow image transfer only when the point depth is within this margin of the nearest LiDAR depth at that pixel (meters, 0 disables).",
            min_value=0.0,
        ),
        projection_occlusion_radius_px=_get_int(
            node,
            "~projection_occlusion_radius_px",
            0,
            "Pixel radius for local min-depth occlusion gating (0 uses only the exact projected pixel).",
            min_value=0,
        ),
        projection_reject_depth_edges=node._get_param_bool(
            "~projection_reject_depth_edges",
            False,
            "If true, reject color/label transfer for projected points that land on strong LiDAR depth discontinuities.",
        ),
        projection_depth_edge_thresh=_get_float(
            node,
            "~projection_depth_edge_thresh",
            0.15,
            "Normalized depth-edge threshold used when ~projection_reject_depth_edges is enabled.",
            min_value=0.0,
            max_value=1.0,
        ),
        projection_depth_edge_radius_px=_get_int(
            node,
            "~projection_depth_edge_radius_px",
            0,
            "Pixel radius used to dilate the LiDAR depth-edge reject mask (helps suppress sky bleed near thin objects).",
            min_value=0,
        ),
        downsample_factor=_get_int(
            node,
            "~downsample_factor",
            1,
            "Integer >=1 stride used to subsample images for CPU/ARM targets.",
            min_value=1,
        ),
    )


def load_debug_config(node: Any) -> DebugConfig:
    debug_output_dir = node._get_param_str(
        "~debug_output_dir",
        "",
        "Directory where sampled debug overlays are written (empty uses <entfac_fusion_ros>/output/debug).",
        allow_empty=True,
    )
    if not debug_output_dir:
        debug_output_dir = _resolve_default_output_dir(
            node,
            param_name="~debug_output_dir",
            subdir="debug",
            fallback_subdir=Path.home() / ".ros" / "entfac_fusion_ros" / "debug",
        )
    Path(debug_output_dir).mkdir(parents=True, exist_ok=True)
    return DebugConfig(
        debug_project_lidar=node._get_param_bool(
            "~debug_project_lidar",
            False,
            "If true (lidar mode), publish a debug image with projected lidar points overlaid.",
        ),
        debug_project_lidar_stride=_get_int(
            node,
            "~debug_project_lidar_stride",
            5,
            "Subsample factor for projected LiDAR debug overlay (1 draws every projected point).",
            min_value=1,
        ),
        debug_project_lidar_radius=_get_int(
            node,
            "~debug_project_lidar_radius",
            0,
            "Marker radius in pixels for the projected LiDAR debug overlay (0 draws single pixels).",
            min_value=0,
        ),
        debug_project_lidar_outline_only=node._get_param_bool(
            "~debug_project_lidar_outline_only",
            False,
            "If true, draw projected LiDAR markers as outlines so the RGB image stays visible underneath.",
        ),
        debug_range_view=node._get_param_bool(
            "~debug_range_view",
            False,
            "If true (lidar mode), publish LiDAR depth/edge images, a reprojection heatmap, and an alignment score.",
        ),
        debug_output_dir=debug_output_dir,
        debug_output_stride=_get_int(
            node,
            "~debug_output_stride",
            20,
            "Save every Nth debug callback per stream (1 saves every frame).",
            min_value=1,
        ),
        debug_publish_fov_points=node._get_param_bool(
            "~debug_publish_fov_points",
            False,
            "If true (lidar mode), publish only the LiDAR points that passed the camera FOV test as a debug PointCloud2 in the LiDAR frame.",
        ),
    )


def load_ply_config(node: Any) -> PlyConfig:
    ply_output_dir = node._get_param_str(
        "~ply_output_dir",
        "",
        "Directory where PLY files are written (empty uses <entfac_fusion_ros>/output/ply).",
        allow_empty=True,
    )
    if not ply_output_dir:
        ply_output_dir = _resolve_default_output_dir(
            node,
            param_name="~ply_output_dir",
            subdir="ply",
            fallback_subdir=Path.home() / ".ros" / "entfac_fusion_ros" / "ply",
        )
    Path(ply_output_dir).mkdir(parents=True, exist_ok=True)
    return PlyConfig(
        ply_output_dir=ply_output_dir,
        ply_recording_enable=node._get_param_bool(
            "~ply_recording_enable",
            False,
            "If true, automatically enable PLY recording at startup (can also be toggled via ~set_ply_recording service).",
        ),
        ply_target_frame=node._get_param_str(
            "~ply_target_frame",
            "",
            "Optional TF frame to transform PLY output to (ply_target_frame <- target_frame). Empty means use target_frame.",
            allow_empty=True,
        ),
        ply_tf_use_latest=node._get_param_bool(
            "~ply_tf_use_latest",
            False,
            "When true, fall back to the latest TF for PLY export if exact-time lookup fails.",
        ),
        ply_tf_tolerance_sec=_get_float(
            node,
            "~ply_tf_tolerance_sec",
            0.02,
            "Max allowed time difference (seconds) when using latest TF for PLY export.",
            min_value=0.0,
        ),
    )


def _get_int(
    node: Any,
    name: str,
    default: int,
    description: str,
    *,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> int:
    value = node._get_param_int(name, default, description)
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be <= {max_value}")
    return value


def _get_float(
    node: Any,
    name: str,
    default: float,
    description: str,
    *,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> float:
    value = node._get_param_float(name, default, description)
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be <= {max_value}")
    return value


def _resolve_default_output_dir(
    node: Any,
    *,
    param_name: str,
    subdir: str,
    fallback_subdir: Path,
) -> str:
    try:
        import rospkg  # lazy import

        pkg_path = rospkg.RosPack().get_path("entfac_fusion_ros")
        output_dir = str(Path(pkg_path) / "output" / subdir)
        node._param_meta[param_name]["value"] = output_dir
        return output_dir
    except Exception as exc:  # noqa: BLE001
        output_dir = str(fallback_subdir)
        node._param_meta[param_name]["value"] = output_dir
        node._log.warn(
            "__init__",
            "Unable to resolve package path for default %s (%s); using %s",
            param_name,
            exc,
            output_dir,
        )
        return output_dir
