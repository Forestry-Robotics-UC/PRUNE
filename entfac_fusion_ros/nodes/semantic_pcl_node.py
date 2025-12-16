#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Derived from Semantic SLAM
#
# Original Author:
#   Xuan Zhang
#
# Subsequent Contributions:
#   David Russell
#
# Modified by:
#   Duda Andrada (ENTFAC Sensor Fusion)
#
# Original project:
#   https://github.com/floatlazer/semantic_slam
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: GNU General Public License v3.0 (GPL-3.0)
# Repository: ENTFAC-Sensor-Fusion
#
# Description:
#   ROS wrapper that turns semantic + geometry inputs into semantic PointCloud2 outputs.

"""ROS wrapper that converts semantic + geometry into semantic PointCloud2."""

import cProfile
import io
import logging
import pstats
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import rospy
import tf2_ros
from message_filters import ApproximateTimeSynchronizer, Subscriber
from scipy.spatial.transform import Rotation as SciRotation
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField

# Ensure core package is importable when running from a monorepo source tree.
# In a proper catkin workspace, this is handled by PYTHONPATH via devel/setup.bash.
_THIS = Path(__file__).resolve()
for parent in _THIS.parents:
    cand = parent / "entfac_fusion_core" / "src"
    if (cand / "entfac_fusion_core").is_dir() and str(cand) not in sys.path:
        sys.path.insert(0, str(cand))
        break

from entfac_fusion_core.semantic_pcl import (
    fuse_depth_semantics,
    fuse_lidar_semantics,
)
from entfac_fusion_core.types import (
    DepthObservation,
    PointObservation,
    SemanticObservation,
)
from entfac_fusion_core.utils.validation import require_homogeneous_transform
from entfac_fusion_core.utils.semantics import (
    count_semantic_groups,
    dominant_packed_colors,
    packed_rgb_to_triplets,
)


class _RospyLogHandler(logging.Handler):
    """Forward Python logging records into rospy logging with node prefix."""

    def __init__(self, node_name):
        super().__init__()
        self._node_name = str(node_name).lstrip("/")
        self._local = threading.local()

    def emit(self, record):
        if getattr(self._local, "in_emit", False):
            return
        self._local.in_emit = True
        try:
            msg = self.format(record)
            tag = f"{self._node_name}:{record.name}.{record.funcName}"
            full = f"[{tag}] {msg}"
            if record.levelno >= logging.ERROR:
                rospy.logerr(full)
            elif record.levelno >= logging.WARNING:
                rospy.logwarn(full)
            elif record.levelno >= logging.INFO:
                rospy.loginfo(full)
            else:
                rospy.logdebug(full)
        finally:
            self._local.in_emit = False


def _configure_core_logging(node_name, debug):
    """Route entfac_fusion_core logs through rospy with node context."""
    logger = logging.getLogger("entfac_fusion_core")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    handler = _RospyLogHandler(node_name)
    handler.setLevel(logging.DEBUG if debug else logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))

    # Avoid duplicate handlers when a node gets restarted.
    for existing in list(logger.handlers):
        if isinstance(existing, _RospyLogHandler):
            logger.removeHandler(existing)
    logger.addHandler(handler)
    logger.propagate = False


def transform_stamped_to_matrix(transform_stamped):
    """Convert geometry_msgs/TransformStamped to 4x4 numpy matrix."""
    t = transform_stamped.transform.translation
    q = transform_stamped.transform.rotation
    translation = np.array([t.x, t.y, t.z], dtype=float)
    quat = np.array([q.x, q.y, q.z, q.w], dtype=float)
    rot_obj = SciRotation.from_quat(quat)
    # SciPy < 1.4 uses as_dcm(); newer versions use as_matrix().
    rot = rot_obj.as_matrix() if hasattr(rot_obj, "as_matrix") else rot_obj.as_dcm()
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = rot
    mat[:3, 3] = translation
    return require_homogeneous_transform(mat.astype(float))


def _build_label_rgb_float_lut(color_map=None):
    """Build a uint16-label -> packed RGB float32 lookup table.

    The packed RGB float encoding is the common ROS/PCL convention where a 24-bit
    RGB integer is reinterpreted as float32.
    """
    labels = np.arange(65536, dtype=np.uint32)
    r = (labels * 37) & 0xFF
    g = (labels * 17) & 0xFF
    b = (labels * 73) & 0xFF
    packed = (r << 16) | (g << 8) | b
    packed[65535] = 0xFFFFFF
    if color_map:
        for label_id, rgb in color_map.items():
            if not (0 <= int(label_id) <= 65535):
                continue
            if not isinstance(rgb, (list, tuple)) or len(rgb) != 3:
                continue
            rr, gg, bb = int(rgb[0]), int(rgb[1]), int(rgb[2])
            packed[int(label_id)] = ((rr & 0xFF) << 16) | ((gg & 0xFF) << 8) | (
                bb & 0xFF
            )
    packed_le = packed.astype("<u4", copy=False)
    return packed_le.view("<f4")


def _labels_to_uint16(labels):
    labels_arr = np.asarray(labels)
    if labels_arr.ndim != 1:
        labels_arr = labels_arr.reshape(-1)
    if labels_arr.dtype.kind not in ("i", "u"):
        raise ValueError("labels must be an integer array")
    if np.any(labels_arr > 65535):
        raise ValueError("label must fit into uint16 (0..65535)")
    if labels_arr.dtype.kind == "u":
        return labels_arr.astype(np.uint16, copy=False)
    labels_u16 = labels_arr.astype(np.uint16, copy=True)
    neg_mask = labels_arr < 0
    if np.any(neg_mask):
        labels_u16[neg_mask] = 65535
    return labels_u16


def semantic_pointcloud_to_msg(
    pcl, frame_id, stamp, colorize_labels=False, rgb_lut=None
):
    """Convert SemanticPointCloud dataclass to PointCloud2."""
    has_conf = pcl.confidence is not None
    has_rgb = bool(colorize_labels)
    num_points = int(pcl.points_xyz.shape[0])

    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
        PointField("label", 12, PointField.UINT16, 1),
    ]
    point_step = 16
    if has_conf:
        fields.append(PointField("confidence", point_step, PointField.FLOAT32, 1))
        point_step += 4
    if has_rgb:
        fields.append(PointField("rgb", point_step, PointField.FLOAT32, 1))
        point_step += 4

    dtype_fields = [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("label", "<u2"),
        ("_pad", "<u2"),
    ]
    if has_conf:
        dtype_fields.append(("confidence", "<f4"))
    if has_rgb:
        dtype_fields.append(("rgb", "<f4"))
    dtype = np.dtype(dtype_fields)
    if dtype.itemsize != point_step:
        raise RuntimeError(
            f"internal dtype mismatch: itemsize={dtype.itemsize} point_step={point_step}"
        )

    cloud = np.empty(num_points, dtype=dtype)
    points = np.asarray(pcl.points_xyz)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must be (N, 3)")
    cloud["x"] = points[:, 0]
    cloud["y"] = points[:, 1]
    cloud["z"] = points[:, 2]
    cloud["_pad"] = 0

    labels_u16 = _labels_to_uint16(pcl.labels)
    if labels_u16.shape[0] != num_points:
        raise ValueError("labels must be (N,) and aligned with points_xyz")
    cloud["label"] = labels_u16

    if has_conf:
        conf = np.asarray(pcl.confidence, dtype=np.float32)
        if conf.shape[0] != num_points:
            raise ValueError("confidence must be (N,) and aligned with points_xyz")
        cloud["confidence"] = conf

    if has_rgb:
        if rgb_lut is None:
            rgb_lut = _build_label_rgb_float_lut()
        cloud["rgb"] = rgb_lut[labels_u16]

    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.height = 1
    msg.width = num_points
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = point_step
    msg.row_step = point_step * num_points
    msg.is_dense = True
    msg.data = cloud.tobytes()
    return msg


def _coerce_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "on")
    return bool(val)


def _rosargv_bool(name, default=False):
    prefix = f"_{name}:="
    for arg in sys.argv:
        if arg.startswith(prefix):
            return _coerce_bool(arg[len(prefix) :])
    return default


def _rosargv_has_private_param(name):
    if not isinstance(name, str):
        return False
    key = name
    if key.startswith("~"):
        key = key[1:]
    if "/" in key:
        key = key.rsplit("/", 1)[-1]
    prefix = f"_{key}:="
    return any(arg.startswith(prefix) for arg in sys.argv)


class SemanticPclNode:
    """ROS node bridging topics to the numpy fusion core."""

    def __init__(self):
        self._param_meta = {}
        self._node_name = rospy.get_name().lstrip("/")

        self.debug = self._get_param_bool(
            "~debug",
            False,
            "Enable debug parameter report at startup (and DEBUG logs if set via launch arg).",
        )
        _configure_core_logging(self._node_name, self.debug)

        # 'depth' uses aligned depth image; 'lidar' projects LiDAR points into the image.
        self.mode = self._get_param_str(
            "~mode",
            "",
            "Force fusion mode ('depth' or 'lidar'); empty string enables auto-detect.",
            allow_empty=True,
        ).lower()

        self.target_frame = self._get_param_str(
            "~target_frame",
            "base_link",
            "Output frame for published semantic point cloud.",
        )
        self.semantic_topic = self._get_param_str(
            "~semantic_topic",
            "/semantic/labels",
            "Semantic label image topic (sensor_msgs/Image).",
        )
        self.conf_topic = self._get_param_str(
            "~confidence_topic",
            None,
            "Optional confidence image topic aligned with semantic labels (sensor_msgs/Image).",
        )
        self.camera_info_topic = self._get_param_str(
            "~camera_info",
            None,
            "CameraInfo topic providing intrinsics and camera frame_id (sensor_msgs/CameraInfo).",
        )
        if not self.camera_info_topic:
            raise ValueError("~camera_info is required")

        self.depth_input_topic = self._get_param_str(
            "~depth_input_topic",
            None,
            "Geometry input topic (sensor_msgs/Image depth or sensor_msgs/PointCloud2 LiDAR). If set, overrides ~depth_topic/~lidar_topic.",
        )
        self.depth_topic = self._get_param_str(
            "~depth_topic",
            None,
            "DEPRECATED: use ~depth_input_topic. Depth image topic for depth mode (sensor_msgs/Image). Leave empty to disable.",
        )
        self.lidar_topic = self._get_param_str(
            "~lidar_topic",
            None,
            "DEPRECATED: use ~depth_input_topic. PointCloud2 topic for LiDAR mode (sensor_msgs/PointCloud2). Leave empty to disable.",
        )
        if self.depth_input_topic and self.mode in ("depth", "lidar"):
            if self.depth_topic or self.lidar_topic:
                self._logwarn(
                    "__init__",
                    "~depth_input_topic is set; ignoring deprecated ~depth_topic/~lidar_topic",
                )
            if self.mode == "depth":
                self.depth_topic = self.depth_input_topic
                self.lidar_topic = None
            else:
                self.lidar_topic = self.depth_input_topic
                self.depth_topic = None

        self.include_unlabeled = self._get_param_bool(
            "~include_unlabeled_pts",
            False,
            "If true, keep points outside the camera FOV (label=-1).",
        )
        self.colorize_labels = self._get_param_bool(
            "~colorize_labels",
            False,
            "If true, publish an extra PointCloud2 field 'rgb' based on label IDs.",
        )
        self.color_map = self._get_color_map(
            "~color_map",
            "Optional dict {label_id: [r,g,b]} used for coloring and for color->label decode when semantic images are RGB.",
        )
        self._color_map_revision = 1 if self.color_map else 0
        self._rgb_lut_revision = -1
        self._rgb_float_lut = None
        self.auto_color_to_label = self._get_param_bool(
            "~auto_color_to_label",
            False,
            "If true and semantic images are RGB/BGR, build a deterministic color->label mapping from observed colors.",
        )
        self.auto_color_to_label_min_fraction = self._get_param_float(
            "~auto_color_to_label_min_fraction",
            0.0005,
            "Auto palette filter: keep colors covering at least this fraction of pixels (helps with JPEG artifacts). Set 0 to disable.",
        )
        if not (0.0 <= self.auto_color_to_label_min_fraction <= 1.0):
            raise ValueError("~auto_color_to_label_min_fraction must be in [0, 1]")
        self.auto_color_to_label_min_count = self._get_param_int(
            "~auto_color_to_label_min_count",
            1,
            "Auto palette filter: keep colors with at least this many pixels (>=1).",
        )
        if self.auto_color_to_label_min_count < 1:
            raise ValueError("~auto_color_to_label_min_count must be >= 1")
        self.auto_color_to_label_max_colors = self._get_param_int(
            "~auto_color_to_label_max_colors",
            64,
            "Auto palette cap: if more colors are detected, keep only the most frequent ones (>=1).",
        )
        if self.auto_color_to_label_max_colors < 1:
            raise ValueError("~auto_color_to_label_max_colors must be >= 1")
        self.semantic_color_quantization_step = self._get_param_int(
            "~semantic_color_quantization_step",
            1,
            "Quantize RGB/BGR semantic images to nearest multiple of this step before color->label decoding (helps with JPEG artifacts).",
        )
        if self.semantic_color_quantization_step < 1:
            raise ValueError("~semantic_color_quantization_step must be >= 1")
        self.auto_color_to_label_merge_distance = self._get_param_int(
            "~auto_color_to_label_merge_distance",
            0,
            "Auto palette merge: merge similar colors within this RGB distance (after quantization) to reduce JPEG palette noise. 0 disables.",
        )
        if self.auto_color_to_label_merge_distance < 0:
            raise ValueError("~auto_color_to_label_merge_distance must be >= 0")
        self.auto_color_to_label_extend = self._get_param_bool(
            "~auto_color_to_label_extend",
            True,
            "If true (auto decode only), extend the color->label mapping when new colors appear in later frames.",
        )
        self._inverse_color_map = None
        self._packed_to_label = None
        self._auto_color_decode = False
        self._auto_palette_packed = None
        self._auto_palette_rgb = None
        self._warned_unknown_colors = False
        self._logged_depth_scaling = False
        self._logged_depth_summary = False
        self._logged_lidar_summary = False
        self.downsample_factor = self._get_param_int(
            "~downsample_factor",
            1,
            "Integer >=1 stride used to subsample images for CPU/ARM targets.",
        )
        if self.downsample_factor < 1:
            raise ValueError("~downsample_factor must be >= 1")
        self.enable_profiling = self._get_param_bool(
            "~enable_profiling",
            False,
            "If true, print a short cProfile summary per callback (future C++/numba profiling hook).",
        )
        self.depth_scale = self._get_param_float(
            "~depth_scale",
            0.0,
            "Scale factor to convert depth values to meters (0=auto: 16UC1/mono16 treated as mm -> 0.001; 32FC1 treated as meters -> 1.0).",
        )
        if self.depth_scale < 0.0:
            raise ValueError("~depth_scale must be >= 0")
        self.static_target_T_depth = self._get_matrix_param(
            "~static_target_T_depth",
            "Optional static 4x4 row-major matrix: depth_frame -> target_frame. Overrides TF.",
        )
        self.static_camera_T_lidar = self._get_matrix_param(
            "~static_camera_T_lidar",
            "Optional static 4x4 row-major matrix: lidar_frame -> camera_frame. Overrides TF.",
        )
        self.static_target_T_lidar = self._get_matrix_param(
            "~static_target_T_lidar",
            "Optional static 4x4 row-major matrix: lidar_frame -> target_frame. Overrides TF.",
        )
        if self.static_target_T_depth is not None:
            self._loginfo("__init__", "Using static target_T_depth parameter")
        if self.static_camera_T_lidar is not None:
            self._loginfo("__init__", "Using static camera_T_lidar parameter")
        if self.static_target_T_lidar is not None:
            self._loginfo("__init__", "Using static target_T_lidar parameter")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self._logdebug(
            "__init__", "Waiting for CameraInfo on topic=%s", self.camera_info_topic
        )
        cam_info = None
        while cam_info is None and not rospy.is_shutdown():
            cam_info = self._wait_for_msg(
                self.camera_info_topic,
                CameraInfo,
                timeout=1.0,
                warn_on_timeout=False,
            )
        if cam_info is None:
            raise rospy.ROSInterruptException("Shutdown while waiting for CameraInfo")
        self.intrinsics = np.asarray(cam_info.K, dtype=float).reshape(3, 3)
        self.camera_frame = cam_info.header.frame_id
        self._logdebug(
            "__init__",
            "CameraInfo received: frame_id=%s stamp=%.6f",
            self.camera_frame,
            cam_info.header.stamp.to_sec(),
        )
        self._logdebug(
            "__init__",
            "Intrinsics K=\n%s",
            np.array2string(self.intrinsics, precision=4, suppress_small=True),
        )
        self.pcl_pub = rospy.Publisher(
            "semantic_pointcloud", PointCloud2, queue_size=1
        )
        self._loginfo(
            "__init__",
            "Publishing PointCloud2 on topic=%s",
            rospy.resolve_name("semantic_pointcloud"),
        )
        self.target_T_depth = None
        self.camera_T_lidar = None
        self.target_T_lidar = None
        if self.mode not in ("depth", "lidar"):
            self.mode = self._detect_mode()
            self._loginfo("__init__", "Auto-detected mode=%s", self.mode)
        self._prime_transforms()

        self._register_subscribers()
        self._loginfo("__init__", "semantic_pcl_node initialized (mode=%s)", self.mode)
        if self.debug:
            self._log_param_report()

    def _logdebug(self, method, msg, *args):
        tag = f"{self._node_name}.{method}" if method else self._node_name
        rospy.logdebug(f"[{tag}] {str(msg)}", *args)

    def _loginfo(self, method, msg, *args):
        tag = f"{self._node_name}.{method}" if method else self._node_name
        rospy.loginfo(f"[{tag}] {str(msg)}", *args)

    def _logwarn(self, method, msg, *args):
        tag = f"{self._node_name}.{method}" if method else self._node_name
        rospy.logwarn(f"[{tag}] {str(msg)}", *args)

    def _logerr(self, method, msg, *args):
        tag = f"{self._node_name}.{method}" if method else self._node_name
        rospy.logerr(f"[{tag}] {str(msg)}", *args)

    def _record_param(self, name, value, source, description):
        self._param_meta[name] = {
            "value": value,
            "source": source,
            "description": description,
        }

    def _get_param(self, name, default, description, *, allow_empty=False):
        has = rospy.has_param(name)
        raw = rospy.get_param(name, default)
        if _rosargv_has_private_param(name):
            source = "cli"
        else:
            source = "param_server" if has else "default"
        if isinstance(raw, str) and not allow_empty and raw.strip() == "":
            raw = default
            if has:
                source = "empty->default"
        self._record_param(name, raw, source, description)
        return raw

    def _get_param_str(self, name, default, description, *, allow_empty=False):
        raw = self._get_param(name, default, description, allow_empty=allow_empty)
        if raw is None:
            val = default
        elif isinstance(raw, str):
            val = raw
        else:
            val = str(raw)
        self._param_meta[name]["value"] = val
        return val

    def _get_param_bool(self, name, default, description):
        raw = self._get_param(name, default, description)
        val = _coerce_bool(raw)
        self._param_meta[name]["value"] = val
        return val

    def _get_param_int(self, name, default, description):
        raw = self._get_param(name, default, description)
        val = int(raw)
        self._param_meta[name]["value"] = val
        return val

    def _get_param_float(self, name, default, description):
        raw = self._get_param(name, default, description)
        val = float(raw)
        self._param_meta[name]["value"] = val
        return val

    def _get_matrix_param(self, name, description):
        has = rospy.has_param(name)
        raw = rospy.get_param(name, [])
        source = "param_server" if has else "default"
        mat = None
        if isinstance(raw, list) and len(raw) == 16:
            try:
                mat = require_homogeneous_transform(
                    np.asarray(raw, dtype=float).reshape(4, 4)
                )
            except ValueError as exc:
                self._logwarn("_get_matrix_param", "%s rejected: %s", name, exc)
        elif raw not in (None, [], {}):
            self._logwarn(
                "_get_matrix_param",
                "%s expected 16-element list (row-major 4x4), got: %r",
                name,
                raw,
            )
        self._record_param(name, mat, source, description)
        return mat

    def _get_color_map(self, name, description):
        has = rospy.has_param(name)
        raw = rospy.get_param(name, {})
        source = "param_server" if has else "default"
        color_map = None
        if isinstance(raw, dict):
            parsed = {}
            for k, v in raw.items():
                try:
                    key_int = int(k)
                    if isinstance(v, (list, tuple)) and len(v) == 3:
                        parsed[key_int] = [int(v[0]), int(v[1]), int(v[2])]
                except Exception:  # noqa: BLE001
                    continue
            color_map = parsed if parsed else None
        self._record_param(name, color_map, source, description)
        return color_map

    def _get_rgb_float_lut(self):
        if not self.colorize_labels:
            return None
        if (
            self._rgb_float_lut is None
            or self._rgb_lut_revision != self._color_map_revision
        ):
            self._rgb_float_lut = _build_label_rgb_float_lut(self.color_map)
            self._rgb_lut_revision = self._color_map_revision
            self._logdebug(
                "_get_rgb_float_lut",
                "Built label->rgb LUT (color_map_entries=%d)",
                len(self.color_map) if self.color_map else 0,
            )
        return self._rgb_float_lut

    def _log_param_report(self):
        self._loginfo("_log_param_report", "semantic_pcl_node debug report:")
        self._loginfo(
            "_log_param_report",
            "  source=cli means passed as _param:=...; source=param_server means set via YAML/launch; source=default means unset."
        )
        for name in sorted(self._param_meta.keys()):
            meta = self._param_meta[name]
            val = meta["value"]
            if isinstance(val, np.ndarray):
                val_str = np.array2string(val, precision=4, suppress_small=True)
            else:
                val_str = repr(val)
            self._loginfo(
                "_log_param_report",
                "  %s=%s (%s) - %s",
                name,
                val_str,
                meta["source"],
                meta["description"],
            )
        self._loginfo("_log_param_report", "derived mode=%s", self.mode)
        self._loginfo(
            "_log_param_report", "derived camera_frame=%s", self.camera_frame
        )
        self._loginfo(
            "_log_param_report",
            "derived intrinsics=%s",
            np.array2string(self.intrinsics, precision=4, suppress_small=True),
        )
        self._loginfo(
            "_log_param_report",
            "active subscriptions: semantic=%s depth=%s lidar=%s confidence=%s",
            self.semantic_topic,
            self.depth_topic,
            self.lidar_topic,
            self.conf_topic,
        )
        self._loginfo(
            "_log_param_report",
            "primed transforms: target_T_depth=%s camera_T_lidar=%s target_T_lidar=%s",
            self.target_T_depth is not None,
            self.camera_T_lidar is not None,
            self.target_T_lidar is not None,
        )
        if self.target_T_depth is not None:
            self._loginfo(
                "_log_param_report",
                "target_T_depth=\n%s",
                np.array2string(
                    self.target_T_depth, precision=4, suppress_small=True
                ),
            )
        if self.camera_T_lidar is not None:
            self._loginfo(
                "_log_param_report",
                "camera_T_lidar=\n%s",
                np.array2string(
                    self.camera_T_lidar, precision=4, suppress_small=True
                ),
            )
        if self.target_T_lidar is not None:
            self._loginfo(
                "_log_param_report",
                "target_T_lidar=\n%s",
                np.array2string(
                    self.target_T_lidar, precision=4, suppress_small=True
                ),
            )

        color_map_len = len(self.color_map) if self.color_map else 0
        self._loginfo(
            "_log_param_report",
            "semantic palette decode: color_map_entries=%d auto_color_to_label=%s extend=%s",
            color_map_len,
            self.auto_color_to_label,
            self.auto_color_to_label_extend,
        )

    def _register_subscribers(self):
        """Setup message_filters synchronizer."""
        self._logdebug(
            "_register_subscribers",
            "Registering subscribers (mode=%s): semantic=%s depth=%s lidar=%s confidence=%s",
            self.mode,
            self.semantic_topic,
            self.depth_topic,
            self.lidar_topic,
            self.conf_topic,
        )
        color_sub = Subscriber(self.semantic_topic, Image, queue_size=1)

        if self.mode == "depth":
            if not self.depth_topic:
                raise ValueError("depth mode requires ~depth_topic")
            depth_sub = Subscriber(self.depth_topic, Image, queue_size=1)
            subs = [color_sub, depth_sub]
            if self.conf_topic:
                conf_sub = Subscriber(self.conf_topic, Image, queue_size=1)
                subs.append(conf_sub)
            self.ts = ApproximateTimeSynchronizer(
                subs, queue_size=5, slop=0.05, allow_headerless=True
            )
            self.ts.registerCallback(self._depth_callback)
        else:
            if not self.lidar_topic:
                raise ValueError("lidar mode requires ~lidar_topic")
            lidar_sub = Subscriber(self.lidar_topic, PointCloud2, queue_size=1)
            subs = [color_sub, lidar_sub]
            if self.conf_topic:
                conf_sub = Subscriber(self.conf_topic, Image, queue_size=1)
                subs.append(conf_sub)
            self.ts = ApproximateTimeSynchronizer(
                subs, queue_size=5, slop=0.05, allow_headerless=True
            )
            self.ts.registerCallback(self._lidar_callback)

    def _lookup_transform(self, target_frame, source_frame, stamp):
        self._logdebug(
            "_lookup_transform",
            "Looking up TF %s -> %s (stamp=%.6f)",
            source_frame,
            target_frame,
            stamp.to_sec() if hasattr(stamp, "to_sec") else 0.0,
        )
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                target_frame, source_frame, stamp, rospy.Duration(0.2)
            )
            mat = transform_stamped_to_matrix(tf_msg)
            self._logdebug(
                "_lookup_transform",
                "TF %s -> %s:\n%s",
                source_frame,
                target_frame,
                np.array2string(mat, precision=4, suppress_small=True),
            )
            return mat
        except tf2_ros.TransformException as ex:
            self._logwarn(
                "_lookup_transform",
                "TF lookup failed %s -> %s: %s", source_frame, target_frame, ex
            )
            return None

    def _wait_for_msg(self, topic, msg_type, timeout=2.0, warn_on_timeout=True):
        if not topic:
            return None
        timeout_str = "None" if timeout is None else f"{float(timeout):.2f}s"
        self._logdebug(
            "_wait_for_msg",
            "Waiting for %s on topic=%s (timeout=%s)",
            msg_type.__name__,
            topic,
            timeout_str,
        )
        try:
            msg = rospy.wait_for_message(topic, msg_type, timeout=timeout)
            self._logdebug(
                "_wait_for_msg",
                "Received %s on topic=%s (stamp=%.6f frame_id=%s)",
                msg_type.__name__,
                topic,
                msg.header.stamp.to_sec() if hasattr(msg, "header") else 0.0,
                msg.header.frame_id if hasattr(msg, "header") else "",
            )
            return msg
        except Exception as exc:  # noqa: BLE001
            if warn_on_timeout:
                self._logwarn(
                    "_wait_for_msg", "Timeout waiting for %s: %s", topic, exc
                )
            else:
                self._logdebug(
                    "_wait_for_msg", "Timeout waiting for %s: %s", topic, exc
                )
            return None

    def _wait_for_topic_type(self, topic, timeout=2.0, warn_on_timeout=True):
        """Wait for a message and return its ROS type string (e.g., sensor_msgs/Image)."""
        if not topic:
            return None
        timeout_str = "None" if timeout is None else f"{float(timeout):.2f}s"
        self._logdebug(
            "_wait_for_topic_type",
            "Waiting for AnyMsg on topic=%s (timeout=%s)",
            topic,
            timeout_str,
        )
        try:
            msg = rospy.wait_for_message(topic, rospy.AnyMsg, timeout=timeout)
            type_str = None
            if hasattr(msg, "_connection_header") and msg._connection_header:
                type_str = msg._connection_header.get("type")
            self._logdebug(
                "_wait_for_topic_type",
                "Received AnyMsg on topic=%s type=%s",
                topic,
                type_str,
            )
            return type_str
        except Exception as exc:  # noqa: BLE001
            if warn_on_timeout:
                self._logwarn(
                    "_wait_for_topic_type",
                    "Timeout waiting for topic type on %s: %s",
                    topic,
                    exc,
                )
            else:
                self._logdebug(
                    "_wait_for_topic_type",
                    "Timeout waiting for topic type on %s: %s",
                    topic,
                    exc,
                )
            return None

    def _detect_mode(self):
        """Detect mode based on configured topics' message types."""
        if self.depth_input_topic:
            if self.depth_topic or self.lidar_topic:
                self._logwarn(
                    "_detect_mode",
                    "~depth_input_topic is set; ignoring deprecated ~depth_topic/~lidar_topic",
                )
            type_str = None
            while type_str is None and not rospy.is_shutdown():
                type_str = self._wait_for_topic_type(
                    self.depth_input_topic, timeout=1.0, warn_on_timeout=False
                )
            if type_str == "sensor_msgs/Image":
                self.depth_topic = self.depth_input_topic
                self.lidar_topic = None
                self._loginfo(
                    "_detect_mode",
                    "Auto-detected depth_input_topic type=%s -> mode=depth",
                    type_str,
                )
                return "depth"
            if type_str == "sensor_msgs/PointCloud2":
                self.lidar_topic = self.depth_input_topic
                self.depth_topic = None
                self._loginfo(
                    "_detect_mode",
                    "Auto-detected depth_input_topic type=%s -> mode=lidar",
                    type_str,
                )
                return "lidar"
            if type_str == "sensor_msgs/CompressedImage":
                raise ValueError(
                    "~depth_input_topic is sensor_msgs/CompressedImage; republish to raw Image via image_transport or set use_republish:=true in launch"
                )
            if type_str:
                raise ValueError(
                    f"Unsupported ~depth_input_topic message type: {type_str} (expected sensor_msgs/Image or sensor_msgs/PointCloud2)"
                )
            raise ValueError(
                f"Unable to determine message type for ~depth_input_topic={self.depth_input_topic}"
            )

        depth_set = bool(self.depth_topic)
        lidar_set = bool(self.lidar_topic)
        self._logdebug(
            "_detect_mode",
            "Detecting mode from topics: depth_topic=%s lidar_topic=%s",
            self.depth_topic,
            self.lidar_topic,
        )

        if depth_set and not lidar_set:
            return "depth"
        if lidar_set and not depth_set:
            return "lidar"
        if not depth_set and not lidar_set:
            raise ValueError("Set ~depth_topic and/or ~lidar_topic")

        # Both are configured: wait briefly for first message to decide.
        if (
            self._wait_for_msg(
                self.depth_topic, Image, timeout=0.2, warn_on_timeout=False
            )
            is not None
        ):
            return "depth"
        if (
            self._wait_for_msg(
                self.lidar_topic, PointCloud2, timeout=0.2, warn_on_timeout=False
            )
            is not None
        ):
            return "lidar"

        self._logwarn(
            "_detect_mode", "Unable to auto-detect mode from topics; defaulting to depth"
        )
        return "depth"

    @staticmethod
    def _build_inverse_color_map(color_map, quantize_step=1):
        keys = []
        values = []
        packed_seen = {}
        step = int(quantize_step)
        half = step // 2
        for label, rgb in color_map.items():
            r, g, b = rgb
            if step > 1:
                r = int(np.clip(((int(r) + half) // step) * step, 0, 255))
                g = int(np.clip(((int(g) + half) // step) * step, 0, 255))
                b = int(np.clip(((int(b) + half) // step) * step, 0, 255))
            packed = (int(r) << 16) | (int(g) << 8) | int(b)
            if packed in packed_seen and packed_seen[packed] != int(label):
                raise ValueError(
                    "color_map entries collide after quantization: "
                    f"packed=0x{packed:06x} labels={packed_seen[packed]} and {int(label)}"
                )
            packed_seen[packed] = int(label)
            keys.append(packed)
            values.append(int(label))

        keys = np.asarray(keys, dtype=np.uint32)
        values = np.asarray(values, dtype=np.int32)
        order = np.argsort(keys)
        return keys[order], values[order]

    @staticmethod
    def _packed_color_to_label(packed, inverse_map):
        keys_sorted, labels_sorted = inverse_map
        packed_flat = packed.reshape(-1)
        idx = np.searchsorted(keys_sorted, packed_flat)

        labels_flat = np.full(packed_flat.shape, -1, dtype=np.int32)
        in_range = idx < keys_sorted.size
        valid = np.zeros_like(in_range, dtype=bool)
        valid[in_range] = keys_sorted[idx[in_range]] == packed_flat[in_range]
        labels_flat[valid] = labels_sorted[idx[valid]]
        return labels_flat.reshape(packed.shape)

    def _set_color_decoder_from_color_map(self):
        self._inverse_color_map = self._build_inverse_color_map(
            self.color_map, quantize_step=self.semantic_color_quantization_step
        )
        keys_sorted, labels_sorted = self._inverse_color_map
        self._packed_to_label = {
            int(k): int(v) for k, v in zip(keys_sorted.tolist(), labels_sorted.tolist())
        }
        self._auto_color_decode = False

    def _set_auto_color_decoder_from_packed(self, packed):
        flat = packed.reshape(-1).astype(np.uint32)
        unique, counts = np.unique(flat, return_counts=True)
        total = int(flat.size)
        min_count_eff = max(
            int(self.auto_color_to_label_min_count),
            int(np.ceil(float(self.auto_color_to_label_min_fraction) * total)),
        )

        raw_unique = int(unique.size)
        filtered_count = int(np.count_nonzero(counts >= int(min_count_eff)))

        palette = dominant_packed_colors(
            packed,
            min_count=int(self.auto_color_to_label_min_count),
            min_fraction=float(self.auto_color_to_label_min_fraction),
            max_colors=int(self.auto_color_to_label_max_colors),
            merge_distance=int(self.auto_color_to_label_merge_distance),
        )
        if palette.size == 0:
            raise ValueError("auto_color_to_label produced an empty palette")
        if palette.size > 65535:
            raise ValueError(
                f"auto_color_to_label found {palette.size} palette colors; exceeds uint16 label range"
            )

        self._auto_palette_packed = palette.astype(np.uint32, copy=False)
        self._auto_palette_rgb = packed_rgb_to_triplets(self._auto_palette_packed)

        # Deterministic label IDs: sort by packed RGB.
        palette_labels = np.arange(self._auto_palette_packed.size, dtype=np.int32)

        # Publishable color_map: label_id -> [r,g,b].
        self.color_map = {}
        for label_id, rgb in enumerate(self._auto_palette_rgb.tolist()):
            self.color_map[int(label_id)] = [int(rgb[0]), int(rgb[1]), int(rgb[2])]
        self._color_map_revision += 1

        # Build a full decoder for all colors seen in this first frame by snapping
        # each observed color to the nearest palette entry (robust to JPEG noise).
        mapped = self._nearest_palette_labels_for_packed(unique)
        self._packed_to_label = {
            int(k): int(v) for k, v in zip(unique.tolist(), mapped.tolist())
        }
        self._inverse_color_map = self._build_inverse_from_mapping_dict(
            self._packed_to_label
        )
        self._auto_color_decode = True

        self._loginfo(
            "_set_auto_color_decoder_from_packed",
            "Auto palette→label decoder ready: palette=%d (raw_unique=%d filtered=%d min_count_eff=%d merge_distance=%d max_colors=%d).",
            int(self._auto_palette_packed.size),
            int(raw_unique),
            int(filtered_count),
            int(min_count_eff),
            int(self.auto_color_to_label_merge_distance),
            int(self.auto_color_to_label_max_colors),
        )

    def _build_inverse_from_mapping_dict(self, mapping_dict):
        keys_sorted = np.asarray(sorted(mapping_dict.keys()), dtype=np.uint32)
        labels_sorted = np.asarray(
            [mapping_dict[int(k)] for k in keys_sorted.tolist()], dtype=np.int32
        )
        return keys_sorted, labels_sorted

    def _nearest_palette_labels_for_packed(self, packed_values):
        if self._auto_palette_rgb is None:
            raise RuntimeError("auto palette not initialized")
        values = np.asarray(packed_values, dtype=np.uint32).reshape(-1)
        rgb_vals = packed_rgb_to_triplets(values).astype(np.int16, copy=False)
        palette_rgb = self._auto_palette_rgb.astype(np.int16, copy=False)
        diff = rgb_vals[:, None, :] - palette_rgb[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        nearest = np.argmin(dist2, axis=1).astype(np.int32)
        return nearest.reshape(np.asarray(packed_values).shape)

    def _extend_auto_color_decoder(self, packed, labels):
        if not self._auto_color_decode:
            return labels
        unknown_mask = labels == -1
        if not np.any(unknown_mask):
            return labels
        unknown_vals = np.unique(packed[unknown_mask].reshape(-1).astype(np.uint32))
        mapped = self._nearest_palette_labels_for_packed(unknown_vals)

        if self.auto_color_to_label_extend:
            # Cache the new colors for faster lookup in later frames (label IDs do
            # NOT change; new colors are snapped to the existing palette).
            for packed_val, label_id in zip(unknown_vals.tolist(), mapped.tolist()):
                self._packed_to_label[int(packed_val)] = int(label_id)
            self._inverse_color_map = self._build_inverse_from_mapping_dict(
                self._packed_to_label
            )
            self._logdebug(
                "_extend_auto_color_decoder",
                "Cached %d new colors to nearest palette labels (mapping_size=%d palette=%d).",
                int(unknown_vals.size),
                int(len(self._packed_to_label)),
                int(self._auto_palette_packed.size)
                if self._auto_palette_packed is not None
                else -1,
            )
            return self._packed_color_to_label(packed, self._inverse_color_map)

        # Decode unknown colors for this frame only (no caching).
        mapped_out = labels.copy()
        unknown_sorted = np.sort(unknown_vals.astype(np.uint32))
        mapped_sorted = mapped.reshape(-1)[np.argsort(unknown_vals)]
        flat_unknown = packed[unknown_mask].reshape(-1).astype(np.uint32)
        idx = np.searchsorted(unknown_sorted, flat_unknown)
        mapped_out[unknown_mask] = mapped_sorted[idx]
        return mapped_out

    @staticmethod
    def _scale_intrinsics(intrinsics, factor):
        scaled = intrinsics.copy()
        scaled[0, 0] /= factor
        scaled[1, 1] /= factor
        scaled[0, 2] /= factor
        scaled[1, 2] /= factor
        return scaled

    @staticmethod
    def _image_to_numpy(msg):
        if msg is None:
            return None
        encoding = msg.encoding.lower()
        if encoding in ("bgr8", "rgb8"):
            dtype = np.uint8
            channels = 3
        elif encoding in ("bgra8", "rgba8"):
            dtype = np.uint8
            channels = 4
        elif encoding in ("32fc1", "32fc"):
            dtype = np.float32
            channels = 1
        elif encoding in ("32sc1",):
            dtype = np.int32
            channels = 1
        elif encoding in ("16uc1", "16sc1", "mono16"):
            dtype = np.uint16
            channels = 1
        elif encoding in ("8uc1", "mono8"):
            dtype = np.uint8
            channels = 1
        else:
            raise ValueError(f"Unsupported image encoding: {msg.encoding}")
        expected_step = msg.width * channels * dtype().nbytes
        if msg.step != expected_step:
            raise ValueError(
                "Unsupported step "
                f"{msg.step} for {msg.encoding}; expected {expected_step}"
            )
        arr = np.frombuffer(msg.data, dtype=dtype)
        if channels == 1:
            return arr.reshape(msg.height, msg.width)
        return arr.reshape(msg.height, msg.width, channels)

    @staticmethod
    def _pointcloud2_to_xyz(msg):
        if msg.is_bigendian:
            raise ValueError("big-endian PointCloud2 not supported in fast path")
        field_offsets = {f.name: f.offset for f in msg.fields}
        for needed in ("x", "y", "z"):
            if needed not in field_offsets:
                raise ValueError("PointCloud2 missing xyz fields")
        dtype = np.dtype(
            {
                "names": ["x", "y", "z"],
                "formats": ["<f4", "<f4", "<f4"],
                "offsets": [
                    field_offsets["x"],
                    field_offsets["y"],
                    field_offsets["z"],
                ],
                "itemsize": msg.point_step,
            }
        )
        count = len(msg.data) // msg.point_step
        cloud = np.frombuffer(msg.data, dtype=dtype, count=count)
        return np.stack([cloud["x"], cloud["y"], cloud["z"]], axis=-1)

    @contextmanager
    def _maybe_profile(self, label):
        if not self.enable_profiling:
            yield
            return
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            yield
        finally:
            profiler.disable()
            s = io.StringIO()
            pstats.Stats(profiler, stream=s).sort_stats("cumulative").print_stats(5)
            self._loginfo(label, "profiling:\n%s", s.getvalue())

    def _prime_transforms(self):
        """Resolve extrinsics once at startup (TF preferred, static fallback)."""
        if self.mode == "depth":
            self._logdebug("_prime_transforms", "Priming depth transforms")
            depth_msg = self._wait_for_msg(
                self.depth_topic, Image, timeout=5.0, warn_on_timeout=False
            )
            depth_frame = depth_msg.header.frame_id if depth_msg else None
            if self.static_target_T_depth is not None:
                self.target_T_depth = self.static_target_T_depth
                self._logdebug(
                    "_prime_transforms",
                    "Using static target_T_depth:\n%s",
                    np.array2string(
                        self.target_T_depth, precision=4, suppress_small=True
                    ),
                )
            elif depth_frame:
                self._logdebug(
                    "_prime_transforms",
                    "Looking up depth->target TF: %s -> %s",
                    depth_frame,
                    self.target_frame,
                )
                self.target_T_depth = self._lookup_transform(
                    self.target_frame, depth_frame, rospy.Time(0)
                )
        else:
            self._logdebug("_prime_transforms", "Priming LiDAR transforms")
            lidar_msg = self._wait_for_msg(
                self.lidar_topic, PointCloud2, timeout=5.0, warn_on_timeout=False
            )
            lidar_frame = lidar_msg.header.frame_id if lidar_msg else None
            if self.static_camera_T_lidar is not None:
                self.camera_T_lidar = self.static_camera_T_lidar
                self._logdebug(
                    "_prime_transforms",
                    "Using static camera_T_lidar:\n%s",
                    np.array2string(
                        self.camera_T_lidar, precision=4, suppress_small=True
                    ),
                )
            elif lidar_frame:
                self._logdebug(
                    "_prime_transforms",
                    "Looking up lidar->camera TF: %s -> %s",
                    lidar_frame,
                    self.camera_frame,
                )
                self.camera_T_lidar = self._lookup_transform(
                    self.camera_frame, lidar_frame, rospy.Time(0)
                )
            if self.static_target_T_lidar is not None:
                self.target_T_lidar = self.static_target_T_lidar
                self._logdebug(
                    "_prime_transforms",
                    "Using static target_T_lidar:\n%s",
                    np.array2string(
                        self.target_T_lidar, precision=4, suppress_small=True
                    ),
                )
            elif lidar_frame:
                self._logdebug(
                    "_prime_transforms",
                    "Looking up lidar->target TF: %s -> %s",
                    lidar_frame,
                    self.target_frame,
                )
                self.target_T_lidar = self._lookup_transform(
                    self.target_frame, lidar_frame, rospy.Time(0)
                )
        self._loginfo(
            "_prime_transforms",
            "Transforms primed (depth=%s, camera_T_lidar=%s, target_T_lidar=%s)",
            self.target_T_depth is not None,
            self.camera_T_lidar is not None,
            self.target_T_lidar is not None,
        )
        if self.mode == "depth" and self.target_T_depth is None:
            self._logwarn(
                "_prime_transforms", "No depth->target transform available at init"
            )
        if self.mode == "lidar":
            if self.camera_T_lidar is None:
                self._logwarn(
                    "_prime_transforms", "No camera<-lidar transform available at init"
                )
            if self.target_T_lidar is None:
                self._logwarn(
                    "_prime_transforms", "No target<-lidar transform available at init"
                )

    def _parse_semantic(self, msg):
        data = self._image_to_numpy(msg)
        encoding = msg.encoding.lower()

        if data.ndim == 3:
            if self._inverse_color_map is None:
                if self.color_map is not None:
                    self._set_color_decoder_from_color_map()
                elif self.auto_color_to_label:
                    kind, count = count_semantic_groups(
                        data[:, :, :3],
                        color_quantize_step=self.semantic_color_quantization_step,
                        color_min_count=self.auto_color_to_label_min_count,
                        color_min_fraction=self.auto_color_to_label_min_fraction,
                        color_max_colors=self.auto_color_to_label_max_colors,
                        color_merge_distance=self.auto_color_to_label_merge_distance,
                    )
                    self._logwarn(
                        "_parse_semantic",
                        "RGB semantic image detected (%s=%d, quantize_step=%d, merge_distance=%d). No ~color_map provided; using auto_color_to_label mapping. "
                        "Provide ~color_map for stable class IDs.",
                        kind,
                        int(count),
                        int(self.semantic_color_quantization_step),
                        int(self.auto_color_to_label_merge_distance),
                    )
                    self._set_auto_color_decoder_from_packed(
                        self._rgb_to_packed(
                            data,
                            encoding,
                            quantize_step=self.semantic_color_quantization_step,
                        )
                    )
                else:
                    raise ValueError(
                        "3-channel semantic image requires either ~color_map (label -> [r,g,b]) "
                        "or ~auto_color_to_label:=true to infer a mapping from the palette"
                    )

            packed = self._rgb_to_packed(
                data, encoding, quantize_step=self.semantic_color_quantization_step
            )
            labels = self._packed_color_to_label(packed, self._inverse_color_map)
            if self._auto_color_decode and np.any(labels == -1):
                return self._extend_auto_color_decoder(packed, labels)

            if (
                not self._auto_color_decode
                and not self._warned_unknown_colors
                and np.any(labels == -1)
            ):
                unknown = int(np.unique(packed[labels == -1]).size)
                self._logwarn(
                    "_parse_semantic",
                    "RGB semantic image contains %d unknown colors not present in ~color_map; those pixels become label=-1.",
                    unknown,
                )
                self._warned_unknown_colors = True
            return labels

        if data.ndim != 2:
            raise ValueError("semantic_topic must be single-channel label image")
        return data

    @staticmethod
    def _quantize_u8(arr_u8, step):
        step = int(step)
        if step <= 1:
            return arr_u8.astype(np.uint32, copy=False)
        vals = arr_u8.astype(np.int16, copy=False)
        half = step // 2
        quant = ((vals + half) // step) * step
        return np.clip(quant, 0, 255).astype(np.uint32, copy=False)

    @classmethod
    def _rgb_to_packed(cls, data, encoding, quantize_step=1):
        if encoding == "bgr8":
            b, g, r = data[:, :, 0], data[:, :, 1], data[:, :, 2]
        elif encoding == "rgb8":
            r, g, b = data[:, :, 0], data[:, :, 1], data[:, :, 2]
        elif encoding == "bgra8":
            b, g, r = data[:, :, 0], data[:, :, 1], data[:, :, 2]
        elif encoding == "rgba8":
            r, g, b = data[:, :, 0], data[:, :, 1], data[:, :, 2]
        else:
            raise ValueError(f"Unsupported 3-channel encoding: {encoding}")
        step = int(quantize_step)
        r_u32 = cls._quantize_u8(r, step)
        g_u32 = cls._quantize_u8(g, step)
        b_u32 = cls._quantize_u8(b, step)
        return (
            (r_u32 << 16) | (g_u32 << 8) | b_u32
        )

    def _depth_callback(self, sem_msg, depth_msg, conf_msg=None):
        with self._maybe_profile("depth_callback"):
            stamp = (
                sem_msg.header.stamp
                if sem_msg.header.stamp > depth_msg.header.stamp
                else depth_msg.header.stamp
            )
            target_T_depth = self.target_T_depth
            if target_T_depth is None:
                depth_frame = depth_msg.header.frame_id
                if depth_frame:
                    self._logdebug(
                        "_depth_callback",
                        "Priming depth->target transform on first callback (%s -> %s)",
                        depth_frame,
                        self.target_frame,
                    )
                    self.target_T_depth = self._lookup_transform(
                        self.target_frame, depth_frame, rospy.Time(0)
                    )
                    target_T_depth = self.target_T_depth
                    if target_T_depth is not None:
                        self._loginfo(
                            "_depth_callback", "Primed depth->target transform"
                        )
                if target_T_depth is None:
                    self._logdebug(
                        "_depth_callback",
                        "Skipping frame: missing depth->target transform",
                    )
                    return

            labels = self._parse_semantic(sem_msg)
            depth_raw = self._image_to_numpy(depth_msg)
            depth_enc = depth_msg.encoding.lower()
            confidence = (
                self._image_to_numpy(conf_msg).astype(float) if conf_msg else None
            )

            if self.downsample_factor > 1:
                f = self.downsample_factor
                labels = labels[::f, ::f]
                depth_raw = depth_raw[::f, ::f]
                if confidence is not None:
                    confidence = confidence[::f, ::f]
                intrinsics = self._scale_intrinsics(self.intrinsics, f)
            else:
                intrinsics = self.intrinsics

            scale = float(self.depth_scale)
            if scale == 0.0:
                if depth_enc in ("16uc1", "16sc1", "mono16"):
                    scale = 0.001
                else:
                    scale = 1.0
            if self.debug and not self._logged_depth_scaling:
                self._loginfo(
                    "_depth_callback",
                    "Depth scaling: encoding=%s scale=%.6f (depth_scale_param=%.6f)",
                    depth_msg.encoding,
                    float(scale),
                    float(self.depth_scale),
                )
                self._logged_depth_scaling = True

            depth = depth_raw.astype(np.float32, copy=False) * float(scale)
            if confidence is not None:
                confidence = confidence.astype(np.float32, copy=False)

            if self.debug and not self._logged_depth_summary:
                valid = np.isfinite(depth) & (depth > 0)
                valid_count = int(np.count_nonzero(valid))
                if valid_count:
                    dmin = float(depth[valid].min())
                    dmax = float(depth[valid].max())
                else:
                    dmin, dmax = float("nan"), float("nan")
                self._loginfo(
                    "_depth_callback",
                    "Depth inputs: semantic_shape=%s depth_shape=%s depth_encoding=%s valid_depth=%d min=%.3f max=%.3f downsample=%d",
                    labels.shape,
                    depth.shape,
                    depth_msg.encoding,
                    valid_count,
                    dmin,
                    dmax,
                    int(self.downsample_factor),
                )

            semantic_obs = SemanticObservation(labels=labels, confidence=confidence)
            depth_obs = DepthObservation(depth=depth)
            pcl = fuse_depth_semantics(
                semantic_obs,
                depth_obs,
                intrinsics,
                target_T_depth,
                include_unlabeled=self.include_unlabeled,
            )
            self._logdebug(
                "_depth_callback",
                "Publishing depth-based semantic PCL with %d points",
                pcl.points_xyz.shape[0],
            )
            if self.debug and not self._logged_depth_summary:
                if pcl.points_xyz.shape[0]:
                    mins = pcl.points_xyz.min(axis=0)
                    maxs = pcl.points_xyz.max(axis=0)
                    self._loginfo(
                        "_depth_callback",
                        "Depth PCL bbox in %s: x=[%.3f, %.3f] y=[%.3f, %.3f] z=[%.3f, %.3f]",
                        self.target_frame,
                        float(mins[0]),
                        float(maxs[0]),
                        float(mins[1]),
                        float(maxs[1]),
                        float(mins[2]),
                        float(maxs[2]),
                    )
                else:
                    self._logwarn(
                        "_depth_callback",
                        "Depth fusion produced empty point cloud (check depth scaling, intrinsics, and transforms)",
                    )
                self._logged_depth_summary = True

            pcl_msg = semantic_pointcloud_to_msg(
                pcl,
                self.target_frame,
                stamp,
                colorize_labels=self.colorize_labels,
                rgb_lut=self._get_rgb_float_lut(),
            )
            self.pcl_pub.publish(pcl_msg)

    def _lidar_callback(self, sem_msg, lidar_msg, conf_msg=None):
        with self._maybe_profile("lidar_callback"):
            stamp = lidar_msg.header.stamp
            camera_T_lidar = self.camera_T_lidar
            target_T_lidar = self.target_T_lidar
            if camera_T_lidar is None or target_T_lidar is None:
                lidar_frame = lidar_msg.header.frame_id
                if lidar_frame:
                    if camera_T_lidar is None:
                        self._logdebug(
                            "_lidar_callback",
                            "Priming lidar->camera transform on first callback (%s -> %s)",
                            lidar_frame,
                            self.camera_frame,
                        )
                        self.camera_T_lidar = self._lookup_transform(
                            self.camera_frame, lidar_frame, rospy.Time(0)
                        )
                        camera_T_lidar = self.camera_T_lidar
                        if camera_T_lidar is not None:
                            self._loginfo(
                                "_lidar_callback", "Primed lidar->camera transform"
                            )
                    if target_T_lidar is None:
                        self._logdebug(
                            "_lidar_callback",
                            "Priming lidar->target transform on first callback (%s -> %s)",
                            lidar_frame,
                            self.target_frame,
                        )
                        self.target_T_lidar = self._lookup_transform(
                            self.target_frame, lidar_frame, rospy.Time(0)
                        )
                        target_T_lidar = self.target_T_lidar
                        if target_T_lidar is not None:
                            self._loginfo(
                                "_lidar_callback", "Primed lidar->target transform"
                            )
                camera_T_lidar = self.camera_T_lidar
                target_T_lidar = self.target_T_lidar
                if camera_T_lidar is None or target_T_lidar is None:
                    self._logdebug(
                        "_lidar_callback",
                        "Skipping frame: missing LiDAR transforms",
                    )
                    return

            labels = self._parse_semantic(sem_msg)
            confidence = (
                self._image_to_numpy(conf_msg).astype(float) if conf_msg else None
            )
            if self.downsample_factor > 1:
                f = self.downsample_factor
                labels = labels[::f, ::f]
                if confidence is not None:
                    confidence = confidence[::f, ::f]
                intrinsics = self._scale_intrinsics(self.intrinsics, f)
            else:
                intrinsics = self.intrinsics

            points = self._pointcloud2_to_xyz(lidar_msg)

            semantic_obs = SemanticObservation(labels=labels, confidence=confidence)
            point_obs = PointObservation(points_xyz=points)
            pcl = fuse_lidar_semantics(
                semantic_obs,
                point_obs,
                intrinsics,
                camera_T_lidar,
                target_T_lidar,
                include_unlabeled=self.include_unlabeled,
            )
            self._logdebug(
                "_lidar_callback",
                "Publishing LiDAR-based semantic PCL with %d points",
                pcl.points_xyz.shape[0],
            )
            if self.debug and not self._logged_lidar_summary:
                if pcl.points_xyz.shape[0]:
                    mins = pcl.points_xyz.min(axis=0)
                    maxs = pcl.points_xyz.max(axis=0)
                    self._loginfo(
                        "_lidar_callback",
                        "LiDAR PCL bbox in %s: x=[%.3f, %.3f] y=[%.3f, %.3f] z=[%.3f, %.3f] (input_points=%d)",
                        self.target_frame,
                        float(mins[0]),
                        float(maxs[0]),
                        float(mins[1]),
                        float(maxs[1]),
                        float(mins[2]),
                        float(maxs[2]),
                        int(points.shape[0]),
                    )
                else:
                    self._logwarn(
                        "_lidar_callback",
                        "LiDAR fusion produced empty point cloud (check intrinsics, transforms, and image alignment)",
                    )
                self._logged_lidar_summary = True
            pcl_msg = semantic_pointcloud_to_msg(
                pcl,
                self.target_frame,
                stamp,
                colorize_labels=self.colorize_labels,
                rgb_lut=self._get_rgb_float_lut(),
            )
            self.pcl_pub.publish(pcl_msg)


def main():
    log_level = rospy.DEBUG if _rosargv_bool("debug", False) else rospy.INFO
    rospy.init_node("semantic_pcl_node", log_level=log_level)
    node = SemanticPclNode()
    rospy.spin()


if __name__ == "__main__":
    main()
