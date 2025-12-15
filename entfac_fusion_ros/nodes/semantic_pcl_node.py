#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: MIT License (open source, free to modify and redistribute)
# Repository: ENTFAC-Sensor-Fusion
#
# Description:
#   ROS wrapper that turns semantic + geometry inputs into semantic PointCloud2 outputs.

"""ROS wrapper that converts semantic + geometry into semantic PointCloud2."""

import cProfile
import io
import pstats
import struct
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import rospy
import tf2_ros
from message_filters import ApproximateTimeSynchronizer, Subscriber
from scipy.spatial.transform import Rotation as SciRotation
from sensor_msgs import point_cloud2
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header

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


def transform_stamped_to_matrix(transform_stamped):
    """Convert geometry_msgs/TransformStamped to 4x4 numpy matrix."""
    t = transform_stamped.transform.translation
    q = transform_stamped.transform.rotation
    translation = np.array([t.x, t.y, t.z], dtype=float)
    quat = np.array([q.x, q.y, q.z, q.w], dtype=float)
    rot = SciRotation.from_quat(quat).as_matrix()
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = rot
    mat[:3, 3] = translation
    return require_homogeneous_transform(mat.astype(float))


def semantic_pointcloud_to_msg(
    pcl, frame_id, stamp, colorize_labels=False, color_map=None
):
    """Convert SemanticPointCloud dataclass to PointCloud2."""
    has_conf = pcl.confidence is not None
    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
    ]
    offset = 12
    fields.append(PointField("label", offset, PointField.UINT16, 1))
    offset += 4
    if has_conf:
        fields.append(PointField("confidence", offset, PointField.FLOAT32, 1))
        offset += 4
    if colorize_labels:
        fields.append(PointField("rgb", offset, PointField.FLOAT32, 1))

    header = Header()
    header.frame_id = frame_id
    header.stamp = stamp

    points = []
    if has_conf:
        for (x, y, z), lbl, conf in zip(pcl.points_xyz, pcl.labels, pcl.confidence):
            label_val = int(lbl)
            if label_val < 0:
                label_val = 65535
            if label_val > 65535:
                raise ValueError("label must fit into uint16 (0..65535)")
            record = [float(x), float(y), float(z), label_val, float(conf)]
            if colorize_labels:
                record.append(_label_to_rgb_float(label_val, color_map))
            points.append(tuple(record))
    else:
        for (x, y, z), lbl in zip(pcl.points_xyz, pcl.labels):
            label_val = int(lbl)
            if label_val < 0:
                label_val = 65535
            if label_val > 65535:
                raise ValueError("label must fit into uint16 (0..65535)")
            record = [float(x), float(y), float(z), label_val]
            if colorize_labels:
                record.append(_label_to_rgb_float(label_val, color_map))
            points.append(tuple(record))

    return point_cloud2.create_cloud(header, fields, points)


def _label_to_rgb_float(label_val, color_map=None):
    # Custom color map if provided (expects dict of int -> [r,g,b])
    if label_val == 65535:
        r, g, b = 255, 255, 255
    elif color_map and label_val in color_map:
        r, g, b = color_map[label_val]
    else:
        r = (label_val * 37) % 256
        g = (label_val * 17) % 256
        b = (label_val * 73) % 256
    packed = (int(r) << 16) | (int(g) << 8) | int(b)
    return struct.unpack("f", struct.pack("I", packed))[0]


class SemanticPclNode:
    """ROS node bridging topics to the numpy fusion core."""

    def __init__(self):
        # 'depth' uses aligned depth image; 'lidar' projects LiDAR into image.
        self.mode = rospy.get_param("~mode", "").lower()
        self.target_frame = self._param_string("~target_frame", "base_link")
        self.semantic_topic = self._param_string("~semantic_topic", "/semantic/labels")
        self.conf_topic = self._param_string("~confidence_topic", None)
        self.camera_info_topic = self._param_string("~camera_info", None)
        self.depth_topic = self._param_string("~depth_topic", None)
        self.lidar_topic = self._param_string("~lidar_topic", None)
        self.include_unlabeled = rospy.get_param("~include_unlabeled_pts", False)
        self.colorize_labels = rospy.get_param("~colorize_labels", False)
        self.color_map = self._load_color_map("~color_map")
        self._inverse_color_map = None
        self.downsample_factor = int(rospy.get_param("~downsample_factor", 1))
        if self.downsample_factor < 1:
            raise ValueError("~downsample_factor must be >= 1")
        self.enable_profiling = rospy.get_param("~enable_profiling", False)
        self.static_target_T_depth = self._load_matrix_param("~static_target_T_depth")
        self.static_camera_T_lidar = self._load_matrix_param("~static_camera_T_lidar")
        self.static_target_T_lidar = self._load_matrix_param("~static_target_T_lidar")
        if self.static_target_T_depth is not None:
            rospy.loginfo("Using static target_T_depth parameter")
        if self.static_camera_T_lidar is not None:
            rospy.loginfo("Using static camera_T_lidar parameter")
        if self.static_target_T_lidar is not None:
            rospy.loginfo("Using static target_T_lidar parameter")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        cam_info = rospy.wait_for_message(self.camera_info_topic, CameraInfo)
        self.intrinsics = np.asarray(cam_info.K, dtype=float).reshape(3, 3)
        self.camera_frame = cam_info.header.frame_id
        self.pcl_pub = rospy.Publisher(
            "semantic_pointcloud", PointCloud2, queue_size=1
        )
        self.target_T_depth = None
        self.camera_T_lidar = None
        self.target_T_lidar = None
        if self.mode not in ("depth", "lidar"):
            self.mode = self._detect_mode()
            rospy.loginfo("Auto-detected mode=%s", self.mode)
        self._prime_transforms()

        self._register_subscribers()
        rospy.loginfo("semantic_pcl_node initialized (mode=%s)", self.mode)

    def _register_subscribers(self):
        """Setup message_filters synchronizer."""
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
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                target_frame, source_frame, stamp, rospy.Duration(0.2)
            )
            return transform_stamped_to_matrix(tf_msg)
        except tf2_ros.TransformException as ex:
            rospy.logwarn(
                "TF lookup failed %s -> %s: %s", source_frame, target_frame, ex
            )
            return None

    @staticmethod
    def _wait_for_msg(topic, msg_type, timeout=2.0):
        try:
            return rospy.wait_for_message(topic, msg_type, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            rospy.logwarn("Timeout waiting for %s: %s", topic, exc)
            return None

    def _detect_mode(self):
        """Detect mode based on configured topics' message types."""
        depth_set = bool(self.depth_topic)
        lidar_set = bool(self.lidar_topic)

        if depth_set and not lidar_set:
            return "depth"
        if lidar_set and not depth_set:
            return "lidar"
        if not depth_set and not lidar_set:
            raise ValueError("Set ~depth_topic and/or ~lidar_topic")

        # Both are configured: wait briefly for first message to decide.
        if self._wait_for_msg(self.depth_topic, Image, timeout=0.2) is not None:
            return "depth"
        if (
            self._wait_for_msg(self.lidar_topic, PointCloud2, timeout=0.2)
            is not None
        ):
            return "lidar"

        rospy.logwarn(
            "Unable to auto-detect mode from topics; defaulting to depth"
        )
        return "depth"

    def _load_matrix_param(self, name):
        """Load a 4x4 matrix parameter if provided."""
        raw = rospy.get_param(name, [])
        if isinstance(raw, list) and len(raw) == 16:
            mat = np.asarray(raw, dtype=float).reshape(4, 4)
            try:
                return require_homogeneous_transform(mat)
            except ValueError as exc:
                rospy.logwarn("%s rejected: %s", name, exc)
        return None

    @staticmethod
    def _param_string(name, default=None):
        """Fetch string param; if empty or None, return default."""
        val = rospy.get_param(name, default)
        if isinstance(val, str) and val.strip() == "":
            return default
        return val

    @staticmethod
    def _load_color_map(name):
        raw = rospy.get_param(name, {})
        if not isinstance(raw, dict):
            return None
        color_map = {}
        for k, v in raw.items():
            try:
                key_int = int(k)
                if isinstance(v, (list, tuple)) and len(v) == 3:
                    color_map[key_int] = [int(v[0]), int(v[1]), int(v[2])]
            except Exception:
                continue
        return color_map if color_map else None

    @staticmethod
    def _build_inverse_color_map(color_map):
        keys = []
        values = []
        for label, rgb in color_map.items():
            r, g, b = rgb
            keys.append((int(r) << 16) | (int(g) << 8) | int(b))
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


def _quat_to_matrix(quat):
    """Convert quaternion (x, y, z, w) to 4x4 matrix without tf dependency."""
    x, y, z, w = quat
    n = x * x + y * y + z * z + w * w
    if n < np.finfo(float).eps:
        return np.eye(4, dtype=float)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s

    mat = np.eye(4, dtype=float)
    mat[0, 0] = 1.0 - (yy + zz)
    mat[0, 1] = xy - wz
    mat[0, 2] = xz + wy
    mat[1, 0] = xy + wz
    mat[1, 1] = 1.0 - (xx + zz)
    mat[1, 2] = yz - wx
    mat[2, 0] = xz - wy
    mat[2, 1] = yz + wx
    mat[2, 2] = 1.0 - (xx + yy)
    return mat

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
            rospy.loginfo("%s profiling:\n%s", label, s.getvalue())

    def _prime_transforms(self):
        """Resolve extrinsics once at startup (TF preferred, static fallback)."""
        if self.mode == "depth":
            depth_msg = self._wait_for_msg(self.depth_topic, Image)
            depth_frame = depth_msg.header.frame_id if depth_msg else None
            if self.static_target_T_depth is not None:
                self.target_T_depth = self.static_target_T_depth
            elif depth_frame:
                self.target_T_depth = self._lookup_transform(
                    self.target_frame, depth_frame, rospy.Time(0)
                )
        else:
            lidar_msg = self._wait_for_msg(self.lidar_topic, PointCloud2)
            lidar_frame = lidar_msg.header.frame_id if lidar_msg else None
            if self.static_camera_T_lidar is not None:
                self.camera_T_lidar = self.static_camera_T_lidar
            elif lidar_frame:
                self.camera_T_lidar = self._lookup_transform(
                    self.camera_frame, lidar_frame, rospy.Time(0)
                )
            if self.static_target_T_lidar is not None:
                self.target_T_lidar = self.static_target_T_lidar
            elif lidar_frame:
                self.target_T_lidar = self._lookup_transform(
                    self.target_frame, lidar_frame, rospy.Time(0)
                )
        rospy.loginfo(
            "Transforms primed (depth=%s, camera_T_lidar=%s, target_T_lidar=%s)",
            self.target_T_depth is not None,
            self.camera_T_lidar is not None,
            self.target_T_lidar is not None,
        )
        if self.mode == "depth" and self.target_T_depth is None:
            rospy.logwarn("No depth->target transform available at init")
        if self.mode == "lidar":
            if self.camera_T_lidar is None:
                rospy.logwarn("No camera<-lidar transform available at init")
            if self.target_T_lidar is None:
                rospy.logwarn("No target<-lidar transform available at init")

    def _parse_semantic(self, msg):
        data = self._image_to_numpy(msg)
        encoding = msg.encoding.lower()

        if data.ndim == 3:
            if self.color_map is None:
                raise ValueError(
                    "3-channel semantic image requires ~color_map (label -> [r,g,b]) "
                    "to convert colors back to label IDs"
                )
            if self._inverse_color_map is None:
                self._inverse_color_map = self._build_inverse_color_map(self.color_map)

            if encoding == "bgr8":
                b, g, r = data[:, :, 0], data[:, :, 1], data[:, :, 2]
            elif encoding == "rgb8":
                r, g, b = data[:, :, 0], data[:, :, 1], data[:, :, 2]
            elif encoding == "bgra8":
                b, g, r = data[:, :, 0], data[:, :, 1], data[:, :, 2]
            elif encoding == "rgba8":
                r, g, b = data[:, :, 0], data[:, :, 1], data[:, :, 2]
            else:
                raise ValueError(f"Unsupported 3-channel encoding: {msg.encoding}")

            packed = (
                (r.astype(np.uint32) << 16)
                | (g.astype(np.uint32) << 8)
                | b.astype(np.uint32)
            )
            return self._packed_color_to_label(packed, self._inverse_color_map)

        if data.ndim != 2:
            raise ValueError("semantic_topic must be single-channel label image")
        return data

    def _depth_callback(self, sem_msg, depth_msg, conf_msg=None):
        with self._maybe_profile("depth_callback"):
            stamp = (
                sem_msg.header.stamp
                if sem_msg.header.stamp > depth_msg.header.stamp
                else depth_msg.header.stamp
            )
            target_T_depth = self.target_T_depth
            if target_T_depth is None:
                return

            labels = self._parse_semantic(sem_msg)
            depth = self._image_to_numpy(depth_msg).astype(float)
            confidence = (
                self._image_to_numpy(conf_msg).astype(float) if conf_msg else None
            )

            if self.downsample_factor > 1:
                f = self.downsample_factor
                labels = labels[::f, ::f]
                depth = depth[::f, ::f]
                if confidence is not None:
                    confidence = confidence[::f, ::f]
                intrinsics = self._scale_intrinsics(self.intrinsics, f)
            else:
                intrinsics = self.intrinsics

            semantic_obs = SemanticObservation(labels=labels, confidence=confidence)
            depth_obs = DepthObservation(depth=depth)
            pcl = fuse_depth_semantics(
                semantic_obs,
                depth_obs,
                intrinsics,
                target_T_depth,
                include_unlabeled=self.include_unlabeled,
            )
            rospy.loginfo(
                "Publishing depth-based semantic PCL with %d points",
                pcl.points_xyz.shape[0],
            )
            pcl_msg = semantic_pointcloud_to_msg(
                pcl,
                self.target_frame,
                stamp,
                colorize_labels=self.colorize_labels,
                color_map=self.color_map,
            )
            self.pcl_pub.publish(pcl_msg)

    def _lidar_callback(self, sem_msg, lidar_msg, conf_msg=None):
        with self._maybe_profile("lidar_callback"):
            stamp = lidar_msg.header.stamp
            camera_T_lidar = self.camera_T_lidar
            target_T_lidar = self.target_T_lidar
            if camera_T_lidar is None or target_T_lidar is None:
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
            rospy.loginfo(
                "Publishing LiDAR-based semantic PCL with %d points",
                pcl.points_xyz.shape[0],
            )
            pcl_msg = semantic_pointcloud_to_msg(
                pcl,
                self.target_frame,
                stamp,
                colorize_labels=self.colorize_labels,
                color_map=self.color_map,
            )
            self.pcl_pub.publish(pcl_msg)


def main():
    rospy.init_node("semantic_pcl_node")
    node = SemanticPclNode()
    rospy.spin()


if __name__ == "__main__":
    main()
