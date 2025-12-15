#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: MIT License (open source, free to modify and redistribute)
# Repository: fruc_ros_utils
#
# Description:
#   ROS wrapper that turns semantic + geometry inputs into semantic PointCloud2 outputs.

"""ROS wrapper that converts semantic + geometry into semantic PointCloud2."""

import cProfile
import io
import pstats
from contextlib import contextmanager
import rostopic

import numpy as np
import rospy
import tf2_ros
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs import point_cloud2
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header

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

    # Convert quaternion to rotation matrix
    x, y, z, w = quat
    rot = np.array(
        [
            [
                1 - 2 * (y**2 + z**2),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
            ],
            [
                2 * (x * y + z * w),
                1 - 2 * (x**2 + z**2),
                2 * (y * z - x * w),
            ],
            [
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x**2 + y**2),
            ],
        ],
        dtype=float,
    )

    mat = np.eye(4, dtype=float)
    mat[:3, :3] = rot
    mat[:3, 3] = translation
    return require_homogeneous_transform(mat)


def semantic_pointcloud_to_msg(pcl, frame_id, stamp):
    """Convert SemanticPointCloud dataclass to PointCloud2."""
    has_conf = pcl.confidence is not None
    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
        PointField("label", 12, PointField.UINT16, 1),
    ]
    if has_conf:
        fields.append(PointField("confidence", 16, PointField.FLOAT32, 1))

    header = Header()
    header.frame_id = frame_id
    header.stamp = stamp

    if has_conf:
        points = [
            (float(x), float(y), float(z), int(lbl), float(conf))
            for (x, y, z), lbl, conf in zip(pcl.points_xyz, pcl.labels, pcl.confidence)
        ]
    else:
        points = [
            (float(x), float(y), float(z), int(lbl))
            for (x, y, z), lbl in zip(pcl.points_xyz, pcl.labels)
        ]

    return point_cloud2.create_cloud(header, fields, points)


class SemanticPclNode:
    """ROS node bridging topics to the numpy fusion core."""

    def __init__(self):
        # 'depth' uses aligned depth image; 'lidar' projects LiDAR into image.
        self.mode = rospy.get_param("~mode", "").lower()
        self.target_frame = rospy.get_param("~target_frame", "base_link")
        self.semantic_topic = rospy.get_param("~semantic_topic")
        self.conf_topic = rospy.get_param("~confidence_topic", "")
        self.camera_info_topic = rospy.get_param("~camera_info")
        self.depth_topic = rospy.get_param("~depth_topic", "")
        self.lidar_topic = rospy.get_param("~lidar_topic", "")
        self.include_unlabeled = rospy.get_param("~include_unlabeled_pts", False)
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
        mode = None
        if self.depth_topic:
            cls, _, _ = rostopic.get_topic_class(self.depth_topic, blocking=False)
            if cls is not None and issubclass(cls, Image):
                mode = "depth"
        if self.lidar_topic:
            cls, _, _ = rostopic.get_topic_class(self.lidar_topic, blocking=False)
            if cls is not None and issubclass(cls, PointCloud2):
                if mode and mode != "lidar":
                    rospy.logwarn(
                        "Both depth and lidar topics valid; defaulting to depth"
                    )
                else:
                    mode = "lidar"
        if mode is None:
            raise ValueError(
                "Unable to auto-detect mode; set ~depth_topic or ~lidar_topic"
            )
        return mode

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
        if encoding in ("32fc1", "32fc"):
            dtype = np.float32
        elif encoding in ("32sc1",):
            dtype = np.int32
        elif encoding in ("16uc1", "16sc1", "mono16"):
            dtype = np.uint16
        elif encoding in ("8uc1", "mono8"):
            dtype = np.uint8
        else:
            raise ValueError(f"Unsupported image encoding: {msg.encoding}")
        arr = np.frombuffer(msg.data, dtype=dtype)
        arr = arr.reshape(msg.height, msg.width)
        return arr

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
        labels = self._image_to_numpy(msg)
        if labels.ndim != 2:
            raise ValueError("semantic_topic must be single-channel label image")
        return labels

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
            pcl_msg = semantic_pointcloud_to_msg(pcl, self.target_frame, stamp)
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
            pcl_msg = semantic_pointcloud_to_msg(pcl, self.target_frame, stamp)
            self.pcl_pub.publish(pcl_msg)


def main():
    rospy.init_node("semantic_pcl_node")
    node = SemanticPclNode()
    rospy.spin()


if __name__ == "__main__":
    main()
