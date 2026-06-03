"""Startup probing and transform priming for colored PCL."""

from __future__ import annotations

import time
from typing import Any

import rospy
from sensor_msgs.msg import Image, PointCloud2


class StartupBootstrap:
    def __init__(self, node: Any, logger: Any):
        self._node = node
        self._log = logger

    def wait_for_msg(self, topic, msg_type, timeout=2.0, warn_on_timeout=True):
        try:
            self._log.debug(
                "_wait_for_msg",
                "Waiting for %s on topic=%s (timeout=%.2fs)",
                msg_type.__name__,
                topic,
                float(timeout),
            )
            msg = rospy.wait_for_message(topic, msg_type, timeout=timeout)
            self._log.debug(
                "_wait_for_msg",
                "Received %s on topic=%s (stamp=%.6f frame_id=%s)",
                msg_type.__name__,
                topic,
                msg.header.stamp.to_sec() if hasattr(msg, "header") else 0.0,
                getattr(getattr(msg, "header", None), "frame_id", ""),
            )
            return msg
        except rospy.ROSException as exc:
            if warn_on_timeout:
                self._log.warn("_wait_for_msg", "Timeout waiting for %s: %s", topic, exc)
            else:
                self._log.debug("_wait_for_msg", "Timeout waiting for %s: %s", topic, exc)
            return None

    def wait_for_topic_type(self, topic, timeout=2.0, warn_on_timeout=True):
        start = time.time()
        while not rospy.is_shutdown():
            try:
                published = rospy.get_published_topics(namespace="/")
                for name, type_str in published:
                    if name == topic:
                        return type_str
                if (time.time() - start) > float(timeout):
                    raise rospy.ROSException("timeout exceeded while waiting for topic type")
                rospy.sleep(0.05)
            except rospy.ROSException as exc:
                if warn_on_timeout:
                    self._log.warn(
                        "_wait_for_topic_type",
                        "Timeout waiting for topic type on %s: %s",
                        topic,
                        exc,
                    )
                else:
                    self._log.debug(
                        "_wait_for_topic_type",
                        "Timeout waiting for topic type on %s: %s",
                        topic,
                        exc,
                    )
                return None

    def detect_mode(self):
        type_str = None
        wait_start = time.time()
        next_warn = wait_start + 5.0
        while type_str is None and not rospy.is_shutdown():
            type_str = self.wait_for_topic_type(
                self._node.depth_input_topic, timeout=1.0, warn_on_timeout=False
            )
            if type_str is None and time.time() >= next_warn:
                self._log.warn(
                    "_detect_mode",
                    "Waiting for %s to appear to auto-detect mode (expected sensor_msgs/Image or sensor_msgs/PointCloud2)",
                    self._node.depth_input_topic,
                )
                next_warn = time.time() + 5.0

        if type_str == "sensor_msgs/Image":
            self._node._mode_source = "auto"
            self._node._mode_detail = (
                f"auto via ~depth_input_topic={self._node.depth_input_topic} ({type_str})"
            )
            return "depth"
        if type_str == "sensor_msgs/PointCloud2":
            self._node._mode_source = "auto"
            self._node._mode_detail = (
                f"auto via ~depth_input_topic={self._node.depth_input_topic} ({type_str})"
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
            f"Unable to determine message type for ~depth_input_topic={self._node.depth_input_topic}"
        )

    def prime_transforms(self):
        if self._node.mode == "depth":
            if self._node.static_target_T_depth is not None:
                self._node.target_T_depth = self._node.static_target_T_depth
                self._node._depth_frame = "<depth_frame>"
                return
            msg = self.wait_for_msg(self._node.depth_input_topic, Image, timeout=5.0)
            if msg is None:
                self._log.debug(
                    "_prime_transforms",
                    "No depth message available at init; will lookup depth->target on first callback",
                )
                return
            depth_frame = msg.header.frame_id
            self._node._depth_frame = depth_frame or ""
            if depth_frame:
                mat = self._node._lookup_transform(
                    self._node.target_frame, depth_frame, rospy.Time(0)
                )
                if mat is not None:
                    self._node.target_T_depth = mat
            return

        if self._node.static_camera_T_lidar is not None:
            self._node.camera_T_lidar = self._node.static_camera_T_lidar
        if self._node.static_target_T_lidar is not None:
            self._node.target_T_lidar = self._node.static_target_T_lidar
        if self._node.camera_T_lidar is not None and self._node.target_T_lidar is not None:
            return

        lidar_msg = self.wait_for_msg(self._node.depth_input_topic, PointCloud2, timeout=5.0)
        if lidar_msg is None:
            self._log.debug(
                "_prime_transforms",
                "No LiDAR message available at init; will lookup transforms on first callback",
            )
            return
        lidar_frame = lidar_msg.header.frame_id
        self._node._lidar_frame = lidar_frame or ""
        if not lidar_frame:
            self._log.warn("_prime_transforms", "LiDAR message has empty frame_id")
            return
        if self._node.camera_T_lidar is None:
            mat = self._node._lookup_transform(self._node.camera_frame, lidar_frame, rospy.Time(0))
            if mat is not None:
                self._node.camera_T_lidar = mat
        if self._node.target_T_lidar is None:
            mat = self._node._lookup_transform(self._node.target_frame, lidar_frame, rospy.Time(0))
            if mat is not None:
                self._node.target_T_lidar = mat
