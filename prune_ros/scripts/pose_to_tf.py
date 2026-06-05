#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Minimal pose -> TF broadcaster for bagged localization topics.

import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image, PointCloud2


class PoseToTF:
    def __init__(self) -> None:
        self.parent_frame = rospy.get_param("~parent_frame", "map")
        self.child_frame = rospy.get_param("~child_frame", "base_link")
        self.pose_topic = rospy.get_param("~pose_topic", "/localization")
        self.input_type = rospy.get_param("~input_type", "odom").strip().lower()
        self.stamp_source_topic = rospy.get_param("~stamp_source_topic", "").strip()
        self.stamp_source_type = (
            rospy.get_param("~stamp_source_type", "image").strip().lower()
        )
        self.stamp_source_max_age_sec = float(
            rospy.get_param("~stamp_source_max_age_sec", 0.02)
        )
        self._last_stamp = None
        self._warned_stamp_source = False
        self._tf_pub = tf2_ros.TransformBroadcaster()

        if self.input_type == "pose":
            msg_type = PoseStamped
            cb = self._pose_cb
        elif self.input_type == "pose_cov":
            msg_type = PoseWithCovarianceStamped
            cb = self._pose_cov_cb
        elif self.input_type == "odom":
            msg_type = Odometry
            cb = self._odom_cb
        else:
            raise ValueError(
                "Unsupported ~input_type. Use 'pose', 'pose_cov', or 'odom'."
            )

        if self.stamp_source_topic:
            self._subscribe_stamp_source()

        rospy.Subscriber(self.pose_topic, msg_type, cb, queue_size=1)
        rospy.loginfo(
            "pose_to_tf: topic=%s type=%s parent=%s child=%s",
            self.pose_topic,
            self.input_type,
            self.parent_frame,
            self.child_frame,
        )

    def _subscribe_stamp_source(self) -> None:
        msg_type = None
        if self.stamp_source_type in ("image", "img"):
            msg_type = Image
        elif self.stamp_source_type in ("pointcloud2", "point_cloud2", "cloud"):
            msg_type = PointCloud2
        elif self.stamp_source_type in ("camerainfo", "camera_info"):
            msg_type = CameraInfo
        else:
            rospy.logwarn(
                "pose_to_tf: unsupported ~stamp_source_type=%r (expected image|pointcloud2|camerainfo)",
                self.stamp_source_type,
            )
            return
        rospy.Subscriber(self.stamp_source_topic, msg_type, self._stamp_cb, queue_size=1)
        rospy.loginfo(
            "pose_to_tf: stamp_source_topic=%s type=%s max_age=%.3fs",
            self.stamp_source_topic,
            self.stamp_source_type,
            self.stamp_source_max_age_sec,
        )

    def _stamp_cb(self, msg) -> None:
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        if stamp is None:
            if not self._warned_stamp_source:
                rospy.logwarn(
                    "pose_to_tf: stamp_source_topic=%s has no header.stamp",
                    self.stamp_source_topic,
                )
                self._warned_stamp_source = True
            return
        self._last_stamp = stamp

    def _resolve_stamp(self, default_stamp: rospy.Time) -> rospy.Time:
        if not self._last_stamp:
            return default_stamp
        if self.stamp_source_max_age_sec <= 0.0:
            return self._last_stamp
        if default_stamp == rospy.Time():
            return self._last_stamp
        delta = abs((self._last_stamp - default_stamp).to_sec())
        if delta <= self.stamp_source_max_age_sec:
            return self._last_stamp
        return default_stamp

    def _send_tf(self, stamp: rospy.Time, pose, frame_id: str) -> None:
        msg = TransformStamped()
        stamp = self._resolve_stamp(stamp)
        msg.header.stamp = stamp if stamp != rospy.Time() else rospy.Time.now()
        msg.header.frame_id = self.parent_frame or frame_id or "map"
        msg.child_frame_id = self.child_frame
        msg.transform.translation.x = pose.position.x
        msg.transform.translation.y = pose.position.y
        msg.transform.translation.z = pose.position.z
        msg.transform.rotation = pose.orientation
        self._tf_pub.sendTransform(msg)

    def _pose_cb(self, msg: PoseStamped) -> None:
        self._send_tf(msg.header.stamp, msg.pose, msg.header.frame_id)

    def _pose_cov_cb(self, msg: PoseWithCovarianceStamped) -> None:
        self._send_tf(msg.header.stamp, msg.pose.pose, msg.header.frame_id)

    def _odom_cb(self, msg: Odometry) -> None:
        self._send_tf(msg.header.stamp, msg.pose.pose, msg.header.frame_id)


def main() -> None:
    rospy.init_node("pose_to_tf", anonymous=False)
    PoseToTF()
    rospy.spin()


if __name__ == "__main__":
    main()
