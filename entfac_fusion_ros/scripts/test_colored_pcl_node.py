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
#   Minimal ROS integration test for colored_pcl_node (rostest).

import unittest

import numpy as np
import rospy
import rostest
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs import point_cloud2


class ColoredPclNodeRosTest(unittest.TestCase):
    def setUp(self):
        rospy.init_node("test_colored_pcl_node", anonymous=True)
        self._got_pcl = None
        self._pcl_sub = rospy.Subscriber(
            "/semantic_pointcloud", PointCloud2, self._on_pcl, queue_size=1
        )
        self._sem_pub = rospy.Publisher("/test/semantic_labels", Image, queue_size=1)
        self._depth_pub = rospy.Publisher("/test/depth", Image, queue_size=1)
        self._cam_pub = rospy.Publisher("/test/camera_info", CameraInfo, queue_size=1)

        deadline = rospy.Time.now() + rospy.Duration(2.0)
        while (
            self._cam_pub.get_num_connections() < 1
            and rospy.Time.now() < deadline
            and not rospy.is_shutdown()
        ):
            rospy.sleep(0.05)

    def _on_pcl(self, msg):
        self._got_pcl = msg

    @staticmethod
    def _make_image_mono8(labels: np.ndarray, stamp: rospy.Time) -> Image:
        labels_u8 = np.asarray(labels, dtype=np.uint8)
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = "base_link"
        msg.height = int(labels_u8.shape[0])
        msg.width = int(labels_u8.shape[1])
        msg.encoding = "mono8"
        msg.is_bigendian = False
        msg.step = msg.width * 1
        msg.data = labels_u8.tobytes()
        return msg

    @staticmethod
    def _make_image_32fc1(depth_m: np.ndarray, stamp: rospy.Time) -> Image:
        depth_f = np.asarray(depth_m, dtype=np.float32)
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = "base_link"
        msg.height = int(depth_f.shape[0])
        msg.width = int(depth_f.shape[1])
        msg.encoding = "32FC1"
        msg.is_bigendian = False
        msg.step = msg.width * 4
        msg.data = depth_f.tobytes()
        return msg

    @staticmethod
    def _make_camera_info(stamp: rospy.Time) -> CameraInfo:
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = "base_link"
        msg.width = 2
        msg.height = 2
        msg.K = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        return msg

    def test_node_publishes_one_cloud(self):
        stamp = rospy.Time.now()
        cam = self._make_camera_info(stamp)
        self._cam_pub.publish(cam)

        labels = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        depth = np.array([[1.0, 0.0], [2.0, np.nan]], dtype=np.float32)
        self._sem_pub.publish(self._make_image_mono8(labels, stamp))
        self._depth_pub.publish(self._make_image_32fc1(depth, stamp))

        deadline = rospy.Time.now() + rospy.Duration(3.0)
        while self._got_pcl is None and rospy.Time.now() < deadline:
            rospy.sleep(0.05)

        self.assertIsNotNone(self._got_pcl, "no PointCloud2 published")
        pcl = self._got_pcl
        self.assertEqual(pcl.header.frame_id, "base_link")
        self.assertEqual(pcl.width, 2)

        pts = list(point_cloud2.read_points(pcl, field_names=("x", "y", "z", "label")))
        self.assertEqual(len(pts), 2)
        # Expected labels: pixels (0,0) and (1,0) are valid.
        self.assertEqual(int(pts[0][3]), 1)
        self.assertEqual(int(pts[1][3]), 3)

        self.assertAlmostEqual(float(pts[0][0]), 0.0, places=6)
        self.assertAlmostEqual(float(pts[0][1]), 0.0, places=6)
        self.assertAlmostEqual(float(pts[0][2]), 1.0, places=6)

        self.assertAlmostEqual(float(pts[1][0]), 0.0, places=6)
        self.assertAlmostEqual(float(pts[1][1]), 2.0, places=6)
        self.assertAlmostEqual(float(pts[1][2]), 2.0, places=6)


if __name__ == "__main__":
    rostest.rosrun("entfac_fusion_ros", "test_colored_pcl_node", ColoredPclNodeRosTest)
