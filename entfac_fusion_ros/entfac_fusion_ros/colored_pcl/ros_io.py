"""ROS wiring helpers for colored PCL."""

from __future__ import annotations

from typing import Any

import rospy
from message_filters import ApproximateTimeSynchronizer, Cache, Subscriber
from sensor_msgs.msg import Image, Imu, PointCloud2
from std_msgs.msg import Float32
from std_srvs.srv import SetBool, Trigger


class ColoredPclRosIo:
    def __init__(self, node: Any):
        self._node = node

    def setup_publishers(self) -> None:
        self._node.pcl_pub = rospy.Publisher("semantic_pointcloud", PointCloud2, queue_size=1)
        self._node._debug_proj_pub = None
        if self._node.debug_project_lidar:
            self._node._debug_proj_pub = rospy.Publisher(self._node.debug_projected_topic, Image, queue_size=1)
        self._node._debug_depth_pub = None
        self._node._debug_edge_pub = None
        self._node._debug_heatmap_pub = None
        self._node._debug_score_pub = None
        self._node._debug_tracked_reprojection_pub = None
        self._node._debug_tracked_reprojection_error_pub = None
        self._node._debug_fov_points_pub = None
        if self._node.debug_range_view:
            self._node._debug_depth_pub = rospy.Publisher(self._node.debug_lidar_depth_topic, Image, queue_size=1)
            self._node._debug_edge_pub = rospy.Publisher(self._node.debug_lidar_edge_topic, Image, queue_size=1)
            self._node._debug_heatmap_pub = rospy.Publisher(self._node.debug_reprojection_heatmap_topic, Image, queue_size=1)
            self._node._debug_score_pub = rospy.Publisher(self._node.debug_alignment_score_topic, Float32, queue_size=1)
        if self._node.tracked_reprojection_enable:
            self._node._debug_tracked_reprojection_pub = rospy.Publisher(self._node.debug_tracked_reprojection_topic, Image, queue_size=1)
            self._node._debug_tracked_reprojection_error_pub = rospy.Publisher(self._node.debug_tracked_reprojection_error_topic, Float32, queue_size=1)
        if self._node.debug_publish_fov_points:
            self._node._debug_fov_points_pub = rospy.Publisher(self._node.debug_fov_points_topic, PointCloud2, queue_size=1)
        self._node._debug_calibration_health_pub = None
        self._node._debug_calibration_uncertainty_pub = None
        if self._node.online_calibration_enable:
            self._node._debug_calibration_health_pub = rospy.Publisher(self._node.debug_calibration_health_topic, Float32, queue_size=1)
            self._node._debug_calibration_uncertainty_pub = rospy.Publisher(self._node.debug_calibration_uncertainty_topic, Float32, queue_size=1)

    def setup_services(self) -> None:
        self._node._srv_set_record = rospy.Service("~set_ply_recording", SetBool, self._node._srv_set_ply_recording)
        self._node._srv_save_ply = rospy.Service("~save_ply", Trigger, self._node._srv_save_ply)

    def register_subscribers(self) -> None:
        semantic_sub = Subscriber(self._node.semantic_topic, Image, queue_size=self._node.sync_queue_size)
        conf_sub = Subscriber(self._node.conf_topic, Image, queue_size=self._node.sync_queue_size) if self._node.conf_topic else None
        invalid_mask_sub = Subscriber(self._node.projection_invalid_mask_topic, Image, queue_size=self._node.sync_queue_size) if self._node.projection_invalid_mask_topic else None
        self._node._configure_rolling_shutter_subscribers()
        self._node._configure_lidar_deskew_subscribers()
        if self._node.mode == "depth":
            depth_sub = Subscriber(self._node.depth_input_topic, Image)
            subs = [semantic_sub, depth_sub]
            if conf_sub is not None:
                subs.append(conf_sub)
            if invalid_mask_sub is not None:
                subs.append(invalid_mask_sub)
            sync = ApproximateTimeSynchronizer(subs, queue_size=self._node.sync_queue_size, slop=self._node.sync_slop_sec)
            sync.registerCallback(self._node._build_depth_sync_callback(conf_sub, invalid_mask_sub))
        else:
            lidar_sub = Subscriber(self._node.depth_input_topic, PointCloud2)
            subs = [semantic_sub, lidar_sub]
            if conf_sub is not None:
                subs.append(conf_sub)
            if invalid_mask_sub is not None:
                subs.append(invalid_mask_sub)
            sync = ApproximateTimeSynchronizer(subs, queue_size=self._node.sync_queue_size, slop=self._node.sync_slop_sec)
            sync.registerCallback(self._node._build_lidar_sync_callback(conf_sub, invalid_mask_sub))
        self._node._sync = sync

