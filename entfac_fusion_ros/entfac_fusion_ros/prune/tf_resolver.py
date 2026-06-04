"""TF lookup and transform priming helpers for prune."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import rospy
import tf2_ros

from entfac_fusion_ros.tf_utils import format_matrix, transform_stamped_to_matrix


class TransformResolver:
    def __init__(self, node: Any, logger: Any):
        self._node = node
        self._log = logger
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_cache: Dict[Tuple[str, str], Tuple[np.ndarray, rospy.Time]] = {}

    def lookup(self, target_frame: str, source_frame: str, stamp: rospy.Time):
        cache_key = (target_frame, source_frame)
        if stamp == rospy.Time(0) and cache_key in self.tf_cache:
            cached_mat, _ = self.tf_cache[cache_key]
            self._log.debug("_lookup_transform", "TF cache hit %s -> %s", source_frame, target_frame)
            return cached_mat
        mat, tf_stamp = self.lookup_with_stamp(target_frame, source_frame, stamp)
        return mat

    def lookup_with_stamp(self, target_frame: str, source_frame: str, stamp: rospy.Time):
        cache_key = (target_frame, source_frame)
        if stamp == rospy.Time(0) and cache_key in self.tf_cache:
            cached_mat, cached_stamp = self.tf_cache[cache_key]
            self._log.debug("_lookup_transform", "TF cache hit %s -> %s", source_frame, target_frame)
            return cached_mat, cached_stamp
        try:
            tf_msg = self.tf_buffer.lookup_transform(target_frame, source_frame, stamp, rospy.Duration(0.1))
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            self._log.warn("_lookup_transform", "TF lookup failed (%s -> %s): %s", source_frame, target_frame, exc)
            return None, None
        try:
            mat = transform_stamped_to_matrix(tf_msg)
        except ValueError as exc:
            self._log.warn("_lookup_transform", "Rejected TF (%s -> %s): %s", source_frame, target_frame, exc)
            return None, None
        if stamp == rospy.Time(0):
            self.tf_cache[cache_key] = (mat, tf_msg.header.stamp)
        self._log.debug("_lookup_transform", "TF %s -> %s:\n%s", source_frame, target_frame, format_matrix(mat))
        return mat, tf_msg.header.stamp
