#!/usr/bin/env python3
"""Tests for the colored PCL timestamp policy."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROS_SRC = Path(__file__).resolve().parents[2] / "entfac_fusion_ros"
if str(ROS_SRC) not in sys.path:
    sys.path.insert(0, str(ROS_SRC))

try:
    import rospy  # type: ignore
except ModuleNotFoundError:
    class _FakeDuration:
        def __init__(self, secs=0.0):
            self._secs = float(secs)

        def to_sec(self):
            return self._secs

    class _FakeTime:
        def __init__(self, secs=0.0):
            self._secs = float(secs)

        def to_sec(self):
            return self._secs

        def __add__(self, other):
            return _FakeTime(self._secs + other.to_sec())

        def __sub__(self, other):
            return _FakeDuration(self._secs - other.to_sec())

        def __gt__(self, other):
            return self._secs > other.to_sec()

        def __lt__(self, other):
            return self._secs < other.to_sec()

        def __eq__(self, other):
            return isinstance(other, _FakeTime) and self._secs == other._secs

        @classmethod
        def from_sec(cls, secs):
            return cls(secs)

    rospy = SimpleNamespace(  # type: ignore[assignment]
        Time=_FakeTime,
        Duration=_FakeDuration,
    )
    sys.modules["rospy"] = rospy

from entfac_fusion_ros.prune.config import SyncConfig
from entfac_fusion_ros.prune.sync_policy import StampPolicy


class MockLogger:
    def __init__(self):
        self.warnings = []

    def warn(self, *args):
        self.warnings.append(args)

    def debug(self, *args, **kwargs):
        pass


class MockNode:
    def __init__(self, mode="depth", cloud_stamp_source="auto"):
        self.mode = mode
        self.cloud_stamp_source = cloud_stamp_source
        self.semantic_time_offset_sec = 0.0
        self.pair_max_dt_sec = 0.03
        self.cloud_time_offset_sec = 0.0
        self.stamp_debug_log_period_sec = 0.0
        self.debug = False
        self._param_meta = {"~cloud_stamp_source": {"value": cloud_stamp_source}}
        self._stamp_debug_last_log_at = 0.0


class StampPolicyTests(unittest.TestCase):
    def _make_policy(self, node):
        config = SyncConfig(
            sync_slop_sec=0.1,
            pair_max_dt_sec=node.pair_max_dt_sec,
            semantic_time_offset_sec=node.semantic_time_offset_sec,
            sync_queue_size=5,
            cloud_time_offset_sec=node.cloud_time_offset_sec,
            cloud_stamp_source=node.cloud_stamp_source,
            stamp_debug_log_period_sec=node.stamp_debug_log_period_sec,
        )
        return StampPolicy(node, config, MockLogger())

    def test_resolve_stamp_source_updates_meta(self):
        node = MockNode(mode="depth", cloud_stamp_source="  Semantic  ")
        policy = self._make_policy(node)

        policy.resolve_cloud_stamp_source()

        self.assertEqual(node.cloud_stamp_source, "semantic")
        self.assertEqual(node._param_meta["~cloud_stamp_source"]["value"], "semantic")

    def test_validate_depth_pair_rejects_large_dt(self):
        node = MockNode(mode="depth")
        node.pair_max_dt_sec = 0.01
        policy = self._make_policy(node)
        policy.resolve_cloud_stamp_source()

        sem_msg = type("Msg", (), {"header": type("Header", (), {"stamp": rospy.Time.from_sec(1.0)})()})()
        depth_msg = type("Msg", (), {"header": type("Header", (), {"stamp": rospy.Time.from_sec(1.2)})()})()

        self.assertIsNone(policy.validate_depth_pair(sem_msg, depth_msg))

    def test_choose_cloud_stamp_midpoint(self):
        node = MockNode(mode="lidar", cloud_stamp_source="midpoint")
        policy = self._make_policy(node)
        policy.resolve_cloud_stamp_source()

        stamp = policy.choose_cloud_stamp(rospy.Time.from_sec(1.0), rospy.Time.from_sec(3.0), "lidar")

        self.assertAlmostEqual(stamp.to_sec(), 2.0)


if __name__ == "__main__":
    unittest.main()
