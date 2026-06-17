#!/usr/bin/env python3
"""Unit tests for IMU cache interpolation."""

import pytest
import numpy as np
import rospy
from sensor_msgs.msg import Imu
from prune_ros.runtime import interpolate_imu_msg


class TestInterpolateImuMsg:
    """Test unified IMU interpolation logic."""

    def _make_imu(self, t: float, omega: list, accel: list) -> Imu:
        """Create an Imu message with given timestamp and values."""
        msg = Imu()
        msg.header.stamp = rospy.Time(t)
        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = omega
        msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = accel
        return msg

    def test_both_available_interpolate(self):
        """Interpolate between two IMU messages at midpoint."""
        before = self._make_imu(0.0, [0, 0, 0], [0, 0, 0])
        after = self._make_imu(1.0, [10, 0, 0], [10, 0, 0])
        stamp = rospy.Time(0.5)

        omega, accel, dt = interpolate_imu_msg(before, after, stamp)

        assert omega is not None
        assert accel is not None
        np.testing.assert_array_almost_equal(omega, [5, 0, 0])
        np.testing.assert_array_almost_equal(accel, [5, 0, 0])
        assert abs(dt - 0.5) < 1e-6

    def test_both_available_interpolate_3d(self):
        """Interpolate 3D motion correctly."""
        before = self._make_imu(0.0, [1, 2, 3], [4, 5, 6])
        after = self._make_imu(2.0, [5, 6, 7], [12, 14, 16])
        stamp = rospy.Time(1.0)

        omega, accel, dt = interpolate_imu_msg(before, after, stamp)

        np.testing.assert_array_almost_equal(omega, [3, 4, 5])
        np.testing.assert_array_almost_equal(accel, [8, 9.5, 11])

    def test_before_only(self):
        """Use before sample when after unavailable."""
        before = self._make_imu(0.0, [1, 2, 3], [4, 5, 6])
        stamp = rospy.Time(0.5)

        omega, accel, dt = interpolate_imu_msg(before, None, stamp)

        np.testing.assert_array_almost_equal(omega, [1, 2, 3])
        np.testing.assert_array_almost_equal(accel, [4, 5, 6])
        assert abs(dt - 0.5) < 1e-6

    def test_after_only(self):
        """Use after sample when before unavailable."""
        after = self._make_imu(1.0, [7, 8, 9], [10, 11, 12])
        stamp = rospy.Time(0.5)

        omega, accel, dt = interpolate_imu_msg(None, after, stamp)

        np.testing.assert_array_almost_equal(omega, [7, 8, 9])
        np.testing.assert_array_almost_equal(accel, [10, 11, 12])
        assert abs(dt - 0.5) < 1e-6

    def test_neither_available(self):
        """Return None when no samples available."""
        omega, accel, dt = interpolate_imu_msg(None, None, rospy.Time(0.5))

        assert omega is None
        assert accel is None
        assert dt == float("inf")

    def test_zero_dt_uses_before(self):
        """When dt=0 between samples, use before."""
        before = self._make_imu(0.0, [1, 0, 0], [0, 0, 0])
        after = self._make_imu(0.0, [2, 0, 0], [0, 0, 0])
        stamp = rospy.Time(0.0)

        omega, accel, dt = interpolate_imu_msg(before, after, stamp)

        np.testing.assert_array_almost_equal(omega, [1, 0, 0])

    def test_clamp_alpha_to_bounds(self):
        """Clamp interpolation parameter to [0, 1]."""
        before = self._make_imu(1.0, [0, 0, 0], [0, 0, 0])
        after = self._make_imu(2.0, [10, 0, 0], [10, 0, 0])

        # Query before interval while keeping ROS time non-negative
        omega, _, _ = interpolate_imu_msg(before, after, rospy.Time(0.0))
        np.testing.assert_array_almost_equal(omega, [0, 0, 0])

        # Query after interval
        omega, _, _ = interpolate_imu_msg(before, after, rospy.Time(2.0))
        np.testing.assert_array_almost_equal(omega, [10, 0, 0])

    def test_best_dt_calculation(self):
        """Verify best_dt is minimum to either sample."""
        before = self._make_imu(0.0, [1, 0, 0], [0, 0, 0])
        after = self._make_imu(10.0, [2, 0, 0], [0, 0, 0])
        stamp = rospy.Time(2.0)

        _, _, dt = interpolate_imu_msg(before, after, stamp)

        assert abs(dt - 2.0) < 1e-6  # min(2.0, 8.0)

    def test_integration_rolling_shutter(self):
        """Test realistic rolling shutter scenario."""
        # IMU running at 100 Hz
        imu_samples = [
            self._make_imu(0.0, [0, 0, 1], [0, 0, 10]),
            self._make_imu(0.01, [0, 0, 1], [0, 0, 10]),
        ]

        # Camera frame arrives at 0.005 (between samples)
        frame_stamp = rospy.Time(0.005)
        omega, accel, dt = interpolate_imu_msg(imu_samples[0], imu_samples[1], frame_stamp)

        assert omega is not None
        assert abs(omega[2] - 1.0) < 1e-6  # Angular velocity
        assert abs(accel[2] - 10.0) < 1e-6  # Linear acceleration

    def test_integration_lidar_deskew(self):
        """Test realistic LiDAR deskew scenario."""
        # IMU at beginning and end of scan
        scan_start = self._make_imu(0.0, [0.1, 0.05, 0.02], [0, 0, -10])
        scan_end = self._make_imu(0.1, [0.1, 0.05, 0.02], [0, 0, -10])

        # Query at midpoint
        stamp = rospy.Time(0.05)
        omega, accel, dt = interpolate_imu_msg(scan_start, scan_end, stamp)

        np.testing.assert_array_almost_equal(omega, [0.1, 0.05, 0.02])
        np.testing.assert_array_almost_equal(accel, [0, 0, -10])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
