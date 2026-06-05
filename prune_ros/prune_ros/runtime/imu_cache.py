#!/usr/bin/env python3
"""IMU Cache Management for rolling shutter and LiDAR deskew corrections."""

from typing import Optional, Tuple
import numpy as np
import rospy
from sensor_msgs.msg import Imu


def interpolate_imu_msg(
    before: Optional[Imu], after: Optional[Imu], stamp: rospy.Time
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """Interpolate IMU message (omega + accel). Returns (omega, accel, best_dt).

    This is the unified 3-way cache interpolation used by both rolling shutter
    (camera IMU) and LiDAR deskew (LiDAR IMU) paths. Handles cases where only
    before or after samples are available, or neither (cache miss).
    """
    if before is None and after is None:
        return None, None, float("inf")

    if before is None:
        omega = np.array(
            [after.angular_velocity.x, after.angular_velocity.y, after.angular_velocity.z],
            dtype=float,
        )
        accel = np.array(
            [after.linear_acceleration.x, after.linear_acceleration.y, after.linear_acceleration.z],
            dtype=float,
        )
        best_dt = abs((stamp - after.header.stamp).to_sec())
        return omega, accel, best_dt

    if after is None:
        omega = np.array(
            [before.angular_velocity.x, before.angular_velocity.y, before.angular_velocity.z],
            dtype=float,
        )
        accel = np.array(
            [before.linear_acceleration.x, before.linear_acceleration.y, before.linear_acceleration.z],
            dtype=float,
        )
        best_dt = abs((stamp - before.header.stamp).to_sec())
        return omega, accel, best_dt

    dt = (after.header.stamp - before.header.stamp).to_sec()
    if dt > 0.0:
        alpha = (stamp - before.header.stamp).to_sec() / dt
        alpha = float(np.clip(alpha, 0.0, 1.0))
        omega_b = np.array(
            [before.angular_velocity.x, before.angular_velocity.y, before.angular_velocity.z],
            dtype=float,
        )
        omega_a = np.array(
            [after.angular_velocity.x, after.angular_velocity.y, after.angular_velocity.z],
            dtype=float,
        )
        accel_b = np.array(
            [before.linear_acceleration.x, before.linear_acceleration.y, before.linear_acceleration.z],
            dtype=float,
        )
        accel_a = np.array(
            [after.linear_acceleration.x, after.linear_acceleration.y, after.linear_acceleration.z],
            dtype=float,
        )
        omega = (1.0 - alpha) * omega_b + alpha * omega_a
        accel = (1.0 - alpha) * accel_b + alpha * accel_a
        best_dt = min(
            abs((stamp - before.header.stamp).to_sec()),
            abs((after.header.stamp - stamp).to_sec()),
        )
    else:
        omega = np.array(
            [before.angular_velocity.x, before.angular_velocity.y, before.angular_velocity.z],
            dtype=float,
        )
        accel = np.array(
            [before.linear_acceleration.x, before.linear_acceleration.y, before.linear_acceleration.z],
            dtype=float,
        )
        best_dt = abs((stamp - before.header.stamp).to_sec())

    return omega, accel, best_dt
