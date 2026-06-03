"""Pure math helpers for pose_to_tf."""

from __future__ import annotations

import math
from typing import Iterable


def quaternion_multiply(
    a: Iterable[float], b: Iterable[float]
) -> tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def normalize_quaternion(
    q: Iterable[float],
) -> tuple[float, float, float, float]:
    x, y, z, w = q
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / norm, y / norm, z / norm, w / norm)


def apply_yaw_offset_deg(
    quaternion_xyzw: Iterable[float], yaw_offset_deg: float
) -> tuple[float, float, float, float]:
    """Rotate a quaternion by a yaw-only offset in degrees."""
    if yaw_offset_deg == 0.0:
        return tuple(quaternion_xyzw)  # type: ignore[return-value]
    half = math.radians(yaw_offset_deg) * 0.5
    yaw_delta = (0.0, 0.0, math.sin(half), math.cos(half))
    return normalize_quaternion(quaternion_multiply(quaternion_xyzw, yaw_delta))
