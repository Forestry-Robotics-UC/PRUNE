#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Duda Andrada
# License: GNU General Public License v3.0 (GPL-3.0)

"""ROS debug publishers for the LiDAR-camera fusion pipeline.

This module is the only Phase-2/3/4 module that imports ROS.  All heavy
numpy computation has been moved to the subsystems that call into this one;
the publisher is a thin "render arrays → publish ROS messages" layer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rospy
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Float32

from prune_core.calibration import CalibrationHealthSnapshot
from prune_ros.calibration import (
    _edge_alignment_score,
    compute_semantic_edge_map,
)


@dataclass
class DebugPublisherParams:
    # Projection overlay
    debug_project_lidar: bool = False
    debug_project_lidar_stride: int = 5
    debug_project_lidar_radius: int = 0
    debug_project_lidar_outline_only: bool = False
    # Range-view (depth + edge + heatmap + score)
    debug_range_view: bool = False
    # FOV point cloud
    debug_publish_fov_points: bool = False
    # Tracked reprojection
    tracked_reprojection_enable: bool = False
    # Calibration health
    online_calibration_enable: bool = False
    # File saving
    debug_output_dir: str = ""
    debug_output_stride: int = 20


class DebugPublisher:
    """Creates and manages all optional debug ROS publishers.

    Only publishers whose corresponding flag is enabled are created.

    Args:
        params: Debug flags and settings.
        node_name: Used for topic names and log context.
        lidar_frame: Frame ID for the FOV-points cloud.
        target_frame: Frame ID for output topics.
        splat_fn: Callable matching the signature of
            ``_splat_reprojection_confidence`` — passed in so this class
            doesn't need to duplicate that logic.
    """

    def __init__(
        self,
        params: DebugPublisherParams,
        node_name: str,
        lidar_frame: str = "",
        target_frame: str = "",
    ) -> None:
        self._p = params
        self._node_name = node_name
        self._lidar_frame = lidar_frame
        self._target_frame = target_frame
        self._callback_seq: int = 0

        self._proj_pub: Optional[rospy.Publisher] = None
        self._depth_pub: Optional[rospy.Publisher] = None
        self._edge_pub: Optional[rospy.Publisher] = None
        self._heatmap_pub: Optional[rospy.Publisher] = None
        self._score_pub: Optional[rospy.Publisher] = None
        self._tracked_pub: Optional[rospy.Publisher] = None
        self._tracked_err_pub: Optional[rospy.Publisher] = None
        self._fov_pts_pub: Optional[rospy.Publisher] = None
        self._calib_health_pub: Optional[rospy.Publisher] = None
        self._calib_uncertainty_pub: Optional[rospy.Publisher] = None

        if params.debug_project_lidar:
            self._proj_pub = rospy.Publisher("/debug/lidar_projection", Image, queue_size=1)
        if params.debug_range_view:
            self._depth_pub = rospy.Publisher("/debug/lidar_depth", Image, queue_size=1)
            self._edge_pub = rospy.Publisher("/debug/lidar_edge", Image, queue_size=1)
            self._heatmap_pub = rospy.Publisher("/debug/reprojection_heatmap", Image, queue_size=1)
            self._score_pub = rospy.Publisher("/debug/alignment_score", Float32, queue_size=1)
        if params.tracked_reprojection_enable:
            self._tracked_pub = rospy.Publisher("/debug/tracked_reprojection", Image, queue_size=1)
            self._tracked_err_pub = rospy.Publisher(
                "/debug/tracked_reprojection_error_px", Float32, queue_size=1
            )
        if params.debug_publish_fov_points:
            self._fov_pts_pub = rospy.Publisher("/debug/lidar_points_in_fov", PointCloud2, queue_size=1)
        if params.online_calibration_enable:
            self._calib_health_pub = rospy.Publisher("/debug/calibration_health", Float32, queue_size=1)
            self._calib_uncertainty_pub = rospy.Publisher(
                "/debug/calibration_uncertainty", Float32, queue_size=1
            )

    def tick(self) -> None:
        """Increment the frame counter (call once per callback)."""
        self._callback_seq += 1

    def update_params(self, params: DebugPublisherParams) -> None:
        """Refresh tunable display params (stride, radius, etc.)."""
        self._p = params

    # ------------------------------------------------------------------
    # Publishing methods
    # ------------------------------------------------------------------

    def publish_lidar_projection(
        self,
        base_rgb: np.ndarray,
        image_shape: Tuple[int, int],
        uv: np.ndarray,
        header,
        colors_u8: Optional[np.ndarray] = None,
    ) -> None:
        if self._proj_pub is None or uv is None or uv.size == 0:
            return
        p = self._p
        h, w = image_shape
        img = np.ascontiguousarray(
            (base_rgb.copy() if base_rgb is not None else np.zeros((h, w, 3), dtype=np.uint8))
        )
        uv_int = np.round(uv).astype(np.int32)
        uu, vv = uv_int[:, 0], uv_int[:, 1]
        in_bounds = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
        uu, vv = uu[in_bounds], vv[in_bounds]
        if colors_u8 is not None:
            colors_u8 = np.asarray(colors_u8, dtype=np.uint8)[in_bounds]
        stride = int(p.debug_project_lidar_stride)
        if stride > 1 and uu.size:
            uu = uu[::stride]
            vv = vv[::stride]
            if colors_u8 is not None:
                colors_u8 = colors_u8[::stride]
        radius = int(p.debug_project_lidar_radius)
        if uu.size:
            if radius <= 0:
                img[vv, uu] = colors_u8 if colors_u8 is not None else np.array([255, 0, 0])
            else:
                for i in range(uu.size):
                    color = tuple(int(c) for c in colors_u8[i]) if colors_u8 is not None else (255, 0, 0)
                    u0, u1 = max(0, int(uu[i]) - radius), min(w - 1, int(uu[i]) + radius)
                    v0, v1 = max(0, int(vv[i]) - radius), min(h - 1, int(vv[i]) + radius)
                    if p.debug_project_lidar_outline_only:
                        img[v0, u0:u1+1] = color; img[v1, u0:u1+1] = color
                        img[v0:v1+1, u0] = color; img[v0:v1+1, u1] = color
                    else:
                        img[v0:v1+1, u0:u1+1] = color
        self._proj_pub.publish(self._make_image_msg(img, header))
        self._save_rgb("lidar_projection", img, header)

    def publish_range_view(
        self,
        *,
        depth_map: np.ndarray,
        edge_map: np.ndarray,
        sem_img: np.ndarray,
        sem_type: str,
        u: np.ndarray,
        v: np.ndarray,
        point_confidence: Optional[np.ndarray],
        header,
    ) -> None:
        if self._depth_pub is None:
            return
        sem_edges = compute_semantic_edge_map(sem_img, sem_type)
        score = _edge_alignment_score(sem_edges, edge_map)
        heat = _splat_confidence(u, v, point_confidence, depth_map.shape)
        self._publish_rgb_image(self._depth_pub, _depth_to_rgb(depth_map), header, "range_depth")
        self._publish_rgb_image(self._edge_pub, _edge_to_rgb(edge_map), header, "range_edge")
        self._publish_rgb_image(self._heatmap_pub, _float_to_heatmap(heat), header, "range_heatmap")
        self._score_pub.publish(Float32(data=score))

    def publish_tracked_reprojection(
        self,
        overlay_img: np.ndarray,
        error_px: float,
        header,
    ) -> None:
        if self._tracked_pub is None:
            return
        self._publish_rgb_image(self._tracked_pub, overlay_img, header, "tracked_reprojection")
        self._tracked_err_pub.publish(Float32(data=float(error_px)))

    def publish_fov_points(
        self,
        points: np.ndarray,
        frame_id: str,
        stamp,
    ) -> None:
        if self._fov_pts_pub is None or not points.shape[0]:
            return
        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id or self._lidar_frame
        msg.height = 1
        msg.width = int(points.shape[0])
        msg.is_dense = True
        msg.is_bigendian = False
        fields = [
            PointField(name="x", offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        msg.fields = fields
        msg.point_step = 12
        msg.row_step = 12 * msg.width
        msg.data = np.ascontiguousarray(points, dtype=np.float32).tobytes()
        self._fov_pts_pub.publish(msg)

    def publish_calibration_health(self, snapshot: CalibrationHealthSnapshot) -> None:
        if self._calib_health_pub is not None:
            self._calib_health_pub.publish(Float32(data=float(snapshot.health)))
        if self._calib_uncertainty_pub is not None:
            self._calib_uncertainty_pub.publish(Float32(data=float(snapshot.uncertainty)))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _publish_rgb_image(self, pub, rgb_u8: np.ndarray, header, save_kind: str) -> None:
        if pub is None:
            return
        pub.publish(self._make_image_msg(rgb_u8, header))
        self._save_rgb(save_kind, rgb_u8, header)

    def _make_image_msg(self, rgb_u8: np.ndarray, header) -> Image:
        h, w = rgb_u8.shape[:2]
        msg = Image()
        msg.header = header
        msg.height = h
        msg.width = w
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = w * 3
        msg.data = np.ascontiguousarray(rgb_u8.astype(np.uint8)).tobytes()
        return msg

    def _save_rgb(self, kind: str, rgb_u8: np.ndarray, header) -> None:
        p = self._p
        if not p.debug_output_dir or p.debug_output_stride < 1:
            return
        if self._callback_seq <= 0:
            return
        if ((self._callback_seq - 1) % int(p.debug_output_stride)) != 0:
            return
        try:
            import cv2
        except ImportError:
            return
        try:
            stamp_ns = int(header.stamp.to_nsec())
        except Exception:  # noqa: BLE001
            stamp_ns = int(time.time() * 1e9)
        out_dir = Path(p.debug_output_dir) / kind
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{kind}_{stamp_ns:019d}_{int(self._callback_seq):06d}.png"
        rgb = np.ascontiguousarray(np.asarray(rgb_u8, dtype=np.uint8))
        if rgb.ndim == 3 and rgb.shape[2] == 3:
            cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# Pure-numpy rendering helpers
# ---------------------------------------------------------------------------


def _float_to_heatmap(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    lo, hi = float(np.min(arr)), float(np.max(arr))
    t = (arr - lo) / (hi - lo + 1e-9) if hi > lo else np.zeros_like(arr)
    t = np.clip(t, 0.0, 1.0)
    r = ((1.0 - t) * 255.0).astype(np.uint8)
    g = (t * 255.0).astype(np.uint8)
    return np.stack((r, g, np.zeros_like(r)), axis=-1)


def _depth_to_rgb(depth_map: np.ndarray) -> np.ndarray:
    depth_map = np.asarray(depth_map, dtype=np.float32)
    valid = np.isfinite(depth_map) & (depth_map > 0.0)
    if not np.any(valid):
        h, w = depth_map.shape
        return np.zeros((h, w, 3), dtype=np.uint8)
    vals = depth_map[valid]
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi <= lo:
        hi = lo + 1e-3
    norm = np.zeros_like(depth_map)
    norm[valid] = (depth_map[valid] - lo) / (hi - lo)
    rgb = _float_to_heatmap(norm)
    rgb[~valid] = 0
    return rgb


def _edge_to_rgb(edge_map: np.ndarray) -> np.ndarray:
    e = np.clip(np.nan_to_num(np.asarray(edge_map, dtype=np.float32)), 0.0, 1.0)
    b = np.sqrt(e)
    r = np.clip(40.0 * b, 0.0, 255.0).astype(np.uint8)
    g = np.clip(220.0 * b, 0.0, 255.0).astype(np.uint8)
    bl = np.clip(255.0 * b, 0.0, 255.0).astype(np.uint8)
    return np.stack((r, g, bl), axis=-1)


def _splat_confidence(
    u: np.ndarray,
    v: np.ndarray,
    values: Optional[np.ndarray],
    image_shape: Tuple[int, int],
) -> np.ndarray:
    h, w = int(image_shape[0]), int(image_shape[1])
    heat = np.zeros((h, w), dtype=np.float32)
    if values is None or len(values) == 0:
        return heat
    u = np.asarray(u, dtype=np.int32).reshape(-1)
    v = np.asarray(v, dtype=np.int32).reshape(-1)
    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    counts = np.zeros((h, w), dtype=np.float32)
    np.add.at(heat, (v, u), vals)
    np.add.at(counts, (v, u), 1.0)
    mask = counts > 0.0
    heat[mask] /= counts[mask]
    return heat
