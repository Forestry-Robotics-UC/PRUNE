"""Tracked reprojection runtime bridge for prune."""

from __future__ import annotations

from typing import Any, Optional

from entfac_fusion_ros.tracked_reprojection import TrackedReprojection, TrackedReprojectionParams


class TrackedReprojectionRuntime:
    def __init__(self, node: Any):
        self._node = node
        self._tracker: Optional[TrackedReprojection] = None

    def build(self) -> Optional[TrackedReprojection]:
        if not self._node.tracked_reprojection_enable:
            self._tracker = None
            return None
        params = TrackedReprojectionParams(
            max_corners=int(self._node.tracked_reprojection_max_corners),
            quality_level=float(self._node.tracked_reprojection_quality_level),
            min_distance_px=float(self._node.tracked_reprojection_min_distance_px),
            min_tracks=int(self._node.tracked_reprojection_min_tracks),
            fb_thresh_px=float(self._node.tracked_reprojection_fb_thresh_px),
            depth_edge_thresh=float(self._node.tracked_reprojection_depth_edge_thresh),
            min_image_edge=float(self._node.tracked_reprojection_min_image_edge),
            log_period_sec=float(self._node.tracked_reprojection_log_period_sec),
        )
        self._tracker = TrackedReprojection(params, self._node._ensure_cv2)
        return self._tracker

    def update(self, *, sem_img, sem_type: str, depth_map, depth_edges):
        if self._tracker is None or sem_img is None:
            return None
        return self._tracker.update(
            sem_img=sem_img,
            sem_type=sem_type,
            depth_map=depth_map,
            depth_edges=depth_edges,
        )
