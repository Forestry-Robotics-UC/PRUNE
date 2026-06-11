"""Tracked reprojection runtime bridge for prune."""

from __future__ import annotations

from typing import Any, Optional

from prune_ros.projection.tracked_reprojection import TrackedReprojection, TrackedReprojectionParams


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
        # TrackedReprojection expects a zero-argument provider that returns
        # the cv2 module (or raises). The node's _ensure_cv2(context) returns
        # a bool and caches the module on node._cv2, so adapt it here.
        node = self._node

        def _cv2_provider():
            if not node._ensure_cv2("tracked_reprojection"):
                raise RuntimeError("OpenCV unavailable for tracked reprojection")
            return node._cv2

        self._tracker = TrackedReprojection(params, _cv2_provider)
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
