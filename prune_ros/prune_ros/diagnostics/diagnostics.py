"""Diagnostics orchestration for prune."""

from __future__ import annotations

from typing import Any

from ..runtime.status import render_status_table


class DiagnosticsOrchestrator:
    def __init__(self, node: Any, debug_pub: Any):
        self._node = node
        self._debug_pub = debug_pub

    def tick(self) -> None:
        if self._debug_pub is not None:
            self._debug_pub.tick()

    def publish_range_view(
        self,
        *,
        depth_map,
        edge_map,
        sem_img,
        sem_type: str,
        u,
        v,
        point_confidence,
        header,
    ) -> None:
        if self._debug_pub is not None:
            self._debug_pub.publish_range_view(
                depth_map=depth_map,
                edge_map=edge_map,
                sem_img=sem_img,
                sem_type=sem_type,
                u=u,
                v=v,
                point_confidence=point_confidence,
                header=header,
            )

    def publish_tracked_reprojection(self, overlay_img, error_px, header) -> None:
        if self._debug_pub is not None:
            self._debug_pub.publish_tracked_reprojection(overlay_img, error_px, header)

    def publish_lidar_projection(self, base_rgb, image_shape, uv, header, colors_u8=None) -> None:
        if self._debug_pub is not None:
            self._debug_pub.publish_lidar_projection(
                base_rgb, image_shape, uv, header, colors_u8=colors_u8
            )

    def publish_fov_points(self, points, frame_id, stamp) -> None:
        if self._debug_pub is not None:
            self._debug_pub.publish_fov_points(points, frame_id, stamp)

    def emit_status(self, *, points: int, callback_sec: float) -> None:
        snap = self._node._status.record(points=int(points), callback_sec=float(callback_sec))
        if snap is None:
            return
        table = render_status_table(
            node_name=self._node._node_name,
            mode=self._node.mode,
            semantic_input_type=self._node.semantic_input_type,
            target_frame=self._node.target_frame,
            output_topic=self._node._output_topic,
            points_last=snap.points_last,
            pub_hz=snap.pub_hz,
            avg_points=snap.avg_points,
            avg_callback_ms=snap.avg_callback_ms,
        )
        self._node._log.debug("status", "\n%s", table)
