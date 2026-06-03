"""LiDAR pipeline orchestration for colored PCL."""

from __future__ import annotations

import time
from typing import Any

from entfac_fusion_ros.lidar_projector import ProjectionMetrics


class LidarFusionPipeline:
    def __init__(self, node: Any):
        self._node = node

    def process(self, sem_msg, lidar_msg, conf_msg=None, invalid_mask_msg=None):
        t0 = time.perf_counter()
        frame_index = int(self._node._results_frame_index)
        self._node._results_frame_index += 1
        pair_dt_sec = self._node._stamp_policy.compute_pair_dt_sec(sem_msg, lidar_msg)
        self._node._maybe_refresh_live_tuning_params()

        result = self._node._stamp_policy.validate_lidar_pair(sem_msg, lidar_msg)
        if result is None:
            self._node._write_lidar_metrics(
                frame_index=frame_index,
                sem_msg=sem_msg,
                lidar_msg=lidar_msg,
                pair_dt_sec=pair_dt_sec,
                pair_accepted=0,
                drop_reason="pair_dt_too_large",
                num_input_points=0,
                projection_metrics=ProjectionMetrics(),
                num_output_points=0,
                runtime_total_ms=1000.0 * (time.perf_counter() - t0),
                runtime_publish_ms=0.0,
            )
            return
        _chosen_stamp, stamp = result

        transforms = self._node._lidar_lookup_transforms(lidar_msg.header.frame_id)
        if transforms is None:
            self._node._write_lidar_metrics(
                frame_index=frame_index,
                sem_msg=sem_msg,
                lidar_msg=lidar_msg,
                pair_dt_sec=pair_dt_sec,
                pair_accepted=0,
                drop_reason="missing_tf",
                num_input_points=0,
                projection_metrics=ProjectionMetrics(),
                num_output_points=0,
                runtime_total_ms=1000.0 * (time.perf_counter() - t0),
                runtime_publish_ms=0.0,
            )
            return
        camera_T_lidar, target_T_lidar = transforms

        (
            labels,
            packed_img,
            confidence,
            projection_invalid_mask,
            rgb_lut,
            include_rgb,
            intrinsics,
            semantic_shape,
            semantic_debug_type,
            semantic_debug_img,
        ) = self._node._prepare_frame_inputs(
            sem_msg, conf_msg, invalid_mask_msg, "_lidar_callback"
        )

        lidar_result = self._node._lidar_process_frame(
            sem_msg=sem_msg,
            lidar_msg=lidar_msg,
            labels=labels,
            packed_img=packed_img,
            confidence=confidence,
            projection_invalid_mask=projection_invalid_mask,
            rgb_lut=rgb_lut,
            include_rgb=include_rgb,
            intrinsics=intrinsics,
            semantic_shape=semantic_shape,
            semantic_debug_type=semantic_debug_type,
            semantic_debug_img=semantic_debug_img,
            camera_T_lidar=camera_T_lidar,
            target_T_lidar=target_T_lidar,
        )

        publish_t0 = time.perf_counter()
        self._node._lidar_assemble_and_publish(
            pcl=lidar_result.pcl,
            debug_colors=lidar_result.debug_colors,
            image_shape=lidar_result.image_shape,
            points=lidar_result.points,
            semantic_debug_img=semantic_debug_img,
            semantic_debug_type=semantic_debug_type,
            intrinsics=intrinsics,
            corrected_camera_T_lidar=lidar_result.corrected_camera_T_lidar,
            include_rgb=include_rgb,
            rgb_values=lidar_result.rgb_values,
            rgb_lut=rgb_lut,
            stamp=stamp,
            sem_msg=sem_msg,
            dt=time.perf_counter() - t0,
            depth_map=lidar_result.depth_map,
            edge_map=lidar_result.edge_map,
        )
        runtime_publish_ms = 1000.0 * (time.perf_counter() - publish_t0)
        lidar_result.projection_metrics.runtime_publish_ms = runtime_publish_ms
        self._node._write_lidar_metrics(
            frame_index=frame_index,
            sem_msg=sem_msg,
            lidar_msg=lidar_msg,
            pair_dt_sec=pair_dt_sec,
            pair_accepted=1,
            drop_reason="none",
            num_input_points=lidar_result.num_input_points,
            projection_metrics=lidar_result.projection_metrics,
            num_output_points=int(lidar_result.pcl.points_xyz.shape[0]),
            runtime_total_ms=1000.0 * (time.perf_counter() - t0),
            runtime_publish_ms=runtime_publish_ms,
        )
        self._node._debug_callback_seq += 1
