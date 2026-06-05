"""LiDAR pipeline orchestration for prune."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import rospy

from prune_core.projection.lidar_projection import project_points_to_image
from prune_core.transforms.se3 import transform_points
from prune_core.types import SemanticPointCloud
from prune_core.utils.semantics import packed_rgb_to_triplets
from prune_ros.runtime import pointcloud2_to_xyz, pointcloud2_to_xyz_t
from prune_ros.runtime import interpolate_imu_msg
from prune_ros.projection import GateMetrics

from prune_ros.pipelines import PipelineResult


@dataclass
class _LidarFrameResult:
    pcl: SemanticPointCloud
    debug_colors: Optional[np.ndarray]
    image_shape: Tuple[int, int]
    points: np.ndarray
    rgb_values: Optional[np.ndarray]
    gate_metrics: GateMetrics
    corrected_camera_T_lidar: np.ndarray
    depth_map: Optional[np.ndarray]
    edge_map: Optional[np.ndarray]
    num_input_points: int


class LidarFusionPipeline:
    def __init__(self, node: Any):
        self._node = node

    def _lookup_transforms(self, lidar_frame: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        camera_T_lidar = self._node.camera_T_lidar
        target_T_lidar = self._node.target_T_lidar
        if camera_T_lidar is None or target_T_lidar is None:
            if lidar_frame:
                self._node._lidar_frame = lidar_frame
                if camera_T_lidar is None:
                    self._node._log.debug('_lidar_callback', 'Priming lidar->camera transform on first callback (%s -> %s)', lidar_frame, self._node.camera_frame)
                    camera_T_lidar = self._node._lookup_transform(self._node.camera_frame, lidar_frame, rospy.Time(0))
                    if camera_T_lidar is not None:
                        self._node.camera_T_lidar = camera_T_lidar
                if target_T_lidar is None:
                    self._node._log.debug('_lidar_callback', 'Priming lidar->target transform on first callback (%s -> %s)', lidar_frame, self._node.target_frame)
                    target_T_lidar = self._node._lookup_transform(self._node.target_frame, lidar_frame, rospy.Time(0))
                    if target_T_lidar is not None:
                        self._node.target_T_lidar = target_T_lidar
        if camera_T_lidar is None or target_T_lidar is None:
            self._node._log.warn('_lidar_callback', 'No lidar transforms available')
            return None
        return camera_T_lidar, target_T_lidar

    def _get_readout_sec(self, stamp: rospy.Time) -> float:
        if not self._node.rolling_shutter_enable:
            return 0.0
        if self._node.metadata_readout_key >= 0 and self._node._metadata_latest:
            entry = self._node._metadata_latest.get(int(self._node.metadata_readout_key))
            if entry is not None:
                meta_stamp, value = entry
                if self._node.metadata_max_dt_sec > 0.0:
                    dt = abs((meta_stamp - stamp).to_sec())
                    if dt > self._node.metadata_max_dt_sec:
                        return float(self._node.rolling_shutter_readout_sec)
                return float(value) * float(self._node.metadata_readout_scale)
        return float(self._node.rolling_shutter_readout_sec)

    def _lookup_imu_omega(self, stamp: rospy.Time) -> Optional[np.ndarray]:
        if self._node._imu_cache is None:
            return None
        before = self._node._imu_cache.getElemBeforeTime(stamp)
        after = self._node._imu_cache.getElemAfterTime(stamp)
        if before is None and after is None:
            return None
        omega, _, best_dt = interpolate_imu_msg(before, after, stamp)
        if omega is None:
            return None
        if self._node.imu_cache_max_dt_sec > 0.0 and best_dt > self._node.imu_cache_max_dt_sec:
            return None
        imu_frame = self._node.imu_frame or (after.header.frame_id if before is None else before.header.frame_id)
        if not imu_frame:
            return None
        if self._node._imu_to_camera_R is None:
            mat = self._node._lookup_transform(self._node.camera_frame, imu_frame, rospy.Time(0))
            if mat is None:
                return None
            self._node._imu_to_camera_R = mat[:3, :3]
        return self._node._imu_to_camera_R @ omega

    def _lookup_lidar_imu(self, stamp: rospy.Time) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._node._lidar_imu_cache is None:
            return None
        before = self._node._lidar_imu_cache.getElemBeforeTime(stamp)
        after = self._node._lidar_imu_cache.getElemAfterTime(stamp)
        if before is None and after is None:
            return None
        omega, accel, best_dt = interpolate_imu_msg(before, after, stamp)
        if omega is None or accel is None:
            return None
        if self._node.lidar_imu_cache_max_dt_sec > 0.0 and best_dt > self._node.lidar_imu_cache_max_dt_sec:
            return None
        imu_frame = self._node.lidar_imu_frame or (before.header.frame_id if before is not None else after.header.frame_id)
        if not imu_frame:
            return None
        if self._node._lidar_imu_to_lidar_R is None:
            if not self._node._lidar_frame:
                return None
            mat = self._node._lookup_transform(self._node._lidar_frame, imu_frame, rospy.Time(0))
            if mat is None:
                return None
            self._node._lidar_imu_to_lidar_R = mat[:3, :3]
        omega_lidar = self._node._lidar_imu_to_lidar_R @ omega
        accel_lidar = self._node._lidar_imu_to_lidar_R @ accel
        return omega_lidar, accel_lidar

    def _deskew_lidar_points(self, points: np.ndarray, t_raw: np.ndarray, scan_stamp: rospy.Time) -> np.ndarray:
        if not self._node.lidar_deskew_enable:
            return points
        if t_raw is None or t_raw.size == 0:
            self._node._lidar_deskew_status = f"armed (missing point time field '{self._node.lidar_time_field}')"
            now = time.time()
            if now - self._node._lidar_deskew_missing_time_warn_at > 2.0:
                self._node._log.warn('_deskew_lidar_points', "Deskew enabled but point cloud has no usable '%s' field; skipping deskew.", self._node.lidar_time_field)
                self._node._lidar_deskew_missing_time_warn_at = now
            return points
        dt = t_raw.astype(np.float64) * float(self._node.lidar_time_scale)
        scan_span = float(np.nanmax(dt)) if dt.size else 0.0
        ref_offset = 0.5 * scan_span if self._node.lidar_deskew_ref == 'mid' else 0.0
        rel_dt = dt - ref_offset
        ref_stamp = scan_stamp + rospy.Duration.from_sec(float(ref_offset))
        sample_count = min(max(1, int(self._node.lidar_deskew_imu_samples)), 32)
        if sample_count <= 1 or scan_span <= 1e-9:
            ref_imu = self._lookup_lidar_imu(ref_stamp)
            if ref_imu is None:
                self._node._lidar_deskew_status = 'armed (waiting for IMU)'
                now = time.time()
                if now - self._node._lidar_deskew_warn_at > 2.0:
                    self._node._log.warn('_deskew_lidar_points', 'Deskew enabled but IMU lookup failed near the selected reference time; skipping deskew.')
                    self._node._lidar_deskew_warn_at = now
                return points
            omega_ref, accel_ref = ref_imu
            omega_eval = np.repeat(np.asarray(omega_ref, dtype=np.float64).reshape(1, 3), points.shape[0], axis=0)
            accel_eval = np.repeat(np.asarray(accel_ref, dtype=np.float64).reshape(1, 3), points.shape[0], axis=0)
        else:
            sample_offsets = np.linspace(0.0, scan_span, sample_count, dtype=np.float64)
            omega_samples = []
            accel_samples = []
            dense_sampling_ok = True
            for offset in sample_offsets:
                imu_sample = self._lookup_lidar_imu(scan_stamp + rospy.Duration.from_sec(float(offset)))
                if imu_sample is None:
                    dense_sampling_ok = False
                    self._node._lidar_deskew_status = 'armed (waiting for dense IMU support)'
                    now = time.time()
                    if now - self._node._lidar_deskew_warn_at > 2.0:
                        self._node._log.warn('_deskew_lidar_points', 'Deskew requested %d IMU samples across the scan but lookup failed at %.6fs; falling back to the single-sample model.', sample_count, float(offset))
                        self._node._lidar_deskew_warn_at = now
                    break
                omega_i, accel_i = imu_sample
                omega_samples.append(np.asarray(omega_i, dtype=np.float64).reshape(3))
                accel_samples.append(np.asarray(accel_i, dtype=np.float64).reshape(3))
            if dense_sampling_ok:
                omega_samples = np.asarray(omega_samples, dtype=np.float64)
                accel_samples = np.asarray(accel_samples, dtype=np.float64)
                sample_rel_dt = sample_offsets - ref_offset
                if sample_rel_dt.size <= 1:
                    omega_eval = np.repeat(omega_samples[:1], points.shape[0], axis=0)
                    accel_eval = np.repeat(accel_samples[:1], points.shape[0], axis=0)
                else:
                    boundaries = 0.5 * (sample_rel_dt[:-1] + sample_rel_dt[1:])
                    sample_idx = np.searchsorted(boundaries, rel_dt, side='right')
                    sample_idx = np.clip(sample_idx, 0, omega_samples.shape[0] - 1)
                    omega_eval = omega_samples[sample_idx]
                    accel_eval = accel_samples[sample_idx]
            else:
                ref_imu = self._lookup_lidar_imu(ref_stamp)
                if ref_imu is None:
                    self._node._lidar_deskew_status = 'armed (waiting for IMU)'
                    return points
                omega_ref, accel_ref = ref_imu
                omega_eval = np.repeat(np.asarray(omega_ref, dtype=np.float64).reshape(1, 3), points.shape[0], axis=0)
                accel_eval = np.repeat(np.asarray(accel_ref, dtype=np.float64).reshape(1, 3), points.shape[0], axis=0)
        now = time.time()
        self._node._lidar_deskew_status = 'active'
        if now - self._node._lidar_deskew_log_at > 2.0:
            self._node._log.info('_deskew_lidar_points', 'Deskew active: mode=%s ref=%s dt_max=%.6f imu_samples=%d', self._node.lidar_deskew_mode, self._node.lidar_deskew_ref, scan_span, sample_count)
            self._node._lidar_deskew_log_at = now
        if self._node.lidar_deskew_mode in ('rotation', 'both'):
            points = points - rel_dt.reshape(-1, 1) * np.cross(omega_eval, points)
        if self._node.lidar_deskew_mode in ('translation', 'both'):
            if not self._node.lidar_imu_accel_is_gravity_compensated:
                self._node._log.warn('_deskew_lidar_points', 'Translation deskew assumes gravity-compensated IMU accel; set ~lidar_imu_accel_gravity_compensated=true if already corrected.')
            points = points - 0.5 * accel_eval * (rel_dt.reshape(-1, 1) ** 2)
        return points

    def _apply_lidar_points_compat(self, points: np.ndarray) -> np.ndarray:
        mat = self._node._compat_declared_lidar_T_points
        if mat is None:
            return points
        return transform_points(mat, points)

    def _process_frame(self, *, sem_msg, lidar_msg, labels: Optional[np.ndarray], packed_img: Optional[np.ndarray], confidence: Optional[np.ndarray], projection_invalid_mask: Optional[np.ndarray], rgb_lut: Optional[np.ndarray], include_rgb: bool, intrinsics: np.ndarray, semantic_shape: Tuple[int, int], semantic_debug_type: str, semantic_debug_img: Optional[np.ndarray], camera_T_lidar: np.ndarray, target_T_lidar: np.ndarray) -> _LidarFrameResult:
        lidar_stamp = sem_msg.header.stamp
        if lidar_msg.header.frame_id and not self._node._lidar_frame:
            self._node._lidar_frame = lidar_msg.header.frame_id
        if self._node.lidar_deskew_enable:
            points, t_raw = pointcloud2_to_xyz_t(lidar_msg, time_field=self._node.lidar_time_field)
            points = self._apply_lidar_points_compat(points)
            points = self._deskew_lidar_points(points, t_raw, lidar_stamp)
        else:
            points = pointcloud2_to_xyz(lidar_msg)
            points = self._apply_lidar_points_compat(points)
        num_input_points = int(points.shape[0])
        corrected_camera_T_lidar = camera_T_lidar
        if self._node._calibration is not None and semantic_debug_img is not None:
            sem_h, sem_w = semantic_debug_img.shape[:2]
            corrected_camera_T_lidar, calib_snapshot = self._node._calibration_bridge.update(points=points, sem_img=semantic_debug_img, sem_type=semantic_debug_type, intrinsics=intrinsics, camera_T_lidar=camera_T_lidar, image_shape=(int(sem_h), int(sem_w)))
            if calib_snapshot is not None:
                self._node._diagnostics.publish_calibration_health(calib_snapshot)
            self._node._online_calibration_status = self._node._calibration_bridge.status
        rolling_shutter_readout_sec = self._get_readout_sec(sem_msg.header.stamp) if self._node.rolling_shutter_enable else 0.0
        rolling_shutter_omega_cam = self._lookup_imu_omega(sem_msg.header.stamp) if rolling_shutter_readout_sec > 0.0 else None
        proj_result = self._node._projector.process_frame(points=points, labels=labels, packed_img=packed_img, confidence=confidence, projection_invalid_mask=projection_invalid_mask, intrinsics=intrinsics, camera_T_lidar=corrected_camera_T_lidar, target_T_lidar=target_T_lidar, semantic_shape=semantic_shape, include_rgb=include_rgb, rolling_shutter_omega_cam=rolling_shutter_omega_cam, rolling_shutter_readout_sec=rolling_shutter_readout_sec, cloud_height=int(lidar_msg.height), cloud_width=int(lidar_msg.width), frame_stamp=lidar_msg.header.stamp.to_sec())
        if proj_result.rolling_shutter_active:
            self._node._rolling_shutter_status = 'active'
        elif self._node.rolling_shutter_enable:
            self._node._rolling_shutter_status = 'idle (readout<=0)' if rolling_shutter_readout_sec <= 0.0 else 'armed (waiting for IMU)'
        pcl = proj_result.cloud
        debug_colors = proj_result.debug_colors
        image_shape = proj_result.image_shape
        rgb_values = proj_result.rgb_values
        gate_metrics = proj_result.metrics
        points = proj_result.points_fov
        if self._node._debug_pub is not None:
            if self._node.debug_project_lidar:
                base_rgb = np.stack([(labels.astype(np.int32) % 256).astype(np.uint8)] * 3, axis=-1) if labels is not None else packed_rgb_to_triplets(packed_img)
                uv, _ = project_points_to_image(points, intrinsics, corrected_camera_T_lidar, (image_shape[1], image_shape[0]))
                self._node._diagnostics.publish_lidar_projection(base_rgb, image_shape, uv, sem_msg.header, colors_u8=debug_colors)
            if self._node.debug_publish_fov_points and points.shape[0]:
                self._node._diagnostics.publish_fov_points(points, lidar_msg.header.frame_id, lidar_msg.header.stamp)
        return _LidarFrameResult(pcl=pcl, debug_colors=debug_colors, image_shape=image_shape, points=points, rgb_values=rgb_values, gate_metrics=gate_metrics, corrected_camera_T_lidar=corrected_camera_T_lidar, depth_map=proj_result.depth_map, edge_map=proj_result.edge_map, num_input_points=num_input_points)

    def _build_result(self, *, lidar_result: _LidarFrameResult, semantic_debug_img, semantic_debug_type: str, intrinsics: np.ndarray, stamp: rospy.Time, sem_msg, lidar_msg, t0: float, frame_index: int, pair_dt_sec: float, rgb_lut: Optional[np.ndarray], include_rgb: bool) -> PipelineResult:
        h, w = lidar_result.image_shape
        self._node._diagnostics.tick()
        if self._node._debug_pub is not None and self._node.debug_range_view and semantic_debug_img is not None and lidar_result.pcl.points_xyz.shape[0]:
            dmap = lidar_result.depth_map if lidar_result.depth_map is not None else self._node._projector._rasterize_depth_map(lidar_result.points, intrinsics, lidar_result.corrected_camera_T_lidar, (h, w))
            emap = lidar_result.edge_map if lidar_result.edge_map is not None else self._node._projector._depth_to_edge_map(dmap)
            self._node._diagnostics.publish_range_view(depth_map=dmap, edge_map=emap, sem_img=semantic_debug_img, sem_type=semantic_debug_type, u=np.arange(w, dtype=np.int32), v=np.arange(h, dtype=np.int32), point_confidence=None, header=sem_msg.header)
        if self._node._tracked_repr is not None and semantic_debug_img is not None:
            dmap = lidar_result.depth_map if lidar_result.depth_map is not None else self._node._projector._rasterize_depth_map(lidar_result.points, intrinsics, lidar_result.corrected_camera_T_lidar, (h, w))
            emap = lidar_result.edge_map if lidar_result.edge_map is not None else self._node._projector._depth_to_edge_map(dmap)
            tracked = self._node._tracked_runtime.update(sem_img=semantic_debug_img, sem_type=semantic_debug_type, depth_map=dmap, depth_edges=emap)
            if tracked is not None:
                self._node._diagnostics.publish_tracked_reprojection(tracked.overlay_img, tracked.error_px, sem_msg.header)
        if self._node.debug and not self._node._logged_lidar_summary:
            if lidar_result.pcl.points_xyz.shape[0]:
                mins = lidar_result.pcl.points_xyz.min(axis=0)
                maxs = lidar_result.pcl.points_xyz.max(axis=0)
                self._node._log.info('_lidar_callback', 'LiDAR PCL bbox in %s: x=[%.3f, %.3f] y=[%.3f, %.3f] z=[%.3f, %.3f] (input_points=%d)', self._node.target_frame, float(mins[0]), float(maxs[0]), float(mins[1]), float(maxs[1]), float(mins[2]), float(maxs[2]), int(lidar_result.points.shape[0]))
            else:
                self._node._log.warn('_lidar_callback', 'LiDAR fusion produced empty point cloud (check intrinsics, transforms, and image alignment)')
            self._node._logged_lidar_summary = True
        def post_publish(publish_ms: float) -> None:
            lidar_result.gate_metrics.runtime_publish_ms = publish_ms
            self._node._metrics_reporter.write_lidar_metrics(frame_index=frame_index, sem_msg=sem_msg, lidar_msg=lidar_msg, pair_dt_sec=pair_dt_sec, pair_accepted=1, drop_reason='none', num_input_points=lidar_result.num_input_points, projection_metrics=lidar_result.gate_metrics, num_output_points=int(lidar_result.pcl.points_xyz.shape[0]), runtime_total_ms=1000.0 * (time.perf_counter() - t0), runtime_publish_ms=publish_ms)
            self._node._debug_callback_seq += 1
        return PipelineResult(cloud=lidar_result.pcl, stamp=stamp, frame_id=self._node.target_frame, callback_sec=time.perf_counter() - t0, debug={'include_rgb': include_rgb, 'rgb_values': lidar_result.rgb_values, 'rgb_lut': rgb_lut, 'post_publish': post_publish})

    def process(self, sem_msg, lidar_msg, conf_msg=None, invalid_mask_msg=None):
        t0 = time.perf_counter()
        frame_index = int(self._node._results_frame_index)
        self._node._results_frame_index += 1
        pair_dt_sec = self._node._stamp_policy.compute_pair_dt_sec(sem_msg, lidar_msg)
        self._node._live_tuning.maybe_refresh_params()
        result = self._node._stamp_policy.validate_lidar_pair(sem_msg, lidar_msg)
        if result is None:
            self._node._metrics_reporter.write_lidar_metrics(frame_index=frame_index, sem_msg=sem_msg, lidar_msg=lidar_msg, pair_dt_sec=pair_dt_sec, pair_accepted=0, drop_reason='pair_dt_too_large', num_input_points=0, projection_metrics=GateMetrics(), num_output_points=0, runtime_total_ms=1000.0 * (time.perf_counter() - t0), runtime_publish_ms=0.0)
            return None
        _chosen_stamp, stamp = result
        transforms = self._lookup_transforms(lidar_msg.header.frame_id)
        if transforms is None:
            self._node._metrics_reporter.write_lidar_metrics(frame_index=frame_index, sem_msg=sem_msg, lidar_msg=lidar_msg, pair_dt_sec=pair_dt_sec, pair_accepted=0, drop_reason='missing_tf', num_input_points=0, projection_metrics=GateMetrics(), num_output_points=0, runtime_total_ms=1000.0 * (time.perf_counter() - t0), runtime_publish_ms=0.0)
            return None
        camera_T_lidar, target_T_lidar = transforms
        labels, packed_img, confidence, projection_invalid_mask, rgb_lut, include_rgb, intrinsics, semantic_shape, semantic_debug_type, semantic_debug_img = self._node._frame_inputs.prepare(sem_msg, conf_msg, invalid_mask_msg, '_lidar_callback')
        lidar_result = self._process_frame(sem_msg=sem_msg, lidar_msg=lidar_msg, labels=labels, packed_img=packed_img, confidence=confidence, projection_invalid_mask=projection_invalid_mask, rgb_lut=rgb_lut, include_rgb=include_rgb, intrinsics=intrinsics, semantic_shape=semantic_shape, semantic_debug_type=semantic_debug_type, semantic_debug_img=semantic_debug_img, camera_T_lidar=camera_T_lidar, target_T_lidar=target_T_lidar)
        return self._build_result(lidar_result=lidar_result, semantic_debug_img=semantic_debug_img, semantic_debug_type=semantic_debug_type, intrinsics=intrinsics, stamp=stamp, sem_msg=sem_msg, lidar_msg=lidar_msg, t0=t0, frame_index=frame_index, pair_dt_sec=pair_dt_sec, rgb_lut=rgb_lut, include_rgb=include_rgb)
