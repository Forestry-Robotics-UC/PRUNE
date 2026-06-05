"""Depth pipeline orchestration for prune."""

from __future__ import annotations

import time
from typing import Any, Optional, Tuple

import numpy as np
import rospy

from prune_core.colored_pcl import fuse_depth_semantics
from prune_core.projection.depth import depth_to_points
from prune_core.transforms.se3 import transform_points
from prune_core.types import DepthObservation, SemanticObservation, SemanticPointCloud
from prune_core.utils.masks import filter_invalid_projection_samples
from prune_core.utils.validation import flatten_masked
from prune_ros.runtime import image_to_numpy

from .results import PipelineResult


class DepthFusionPipeline:
    def __init__(self, node: Any):
        self._node = node

    def _lookup_transforms(self) -> Optional[np.ndarray]:
        target_T_depth = self._node.target_T_depth
        if target_T_depth is None:
            depth_frame = getattr(self._node, '_current_depth_frame', None)
            if depth_frame:
                self._node._log.debug('_depth_callback', 'Priming depth->target transform on first callback (%s -> %s)', depth_frame, self._node.target_frame)
                target_T_depth = self._node._lookup_transform(self._node.target_frame, depth_frame, rospy.Time(0))
                if target_T_depth is not None:
                    self._node.target_T_depth = target_T_depth
        if target_T_depth is None:
            self._node._log.warn('_depth_callback', 'No depth->target transform available')
            return None
        return target_T_depth

    def _unproject_and_fuse(self, *, depth: np.ndarray, labels: Optional[np.ndarray], packed_img: Optional[np.ndarray], confidence: Optional[np.ndarray], projection_invalid_mask: Optional[np.ndarray], intrinsics: np.ndarray, target_T_depth: np.ndarray, rgb_lut: Optional[np.ndarray], include_rgb: bool) -> Tuple[SemanticPointCloud, Optional[np.ndarray]]:
        rgb_values = None
        if labels is not None:
            if projection_invalid_mask is not None:
                labels = labels.copy()
                labels[projection_invalid_mask] = -1
                if confidence is not None:
                    confidence = confidence.copy()
                    confidence[projection_invalid_mask] = 0.0
            semantic_obs = SemanticObservation(labels=labels, confidence=confidence)
            depth_obs = DepthObservation(depth=depth)
            pcl = fuse_depth_semantics(semantic_obs, depth_obs, intrinsics, target_T_depth, include_unlabeled=self._node.include_unlabeled, max_depth_m=self._node.max_depth_m)
        else:
            points_cam, valid_mask = depth_to_points(depth, intrinsics, max_depth_m=self._node.max_depth_m)
            if points_cam.shape[0] == 0:
                pcl = SemanticPointCloud(np.empty((0, 3)), np.empty((0,), dtype=np.int64), None)
            else:
                labels_all = np.full(points_cam.shape[0], -1, dtype=np.int64)
                conf_flat = flatten_masked(confidence, valid_mask) if confidence is not None else None
                invalid_flat = flatten_masked(projection_invalid_mask, valid_mask) if projection_invalid_mask is not None else None
                if conf_flat is not None and invalid_flat is not None:
                    conf_flat = conf_flat.astype(np.float32, copy=True)
                    conf_flat[invalid_flat] = 0.0
                points_target = transform_points(target_T_depth, points_cam)
                pcl = SemanticPointCloud(points_target, labels_all, conf_flat)
                if include_rgb and packed_img is not None:
                    colors_packed = packed_img[valid_mask].reshape(-1).astype(np.uint32, copy=True)
                    if invalid_flat is not None:
                        points_target, labels_all, conf_flat, colors_packed = filter_invalid_projection_samples(invalid_flat, points=points_target, labels=labels_all, confidence=conf_flat, rgb_values=colors_packed)
                        pcl = SemanticPointCloud(points_target, labels_all, conf_flat)
                    rgb_values = colors_packed.astype('<u4', copy=False).view('<f4')
        if include_rgb and rgb_values is None and rgb_lut is not None:
            from prune_ros.runtime import labels_to_uint16
            rgb_values = rgb_lut[labels_to_uint16(pcl.labels)]
        return pcl, rgb_values

    def process(self, sem_msg, depth_msg, conf_msg=None, invalid_mask_msg=None):
        t0 = time.perf_counter()
        self._node._live_tuning.maybe_refresh_params()
        result = self._node._stamp_policy.validate_depth_pair(sem_msg, depth_msg)
        if result is None:
            return None
        _chosen_stamp, stamp = result
        self._node._current_depth_frame = depth_msg.header.frame_id
        target_T_depth = self._lookup_transforms()
        if target_T_depth is None:
            return None
        labels, packed_img, confidence, projection_invalid_mask, rgb_lut, include_rgb, intrinsics, _semantic_shape, _semantic_debug_type, semantic_debug_img = self._node._frame_inputs.prepare(sem_msg, conf_msg, invalid_mask_msg, '_depth_callback')
        depth_raw = self._node.image_to_numpy(depth_msg) if hasattr(self._node, 'image_to_numpy') else image_to_numpy(depth_msg)
        depth_enc = depth_msg.encoding.lower()
        invalid_raw = None
        if self._node.filter_invalid_depth and depth_enc in ('16uc1', 'mono16', '16sc1'):
            invalid_raw = (depth_raw == 0) | (depth_raw == np.iinfo(np.uint16).max)
        if self._node.downsample_factor > 1:
            f = self._node.downsample_factor
            depth_raw = depth_raw[::f, ::f]
            if invalid_raw is not None:
                invalid_raw = invalid_raw[::f, ::f]
        scale = float(self._node.depth_scale)
        if scale == 0.0:
            scale = 0.001 if depth_enc in ('16uc1', '16sc1', 'mono16') else 1.0
        if self._node.debug and not self._node._logged_depth_scaling:
            self._node._log.info('_depth_callback', 'Depth scaling: encoding=%s scale=%.6f (depth_scale_param=%.6f)', depth_msg.encoding, float(scale), float(self._node.depth_scale))
            self._node._logged_depth_scaling = True
        depth = depth_raw.astype(np.float32, copy=False) * float(scale)
        if invalid_raw is not None:
            depth[invalid_raw] = 0.0
        if confidence is not None:
            confidence = confidence.astype(np.float32, copy=False)
        if self._node.debug and not self._node._logged_depth_summary:
            valid = np.isfinite(depth) & (depth > 0)
            valid_count = int(np.count_nonzero(valid))
            dmin = float(depth[valid].min()) if valid_count else float('nan')
            dmax = float(depth[valid].max()) if valid_count else float('nan')
            self._node._log.info('_depth_callback', 'Depth inputs: semantic_shape=%s depth_shape=%s depth_encoding=%s valid_depth=%d min=%.3f max=%.3f downsample=%d', semantic_debug_img.shape[:2], depth.shape, depth_msg.encoding, valid_count, dmin, dmax, int(self._node.downsample_factor))
            self._node._logged_depth_summary = True
        pcl, rgb_values = self._unproject_and_fuse(depth=depth, labels=labels, packed_img=packed_img, confidence=confidence, projection_invalid_mask=projection_invalid_mask, intrinsics=intrinsics, target_T_depth=target_T_depth, rgb_lut=rgb_lut, include_rgb=include_rgb)
        return PipelineResult(
            cloud=pcl,
            stamp=stamp,
            frame_id=self._node.target_frame,
            callback_sec=time.perf_counter() - t0,
            debug={
                'include_rgb': include_rgb,
                'rgb_values': rgb_values,
                'rgb_lut': rgb_lut,
            },
        )
