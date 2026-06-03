"""Depth pipeline orchestration for colored PCL."""

from __future__ import annotations

import time
from typing import Any

import numpy as np


class DepthFusionPipeline:
    def __init__(self, node: Any):
        self._node = node

    def process(self, sem_msg, depth_msg, conf_msg=None, invalid_mask_msg=None):
        t0 = time.perf_counter()
        self._node._maybe_refresh_live_tuning_params()

        result = self._node._stamp_policy.validate_depth_pair(sem_msg, depth_msg)
        if result is None:
            return
        _chosen_stamp, stamp = result

        self._node._current_depth_frame = depth_msg.header.frame_id
        target_T_depth = self._node._depth_lookup_transforms()
        if target_T_depth is None:
            return

        (
            labels,
            packed_img,
            confidence,
            projection_invalid_mask,
            rgb_lut,
            include_rgb,
            intrinsics,
            _semantic_shape,
            _semantic_debug_type,
            semantic_debug_img,
        ) = self._node._prepare_frame_inputs(
            sem_msg, conf_msg, invalid_mask_msg, "_depth_callback"
        )

        depth_raw = self._node.image_to_numpy(depth_msg) if hasattr(self._node, "image_to_numpy") else None
        if depth_raw is None:
            from entfac_fusion_ros.conversions import image_to_numpy
            depth_raw = image_to_numpy(depth_msg)
        depth_enc = depth_msg.encoding.lower()
        invalid_raw = None
        if self._node.filter_invalid_depth and depth_enc in ("16uc1", "mono16", "16sc1"):
            invalid_raw = (depth_raw == 0) | (depth_raw == np.iinfo(np.uint16).max)
        if self._node.downsample_factor > 1:
            f = self._node.downsample_factor
            depth_raw = depth_raw[::f, ::f]
            if invalid_raw is not None:
                invalid_raw = invalid_raw[::f, ::f]

        scale = float(self._node.depth_scale)
        if scale == 0.0:
            scale = 0.001 if depth_enc in ("16uc1", "16sc1", "mono16") else 1.0
        if self._node.debug and not self._node._logged_depth_scaling:
            self._node._log.info(
                "_depth_callback",
                "Depth scaling: encoding=%s scale=%.6f (depth_scale_param=%.6f)",
                depth_msg.encoding,
                float(scale),
                float(self._node.depth_scale),
            )
            self._node._logged_depth_scaling = True

        depth = depth_raw.astype(np.float32, copy=False) * float(scale)
        if invalid_raw is not None:
            depth[invalid_raw] = 0.0
        if confidence is not None:
            confidence = confidence.astype(np.float32, copy=False)

        if self._node.debug and not self._node._logged_depth_summary:
            valid = np.isfinite(depth) & (depth > 0)
            valid_count = int(np.count_nonzero(valid))
            if valid_count:
                dmin = float(depth[valid].min())
                dmax = float(depth[valid].max())
            else:
                dmin, dmax = float("nan"), float("nan")
            self._node._log.info(
                "_depth_callback",
                "Depth inputs: semantic_shape=%s depth_shape=%s depth_encoding=%s valid_depth=%d min=%.3f max=%.3f downsample=%d",
                semantic_debug_img.shape[:2],
                depth.shape,
                depth_msg.encoding,
                valid_count,
                dmin,
                dmax,
                int(self._node.downsample_factor),
            )
            self._node._logged_depth_summary = True

        pcl, rgb_values = self._node._depth_unproject_and_fuse(
            depth=depth,
            labels=labels,
            packed_img=packed_img,
            confidence=confidence,
            projection_invalid_mask=projection_invalid_mask,
            intrinsics=intrinsics,
            target_T_depth=target_T_depth,
            rgb_lut=rgb_lut,
            include_rgb=include_rgb,
        )
        self._node._depth_assemble_and_publish(
            pcl=pcl,
            include_rgb=include_rgb,
            rgb_values=rgb_values,
            rgb_lut=rgb_lut,
            stamp=stamp,
            dt=time.perf_counter() - t0,
        )
