"""Semantic input parsing helpers for the colored PCL node."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from entfac_fusion_ros.conversions import image_to_numpy, rgb_to_packed_u32
from entfac_fusion_core.utils.masks import invalid_image_to_mask

from .config import ColorConfig, ProjectionConfig
from .results import SemanticInputs


class SemanticInputParser:
    """Parse semantic images, confidence, and invalid masks."""

    def __init__(self, node: Any, color_config: ColorConfig, projection_config: ProjectionConfig, logger: Any):
        self._node = node
        self._color = color_config
        self._projection = projection_config
        self._log = logger

    def parse(self, sem_msg, conf_msg, invalid_mask_msg, callback_name: str) -> SemanticInputs:
        include_rgb = bool(self._node.colorize_labels) if self._node.semantic_input_type == "labels" else True
        rgb_lut = None

        if self._node.semantic_input_type == "labels":
            labels = self._parse_semantic_labels(sem_msg)
            invalid_from_perception = labels == self._node.perception_invalid_label
            if np.any(invalid_from_perception):
                labels = labels.copy().astype(np.int64)
                labels[invalid_from_perception] = -1
            if include_rgb:
                rgb_lut = self._get_rgb_float_lut(labels)
            packed_img = None
        else:
            packed_img = self._parse_semantic_rgb_packed(sem_msg)
            labels = None
            if include_rgb and self._node.color_map and not self._node._warned_rgb_color_map:
                self._log.warn(
                    callback_name,
                    "~color_map is ignored when semantic_input_type=rgb (colors are passed through)",
                )
                self._node._warned_rgb_color_map = True

        confidence = image_to_numpy(conf_msg).astype(float) if conf_msg else None
        if confidence is not None and self._node._undistort_active:
            confidence = self._node._undistort_array(confidence, interpolation="linear")
        semantic_shape = labels.shape if labels is not None else packed_img.shape
        projection_invalid_mask = self._parse_projection_invalid_mask(invalid_mask_msg, semantic_shape)
        return SemanticInputs(
            labels=labels,
            packed_rgb=packed_img,
            confidence=confidence,
            projection_invalid_mask=projection_invalid_mask,
            rgb_lut=rgb_lut,
        )

    def _parse_semantic_labels(self, msg):
        data = image_to_numpy(msg)
        if self._node._undistort_active:
            data = self._node._undistort_array(data, interpolation="nearest")
        if data.ndim == 3:
            data = data[..., 0]
        if data.ndim != 2:
            raise ValueError(
                "semantic_input_type=labels requires a single-channel label image (e.g., mono8/16UC1/32SC1). "
                f"Got encoding={msg.encoding} shape={data.shape}."
            )
        return data

    def _parse_semantic_rgb_packed(self, msg):
        data = image_to_numpy(msg)
        if self._node._undistort_active:
            data = self._node._undistort_array(data, interpolation="linear")
        if data.ndim != 3:
            raise ValueError(
                "semantic_input_type=rgb requires a 3/4-channel image (rgb8/bgr8/rgba8/bgra8). "
                f"Got encoding={msg.encoding} shape={data.shape}."
            )
        return rgb_to_packed_u32(
            data,
            msg.encoding,
            quantize_step=int(self._node.semantic_color_quantization_step),
        )

    def _parse_projection_invalid_mask(self, msg, expected_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        if msg is None:
            return None
        data = image_to_numpy(msg)
        if self._node._undistort_active:
            data = self._node._undistort_array(data, interpolation="nearest")
        invalid = invalid_image_to_mask(
            data,
            invalid_value=int(self._node.projection_invalid_mask_value),
            dilate_px=int(self._node.projection_invalid_mask_dilate_px),
        )
        if invalid.shape != tuple(expected_shape):
            raise ValueError(
                "projection invalid mask shape "
                f"{invalid.shape} must match semantic image shape {tuple(expected_shape)}"
            )
        return invalid

    def _get_rgb_float_lut(self, labels_img: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        return self._node._projector._get_rgb_float_lut(labels_img)
