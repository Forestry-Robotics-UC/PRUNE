"""Frame input preparation helpers for prune."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np


class FrameInputPreparer:
    def __init__(self, node: Any):
        self._node = node

    @staticmethod
    def scale_intrinsics(intrinsics: np.ndarray, factor: int) -> np.ndarray:
        scaled = intrinsics.copy()
        scaled[0, 0] /= factor
        scaled[1, 1] /= factor
        scaled[0, 2] /= factor
        scaled[1, 2] /= factor
        return scaled

    def prepare(
        self,
        sem_msg,
        conf_msg,
        invalid_mask_msg,
        callback_name: str,
    ) -> Tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        bool,
        np.ndarray,
        Tuple[int, int],
        str,
        np.ndarray,
    ]:
        include_rgb = (
            bool(self._node.colorize_labels)
            if self._node.semantic_input_type == "labels"
            else True
        )
        parsed = self._node._semantic_parser.parse(
            sem_msg, conf_msg, invalid_mask_msg, callback_name
        )
        labels = parsed.labels
        packed_img = parsed.packed_rgb
        confidence = parsed.confidence
        projection_invalid_mask = parsed.projection_invalid_mask
        rgb_lut = parsed.rgb_lut

        semantic_debug_type = "labels" if labels is not None else "rgb"
        semantic_debug_img = labels if labels is not None else packed_img
        if semantic_debug_img is None:
            raise ValueError("semantic input could not be prepared")

        if self._node.downsample_factor > 1:
            factor = self._node.downsample_factor
            if labels is not None:
                labels = labels[::factor, ::factor]
            else:
                packed_img = packed_img[::factor, ::factor]
            if confidence is not None:
                confidence = confidence[::factor, ::factor]
            if projection_invalid_mask is not None:
                projection_invalid_mask = projection_invalid_mask[::factor, ::factor]
            intrinsics = self.scale_intrinsics(self._node.intrinsics, factor)
        else:
            intrinsics = self._node.intrinsics

        semantic_shape = labels.shape if labels is not None else packed_img.shape[:2]
        semantic_debug_img = labels if labels is not None else packed_img
        if semantic_debug_img is None:
            raise ValueError("semantic input could not be prepared")

        return (
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
        )
