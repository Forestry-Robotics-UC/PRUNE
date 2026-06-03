#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ENTFAC Sensor Fusion implementation.
#
# Note:
#   This file was developed specifically for ENTFAC Sensor Fusion.
#   Project-level upstream attribution is documented in README.md.
#
# Modified by:
#   Duda Andrada (ENTFAC Sensor Fusion)
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: GNU General Public License v3.0 (GPL-3.0)
# Repository: ENTFAC-Sensor-Fusion
#
# Description:
#   Helpers for image-space invalid masks used by projection quality gates.

"""Image-space invalid-mask helpers."""

import numpy as np


def dilate_mask(mask: np.ndarray, radius_px: int = 0) -> np.ndarray:
    """Dilate a 2D boolean mask by a square pixel radius.

    Args:
        mask: Boolean image mask with shape ``(H, W)``.
        radius_px: Non-negative dilation radius in pixels.

    Returns:
        Dilated boolean mask with the same shape.

    Raises:
        ValueError: If ``mask`` is not 2D boolean or radius is negative.
    """
    src = np.asarray(mask)
    if src.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {src.shape}")
    if src.dtype != np.bool_:
        raise ValueError("mask must be boolean")
    radius = int(radius_px)
    if radius < 0:
        raise ValueError("radius_px must be >= 0")
    if radius == 0 or src.size == 0:
        return src.copy()

    h, w = src.shape
    out = np.zeros_like(src, dtype=bool)
    for dy in range(-radius, radius + 1):
        dst_y0 = max(0, dy)
        dst_y1 = min(h, h + dy)
        src_y0 = max(0, -dy)
        src_y1 = min(h, h - dy)
        if dst_y0 >= dst_y1:
            continue
        for dx in range(-radius, radius + 1):
            dst_x0 = max(0, dx)
            dst_x1 = min(w, w + dx)
            src_x0 = max(0, -dx)
            src_x1 = min(w, w - dx)
            if dst_x0 >= dst_x1:
                continue
            out[dst_y0:dst_y1, dst_x0:dst_x1] |= src[
                src_y0:src_y1, src_x0:src_x1
            ]
    return out


def invalid_image_to_mask(
    image: np.ndarray,
    *,
    invalid_value: int = 255,
    dilate_px: int = 0,
) -> np.ndarray:
    """Convert an invalid-mask image to a boolean invalid mask.

    Args:
        image: Single-channel invalid-mask image. Pixels equal to
            ``invalid_value`` are invalid.
        invalid_value: Pixel value used to mark invalid samples.
        dilate_px: Optional non-negative dilation radius.

    Returns:
        Boolean mask where ``True`` means invalid/rejected.
    """
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError(f"invalid mask image must be 2D, got shape {arr.shape}")
    if int(dilate_px) < 0:
        raise ValueError("dilate_px must be >= 0")
    invalid = arr.astype(np.int64, copy=False) == int(invalid_value)
    return dilate_mask(invalid.astype(bool, copy=False), int(dilate_px))


def sample_invalid_mask(
    mask: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    default_invalid: bool = True,
) -> np.ndarray:
    """Sample invalid-mask values at projected pixel coordinates.

    Out-of-bounds coordinates are treated as invalid by default.
    """
    mask_arr = np.asarray(mask)
    if mask_arr.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask_arr.shape}")
    if mask_arr.dtype != np.bool_:
        raise ValueError("mask must be boolean")

    u_arr = np.asarray(u, dtype=np.int64).reshape(-1)
    v_arr = np.asarray(v, dtype=np.int64).reshape(-1)
    if u_arr.shape != v_arr.shape:
        raise ValueError("u and v must have the same shape")

    h, w = mask_arr.shape
    inside = (u_arr >= 0) & (u_arr < w) & (v_arr >= 0) & (v_arr < h)
    sampled = np.full(u_arr.shape, bool(default_invalid), dtype=bool)
    if np.any(inside):
        sampled[inside] = mask_arr[v_arr[inside], u_arr[inside]]
    return sampled


def apply_invalid_projection_samples(
    invalid: np.ndarray,
    *,
    labels: np.ndarray = None,
    confidence: np.ndarray = None,
    rgb_values: np.ndarray = None,
):
    """Suppress projected samples marked invalid.

    Invalid samples keep their geometry but lose transferred semantic evidence:
    label IDs become ``-1`` and confidence/RGB payloads become zero.
    """
    invalid_arr = np.asarray(invalid)
    if invalid_arr.ndim != 1:
        invalid_arr = invalid_arr.reshape(-1)
    if invalid_arr.dtype != np.bool_:
        raise ValueError("invalid must be boolean")

    n = int(invalid_arr.shape[0])

    labels_out = labels
    if labels is not None:
        labels_out = np.asarray(labels).copy()
        if labels_out.shape[0] != n:
            raise ValueError("labels must be aligned with invalid")
        labels_out[invalid_arr] = -1

    confidence_out = confidence
    if confidence is not None:
        confidence_out = np.asarray(confidence, dtype=np.float32).copy()
        if confidence_out.shape[0] != n:
            raise ValueError("confidence must be aligned with invalid")
        confidence_out[invalid_arr] = 0.0

    rgb_out = rgb_values
    if rgb_values is not None:
        rgb_out = np.asarray(rgb_values).copy()
        if rgb_out.shape[0] != n:
            raise ValueError("rgb_values must be aligned with invalid")
        rgb_out[invalid_arr] = 0.0

    return labels_out, confidence_out, rgb_out


def filter_invalid_projection_samples(
    invalid: np.ndarray,
    *,
    points: np.ndarray = None,
    labels: np.ndarray = None,
    confidence: np.ndarray = None,
    rgb_values: np.ndarray = None,
):
    """Drop projected samples marked invalid while preserving array alignment.

    Invalid-mask pixels from perception represent rejected image evidence, not
    black color observations.  This helper removes those projected samples from
    every aligned output payload before the semantic point cloud is published.
    """
    invalid_arr = np.asarray(invalid)
    if invalid_arr.ndim != 1:
        invalid_arr = invalid_arr.reshape(-1)
    if invalid_arr.dtype != np.bool_:
        raise ValueError("invalid must be boolean")

    n = int(invalid_arr.shape[0])
    keep = ~invalid_arr

    def _filter_aligned(name: str, values):
        if values is None:
            return None
        arr = np.asarray(values)
        if arr.shape[0] != n:
            raise ValueError(f"{name} must be aligned with invalid")
        return arr[keep]

    return (
        _filter_aligned("points", points),
        _filter_aligned("labels", labels),
        _filter_aligned("confidence", confidence),
        _filter_aligned("rgb_values", rgb_values),
    )
