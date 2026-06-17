#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic image/depth edge-map helpers used by projection diagnostics."""

from __future__ import annotations

import numpy as np


def compute_image_edge_map(image: np.ndarray, image_type: str) -> np.ndarray:
    """Return a normalized float32 edge map for labels or RGB images."""
    image_type = (image_type or "").strip().lower()
    if image_type == "labels":
        labels = np.asarray(image)
        if labels.ndim == 3:
            labels = labels[:, :, 0]
        if labels.ndim != 2:
            return np.zeros(labels.shape[:2], dtype=np.float32)
        edges = np.zeros_like(labels, dtype=np.float32)
        edges[:, 1:] = (labels[:, 1:] != labels[:, :-1]).astype(np.float32)
        edges[1:, :] = np.maximum(
            edges[1:, :], (labels[1:, :] != labels[:-1, :]).astype(np.float32)
        )
        return edges

    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return np.zeros(rgb.shape[:2], dtype=np.float32)
    rgb = rgb[:, :, :3].astype(np.float32, copy=False)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    edges = np.zeros_like(gray, dtype=np.float32)
    edges[:, 1:] += np.abs(gray[:, 1:] - gray[:, :-1])
    edges[1:, :] += np.abs(gray[1:, :] - gray[:-1, :])
    max_val = float(np.max(edges)) if edges.size else 0.0
    if max_val > 0.0:
        edges /= max_val
    return edges


def edge_alignment_score(image_edges: np.ndarray, depth_edges: np.ndarray) -> float:
    """Cosine-style overlap score for same-shaped edge maps."""
    image_edges = np.asarray(image_edges, dtype=np.float32)
    depth_edges = np.asarray(depth_edges, dtype=np.float32)
    if image_edges.shape != depth_edges.shape or image_edges.size == 0:
        return 0.0
    denom = float(
        np.sqrt(np.sum(image_edges * image_edges) * np.sum(depth_edges * depth_edges))
    ) + 1e-6
    return float(np.sum(image_edges * depth_edges) / denom)
