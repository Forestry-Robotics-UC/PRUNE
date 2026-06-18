#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: GNU General Public License v3.0 (GPL-3.0)
# Repository: PRUNE
#
# Description:
#   GLIM-inspired local geometric reliability cues for projection rejection:
#   per-point surface normals, planarity confidence, surface-discontinuity
#   detection, and semantic-normal consistency. Pure numpy, no ROS imports.

"""Local surface-normal estimation and geometric reliability scoring.

GLIM-like systems exploit local geometric consistency as a reliability
signal. Here those ideas are adapted as projection-rejection cues for
ENTFAC-Sensor-Fusion / PRUNE: points whose local neighborhood is too
sparse, too scattered, or geometrically inconsistent with their semantic
label are flagged before semantic transfer. This module is not a SLAM
component and performs no registration.

Frame and failure conventions:

- All geometry is evaluated in the frame of the input points (the LiDAR
  frame in the PRUNE pipeline). Normals are oriented toward ``viewpoint``
  (the sensor origin by default).
- Points whose neighborhood cannot support a normal (fewer than
  ``min_neighbors`` neighbors inside ``radius_m``, or a degenerate
  covariance) are *marked invalid, never guessed*: their normal is zero,
  their reliability is zero, and they are never rejected by the gate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree as _cKDTree
except ImportError:
    _cKDTree = None

_LOG = logging.getLogger(__name__)

# Brute-force neighbor search is quadratic; only allow it for small clouds
# (tests, degraded environments without scipy).
_BRUTE_FORCE_MAX_REFERENCE = 4096

_EIG_EPS = 1e-12

_warned_no_neighbor_backend = False


@dataclass(frozen=True)
class GeometricReliabilityParams:
    """Tunables for :func:`evaluate_geometric_reliability`.

    Attributes:
        k_neighbors: Neighbors requested per query point (includes the
            point itself when it is part of the reference cloud).
        radius_m: Maximum neighbor distance in meters; neighbors beyond
            this radius are discarded before the normal fit.
        min_neighbors: Minimum surviving neighbors required for a valid
            normal (floored at 3, the minimum for a plane fit).
        curvature_max: Surface-variation threshold ``λ0/(λ0+λ1+λ2)``
            above which a point is flagged as a surface discontinuity.
            ``0`` disables the discontinuity check.
        up_labels: Semantic label ids whose surfaces are expected to face
            "up" (terrain, trail, ground). Empty disables the
            semantic-normal consistency check.
        up_max_angle_deg: Maximum angle between the (unsigned) normal and
            the up direction for ``up_labels`` points to count as
            consistent.
        score_min: Reliability score in ``[0, 1]`` below which a valid
            point is flagged for rejection. ``0`` disables the score
            criterion.
    """

    k_neighbors: int = 12
    radius_m: float = 0.5
    min_neighbors: int = 5
    curvature_max: float = 0.12
    up_labels: Tuple[int, ...] = ()
    up_max_angle_deg: float = 60.0
    score_min: float = 0.0


@dataclass
class GeometricReliabilityResult:
    """Per-point outputs of :func:`evaluate_geometric_reliability`.

    All arrays have length ``N`` (the number of query points). Invalid
    points (``normal_valid == False``) carry zero normals, zero
    planarity/curvature, zero reliability, and are never in ``reject``.
    """

    normals: np.ndarray
    normal_valid: np.ndarray
    planarity: np.ndarray
    curvature: np.ndarray
    discontinuity_hit: np.ndarray
    semantic_inconsistent: np.ndarray
    reliability: np.ndarray
    reject: np.ndarray

    @classmethod
    def empty(cls, n: int) -> "GeometricReliabilityResult":
        return cls(
            normals=np.zeros((n, 3), dtype=np.float32),
            normal_valid=np.zeros(n, dtype=bool),
            planarity=np.zeros(n, dtype=np.float32),
            curvature=np.zeros(n, dtype=np.float32),
            discontinuity_hit=np.zeros(n, dtype=bool),
            semantic_inconsistent=np.zeros(n, dtype=bool),
            reliability=np.zeros(n, dtype=np.float32),
            reject=np.zeros(n, dtype=bool),
        )


def _neighbor_indices(
    reference: np.ndarray,
    queries: np.ndarray,
    k: int,
    radius_m: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return ``(idx, valid)`` neighbor indices/mask of shape ``(M, k)``.

    Uses a KD-tree when scipy is available; falls back to brute force for
    small reference clouds. Returns ``None`` when no backend can run (so
    the caller marks every normal invalid instead of stalling the node).
    """
    global _warned_no_neighbor_backend
    n_ref = reference.shape[0]
    k = min(int(k), n_ref)
    if k < 1:
        return None

    if _cKDTree is not None:
        tree = _cKDTree(reference)
        dists, idx = tree.query(queries, k=k)
        if k == 1:
            dists = dists[:, None]
            idx = idx[:, None]
        valid = np.isfinite(dists) & (dists <= float(radius_m)) & (idx < n_ref)
        idx = np.clip(idx, 0, n_ref - 1)
        return idx.astype(np.int64, copy=False), valid

    if n_ref <= _BRUTE_FORCE_MAX_REFERENCE:
        d2 = (
            np.sum(queries.astype(np.float64) ** 2, axis=1)[:, None]
            - 2.0 * queries.astype(np.float64) @ reference.astype(np.float64).T
            + np.sum(reference.astype(np.float64) ** 2, axis=1)[None, :]
        )
        idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
        d2_sel = np.take_along_axis(d2, idx, axis=1)
        valid = d2_sel <= float(radius_m) ** 2
        return idx.astype(np.int64, copy=False), valid

    if not _warned_no_neighbor_backend:
        _LOG.warning(
            "geometric reliability: scipy is unavailable and the reference "
            "cloud has %d points (> %d brute-force cap); all normals will "
            "be marked invalid and the gate stays inert",
            n_ref,
            _BRUTE_FORCE_MAX_REFERENCE,
        )
        _warned_no_neighbor_backend = True
    return None


def estimate_local_normals(
    queries: np.ndarray,
    reference: Optional[np.ndarray] = None,
    *,
    k_neighbors: int = 12,
    radius_m: float = 0.5,
    min_neighbors: int = 5,
    viewpoint: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate per-point surface normals via local PCA.

    Args:
        queries: ``(M, 3)`` points whose normals are estimated.
        reference: ``(N, 3)`` neighbor pool; defaults to ``queries``.
            Passing the full FOV cloud improves neighborhoods at crop
            borders.
        k_neighbors: Neighbors requested per query.
        radius_m: Neighbor distance cap in meters.
        min_neighbors: Minimum neighbors for a valid fit (floored at 3).
        viewpoint: ``(3,)`` sensor origin used to orient normals;
            defaults to the frame origin.

    Returns:
        ``(normals, planarity, curvature, valid)`` where ``normals`` is
        ``(M, 3)`` float32 (zero rows where invalid), ``planarity`` is
        ``(λ1-λ0)/λ2`` in ``[0, 1]``, ``curvature`` is the surface
        variation ``λ0/(λ0+λ1+λ2)``, and ``valid`` marks points whose
        neighborhood supported the fit.
    """
    queries = np.asarray(queries, dtype=np.float64).reshape(-1, 3)
    m = queries.shape[0]
    normals = np.zeros((m, 3), dtype=np.float32)
    planarity = np.zeros(m, dtype=np.float32)
    curvature = np.zeros(m, dtype=np.float32)
    valid = np.zeros(m, dtype=bool)
    if m == 0:
        return normals, planarity, curvature, valid

    reference = (
        queries
        if reference is None
        else np.asarray(reference, dtype=np.float64).reshape(-1, 3)
    )
    if reference.shape[0] == 0:
        return normals, planarity, curvature, valid

    neighbors = _neighbor_indices(reference, queries, k_neighbors, radius_m)
    if neighbors is None:
        return normals, planarity, curvature, valid
    idx, nb_valid = neighbors

    counts = nb_valid.sum(axis=1)
    min_neighbors = max(3, int(min_neighbors))
    fit = counts >= min_neighbors
    if not np.any(fit):
        return normals, planarity, curvature, valid

    idx_f = idx[fit]
    w = nb_valid[fit].astype(np.float64)[:, :, None]
    nb = reference[idx_f] * w
    counts_f = counts[fit].astype(np.float64)[:, None]
    mean = nb.sum(axis=1) / counts_f
    diffs = (reference[idx_f] - mean[:, None, :]) * w
    cov = np.einsum("nki,nkj->nij", diffs, diffs) / counts_f[:, :, None]

    evals, evecs = np.linalg.eigh(cov)
    e0 = np.maximum(evals[:, 0], 0.0)
    e1 = np.maximum(evals[:, 1], 0.0)
    e2 = np.maximum(evals[:, 2], 0.0)
    ok = np.isfinite(evals).all(axis=1) & (e2 > _EIG_EPS)

    n_fit = evecs[:, :, 0]
    view = np.zeros(3) if viewpoint is None else np.asarray(viewpoint, dtype=np.float64).reshape(3)
    to_view = view[None, :] - queries[fit]
    flip = np.einsum("ni,ni->n", n_fit, to_view) < 0.0
    n_fit = np.where(flip[:, None], -n_fit, n_fit)

    e_sum = e0 + e1 + e2
    pl = np.where(ok, (e1 - e0) / np.maximum(e2, _EIG_EPS), 0.0)
    cu = np.where(ok & (e_sum > _EIG_EPS), e0 / np.maximum(e_sum, _EIG_EPS), 0.0)

    fit_idx = np.nonzero(fit)[0]
    ok_idx = fit_idx[ok]
    normals[ok_idx] = n_fit[ok].astype(np.float32)
    planarity[ok_idx] = np.clip(pl[ok], 0.0, 1.0).astype(np.float32)
    curvature[ok_idx] = np.clip(cu[ok], 0.0, 1.0).astype(np.float32)
    valid[ok_idx] = True
    return normals, planarity, curvature, valid


def semantic_normal_inconsistency(
    labels: np.ndarray,
    normals: np.ndarray,
    normal_valid: np.ndarray,
    up_vector: np.ndarray,
    up_labels: Tuple[int, ...],
    up_max_angle_deg: float,
) -> np.ndarray:
    """Flag points whose label expects an up-facing surface but whose
    normal disagrees.

    The comparison is unsigned (``|n · up|``) so normal orientation does
    not matter. Points with invalid normals or labels outside
    ``up_labels`` are never flagged.
    """
    n = normals.shape[0]
    if not up_labels or n == 0:
        return np.zeros(n, dtype=bool)
    up = np.asarray(up_vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(up))
    if norm <= 0.0:
        return np.zeros(n, dtype=bool)
    up /= norm
    labels = np.asarray(labels).reshape(-1)
    checked = np.isin(labels, np.asarray(up_labels)) & normal_valid
    cos_min = float(np.cos(np.deg2rad(float(up_max_angle_deg))))
    cos_angle = np.abs(normals.astype(np.float64) @ up)
    return checked & (cos_angle < cos_min)


def evaluate_geometric_reliability(
    points: np.ndarray,
    *,
    params: GeometricReliabilityParams,
    reference_points: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    up_vector: Optional[np.ndarray] = None,
    viewpoint: Optional[np.ndarray] = None,
) -> GeometricReliabilityResult:
    """Compute per-point geometric reliability and the rejection mask.

    The reliability score is ``planarity * (1 - curvature/curvature_max)``
    clipped to ``[0, 1]`` (just ``planarity`` when the discontinuity check
    is disabled). A point is flagged for rejection only when its normal is
    valid and it hits the discontinuity check, the semantic-normal
    consistency check, or falls below ``score_min``.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = points.shape[0]
    if n == 0:
        return GeometricReliabilityResult.empty(0)

    normals, planarity, curvature, valid = estimate_local_normals(
        points,
        reference_points,
        k_neighbors=params.k_neighbors,
        radius_m=params.radius_m,
        min_neighbors=params.min_neighbors,
        viewpoint=viewpoint,
    )

    curvature_max = float(params.curvature_max)
    if curvature_max > 0.0:
        discontinuity = valid & (curvature > curvature_max)
        penalty = np.clip(curvature / curvature_max, 0.0, 1.0)
    else:
        discontinuity = np.zeros(n, dtype=bool)
        penalty = np.zeros(n, dtype=np.float32)
    reliability = np.where(valid, planarity * (1.0 - penalty), 0.0).astype(np.float32)

    if labels is not None and up_vector is not None and params.up_labels:
        inconsistent = semantic_normal_inconsistency(
            labels,
            normals,
            valid,
            up_vector,
            tuple(params.up_labels),
            params.up_max_angle_deg,
        )
    else:
        inconsistent = np.zeros(n, dtype=bool)

    reject = discontinuity | inconsistent
    if params.score_min > 0.0:
        reject |= valid & (reliability < float(params.score_min))

    return GeometricReliabilityResult(
        normals=normals,
        normal_valid=valid,
        planarity=planarity,
        curvature=curvature,
        discontinuity_hit=discontinuity,
        semantic_inconsistent=inconsistent,
        reliability=reliability,
        reject=reject,
    )
