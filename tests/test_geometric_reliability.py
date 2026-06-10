#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: GNU General Public License v3.0 (GPL-3.0)
# Repository: PRUNE
#
# Description:
#   Unit tests for the GLIM-inspired local geometric reliability core.

import sys
from pathlib import Path

import numpy as np

CORE_SRC = Path(__file__).resolve().parents[1] / "prune_core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from prune_core.geometry import local_reliability as lr
from prune_core.geometry import (
    GeometricReliabilityParams,
    estimate_local_normals,
    evaluate_geometric_reliability,
    semantic_normal_inconsistency,
)


def _plane_points(normal, d=5.0, extent=2.0, step=0.1, noise=0.0, seed=0):
    """Dense grid on the plane normal·x = d, expressed in world coords."""
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    # Build an orthonormal basis (t1, t2, normal).
    helper = np.array([1.0, 0.0, 0.0])
    if abs(normal @ helper) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    t1 = np.cross(normal, helper)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)
    coords = np.arange(-extent, extent + step / 2, step)
    a, b = np.meshgrid(coords, coords, indexing="ij")
    pts = (
        d * normal[None, :]
        + a.reshape(-1, 1) * t1[None, :]
        + b.reshape(-1, 1) * t2[None, :]
    )
    if noise > 0.0:
        rng = np.random.default_rng(seed)
        pts = pts + rng.normal(scale=noise, size=pts.shape)
    return pts


class TestEstimateLocalNormals:
    def test_plane_normals_match_plane(self):
        true_normal = np.array([0.0, 0.0, 1.0])
        pts = _plane_points(true_normal, noise=0.002)
        normals, planarity, curvature, valid = estimate_local_normals(
            pts, k_neighbors=12, radius_m=0.5, min_neighbors=5
        )
        assert valid.all()
        alignment = np.abs(normals @ true_normal)
        assert float(alignment.min()) > 0.99
        # k-NN neighborhoods on a regular grid are anisotropic, so planarity
        # saturates well below 1 even on a perfect plane; curvature is the
        # discriminative signal.
        assert float(planarity[valid].mean()) > 0.5
        assert float(curvature[valid].max()) < 0.05

    def test_normals_oriented_toward_viewpoint(self):
        pts = _plane_points([0.0, 0.0, 1.0], d=5.0)
        normals, _, _, valid = estimate_local_normals(
            pts, viewpoint=np.zeros(3)
        )
        # Viewpoint is at the origin below the z=5 plane: oriented normals
        # must point back toward -z.
        assert (normals[valid][:, 2] < 0.0).all()

    def test_sparse_points_marked_invalid_not_guessed(self):
        pts = np.array([[0.0, 0.0, 5.0], [10.0, 0.0, 5.0], [0.0, 10.0, 5.0]])
        normals, planarity, curvature, valid = estimate_local_normals(
            pts, k_neighbors=12, radius_m=0.5, min_neighbors=5
        )
        assert not valid.any()
        assert np.all(normals == 0.0)
        assert np.all(planarity == 0.0)
        assert np.all(curvature == 0.0)

    def test_scattered_blob_has_high_curvature(self):
        rng = np.random.default_rng(7)
        pts = rng.uniform(-1.0, 1.0, size=(2000, 3))
        _, planarity, curvature, valid = estimate_local_normals(
            pts, k_neighbors=12, radius_m=1.0, min_neighbors=5
        )
        assert valid.any()
        assert float(curvature[valid].mean()) > 0.12

    def test_reference_cloud_extends_neighborhoods(self):
        reference = _plane_points([0.0, 0.0, 1.0])
        queries = reference[:5]
        normals, _, _, valid = estimate_local_normals(queries, reference)
        assert valid.all()
        assert float(np.abs(normals[:, 2]).min()) > 0.99

    def test_empty_input(self):
        normals, planarity, curvature, valid = estimate_local_normals(
            np.empty((0, 3))
        )
        assert normals.shape == (0, 3)
        assert valid.shape == (0,)

    def test_brute_force_fallback_matches_kdtree(self, monkeypatch):
        pts = _plane_points([0.0, 1.0, 0.0], extent=1.0, step=0.2, noise=0.001)
        kd_normals, _, _, kd_valid = estimate_local_normals(pts)
        monkeypatch.setattr(lr, "_cKDTree", None)
        bf_normals, _, _, bf_valid = estimate_local_normals(pts)
        assert (kd_valid == bf_valid).all()
        alignment = np.abs(np.einsum("ni,ni->n", kd_normals, bf_normals))
        assert float(alignment[kd_valid].min()) > 0.99

    def test_no_backend_marks_all_invalid(self, monkeypatch):
        monkeypatch.setattr(lr, "_cKDTree", None)
        monkeypatch.setattr(lr, "_BRUTE_FORCE_MAX_REFERENCE", 1)
        pts = _plane_points([0.0, 0.0, 1.0], extent=0.5, step=0.1)
        _, _, _, valid = estimate_local_normals(pts)
        assert not valid.any()


class TestSemanticNormalInconsistency:
    def test_up_facing_terrain_is_consistent(self):
        normals = np.tile([0.0, 0.0, 1.0], (10, 1)).astype(np.float32)
        labels = np.full(10, 3)
        flags = semantic_normal_inconsistency(
            labels, normals, np.ones(10, dtype=bool), [0, 0, 1], (3,), 60.0
        )
        assert not flags.any()

    def test_vertical_surface_labeled_terrain_is_inconsistent(self):
        normals = np.tile([1.0, 0.0, 0.0], (10, 1)).astype(np.float32)
        labels = np.full(10, 3)
        flags = semantic_normal_inconsistency(
            labels, normals, np.ones(10, dtype=bool), [0, 0, 1], (3,), 60.0
        )
        assert flags.all()

    def test_unsigned_comparison_ignores_normal_orientation(self):
        normals = np.tile([0.0, 0.0, -1.0], (4, 1)).astype(np.float32)
        labels = np.full(4, 3)
        flags = semantic_normal_inconsistency(
            labels, normals, np.ones(4, dtype=bool), [0, 0, 1], (3,), 60.0
        )
        assert not flags.any()

    def test_other_labels_and_invalid_normals_never_flagged(self):
        normals = np.tile([1.0, 0.0, 0.0], (4, 1)).astype(np.float32)
        labels = np.array([1, 1, 3, 3])
        valid = np.array([True, True, False, False])
        flags = semantic_normal_inconsistency(
            labels, normals, valid, [0, 0, 1], (3,), 60.0
        )
        assert not flags.any()

    def test_empty_up_labels_disables_check(self):
        normals = np.tile([1.0, 0.0, 0.0], (4, 1)).astype(np.float32)
        flags = semantic_normal_inconsistency(
            np.full(4, 3), normals, np.ones(4, dtype=bool), [0, 0, 1], (), 60.0
        )
        assert not flags.any()


class TestEvaluateGeometricReliability:
    def test_plane_is_reliable_and_kept(self):
        pts = _plane_points([0.0, 0.0, 1.0], noise=0.002)
        result = evaluate_geometric_reliability(
            pts, params=GeometricReliabilityParams()
        )
        assert result.normal_valid.all()
        assert not result.reject.any()
        assert float(result.reliability[result.normal_valid].mean()) > 0.5

    def test_scattered_blob_hits_discontinuity(self):
        rng = np.random.default_rng(11)
        pts = rng.uniform(-1.0, 1.0, size=(2000, 3))
        result = evaluate_geometric_reliability(
            pts,
            params=GeometricReliabilityParams(radius_m=1.0, curvature_max=0.12),
        )
        hit_ratio = result.discontinuity_hit.sum() / max(result.normal_valid.sum(), 1)
        assert hit_ratio > 0.5
        assert (result.reject & ~result.normal_valid).sum() == 0

    def test_semantic_inconsistency_rejects_mislabeled_wall(self):
        wall = _plane_points([1.0, 0.0, 0.0], d=4.0)
        floor = _plane_points([0.0, 0.0, 1.0], d=-1.0)
        pts = np.vstack([wall, floor])
        labels = np.full(pts.shape[0], 3)
        result = evaluate_geometric_reliability(
            pts,
            params=GeometricReliabilityParams(up_labels=(3,), up_max_angle_deg=60.0),
            labels=labels,
            up_vector=[0.0, 0.0, 1.0],
        )
        wall_flags = result.semantic_inconsistent[: wall.shape[0]]
        floor_flags = result.semantic_inconsistent[wall.shape[0]:]
        assert wall_flags.mean() > 0.9
        assert floor_flags.mean() < 0.1
        assert result.reject[: wall.shape[0]].mean() > 0.9

    def test_invalid_normals_never_rejected(self):
        pts = np.array([[0.0, 0.0, 5.0], [10.0, 0.0, 5.0]])
        result = evaluate_geometric_reliability(
            pts,
            params=GeometricReliabilityParams(score_min=0.9, up_labels=(1,)),
            labels=np.array([1, 1]),
            up_vector=[0.0, 0.0, 1.0],
        )
        assert not result.normal_valid.any()
        assert not result.reject.any()
        assert np.all(result.reliability == 0.0)

    def test_score_min_rejects_low_reliability(self):
        rng = np.random.default_rng(3)
        pts = rng.uniform(-1.0, 1.0, size=(1500, 3))
        loose = evaluate_geometric_reliability(
            pts,
            params=GeometricReliabilityParams(
                radius_m=1.0, curvature_max=0.0, score_min=0.0
            ),
        )
        strict = evaluate_geometric_reliability(
            pts,
            params=GeometricReliabilityParams(
                radius_m=1.0, curvature_max=0.0, score_min=0.95
            ),
        )
        assert not loose.reject.any()
        assert strict.reject.sum() > 0

    def test_empty_input(self):
        result = evaluate_geometric_reliability(
            np.empty((0, 3)), params=GeometricReliabilityParams()
        )
        assert result.reject.shape == (0,)
