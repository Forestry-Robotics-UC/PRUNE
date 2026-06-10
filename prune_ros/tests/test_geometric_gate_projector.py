#!/usr/bin/env python3
"""Projector-level tests for the GLIM-inspired geometric reliability gate.

ROS-free: exercises :class:`LidarProjector` directly with synthetic scenes.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
ROS_SRC = _REPO / "prune_ros"
CORE_SRC = _REPO / "prune_core" / "src"
for _p in (str(ROS_SRC), str(CORE_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from prune_ros.projection.lidar_projector import (  # noqa: E402
    LidarProjector,
    LidarProjectorParams,
)

IMAGE_W, IMAGE_H = 128, 96
INTRINSICS = np.array(
    [[100.0, 0.0, 64.0], [0.0, 100.0, 48.0], [0.0, 0.0, 1.0]]
)
TERRAIN_LABEL = 3


def _wall_and_floor_points():
    """Synthetic scene in a camera-aligned LiDAR frame (z forward, y down).

    A frontal wall at z=6 (normal along the view axis) and a floor at
    y=+1.5 (normal along y, perpendicular to target-frame up = +z under an
    identity target transform).
    """
    coords = np.arange(-1.5, 1.5, 0.05)
    wx, wy = np.meshgrid(coords, coords, indexing="ij")
    wall = np.column_stack(
        [wx.ravel(), wy.ravel(), np.full(wx.size, 6.0)]
    )
    fx, fz = np.meshgrid(coords, np.arange(4.0, 7.0, 0.05), indexing="ij")
    floor = np.column_stack(
        [fx.ravel(), np.full(fx.size, 1.5), fz.ravel()]
    )
    return wall, floor


def _run_projector(params, points):
    projector = LidarProjector(params)
    labels = np.full((IMAGE_H, IMAGE_W), TERRAIN_LABEL, dtype=np.int32)
    return projector.process_frame(
        points=points.astype(np.float32),
        labels=labels,
        packed_img=None,
        confidence=None,
        projection_invalid_mask=None,
        intrinsics=INTRINSICS,
        camera_T_lidar=np.eye(4),
        target_T_lidar=np.eye(4),
        semantic_shape=(IMAGE_H, IMAGE_W),
        include_rgb=False,
    )


class TestGeometricGateDefaultOff(unittest.TestCase):
    def test_disabled_gate_preserves_output_and_counters(self):
        wall, floor = _wall_and_floor_points()
        points = np.vstack([wall, floor])
        result = _run_projector(LidarProjectorParams(), points)
        self.assertEqual(result.metrics.num_would_hit_geometric, 0)
        self.assertEqual(result.metrics.num_rejected_geometric, 0)
        self.assertEqual(result.metrics.runtime_geometric_ms, 0.0)
        self.assertEqual(
            result.cloud.points_xyz.shape[0],
            result.metrics.num_points_projected_in_image,
        )
        # No suppression: every projected point keeps its sampled label.
        self.assertEqual(int(np.count_nonzero(result.cloud.labels < 0)), 0)


class TestGeometricGateEnabled(unittest.TestCase):
    def test_semantic_normal_consistency_rejects_mislabeled_wall(self):
        wall, floor = _wall_and_floor_points()
        points = np.vstack([wall, floor])
        baseline = _run_projector(LidarProjectorParams(), points)
        params = LidarProjectorParams(
            projection_geometric_enable=True,
            geometric_up_labels=(TERRAIN_LABEL,),
            geometric_up_max_angle_deg=60.0,
            geometric_curvature_max=0.0,
        )
        result = _run_projector(params, points)
        # The frontal wall's normal is along the view axis, which is the
        # target-frame up here (identity transform); the floor's normal is
        # perpendicular to it, so floor points are label-inconsistent.
        self.assertGreater(result.metrics.num_would_hit_geometric, 0)
        self.assertEqual(
            result.metrics.num_rejected_geometric,
            result.metrics.num_would_hit_geometric,
        )
        # In labels mode the gates suppress semantics but keep geometry:
        # same point count, more invalidated labels than the baseline.
        self.assertEqual(
            result.cloud.points_xyz.shape[0],
            baseline.cloud.points_xyz.shape[0],
        )
        self.assertGreater(
            int(np.count_nonzero(result.cloud.labels < 0)),
            int(np.count_nonzero(baseline.cloud.labels < 0)),
        )
        self.assertGreater(result.metrics.runtime_geometric_ms, 0.0)

    def test_suppression_mode_counts_without_rejecting(self):
        wall, floor = _wall_and_floor_points()
        points = np.vstack([wall, floor])
        params = LidarProjectorParams(
            projection_geometric_enable=True,
            use_geometric_gate=False,
            geometric_up_labels=(TERRAIN_LABEL,),
            geometric_up_max_angle_deg=60.0,
            geometric_curvature_max=0.0,
        )
        baseline = _run_projector(LidarProjectorParams(), points)
        result = _run_projector(params, points)
        self.assertGreater(result.metrics.num_would_hit_geometric, 0)
        self.assertEqual(result.metrics.num_rejected_geometric, 0)
        self.assertEqual(
            result.cloud.points_xyz.shape[0],
            baseline.cloud.points_xyz.shape[0],
        )
        # Diagnostics-only: labels match the baseline exactly.
        self.assertEqual(
            int(np.count_nonzero(result.cloud.labels < 0)),
            int(np.count_nonzero(baseline.cloud.labels < 0)),
        )

    def test_scattered_points_hit_discontinuity_check(self):
        rng = np.random.default_rng(5)
        # Vegetation-like scatter in front of the camera.
        scatter = rng.uniform(
            low=(-1.0, -1.0, 4.0), high=(1.0, 1.0, 6.0), size=(4000, 3)
        )
        params = LidarProjectorParams(
            projection_geometric_enable=True,
            geometric_radius_m=1.0,
            geometric_curvature_max=0.12,
        )
        result = _run_projector(params, scatter)
        self.assertGreater(result.metrics.num_would_hit_geometric, 0)


if __name__ == "__main__":
    unittest.main()
