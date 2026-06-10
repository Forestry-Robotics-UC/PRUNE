# Changelog

All notable changes to PRUNE are documented in this file.

This project follows Semantic Versioning for release tags.

## Unreleased

### Added
- GLIM-inspired geometric reliability gate (off by default, behavior-preserving):
  - `prune_core.geometry.local_reliability` — ROS-free local surface-normal
    estimation (k-NN/radius PCA with explicit sparse-failure handling),
    planarity confidence, 3D surface-discontinuity detection, semantic-normal
    consistency, and a per-point geometric reliability score. Unestimable
    normals are marked invalid, never guessed, and never rejected.
  - Projector integration behind `~projection_geometric_enable` with the
    `~use_geometric_gate` ablation switch (suppression-vs-filtering semantics
    match the existing gates), new `num_rejected_geometric` /
    `num_would_hit_geometric` counters, `runtime_geometric_ms` timing, and
    per-point normals/reliability stashed on `ProjectionQualityResult` for
    the future enriched-output interface to ENTFAC-Mapping.
  - New parameters: `~geometric_k_neighbors`, `~geometric_radius_m`,
    `~geometric_min_neighbors`, `~geometric_curvature_max`,
    `~geometric_up_labels`, `~geometric_up_max_angle_deg`,
    `~geometric_score_min`; startup report rows; metrics CSV/summary columns;
    `prune.launch` overrides.
  - Tests: `tests/test_geometric_reliability.py` (core) and
    `prune_ros/tests/test_geometric_gate_projector.py` (projector-level
    default-off equivalence, suppression, and rejection).

## v1.0.0 (ROS Noetic)

### Public API (locked)
- Core API: `prune_core.colored_pcl.fuse_depth_semantics` and `prune_core.colored_pcl.fuse_lidar_semantics`
- Core data model: observation/measurement dataclasses in `prune_core.types.observations`
- Core validation utilities in `prune_core.utils.validation`
- ROS API: `prune_ros` `prune_node` topics/params/TF/services

### Highlights
- Stateless single-frame fusion for depth and LiDAR-projection modes
- Standard ROS message interface (`sensor_msgs/Image`, `CameraInfo`, `PointCloud2`, TF2)
- Optional `rgb` and `confidence` fields in output `PointCloud2`
- Low-noise logging with startup and periodic ASCII status tables
- Core unit tests and a minimal ROS integration rostest scaffold
- GPL-3.0 licensing normalized with upstream attribution preserved
