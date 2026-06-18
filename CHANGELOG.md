# Changelog

All notable changes to PRUNE are documented in this file.

This project follows Semantic Versioning for release tags.

## 1.1.1

### Added
- GLIM-inspired geometric reliability gate (off by default, behavior-preserving):
  - `prune_core.geometry.local_reliability` — ROS-free local surface-normal
    estimation (k-NN/radius PCA with explicit sparse-failure handling),
    planarity confidence, 3D surface-discontinuity detection, semantic-normal
    consistency, and a per-point geometric reliability score. Unestimable
    normals are marked invalid, never guessed, and never rejected.
  - Projector integration behind `~projection_geometric_enable` with the
    `~use_geometric_gate` ablation switch: when the switch is false the gate
    only computes would-hit diagnostics and never alters the output cloud.
    New `num_rejected_geometric` / `num_would_hit_geometric` counters,
    `runtime_geometric_ms` timing, and per-point normals/reliability stashed
    on `ProjectionQualityResult` for the future enriched-output interface to
    ENTFAC-Mapping.
  - Optional `~geometric_fold_into_confidence` coupling (default false):
    min-combines the geometric reliability score into the G4 confidence
    evidence. Kept off by default, and inert in suppression mode, so the G5
    ablation row isolates the geometric gate.
  - New parameters: `~geometric_k_neighbors`, `~geometric_radius_m`,
    `~geometric_min_neighbors`, `~geometric_curvature_max`,
    `~geometric_up_labels`, `~geometric_up_max_angle_deg`,
    `~geometric_score_min`, `~geometric_fold_into_confidence`; startup report
    rows; metrics CSV/summary columns; `prune.launch` overrides; defaults
    documented in `prune_ros/config/expert.yaml`.
  - Requires scipy for KD-tree neighbor search; without scipy the gate falls
    back to brute force for small clouds and otherwise stays inert (one
    warning, all normals marked invalid).
  - Tests: `tests/test_geometric_reliability.py` (core) and
    `prune_ros/tests/test_geometric_gate_projector.py` (projector-level
    default-off equivalence, suppression purity with an active confidence
    gate, fold opt-in, and rejection).
- Gate numbering G1–G5 documented in `docs/manual/parameters.md`:
  G1 invalid mask, G2 depth edge, G3 occlusion, G4 confidence, G5 geometric
  reliability.
- Tracked-reprojection diagnostics are now recorded in the metrics CSV:
  per-frame `tracked_reprojection_error_px` and
  `num_tracked_reprojection_tracks` columns, plus
  `mean_tracked_reprojection_error_px` / `num_tracked_reprojection_frames`
  summary fields averaged only over frames with an actual measurement.

- Dynamic reconfigure Gates group (replayed from `forestsphere` and extended):
  `use_invalid_mask`, `projection_invalid_mask_dilate_px`,
  `use_depth_edge_rejection`, `use_occlusion_gate`, plus the G5 toggles
  `projection_geometric_enable`, `use_geometric_gate`,
  `geometric_curvature_max`, and `geometric_score_min`; the same parameters
  were added to live tuning (`TUNING_PARAMS`).
- Gate-colored projected-LiDAR debug overlay (replayed from `forestsphere`
  and extended with G5): when `debug_project_lidar` is enabled,
  `/debug/lidar_projection` colors points by gate outcome — accepted=green,
  G5 geometric=cyan, G3 occlusion=magenta, G2 depth-edge=orange, G1 invalid
  mask=red. `gate_debug_colors` / `uv_inside` on `ProjectionResult` are
  aligned to the in-image `inside` subset.
- `python3-scipy` declared in `prune_core/package.xml` (used by
  `prune_core.geometry`).

### Fixed
- `~tracked_reprojection_min_image_edge` ROS default corrected from `3.0` to
  `0.05`: the threshold compares against a normalized [0,1] image-edge map,
  so the old default silently disabled the image-edge filter when YAML
  defaults were not loaded.
- Tracked reprojection produced no output on any frame because the node's
  `_ensure_cv2(context) -> bool` was passed where the tracker expects a
  zero-argument provider returning the cv2 module; the resulting TypeError
  was swallowed and `update()` returned `None` forever. The runtime now
  adapts the node helper to the documented provider contract.
- Tracked reprojection silently no-oped in RGB mode: the pipeline hands the
  tracker PRUNE's packed uint32 RGB image, which was treated as an all-zero
  grayscale frame. The tracker now unpacks 2-D packed images to (H, W, 3)
  and feeds the unpacked image to the image-edge filter.
- Live changes to `tracked_reprojection_*` parameters now actually take
  effect: the tracker snapshots its params at build time, so the live-tuning
  controller rebuilds it when one of those parameters changes.

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
