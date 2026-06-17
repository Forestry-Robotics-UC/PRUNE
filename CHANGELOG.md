# Changelog

All notable changes to PRUNE are documented in this file.

This project follows Semantic Versioning for release tags.

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
- Temporal pairing with sync-drop logging, plus optional rolling-shutter (IMU)
  and LiDAR-deskew motion pre-correction
- Sequential evidence-gate pipeline: G1 invalid-mask rejection, G2 depth-edge
  rejection, G3 occlusion consistency, G4 confidence threshold, each with
  per-gate would-hit diagnostics
- Per-frame projection-health score with optional adaptive gate tightening
- Low-noise logging with startup and periodic ASCII status tables
- Core unit tests and a minimal ROS integration rostest scaffold
- GPL-3.0 licensing normalized with upstream attribution preserved

See `RELEASE_NOTES_v1.0.0.md` for the full feature breakdown.
