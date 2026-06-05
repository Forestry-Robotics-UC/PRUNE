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
- Low-noise logging with startup and periodic ASCII status tables
- Core unit tests and a minimal ROS integration rostest scaffold
- GPL-3.0 licensing normalized with upstream attribution preserved
