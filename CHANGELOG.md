# Changelog

All notable changes to ENTFAC Sensor Fusion are documented in this file.

This project follows Semantic Versioning for release tags.

## v1.0.0 (ROS Noetic)

### Public API (locked)
- Core API: `entfac_fusion_core.semantic_pcl.fuse_depth_semantics` and `entfac_fusion_core.semantic_pcl.fuse_lidar_semantics`
- Core data model: observation/measurement dataclasses in `entfac_fusion_core.types.observations`
- Core validation utilities in `entfac_fusion_core.utils.validation`
- ROS API: `entfac_fusion_ros` `semantic_pcl_node` topics/params/TF/services

### Highlights
- Stateless single-frame fusion for depth and LiDAR-projection modes
- Standard ROS message interface (`sensor_msgs/Image`, `CameraInfo`, `PointCloud2`, TF2)
- Optional `rgb` and `confidence` fields in output `PointCloud2`
- Low-noise logging with startup and periodic ASCII status tables
- Core unit tests and a minimal ROS integration rostest scaffold
- GPL-3.0 licensing normalized with upstream attribution preserved

