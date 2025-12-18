# Public API (v1.0)

Only the following interfaces are considered **public and stable** for v1.0.
Everything else in the repository is internal and may change without notice.

## `entfac_fusion_core` (ROS-agnostic)

- Fusion functions:
  - `entfac_fusion_core.semantic_pcl.fuse_depth_semantics`
  - `entfac_fusion_core.semantic_pcl.fuse_lidar_semantics`
- Dataclasses:
  - `entfac_fusion_core.types.observations.SemanticObservation`
  - `entfac_fusion_core.types.observations.DepthObservation`
  - `entfac_fusion_core.types.observations.PointObservation`
  - `entfac_fusion_core.types.observations.SemanticPointCloud`
- Validation helpers:
  - `entfac_fusion_core.utils.validation.ensure_float_matrix`
  - `entfac_fusion_core.utils.validation.require_homogeneous_transform`
  - `entfac_fusion_core.utils.validation.flatten_masked`

## `entfac_fusion_ros` (ROS1 Noetic)

- Node:
  - `entfac_fusion_ros/scripts/semantic_pcl_node.py` (executable entrypoint)
  - Interface contract documented in `entfac_fusion_ros.semantic_pcl_node`
    docstring and in `manual/ros_contract`.

