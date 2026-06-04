# Public API (v1.0)

Only these interfaces are considered stable.

## `entfac_fusion_core` (ROS-agnostic)

- Fusion functions:
  - `entfac_fusion_core.colored_pcl.fuse_depth_semantics`
  - `entfac_fusion_core.colored_pcl.fuse_lidar_semantics`
- Dataclasses:
  - `entfac_fusion_core.types.observations.SemanticObservation`
  - `entfac_fusion_core.types.observations.DepthObservation`
  - `entfac_fusion_core.types.observations.PointObservation`
  - `entfac_fusion_core.types.observations.SemanticPointCloud`
- Validation helpers:
  - `entfac_fusion_core.utils.validation.ensure_float_matrix`
  - `entfac_fusion_core.utils.validation.require_homogeneous_transform`
  - `entfac_fusion_core.utils.validation.flatten_masked`
- Mask helpers:
  - `entfac_fusion_core.utils.masks.invalid_image_to_mask`
  - `entfac_fusion_core.utils.masks.sample_invalid_mask`
  - `entfac_fusion_core.utils.masks.apply_invalid_projection_samples`

## `entfac_fusion_ros` (ROS1 Noetic)

- Node executable:
  - `entfac_fusion_ros/scripts/prune_node.py`
- Node implementation:
  - `entfac_fusion_ros/entfac_fusion_ros/prune_node.py`
- ROS contract:
  - `docs/manual/ros_contract.md`
