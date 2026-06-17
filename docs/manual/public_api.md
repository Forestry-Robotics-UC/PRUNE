# Public API (v1.0)

Only these interfaces are considered stable.

## `prune_core` (ROS-agnostic)

- Fusion functions:
  - `prune_core.colored_pcl.fuse_depth_semantics`
  - `prune_core.colored_pcl.fuse_lidar_semantics`
- Dataclasses:
  - `prune_core.types.observations.SemanticObservation`
  - `prune_core.types.observations.DepthObservation`
  - `prune_core.types.observations.PointObservation`
  - `prune_core.types.observations.SemanticPointCloud`
- Validation helpers:
  - `prune_core.utils.validation.ensure_float_matrix`
  - `prune_core.utils.validation.require_homogeneous_transform`
  - `prune_core.utils.validation.flatten_masked`
- Mask helpers:
  - `prune_core.utils.masks.invalid_image_to_mask`
  - `prune_core.utils.masks.sample_invalid_mask`
  - `prune_core.utils.masks.apply_invalid_projection_samples`

## `prune_ros` (ROS1 Noetic)

- Node executable:
  - `prune_ros/scripts/prune_node.py`
- Node implementation:
  - `prune_ros/prune_ros/node/prune_node.py`
- ROS contract:
  - `docs/manual/ros_contract.md`
