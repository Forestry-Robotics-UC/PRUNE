# Architecture summary

## Module map

### `entfac_fusion_core` (ROS-agnostic)

- `types/observations.py`: NumPy dataclasses
- `utils/validation.py`: shape/dtype/SE(3) validation helpers
- `projection/depth.py`: depth image -> 3D points (with optional `max_depth_m`)
- `projection/lidar_projection.py`: LiDAR -> image projection
- `transforms/se3.py`: SE(3) point transforms
- `colored_pcl/fusion.py`: public fusion entry points

### `entfac_fusion_ros` (ROS wrappers)

- `scripts/colored_pcl_node.py`: roslaunch entrypoint
- `entfac_fusion_ros/colored_pcl_node.py`: main ROS node
- `entfac_fusion_ros/conversions.py`: fast `sensor_msgs` -> NumPy conversion
- `entfac_fusion_ros/pc2.py`: NumPy -> `PointCloud2` packing
- `entfac_fusion_ros/ply.py`: async PLY export
- `entfac_fusion_ros/tf_utils.py`: TF -> matrix utilities
- `entfac_fusion_ros/status.py`: periodic status reporting

## Data flow (single frame)

1. Subscribed inputs:
   - semantic image
   - geometry (depth image or LiDAR cloud)
   - camera info
   - optional confidence image
2. Optional sensor-domain correction:
   - semantic undistort
   - rolling-shutter correction
   - LiDAR deskew
3. Core fusion:
   - depth mode: `fuse_depth_semantics(...)`
   - LiDAR mode: `fuse_lidar_semantics(...)`
4. Publish semantic `PointCloud2`

## Configuration

- Base defaults: `config/core.yaml`
- Advanced timing/correction: `config/expert.yaml`
- Launch-time `<param>` overrides YAML values.
