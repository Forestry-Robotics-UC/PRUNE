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
   - optional invalid-mask image from ENTFAC Perception
2. Optional sensor-domain correction:
   - semantic undistort
   - rolling-shutter correction
   - LiDAR deskew
3. Core fusion:
   - depth mode: `fuse_depth_semantics(...)`
   - LiDAR mode: `fuse_lidar_semantics(...)`
4. Optional projection-quality gates:
   - confidence threshold
   - invalid mask
   - LiDAR depth-edge / occlusion checks
5. Publish semantic `PointCloud2`

## Configuration entry points

- **YAML defaults**: `config/core.yaml` + `config/expert.yaml` loaded via `rosparam` under `colored_pcl_node/*`
- **Launch overrides**: `<param>` overrides YAML if set (avoid empty-string overrides)
- **Project overrides**: optional site-specific YAML layered after core/expert in launch
- **Extrinsics**:
  - preferred: TF/URDF (read at init / first callback)
  - bag replay / fixed rigs: static `~static_*` matrices to avoid TF waits
