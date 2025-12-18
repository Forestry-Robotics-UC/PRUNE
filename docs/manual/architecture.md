# Architecture summary

## Module map

### `entfac_fusion_core` (ROS-agnostic)

- `types/observations.py`: numpy dataclasses (`SemanticObservation`, `DepthObservation`, `PointObservation`, `SemanticPointCloud`)
- `utils/validation.py`: shape/dtype/transform validation helpers
- `projection/depth.py`: depth back-projection (depth image → 3D points in camera/depth frame)
- `projection/lidar_projection.py`: LiDAR → image projection (points → pixel `(u,v)` + in-bounds mask)
- `transforms/se3.py`: apply 4×4 transforms to point arrays
- `semantic_pcl/fusion.py`: public fusion entry points (`fuse_depth_semantics`, `fuse_lidar_semantics`)

### `entfac_fusion_ros` (ROS1 wrappers)

- `scripts/semantic_pcl_node.py`: roslaunch entrypoint (thin wrapper)
- `src/entfac_fusion_ros/semantic_pcl_node.py`: node implementation (v1 ROS interface)
- `src/entfac_fusion_ros/conversions.py`: `sensor_msgs` → numpy conversions (no `ros_numpy`)
- `src/entfac_fusion_ros/pc2.py`: numpy → `sensor_msgs/PointCloud2` packing (`label`, optional `confidence`, optional `rgb`)
- `src/entfac_fusion_ros/tf_utils.py`: TF `TransformStamped` → 4×4 numpy matrix
- `src/entfac_fusion_ros/status.py`: low-noise periodic ASCII status tables
- `src/entfac_fusion_ros/ply.py`: async PLY writer used by services

## Dataflow (single frame)

1. **Inputs (ROS topics)**
   - `~semantic_topic` (`sensor_msgs/Image`)
   - `~camera_info` (`sensor_msgs/CameraInfo`)
   - `~depth_input_topic` (`sensor_msgs/Image` depth *or* `sensor_msgs/PointCloud2` LiDAR)
   - optional `~confidence_topic` (`sensor_msgs/Image`)
2. **Convert messages → numpy**
   - images via `entfac_fusion_ros.conversions.image_to_numpy`
   - LiDAR via `entfac_fusion_ros.conversions.pointcloud2_to_xyz`
3. **Build core observations**
   - labels mode: `SemanticObservation(labels, confidence)` + `DepthObservation(depth)` / `PointObservation(points_xyz)`
   - rgb mode: labels are not available; output `label=unknown` and (optionally) pass-through `rgb`
4. **Apply fusion (core)**
   - depth mode: `fuse_depth_semantics(...)`
   - lidar mode: `fuse_lidar_semantics(...)`
5. **Publish measurement**
   - `SemanticPointCloud` → `PointCloud2` via `entfac_fusion_ros.pc2.semantic_pointcloud_to_msg`

## Configuration entry points

- **YAML defaults**: loaded via `rosparam` under `semantic_pcl_node/*`
- **Launch overrides**: `<param>` overrides YAML if set (avoid empty-string overrides)
- **Extrinsics**:
  - preferred: TF/URDF (read at init / first callback)
  - bag replay / fixed rigs: static `~static_*` matrices to avoid TF waits

