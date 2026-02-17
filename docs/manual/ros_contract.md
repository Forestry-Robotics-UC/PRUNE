# ROS interface contract

The v1 ROS interface is defined by `colored_pcl_node`.

## Topics

### Subscribed

- `~semantic_topic` (`sensor_msgs/Image`) - required
- `~camera_info` (`sensor_msgs/CameraInfo`) - required
- `~depth_input_topic` (`sensor_msgs/Image` or `sensor_msgs/PointCloud2`) - required
- `~confidence_topic` (`sensor_msgs/Image`) - optional
- `~imu_topic` / `~lidar_imu_topic` (`sensor_msgs/Imu`) - optional for correction paths
- `~camera_metadata_topic` (RealSense metadata) - optional

### Published

- `semantic_pointcloud` (`sensor_msgs/PointCloud2`)
  - fields: `x y z label [confidence] [rgb]`
  - `label` uses `uint16` unknown = `65535`
  - output frame = `~target_frame`

## Parameters

See `docs/manual/parameters.md` (generated from node source).

## TF / extrinsics

- Depth mode requires `depth_frame -> target_frame`
- LiDAR mode requires:
  - `lidar_frame -> camera_frame`
  - `lidar_frame -> target_frame`

Either provide TF/URDF or static `~static_*` 4x4 matrices.

## Services

- `~save_ply` (`std_srvs/Trigger`)
- `~set_ply_recording` (`std_srvs/SetBool`)
