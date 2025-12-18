# ROS interface contract

The v1 ROS interface is defined by the `semantic_pcl_node` node.

## Topics

### Subscribed
- `~semantic_topic` (`sensor_msgs/Image`) — required
- `~camera_info` (`sensor_msgs/CameraInfo`) — required
- `~depth_input_topic` (`sensor_msgs/Image` or `sensor_msgs/PointCloud2`) — recommended (auto-detect)
- `~depth_topic` (`sensor_msgs/Image`) — deprecated
- `~lidar_topic` (`sensor_msgs/PointCloud2`) — deprecated
- `~confidence_topic` (`sensor_msgs/Image`) — optional

### Published
- `semantic_pointcloud` (`sensor_msgs/PointCloud2`)
  - fields: `x y z label [confidence] [rgb]`
  - `label` is `uint16` and encodes unknown as `65535`
  - output frame: `~target_frame`

## Parameters

See `manual/parameters` (generated).

## TF / extrinsics

- Depth mode requires `depth_frame -> target_frame`
- LiDAR mode requires `lidar_frame -> camera_frame` and `lidar_frame -> target_frame`

You can supply static 4×4 transforms via parameters or use TF/URDF.

## Services

- `~save_ply` (`std_srvs/Trigger`)
- `~set_ply_recording` (`std_srvs/SetBool`)

