# ROS interface contract

The v1 ROS interface is defined by the `prune_node` node.

## Topics

### Subscribed
- `~semantic_topic` (`sensor_msgs/Image`) — required
- `~camera_info` (`sensor_msgs/CameraInfo`) — optional when `~camera_info_txt` is used
- `~depth_input_topic` (`sensor_msgs/Image` or `sensor_msgs/PointCloud2`) — required (auto-detect)
- `~confidence_topic` (`sensor_msgs/Image`) — optional
- `~projection_invalid_mask_topic` (`sensor_msgs/Image`) — optional, aligned to
  `~semantic_topic`; invalid samples reject transferred labels/RGB evidence

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
