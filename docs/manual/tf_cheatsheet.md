# TF & extrinsics cheat sheet

## Frames

- `camera_frame`: from first `~camera_info.header.frame_id`, or from `~camera_frame` / `frame_id` when using `~camera_info_txt`
- `depth_frame`: taken from `~depth_input_topic` when it is a `sensor_msgs/Image`
- `lidar_frame`: taken from `~depth_input_topic` when it is a `sensor_msgs/PointCloud2`
- `target_frame`: configured via `~target_frame` (default: `base_link`)

## Required transforms

### Depth mode (`sensor_msgs/Image` depth)

Needs `depth_frame -> target_frame`.

Resolution order:
1. `~static_target_T_depth` (if set)
2. TF lookup `target_frame <- depth_frame` (resolved once)

### LiDAR mode (`sensor_msgs/PointCloud2` LiDAR)

Needs:
- `lidar_frame -> camera_frame` (for projection)
- `lidar_frame -> target_frame` (for output frame)

Resolution order:
1. `~static_camera_T_lidar` / `~static_target_T_lidar` (if set)
2. TF lookups (resolved once)

If transforms are missing at runtime, the node logs a warning and skips publishing until the transform becomes available.

## Static matrix format (YAML)

All static matrices are **row-major 4×4** lists with 16 elements:

```yaml
prune_node:
  static_target_T_depth: [1, 0, 0, 0,
                          0, 1, 0, 0,
                          0, 0, 1, 0,
                          0, 0, 0, 1]
```

## Best practice

- Prefer **TF/URDF** for robots with a TF tree.
- Prefer **static matrices** for bag replay and fixed sensor rigs to avoid TF lookup waits.
