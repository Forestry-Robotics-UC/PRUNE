# TF & extrinsics cheat sheet

## Frames

- `camera_frame`: from first `~camera_info.header.frame_id`
- `depth_frame`: frame of `~depth_input_topic` when depth mode
- `lidar_frame`: frame of `~depth_input_topic` when LiDAR mode
- `target_frame`: configured via `~target_frame`

## Required transforms

### Depth mode

Needs `depth_frame -> target_frame`.

Resolution order:
1. `~static_target_T_depth`
2. TF lookup `target_frame <- depth_frame`

### LiDAR mode

Needs:
- `lidar_frame -> camera_frame`
- `lidar_frame -> target_frame`

Resolution order:
1. `~static_camera_T_lidar` / `~static_target_T_lidar`
2. TF lookups

## Static matrix format (YAML)

All static matrices are row-major 4x4 with 16 entries:

```yaml
colored_pcl_node:
  static_target_T_depth: [1, 0, 0, 0,
                          0, 1, 0, 0,
                          0, 0, 1, 0,
                          0, 0, 0, 1]
```

## Best practice

- Prefer TF/URDF on live robots.
- Prefer static matrices for bag replay/fixed rigs.
