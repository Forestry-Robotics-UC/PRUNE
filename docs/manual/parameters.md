# Parameter reference

Generated from `entfac_fusion_ros/entfac_fusion_ros/colored_pcl_node.py` using:

```bash
python docs/tools/extract_ros_params.py > docs/manual/parameters.md
```

| Param | Type | Default | Description |
|---|---|---|---|
| `~camera_info` | `str` | `None` | CameraInfo topic providing intrinsics and camera frame_id (sensor_msgs/CameraInfo). |
| `~camera_metadata_topic` | `str` | `''` | Camera metadata topic for rolling shutter readout (realsense2_camera_msgs/Metadata). |
| `~cloud_stamp_source` | `str` | `''` | Timestamp source for published PointCloud2: auto, semantic, depth, lidar, latest, earliest, midpoint. |
| `~cloud_time_offset_sec` | `float` | `0.0` | Signed offset (seconds) added to published cloud timestamps (negative shifts earlier). |
| `~color_map` | `dict` | `None` | Optional dict {label_id: [r,g,b]} used to colorize labels when ~semantic_input_type='labels'. YAML keys must be quoted (e.g. "0": [0,0,0]). |
| `~colorize_labels` | `bool` | `false` | If true, publish an extra PointCloud2 field 'rgb' (label palette in 'labels' mode; passthrough colors in 'rgb' mode). |
| `~confidence_topic` | `str` | `None` | Optional confidence image topic aligned with semantic labels (sensor_msgs/Image). |
| `~core_debug` | `bool` | `false` | Enable entfac_fusion_core DEBUG logs (can be noisy at 10–30 Hz). |
| `~debug` | `bool` | `false` | Enable debug parameter report at startup (and DEBUG logs if set via launch arg). |
| `~debug_project_lidar` | `bool` | `false` | If true (lidar mode), publish a debug image with projected lidar points overlaid. |
| `~depth_input_topic` | `str` | `None` | Geometry input topic: depth (sensor_msgs/Image) or LiDAR (sensor_msgs/PointCloud2). The node auto-detects which message type is published and selects the fusion mode. |
| `~depth_scale` | `float` | `0.0` | Scale factor to convert depth values to meters (0=auto: 16UC1/mono16 treated as mm -> 0.001; 32FC1 treated as meters -> 1.0). |
| `~downsample_factor` | `int` | `1` | Integer >=1 stride used to subsample images for CPU/ARM targets. |
| `~enable_profiling` | `bool` | `false` | If true, print a short cProfile summary per callback (future C++/numba profiling hook). |
| `~filter_invalid_depth` | `bool` | `true` | If true, treat common uint16 depth sentinels (0, 65535) as invalid before scaling. |
| `~imu_cache_max_dt_sec` | `float` | `0.02` | Max allowed dt (seconds) between semantic frame and IMU for correction. |
| `~imu_cache_size` | `int` | `2000` | IMU cache size for rolling shutter correction. |
| `~imu_frame` | `str` | `''` | Optional IMU frame override for rolling shutter correction. |
| `~imu_topic` | `str` | `''` | IMU topic used for rolling shutter correction (sensor_msgs/Imu). |
| `~include_unlabeled_pts` | `bool` | `false` | If true, keep points outside the camera FOV (label=-1). |
| `~lidar_deskew_enable` | `bool` | `false` | Enable LiDAR deskew using per-point time + IMU. |
| `~lidar_deskew_mode` | `str` | `'rotation'` | Deskew mode: rotation, translation, or both. |
| `~lidar_deskew_ref` | `str` | `'start'` | Deskew reference time: start or mid (scan start recommended). |
| `~lidar_imu_accel_gravity_compensated` | `bool` | `true` | If true, IMU linear_acceleration is gravity-compensated (recommended). |
| `~lidar_imu_cache_max_dt_sec` | `float` | `0.02` | Max allowed dt (seconds) between LiDAR scan time and IMU for deskew. |
| `~lidar_imu_cache_size` | `int` | `2000` | IMU cache size for LiDAR deskew. |
| `~lidar_imu_frame` | `str` | `''` | Optional IMU frame override for LiDAR deskew. |
| `~lidar_imu_topic` | `str` | `''` | IMU topic used for LiDAR deskew (sensor_msgs/Imu). |
| `~lidar_time_field` | `str` | `'t'` | PointCloud2 field name for per-point time (default: t). |
| `~lidar_time_scale` | `float` | `1e-09` | Scale factor to convert per-point time to seconds (e.g., ns -> 1e-9). |
| `~max_depth_m` | `float` | `0.0` | Optional maximum depth in meters (<=0 disables). |
| `~metadata_max_dt_sec` | `float` | `0.1` | Max allowed dt (seconds) between metadata and semantic frame for readout. |
| `~metadata_readout_key` | `int` | `None` | Metadata key for readout time; set -1 to disable metadata readout. |
| `~metadata_readout_scale` | `float` | `1e-06` | Scale applied to metadata value to convert to seconds (e.g., use 1e-6 for usec). |
| `~mode` | `str` | `''` | Force fusion mode ('depth' or 'lidar'); empty string enables auto-detect. |
| `~num_labels` | `int` | `0` | Optional number of label IDs (0=auto from first label image). Used only when ~semantic_input_type='labels' and ~colorize_labels is true with empty ~color_map. |
| `~pair_max_dt_sec` | `float` | `0.03` | Hard max allowed |Δt| (seconds) between semantic and geometry; <=0 disables. |
| `~ply_output_dir` | `str` | `''` | Directory where PLY files are written (empty uses <entfac_fusion_ros>/output/ply). |
| `~ply_target_frame` | `str` | `''` | Optional TF frame to transform PLY output to (ply_target_frame <- target_frame). Empty means use target_frame. |
| `~ply_tf_tolerance_sec` | `float` | `0.02` | Max allowed time difference (seconds) when using latest TF for PLY export. |
| `~ply_tf_use_latest` | `bool` | `false` | When true, fall back to the latest TF for PLY export if exact-time lookup fails. |
| `~random_color_seed` | `int` | `1` | Seed for deterministic random label palette when ~colorize_labels is true and ~color_map is empty. |
| `~rolling_shutter_direction` | `str` | `'top_to_bottom'` | Rolling shutter readout direction: top_to_bottom or bottom_to_top. |
| `~rolling_shutter_enable` | `bool` | `false` | Apply rotation-only rolling shutter correction using IMU. |
| `~rolling_shutter_readout_sec` | `float` | `0.0` | Rolling shutter total readout time in seconds (0 disables). |
| `~semantic_color_quantization_step` | `int` | `1` | Quantize RGB/BGR semantic images to nearest multiple of this step before packing for the PointCloud2 rgb field (helps with JPEG artifacts). |
| `~semantic_input_type` | `str` | `'labels'` | Semantic image representation: 'labels' (single-channel label IDs) or 'rgb' (3-channel colors used directly for output coloring). |
| `~semantic_topic` | `str` | `'/semantic/labels'` | Semantic label image topic (sensor_msgs/Image). |
| `~static_camera_T_lidar` | `list[16]` | `None` | Optional static 4x4 row-major matrix: lidar_frame -> camera_frame. Overrides TF. |
| `~static_target_T_depth` | `list[16]` | `None` | Optional static 4x4 row-major matrix: depth_frame -> target_frame. Overrides TF. |
| `~static_target_T_lidar` | `list[16]` | `None` | Optional static 4x4 row-major matrix: lidar_frame -> target_frame. Overrides TF. |
| `~status_period` | `raw` | `''` | Seconds between periodic status table prints. Empty=auto (1s when debug=true, else disabled). Set to 0 to disable explicitly. |
| `~sync_queue_size` | `int` | `5` | ApproximateTimeSynchronizer queue size for semantic/depth or semantic/lidar pairing. |
| `~sync_slop_sec` | `float` | `0.1` | ApproximateTimeSynchronizer slop in seconds for semantic/depth or semantic/lidar pairing. |
| `~target_frame` | `str` | `'base_link'` | Output frame for published semantic point cloud. |
| `~undistort_alpha` | `float` | `0.0` | Undistort balance/alpha in [0,1]; 0=crop to valid pixels, 1=keep all pixels. |
| `~undistort_semantic` | `bool` | `false` | If true, undistort semantic images using CameraInfo distortion before projection (lidar mode only). |
