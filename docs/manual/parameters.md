# Parameter reference

Generated from `entfac_fusion_ros/entfac_fusion_ros/prune_node.py` using:

```bash
python docs/tools/extract_ros_params.py > docs/manual/parameters.md
```

Defaults are primarily defined in `entfac_fusion_ros/config/core.yaml` and `entfac_fusion_ros/config/expert.yaml`, then overridden by launch-time params when set.

| Param | Type | Default | Description |
|---|---|---|---|
| `~camera_frame` | `str` | `''` | Optional camera frame override used when ~camera_info_txt does not include frame_id. |
| `~camera_info` | `str` | `None` | CameraInfo topic providing intrinsics and camera frame_id (sensor_msgs/CameraInfo). |
| `~camera_info_txt` | `str` | `''` | Optional path to a camera calibration text file. When set, intrinsics are loaded from file and ~camera_info topic is optional. |
| `~camera_metadata_topic` | `str` | `''` | Camera metadata topic for rolling shutter readout (realsense2_camera_msgs/Metadata). |
| `~cloud_stamp_source` | `str` | `''` | Timestamp source for published PointCloud2: auto, semantic, depth, lidar, latest, earliest, midpoint. |
| `~cloud_time_offset_sec` | `float` | `0.0` | Signed offset (seconds) added to published cloud timestamps (negative shifts earlier). |
| `~color_map` | `dict` | `None` | Optional dict {label_id: [r,g,b]} used to colorize labels when ~semantic_input_type='labels'. YAML keys must be quoted (e.g. "0": [0,0,0]). |
| `~colorize_labels` | `bool` | `false` | If true, publish an extra PointCloud2 field 'rgb' (label palette in 'labels' mode; passthrough colors in 'rgb' mode). |
| `~compat_declared_lidar_T_points` | `list[16]` | `None` | Optional static 4x4 row-major matrix mapping incoming point-data coordinates into the declared LiDAR frame. Applied before deskew/projection. Overrides the built-in ~compat_ouster_sensor_frame transform when provided. |
| `~compat_ouster_sensor_frame` | `bool` | `false` | Legacy-bag compatibility: treat incoming Ouster PointCloud2 XYZ as sensor-frame points mislabeled as the LiDAR frame and convert them back into the declared LiDAR frame before deskew/projection. |
| `~confidence_topic` | `str` | `None` | Optional confidence image topic aligned with semantic labels (sensor_msgs/Image). |
| `~core_debug` | `bool` | `false` | Enable entfac_fusion_core DEBUG logs (can be noisy at 10–30 Hz). |
| `~debug` | `bool` | `false` | Enable debug parameter report at startup (and DEBUG logs if set via launch arg). |
| `~debug_project_lidar` | `bool` | `false` | If true (lidar mode), publish a debug image with projected lidar points overlaid. |
| `~debug_project_lidar_outline_only` | `bool` | `false` | If true, draw projected LiDAR markers as outlines so the RGB image stays visible underneath. |
| `~debug_project_lidar_radius` | `int` | `0` | Marker radius in pixels for the projected LiDAR debug overlay (0 draws single pixels). |
| `~debug_project_lidar_stride` | `int` | `5` | Subsample factor for projected LiDAR debug overlay (1 draws every projected point). |
| `~debug_output_dir` | `str` | `''` | Directory where sampled debug overlays are written (empty uses `<entfac_fusion_ros>/output/debug`). |
| `~debug_output_stride` | `int` | `20` | Save every Nth debug callback per stream (1 saves every frame). |
| `~debug_publish_fov_points` | `bool` | `false` | If true (lidar mode), publish only the LiDAR points that passed the camera FOV test as a debug PointCloud2 in the LiDAR frame. |
| `~debug_range_view` | `bool` | `false` | If true (lidar mode), publish LiDAR depth/edge images, a reprojection heatmap, and an alignment score. |
| `~depth_input_topic` | `str` | `None` | Geometry input topic: depth (sensor_msgs/Image) or LiDAR (sensor_msgs/PointCloud2). The node auto-detects which message type is published and selects the fusion mode. |
| `~depth_scale` | `float` | `0.0` | Scale factor to convert depth values to meters (0=auto: 16UC1/mono16 treated as mm -> 0.001; 32FC1 treated as meters -> 1.0). |
| `~downsample_factor` | `int` | `1` | Integer >=1 stride used to subsample images for CPU/ARM targets. |
| `~enable_profiling` | `bool` | `false` | If true, print a short cProfile summary per callback (future C++/numba profiling hook). |
| `~filter_invalid_depth` | `bool` | `true` | If true, treat common uint16 depth sentinels (0, 65535) as invalid before scaling. |
| `~imu_cache_max_dt_sec` | `float` | `0.02` | Max allowed dt (seconds) between semantic frame and IMU for correction. |
| `~imu_cache_size` | `int` | `2000` | IMU cache size for rolling shutter correction. |
| `~imu_frame` | `str` | `''` | Optional IMU frame override for rolling shutter correction. |
| `~imu_topic` | `str` | `''` | IMU topic used for rolling shutter correction (sensor_msgs/Imu). |
| `~include_unlabeled` | `bool` | `false` | If true, keep points outside the camera FOV (label=-1). |
| `~lidar_deskew_enable` | `bool` | `false` | Enable LiDAR deskew using per-point time + IMU. |
| `~lidar_deskew_imu_samples` | `int` | `1` | Number of IMU samples used across each scan for LiDAR deskew (1 keeps the lightweight single-sample model; values >1 better handle fast motion). |
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
| `~online_calibration_edge_threshold` | `float` | `0.2` | Edge threshold in [0,1] used for observability density checks on semantic/depth edge maps. |
| `~online_calibration_enable` | `bool` | `false` | Enable lightweight online LiDAR-camera misalignment estimation with health/uncertainty and small projection correction (classical, no neural models). |
| `~online_calibration_every_n_frames` | `int` | `10` | Run online calibration update every N lidar callbacks (>=1). |
| `~online_calibration_health_ema_alpha` | `float` | `0.15` | EMA alpha for calibration health score smoothing. |
| `~online_calibration_health_score_center` | `float` | `0.25` | Alignment-score midpoint used by the health logistic transfer. |
| `~online_calibration_health_score_scale` | `float` | `0.1` | Alignment-score scale used by the health logistic transfer. |
| `~online_calibration_health_std_scale` | `float` | `0.08` | Scale that maps alignment-score std into stability confidence. |
| `~online_calibration_health_std_window` | `int` | `40` | Sliding window size used to estimate alignment-score stability. |
| `~online_calibration_learning_rate` | `float` | `0.25` | Update gain for online rotational correction (smaller is more conservative). |
| `~online_calibration_log_period_sec` | `float` | `2.0` | Minimum seconds between online calibration status logs. |
| `~online_calibration_max_correction_deg` | `float` | `3.0` | Clamp for each correction angle component (roll/pitch/yaw) in degrees. |
| `~online_calibration_max_points` | `int` | `8000` | Max number of LiDAR points used by online calibration updates (uniform stride subsampling above this). |
| `~online_calibration_min_depth_edge_density` | `float` | `0.01` | Minimum LiDAR depth-edge density expected for well-observable frames. |
| `~online_calibration_min_fov_points` | `int` | `500` | Minimum in-FOV LiDAR points required by the online calibration update. |
| `~online_calibration_min_observability` | `float` | `0.15` | Minimum observability required before correction updates are applied. |
| `~online_calibration_min_sem_edge_density` | `float` | `0.01` | Minimum semantic edge density expected for well-observable frames. |
| `~online_calibration_step_deg` | `float` | `0.2` | Finite-difference perturbation step in degrees for rotational misalignment estimation. |
| `~pair_max_dt_sec` | `float` | `0.03` | Hard max allowed |Δt| (seconds) between semantic and geometry; <=0 disables. |
| `~ply_output_dir` | `str` | `''` | Directory where PLY files are written (empty uses <entfac_fusion_ros>/output/ply). |
| `~ply_target_frame` | `str` | `''` | Optional TF frame to transform PLY output to (ply_target_frame <- target_frame). Empty means use target_frame. |
| `~ply_tf_tolerance_sec` | `float` | `0.02` | Max allowed time difference (seconds) when using latest TF for PLY export. |
| `~ply_tf_use_latest` | `bool` | `false` | When true, fall back to the latest TF for PLY export if exact-time lookup fails. |
| `~projection_confidence_min` | `float` | `0.0` | Minimum patch confidence required to trust transferred image color/label (0 disables). |
| `~projection_depth_edge_radius_px` | `int` | `0` | Pixel radius used to dilate the LiDAR depth-edge reject mask (helps suppress sky bleed near thin objects). |
| `~projection_depth_edge_thresh` | `float` | `0.15` | Normalized depth-edge threshold used when ~projection_reject_depth_edges is enabled. |
| `~projection_invalid_mask_dilate_px` | `int` | `0` | Optional dilation radius in pixels applied to the invalid mask before projection sampling. |
| `~projection_invalid_mask_topic` | `str` | `''` | Optional single-channel invalid-mask image topic aligned with ~semantic_topic; pixels equal to ~projection_invalid_mask_value reject transferred labels/RGB. |
| `~projection_invalid_mask_value` | `int` | `255` | Pixel value in ~projection_invalid_mask_topic that marks invalid/rejected samples. |
| `~projection_occlusion_epsilon_m` | `float` | `0.0` | Allow image transfer only when the point depth is within this margin of the nearest LiDAR depth at that pixel (meters, 0 disables). |
| `~projection_occlusion_radius_px` | `int` | `0` | Pixel radius for local min-depth occlusion gating (0 uses only the exact projected pixel). |
| `~projection_patch_size` | `int` | `1` | Odd patch size for robust LiDAR-to-image sampling (1=center pixel, 3=3x3, 5=5x5). |
| `~projection_reject_depth_edges` | `bool` | `false` | If true, reject color/label transfer for projected points that land on strong LiDAR depth discontinuities. |
| `~random_color_seed` | `int` | `1` | Seed for deterministic random label palette when ~colorize_labels is true and ~color_map is empty. |
| `~rolling_shutter_direction` | `str` | `'top_to_bottom'` | Rolling shutter readout direction: top_to_bottom or bottom_to_top. |
| `~rolling_shutter_enable` | `bool` | `false` | Apply rotation-only rolling shutter correction using IMU. |
| `~rolling_shutter_readout_sec` | `float` | `0.0` | Rolling shutter total readout time in seconds (0 disables). |
| `~semantic_color_quantization_step` | `int` | `1` | Quantize RGB/BGR semantic images to nearest multiple of this step before packing for the PointCloud2 rgb field (helps with JPEG artifacts). |
| `~semantic_input_type` | `str` | `'labels'` | Semantic image representation: 'labels' (single-channel label IDs) or 'rgb' (3-channel colors used directly for output coloring). |
| `~semantic_time_offset_sec` | `float` | `0.0` | Signed offset (seconds) applied to semantic timestamps for pairing and timestamp selection (negative shifts semantic earlier). |
| `~semantic_topic` | `str` | `'/semantic/labels'` | Semantic label image topic (sensor_msgs/Image). |
| `~stamp_debug_log_period_sec` | `float` | `2.0` | Minimum period (seconds) between timestamp/offset debug logs; set 0 to log every callback when debug=true. |
| `~static_camera_T_lidar` | `list[16]` | `None` | Optional static 4x4 row-major matrix: lidar_frame -> camera_frame. Overrides TF. |
| `~static_target_T_depth` | `list[16]` | `None` | Optional static 4x4 row-major matrix: depth_frame -> target_frame. Overrides TF. |
| `~static_target_T_lidar` | `list[16]` | `None` | Optional static 4x4 row-major matrix: lidar_frame -> target_frame. Overrides TF. |
| `~status_period` | `raw` | `''` | Seconds between periodic status table prints. Empty=auto (1s when debug=true, else disabled). Set to 0 to disable explicitly. |
| `~sync_queue_size` | `int` | `5` | ApproximateTimeSynchronizer queue size for semantic/depth or semantic/lidar pairing. |
| `~sync_slop_sec` | `float` | `0.1` | ApproximateTimeSynchronizer slop in seconds for semantic/depth or semantic/lidar pairing. |
| `~target_frame` | `str` | `'base_link'` | Output frame for published semantic point cloud. |
| `~tracked_reprojection_depth_edge_thresh` | `float` | `0.15` | Normalized LiDAR depth-edge threshold used to convert the projected depth map into an edge target for tracked reprojection. |
| `~tracked_reprojection_enable` | `bool` | `false` | Enable stateful feature-tracked LiDAR reprojection diagnostics. This is heavier than the online edge score and is intended mainly for offline rosbag review. |
| `~tracked_reprojection_fb_thresh_px` | `float` | `1.5` | Forward-backward optical-flow consistency threshold in pixels. |
| `~tracked_reprojection_log_period_sec` | `float` | `2.0` | Minimum seconds between tracked reprojection status logs. |
| `~tracked_reprojection_max_corners` | `int` | `300` | Maximum number of tracked image features used by the tracked reprojection diagnostic. |
| `~tracked_reprojection_min_distance_px` | `float` | `8.0` | Minimum pixel spacing between tracked reprojection features. |
| `~tracked_reprojection_min_image_edge` | `float` | `0.05` | Minimum image-edge strength required for a tracked feature to contribute to the reprojection error metric. |
| `~tracked_reprojection_min_tracks` | `int` | `80` | Minimum number of active tracks to maintain before replenishing features. |
| `~tracked_reprojection_quality_level` | `float` | `0.01` | Shi-Tomasi quality level for tracked reprojection feature detection. |
| `~undistort_alpha` | `float` | `0.0` | Undistort balance/alpha in [0,1]; 0=crop to valid pixels, 1=keep all pixels. |
| `~undistort_semantic` | `bool` | `false` | If true, undistort semantic images using CameraInfo distortion before projection (lidar mode only). |
