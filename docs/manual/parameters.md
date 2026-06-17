# Parameter reference

Generated from `prune_ros/prune_ros/` (all `config/*.py`, `node/`, `startup/`) using:

```bash
python docs/tools/extract_ros_params.py > docs/manual/parameters.md
```

Defaults are primarily defined in `prune_ros/config/core.yaml` and `prune_ros/config/expert.yaml`, then overridden by launch-time params when set.

| Param | Type | Default | Description |
|---|---|---|---|
| `~adaptive_confidence_threshold_offset` | `float` | `0.1` | Offset added to the confidence threshold when adaptive projection health is enabled and health is bad. |
| `~adaptive_depth_edge_threshold_scale` | `float` | `0.8` | Scale applied to depth-edge threshold when adaptive projection health is enabled and health is bad; values below 1 reject more edge-near points. |
| `~adaptive_prefer_suppression_on_bad_health` | `bool` | `true` | When adaptive projection health is enabled and health is bad, prefer suppressing labels over deleting geometry where the active PRUNE path supports suppression. |
| `~bag_name` | `str` | `''` | Optional bag identifier recorded in metrics CSV outputs. |
| `~camera_fov_gate_enable` | `bool` | `true` | Drop LiDAR points outside the camera FoV before projection. Reduces processed point count from 360-deg LiDAR to ~18% on a typical 70-deg camera, giving ~5x speedup on downstream projection/sampling steps. |
| `~camera_fov_gate_margin_deg` | `float` | `5.0` | Angular margin in degrees added to each side of the camera FoV gate to avoid hard cutoffs at image edges. |
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
| `~core_debug` | `bool` | `false` | Enable prune_core DEBUG logs (can be noisy at 10–30 Hz). |
| `~debug` | `bool` | `false` | Enable debug parameter report at startup (and DEBUG logs if set via launch arg). |
| `~debug_output_dir` | `str` | `None` | Directory for saved debug images and sidecar outputs. |
| `~debug_output_stride` | `int` | `1` | Save every Nth debug frame to disk. |
| `~debug_project_lidar` | `bool` | `false` | Publish LiDAR projection overlay debug images. |
| `~debug_project_lidar_outline_only` | `bool` | `false` | Render LiDAR projection overlay as outlines only. |
| `~debug_project_lidar_radius` | `int` | `1` | Point radius in pixels for LiDAR projection overlay rendering. |
| `~debug_project_lidar_stride` | `int` | `1` | Subsample stride used when rendering LiDAR projection overlays. |
| `~debug_publish_fov_points` | `bool` | `false` | Publish LiDAR points that survive the camera FoV gate. |
| `~debug_range_view` | `bool` | `false` | Publish depth/edge/range-view debug images. |
| `~depth_input_topic` | `str` | `None` | Geometry input topic: depth (sensor_msgs/Image) or LiDAR (sensor_msgs/PointCloud2). The node auto-detects which message type is published and selects the fusion mode. |
| `~depth_map_subsample` | `int` | `1` | Depth buffer resolution divisor (1=full, 2=half, 4=quarter). Reduces rasterization and edge-map cost. |
| `~depth_scale` | `float` | `0.0` | Scale factor to convert depth values to meters (0=auto: 16UC1/mono16 treated as mm -> 0.001; 32FC1 treated as meters -> 1.0). |
| `~downsample_factor` | `int` | `1` | Uniform integer downsample applied to semantic/depth images before fusion. |
| `~edge_cache_max_age_sec` | `float` | `0.0` | Reuse the edge map for frames within this time window (seconds). 0 disables caching. |
| `~enable_adaptive_projection_health` | `bool` | `false` | If true, make PRUNE projection gates more conservative when projection-health diagnostics are poor. |
| `~enable_metrics_csv` | `bool` | `false` | Write per-frame metrics CSV for LiDAR experiments. |
| `~enable_profiling` | `bool` | `false` | If true, print a short cProfile summary per callback (future C++/numba profiling hook). |
| `~experiment_variant_name` | `str` | `''` | Optional experiment label recorded in metrics CSV outputs. |
| `~filter_invalid_depth` | `bool` | `true` | If true, treat common uint16 depth sentinels (0, 65535) as invalid before scaling. |
| `~imu_cache_max_dt_sec` | `float` | `0.02` | Max allowed dt (seconds) between semantic frame and IMU for correction. |
| `~imu_cache_size` | `int` | `2000` | IMU cache size for rolling shutter correction. |
| `~imu_frame` | `str` | `''` | Optional IMU frame override for rolling shutter correction. |
| `~imu_topic` | `str` | `''` | IMU topic used for rolling shutter correction (sensor_msgs/Imu). |
| `~include_unlabeled` | `bool` | `false` | If true, keep points outside the camera FoV as unlabeled samples instead of dropping them. |
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
| `~overlay_dot_radius` | `int` | `2` | Dot radius in pixels for overlay layers. |
| `~overlay_output_dir` | `str` | `''` | Directory for 3-layer GIMP overlays (base, accepted, depth). Empty = disabled. |
| `~overlay_output_stride` | `int` | `20` | Save overlay every Nth accepted frame. |
| `~pair_max_dt_sec` | `float` | `0.03` | Hard max allowed |Δt| (seconds) between semantic and geometry; <=0 disables. |
| `~perception_invalid_label` | `int` | `65535` | Label value from Perception indicating invalid/low-confidence pixels; mapped to -1 (unlabeled) before fusion. The perception stack uses 65535 by default. |
| `~ply_output_dir` | `str` | `''` | Optional directory for saved PLY files. Empty uses the current debug/output directory. |
| `~ply_recording_enable` | `bool` | `false` | If true, write every published cloud to PLY asynchronously. |
| `~ply_target_frame` | `str` | `''` | Optional target frame for saved PLY clouds. Empty uses the published cloud frame. |
| `~ply_tf_tolerance_sec` | `float` | `0.0` | Tolerance window for exact-time PLY TF lookup. Zero requires an exact transform stamp. |
| `~ply_tf_use_latest` | `bool` | `false` | When true, PLY export uses the latest TF instead of exact timestamp lookup. |
| `~projection_confidence_min` | `float` | `0.0` | Minimum patch confidence required to trust transferred image color/label (0 disables). |
| `~projection_depth_edge_radius_px` | `int` | `0` | Optional radius used to expand depth-edge rejection neighborhoods. |
| `~projection_depth_edge_thresh` | `float` | `0.15` | Depth discontinuity threshold in meters/pixel for the depth-edge gate. |
| `~projection_health_bad_threshold` | `float` | `0.25` | Projection-health score below this value is treated as bad for optional adaptive gates. |
| `~projection_health_warn_threshold` | `float` | `0.5` | Projection-health score below this value is reported as warning quality. |
| `~projection_invalid_mask_dilate_px` | `int` | `0` | Optional dilation radius in pixels applied to the invalid mask before projection sampling. |
| `~projection_invalid_mask_topic` | `str` | `''` | Optional single-channel invalid-mask image topic aligned with ~semantic_topic; pixels equal to ~projection_invalid_mask_value reject transferred labels/RGB. |
| `~projection_invalid_mask_value` | `int` | `255` | Pixel value in ~projection_invalid_mask_topic that marks invalid/rejected samples. |
| `~projection_occlusion_epsilon_m` | `float` | `0.0` | Allow image transfer only when the point depth is within this margin of the nearest LiDAR depth at that pixel (meters, 0 disables). |
| `~projection_occlusion_radius_px` | `int` | `0` | Optional pixel radius used when evaluating the LiDAR depth support map for occlusion rejection. |
| `~projection_patch_size` | `int` | `1` | Odd patch size for robust LiDAR-to-image sampling (1=center pixel, 3=3x3, 5=5x5). |
| `~projection_reject_depth_edges` | `bool` | `false` | Reject image transfer near strong LiDAR depth discontinuities. |
| `~random_color_seed` | `int` | `1` | Seed for deterministic random label palette when ~colorize_labels is true and ~color_map is empty. |
| `~results_dir` | `str` | `''` | Optional output directory for metrics/debug result bundles. |
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
| `~tracked_reprojection_depth_edge_thresh` | `float` | `0.15` | Depth-edge threshold reused by tracked reprojection diagnostics. |
| `~tracked_reprojection_enable` | `bool` | `false` | Enable tracked reprojection diagnostics. |
| `~tracked_reprojection_fb_thresh_px` | `float` | `1.0` | Forward-backward pixel error threshold for tracked reprojection. |
| `~tracked_reprojection_log_period_sec` | `float` | `2.0` | Minimum seconds between tracked reprojection log messages. |
| `~tracked_reprojection_max_corners` | `int` | `200` | Maximum number of tracked reprojection corners. |
| `~tracked_reprojection_min_distance_px` | `float` | `7.0` | Minimum feature spacing for tracked reprojection. |
| `~tracked_reprojection_min_image_edge` | `float` | `3.0` | Minimum image-edge margin for tracked reprojection features. |
| `~tracked_reprojection_min_tracks` | `int` | `20` | Minimum active tracks before the tracked reprojection diagnostic resets. |
| `~tracked_reprojection_quality_level` | `float` | `0.01` | Feature quality threshold for tracked reprojection. |
| `~undistort_alpha` | `float` | `0.0` | Undistort balance/alpha in [0,1]; 0=crop to valid pixels, 1=keep all pixels. |
| `~undistort_semantic` | `bool` | `false` | If true, undistort semantic images using CameraInfo distortion before projection (lidar mode only). |
| `~use_depth_edge_rejection` | `bool` | `true` | Enable G2 depth-edge evidence gate in experiment metrics and projector runtime. |
| `~use_invalid_mask` | `bool` | `true` | Enable G1 invalid-mask evidence gate in experiment metrics and projector runtime. |
| `~use_occlusion_gate` | `bool` | `true` | Enable G3 occlusion evidence gate in experiment metrics and projector runtime. |
| `~use_range_image_edges` | `str` | `'auto'` | Edge-map strategy: "auto" uses range-image path for organized clouds, "always", or "never". |
