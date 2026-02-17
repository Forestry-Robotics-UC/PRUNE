# Failure-mode playbook

## Startup / configuration

- **Stuck waiting for `~camera_info`**
  - Symptom: repeated timeout logs during init.
  - Likely causes: wrong topic name, bag not playing, or `/use_sim_time` without `/clock`.

- **Auto-detect never resolves `~depth_input_topic`**
  - Symptom: periodic warning: “Waiting for … to appear to auto-detect mode”.
  - Likely causes: topic not published, wrong namespace, or using a transport-specific topic (e.g. `/compressed`).

- **TF lookup warnings / no output**
  - Symptom: “No depth->target transform available” or “No lidar transforms available”.
  - Likely causes: missing URDF/robot_state_publisher, wrong `~target_frame`, wrong sensor frame IDs, or missing static params.

## Runtime output issues

- **Published cloud is empty**
  - Depth mode: depth image invalid (all zeros/NaNs), wrong `~depth_scale`, or wrong `~static_target_T_depth`.
  - LiDAR mode: points behind camera, wrong `~static_camera_T_lidar`, wrong intrinsics, or image/LiDAR not time-aligned.

- **Frequent pair drops**
  - Symptom: warnings like “Dropping pair: |Δt| > ...”.
  - Likely causes: strict `~pair_max_dt_sec`, clock drift, bag timestamp issues, or incorrect topic pairing.
  - Mitigation: verify stamps and tune `~pair_max_dt_sec` conservatively.

- **Rolling-shutter correction not applied**
  - Symptom: correction enabled but behavior unchanged.
  - Likely causes: missing `~imu_topic`, stale IMU cache (`~imu_cache_max_dt_sec` too small), or missing metadata when relying on `~camera_metadata_topic`.
  - Mitigation: validate IMU stream timing and optionally set fixed `~rolling_shutter_readout_sec`.

- **LiDAR deskew not applied**
  - Symptom: deskew enabled but warnings or unchanged cloud quality.
  - Likely causes: missing `~lidar_imu_topic`, missing per-point time field (`~lidar_time_field`), bad `~lidar_time_scale`, or poor IMU/LiDAR sync.
  - Mitigation: confirm PointCloud2 contains the configured time field and verify IMU coverage of scan interval.

- **Cloud exists but not visible in RViz**
  - Symptom: RViz shows topic activity but no points.
  - Likely causes: RViz fixed frame mismatch (`~target_frame`), points far away due to wrong transform scale/units, or depth scale mismatch.

- **RGB semantic images show “many colors”**
  - Symptom: unexpected palette size when inspecting colors.
  - Likely causes: lossy JPEG transport for palette images. Prefer label-ID images or lossless PNG compression upstream; `~semantic_color_quantization_step` can reduce noise for visualization but cannot recover exact class IDs.

## Debugging knobs

- `~debug:=true`: prints startup param report + one-time input summaries.
- `~core_debug:=true`: enables core DEBUG logs (can be noisy at 10–30 Hz).
- `~status_period`: periodic ASCII status table (set to `1.0` for 1 Hz, `0` disables).
