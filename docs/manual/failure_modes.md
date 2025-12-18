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

