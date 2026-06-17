# PRUNE v1.0.0 (ROS 1 Noetic)

PRUNE converts single-frame semantic perception + geometry into semantic 3D
measurements (`sensor_msgs/PointCloud2`) for downstream mapping. It replaces
standard geometric LiDAR-to-camera projection with temporal input validation,
optional motion pre-correction, and a sequence of point-level evidence gates.

This release is **stateless**: PRUNE maintains no semantic map, object map, or
accumulated voxel state. Persistent map integration is the responsibility of a
downstream mapper.

---

## Features

### Temporal pairing and motion pre-correction
- `sync_slop_sec` / `pair_max_dt_sec` candidate pairing with a maximum allowed
  semantic/geometry timestamp offset; out-of-window pairs are logged and dropped
  rather than projected.
- Rolling-shutter correction: per-row rotational correction from an IMU ring
  buffer integrated over the readout interval.
- LiDAR deskew: per-point rotational (and optional translational,
  constant-acceleration) correction to a common reference time.

### Evidence gates (selective projection)
- **G1 — Invalid-mask rejection**: patch-based test against an upstream
  invalid-region mask (e.g. sky), with optional dilation; majority-vote label
  sampling over non-unknown labels in the same patch.
- **G2 — Depth-edge rejection**: Sobel-based boundary-risk detection on the
  projected LiDAR depth image; rejects ambiguous foreground/background
  silhouette regions (canopy, branches, vegetation edges).
- **G3 — Occlusion consistency**: local z-buffer-style check against the
  nearest visible surface in the full-scan depth image; rejects points with no
  local depth support or that lie substantially behind the nearest surface.
- **G4 — Confidence threshold**: suppresses semantic evidence below a
  configurable confidence floor while preserving point geometry.
- Gates run sequentially; rejected points are excluded from later gates.
  Suppression-mode gates (confidence, invalid-mask) keep the 3D point but reset
  its label to the unknown sentinel and confidence to zero.
- Diagnostic would-hit counters are computed per gate, including gates that are
  disabled in the active configuration.

### Online-adjustable projection quality
- Per-frame `projection_health_score` ([0,1]) from in-front, in-image, and
  gate-rejection ratios.
- Optional adaptive mode: tightens the confidence threshold and depth-edge
  threshold automatically when projection health drops below a configured
  bad-health threshold, with no retraining or pipeline restructuring required.

### Fusion modes
- Depth mode: semantic image + aligned depth image → semantic point cloud.
- LiDAR mode: project LiDAR points into the semantic image → semantic point
  cloud.
- Output `PointCloud2` fields: `x y z` (float32), `label` (uint16; unknown =
  `65535`), `confidence` (float32, optional), `rgb` (float32 packed, optional).

### Extrinsics and I/O
- Preferred: TF2/URDF. Optional: static 4x4 matrices for bag replay or fixed
  rigs.
- Optional async PLY export with TF-aware target-frame transform.
- Optional debug overlays: projected-LiDAR image debug, range-view (depth/edge
  images, reprojection heatmap), and offline tracked-reprojection diagnostics.

### Tooling
- `tools/diagnostics/rosbag_time_skew.py`: nearest-neighbor timestamp skew
  stats between two topics.
- `tools/results/`: ablation-suite runner, metrics summarizer, paper-table and
  plot generation, GIMP/labeling overlay layer export.
- `tools/validation/`: deterministic ICNF results-directory validation and bag
  workflow checks.

### Architecture
- `prune_core` (ROS-agnostic): projection, fusion, transforms, validation —
  pure NumPy, unit-tested independent of ROS.
- `prune_ros`: ROS 1 Noetic node, conversions, PointCloud2 packing, TF
  utilities, status reporting.

---

## ROS interface (v1.0)

### Subscribed
- `semantic_topic` (`sensor_msgs/Image`) — required
- `camera_info` (`sensor_msgs/CameraInfo`) — required
- `depth_input_topic` (`sensor_msgs/Image` or `sensor_msgs/PointCloud2`) —
  required (auto-detected)
- `confidence_topic` (`sensor_msgs/Image`) — optional
- `projection_invalid_mask_topic` (`sensor_msgs/Image`) — optional
- `imu_topic` / `lidar_imu_topic` (`sensor_msgs/Imu`) — optional, for motion
  pre-correction
- `camera_metadata_topic` (`realsense2_camera_msgs/Metadata`) — optional, for
  rolling-shutter readout timing

### Published
- `semantic_pointcloud` (`sensor_msgs/PointCloud2`) — in `target_frame`

See `docs/manual/parameters.md` for the full parameter reference.

---

## Quickstart

```bash
rosdep update
rosdep install --from-paths src --ignore-src -r -y
catkin_make
source devel/setup.bash
roslaunch prune_ros prune.launch
```

Core unit tests:

```bash
pytest -q
```

---

## Attribution and license

PRUNE is a derivative work based on **Semantic SLAM** by **Xuan Zhang**, with
subsequent contributions by **David Russell**.

Original project: https://github.com/floatlazer/semantic_slam

License: **GPL-3.0**
