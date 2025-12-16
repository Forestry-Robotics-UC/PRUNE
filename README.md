# ENTFAC Sensor Fusion

Stateless sensor fusion module for ENTFAC that turns single-frame semantic
perception outputs plus geometry into semantic point cloud measurements. Mapping
and temporal fusion live elsewhere.

## Attribution and Licensing

ENTFAC Sensor Fusion is a derivative work based on the open-source **Semantic SLAM**
implementation originally developed by **Xuan Zhang**, and later extended by
**David Russell**, particularly in the lidar-to-camera projection and semantic
point cloud generation components.

Original project:
https://github.com/floatlazer/semantic_slam

Significant modifications, refactoring, and architectural changes have been made
to support the ENTFAC modular sensor fusion framework, including ROS1 integration,
semantic fusion strategies, and system-level reorganization.

This project is distributed under the **GNU General Public License v3.0 (GPL-3.0)**.
See `LICENSE`.

## Responsibilities
- Input: semantic labels (+ optional confidence), depth image or LiDAR points,
  camera intrinsics, and frame transforms.
- Output: `SemanticPointCloud` in a target frame with per-point label and optional
  confidence. No map state is stored.
- Modes: depth-based fusion (RGB-D) and LiDAR projection fusion (LiDAR + camera).

## Repository Layout
- `entfac_fusion_core/`: catkin Python package; numpy-only, ROS-agnostic core.
  - Python sources live in `entfac_fusion_core/src/entfac_fusion_core/`.
- `entfac_fusion_ros/`: catkin ROS wrapper package.
  - `nodes/semantic_pcl_node.py`: bridge topics to the core and publish
    `sensor_msgs/PointCloud2` with fields `x y z label [confidence]`.
  - `config/semantic_pcl.yaml`: default parameters with optional static extrinsics.
  - `launch/semantic_pcl.launch`: generic launch (optional image_transport republish).
  - `launch/choupal_semantic_pcl.launch`: Choupal bag demo (bag play + TF + republish).
- `tests/`: pytest coverage for core fusion paths.

## Core Usage (numpy)
```python
from entfac_fusion_core.semantic_pcl import fuse_depth_semantics
from entfac_fusion_core.types import SemanticObservation, DepthObservation
import numpy as np

labels = np.zeros((480, 640), dtype=np.int32)
depth = np.ones((480, 640), dtype=float)
intrinsics = np.eye(3)
target_T_depth = np.eye(4)

pcl = fuse_depth_semantics(
    SemanticObservation(labels=labels),
    DepthObservation(depth=depth),
    intrinsics,
    target_T_depth,
)
# pcl.points_xyz, pcl.labels, pcl.confidence
```

## ROS Node (`semantic_pcl_node.py`)
- Parameters:
  - `~semantic_topic`: single-channel semantic labels (Image).
  - `~confidence_topic` (optional): confidence image aligned to semantic labels.
  - `~camera_info`: CameraInfo for intrinsics + frame id.
  - `~depth_input_topic`: geometry input topic; set to either a depth `sensor_msgs/Image` or a `sensor_msgs/PointCloud2` (LiDAR). The node auto-detects which and selects the fusion mode.
  - Deprecated: `~depth_topic` and `~lidar_topic` (still supported for backwards-compat).
  - Mode auto-detected from `~depth_input_topic` (depth if Image, lidar if PointCloud2). You can still set `~mode` to force.
  - `~target_frame`: frame for output cloud (default `base_link`).
  - `~include_unlabeled_pts`: keep points outside the camera FOV as label `-1`.
  - `~auto_color_to_label`: if the semantic image is RGB/BGR and `~color_map` is empty, infer a deterministic palette→label mapping (sorted by packed RGB).
  - `~auto_color_to_label_extend`: cache new colors when they appear by snapping them to the inferred palette (label IDs do not grow).
  - `~auto_color_to_label_min_fraction`, `~auto_color_to_label_min_count`, `~auto_color_to_label_max_colors`: filters/caps the inferred palette (useful for JPEG artifacts).
  - `~semantic_color_quantization_step`: quantize RGB/BGR semantic images before color→label decode (set to 8/16 for JPEG artifacts; 1 disables).
  - `~auto_color_to_label_merge_distance`: merge similar colors (after quantization) to reduce JPEG palette noise.
  - `~colorize_labels`: add an `rgb` field based on label IDs (default false).
  - `~downsample_factor`: integer >=1 to subsample labels/depth for CPU-bound/ARM.
  - `~enable_profiling`: cProfile summary per callback (off by default).
  - Extrinsics: provide static 4×4 matrices (`~static_target_T_depth`,
    `~static_camera_T_lidar`, `~static_target_T_lidar`) or rely on TF/URDF.
- Publishes: `semantic_pointcloud` (`PointCloud2`) with fields `label` and optional
  `confidence`.
- TF: looks up transforms from depth/LiDAR frame to `target_frame`, and from
  LiDAR frame to camera frame when in `lidar` mode.
  Static matrices override TF if provided; TF is resolved once at startup (URDF).
- Conversions use numpy buffer parsing (no `ros_numpy`) for lower overhead.
- Logging: core uses Python `logging`; ROS node logs via `rospy` (info for counts,
  warnings when TF/extrinsics are missing or no valid points are found).

## Extrinsics options
- Use TF/URDF: provide proper static transforms for camera ↔ depth ↔ target frames.
- Or provide static 4×4 row-major matrices via params:
  - `static_target_T_depth` (depth frame → target frame)
  - `static_camera_T_lidar` (lidar frame → camera frame)
  - `static_target_T_lidar` (lidar frame → target frame)
  See `entfac_fusion_ros/config/semantic_pcl.yaml` for layout.

## Dependencies
- Python (core/tests): `numpy`, `pytest` (see `requirements.txt`).
- ROS (wrappers): see `entfac_fusion_core/package.xml` and `entfac_fusion_ros/package.xml`.
  Recommended:
  - `rosdep update`
  - `rosdep install --from-paths src --ignore-src -r -y`

## Docker (core tests)
```bash
docker build -t entfac-sensor-fusion -f Docker/entfac-sensor-fusion.Dockerfile .
docker run --rm entfac-sensor-fusion
```

## Docker (ROS)
- Run with bags mounted (edit `docker-compose.yml` as needed):
  ```bash
  docker compose run --rm sensor-fusion-ros
  # inside:
  source /opt/ros/noetic/setup.bash
  rosdep update
  rosdep install --from-paths src --ignore-src -r -y
  catkin_make
  source devel/setup.bash
  roslaunch entfac_fusion_ros choupal_semantic_pcl.launch
  ```
  Bags are available under `/bags`.

## Docker (ROS + GUI / RViz)
- Optional X11-forwarding service for debugging GUI tools (RViz, rqt) from inside the container:
  ```bash
  xhost +si:localuser:$(whoami)
  docker compose run --rm sensor-fusion-ros-gui
  # inside:
  rviz
  ```
  To revoke access: `xhost -si:localuser:$(whoami)`.
  If `rviz` is not installed in your image, install `ros-noetic-rviz` (or run RViz on the host).

## Testing
```bash
pytest -q
```

## ROS launch
- Build your workspace, source setup, set topics/extrinsics in `entfac_fusion_ros/config/semantic_pcl.yaml`.
- Launch (auto-detects depth vs. LiDAR based on the provided topics):
  ```bash
  roslaunch entfac_fusion_ros semantic_pcl.launch
  ```
- Debug startup report + DEBUG logs:
  ```bash
  roslaunch entfac_fusion_ros semantic_pcl.launch debug:=true
  ```
- To decompress compressed topics, set `use_republish:=true` and provide base input topics (no `/compressed` suffix), e.g. `semantic_in_topic:=/segmentation/test`.
- Choupal bag example (plays bags, republishes `/segmentation/test` from compressed, and loads TF from `/bags/sensor-box.urdf`):
  ```bash
  roslaunch entfac_fusion_ros choupal_semantic_pcl.launch
  ```
- Param precedence: YAML loaded via `rosparam` sets defaults; later `<param>` tags override. Avoid setting empty-string params in launch files since they overwrite YAML.

## Semantic colors
- If the semantic topic is a single-channel label image (`mono8`, `16UC1`, `32SC1`), no palette is needed.
- If the semantic topic is a 3/4-channel palette image (`rgb8`, `bgr8`, `rgba8`, `bgra8`), the node must convert colors → label IDs to populate the `label` field and run fusion.
  Options:
  - Recommended: set `semantic_pcl_node/color_map` for stable, correct class IDs.
  - Debug-friendly: set `semantic_pcl_node/auto_color_to_label:=true` to infer a palette and snap observed colors to the nearest palette entry (robust to JPEG artifacts).

Notes:
- If your semantic image is transported as `image_transport/compressed` using JPEG, the decoded image can contain many near-duplicate colors even if you only have a handful of classes. Republish cannot recover the original palette; prefer publishing class IDs (single-channel) or use lossless PNG compression upstream.
- For JPEG/noisy palette streams, tune `semantic_color_quantization_step` (8/16), `auto_color_to_label_merge_distance`, and cap `auto_color_to_label_max_colors` to match your expected class count.

Example `color_map` (original semfire palette):
```yaml
color_map:
  0: [0, 0, 0]        # Background
  1: [0, 0, 128]      # Fuel
  2: [0, 50, 100]     # Trunks
  3: [0, 213, 255]    # Humans
  4: [163, 0, 128]    # Animals
  5: [0, 51, 0]       # Canopies
  6: [165, 165, 165]  # Traversable
```

## Notes on design and performance
- Separation of core vs. ROS keeps the math testable without ROS, and supports
  future backends (mapping/TSDF) with minimal coupling; even for a small module
  this reduces ROS message churn in tests and makes portability easier.
- LiDAR projection assumes standard XYZ in sensor frame, compatible with common
  vendors (Ouster, Livox, Velodyne) once their drivers publish `PointCloud2`.
  Extrinsics can come from TF/URDF or static params for bag replay.
- Potential bottlenecks: full-image meshgrid creation (now cached), image
  copies, and PointCloud2 packing; the node avoids per-point Python loops by
  packing output clouds with NumPy (structured array → `.tobytes()`), but higher
  resolutions may still benefit from `downsample_factor` or upstream
  downsampling.
- LiDAR compatibility: projection expects XYZ in sensor frame; standard Ouster,
  Livox, and Velodyne ROS drivers publish `PointCloud2` in this form, so only
  extrinsics and intrinsics are required.
- Separation rationale: keeping a ROS-free core makes unit testing and future
  backend swaps (e.g., TSDF/octree) cheaper, even if the fusion surface is small.
- ARM/Jetson tips:
  - Use `downsample_factor` to reduce per-frame work.
  - Ensure OpenBLAS/BLIS is installed; set `OPENBLAS_NUM_THREADS` to the count of
    big cores to avoid oversubscription.
  - Pin the node to a big core if needed (`taskset`) and keep labels single-channel
    to avoid extra copies.
  - Downsampling uses stride slicing (nearest-neighbor), so labels remain valid.
