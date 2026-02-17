# ENTFAC Sensor Fusion

Stateless single-frame fusion for ENTFAC.

- Inputs: semantic image (+ optional confidence), geometry (depth image or LiDAR), camera intrinsics, extrinsics/TF.
- Output: semantic `PointCloud2` measurement in a target frame.
- No map state is stored here. Mapping/accumulation belongs in semantic mapping.

## Attribution and License

ENTFAC Sensor Fusion is derived from Semantic SLAM work by Xuan Zhang and David Russell:
https://github.com/floatlazer/semantic_slam

This repository is distributed under GPL-3.0 (`LICENSE`).

## Repository layout

- `entfac_fusion_core/`: ROS-agnostic NumPy fusion core.
  - Public fusion API: `entfac_fusion_core.colored_pcl`
- `entfac_fusion_ros/`: ROS1 Noetic wrapper node and launch/config.
  - Node executable: `scripts/colored_pcl_node.py`
  - Node implementation: `entfac_fusion_ros/colored_pcl_node.py`
  - Defaults: `config/core.yaml` + `config/expert.yaml`
  - Generic launch: `launch/colored_pcl.launch`
  - Forestsphere launch/profile: `launch/forestsphere.launch`, `config/forestsphere.yaml`
- `tests/`: core unit tests.
- `docs/`: Sphinx docs/manual.

## Core usage (NumPy)

```python
from entfac_fusion_core.colored_pcl import fuse_depth_semantics
from entfac_fusion_core.types import SemanticObservation, DepthObservation
import numpy as np

labels = np.zeros((480, 640), dtype=np.int32)
depth = np.ones((480, 640), dtype=np.float32)
intrinsics = np.eye(3, dtype=np.float32)
target_T_depth = np.eye(4, dtype=np.float32)

pcl = fuse_depth_semantics(
    SemanticObservation(labels=labels),
    DepthObservation(depth=depth),
    intrinsics,
    target_T_depth,
    max_depth_m=30.0,  # optional depth clipping
)
```

## ROS node (`colored_pcl_node`)

### Core behavior

- Auto-detects mode from `~depth_input_topic` type:
  - `sensor_msgs/Image` -> depth mode
  - `sensor_msgs/PointCloud2` -> LiDAR mode
- Supports:
  - semantic labels (`~semantic_input_type:=labels`)
  - semantic RGB passthrough (`~semantic_input_type:=rgb`)
- Publishes `semantic_pointcloud` with fields:
  - `x y z label [confidence] [rgb]`

### Sensor-domain correction features (fusion-side)

- Semantic undistortion:
  - `~undistort_semantic`, `~undistort_alpha`
- Rolling-shutter correction with IMU (+ optional metadata readout):
  - `~rolling_shutter_*`, `~imu_topic`, `~camera_metadata_topic`, `~metadata_*`
- LiDAR deskew using per-point time + IMU:
  - `~lidar_deskew_*`, `~lidar_imu_topic`, `~lidar_time_field`, `~lidar_time_scale`
- Pair hard gate (timestamp validity):
  - `~pair_max_dt_sec`
- Projection debug stream:
  - `~debug_project_lidar`

### Extrinsics

Either TF/URDF or static matrices:

- `~static_target_T_depth`
- `~static_camera_T_lidar`
- `~static_target_T_lidar`

All are row-major 4x4 lists.

### PLY services

```bash
rosservice call /colored_pcl_node/save_ply "{}"
rosservice call /colored_pcl_node/set_ply_recording "data: true"
rosservice call /colored_pcl_node/set_ply_recording "data: false"
```

## Quick run (ROS)

```bash
rosdep update
rosdep install --from-paths src --ignore-src -r -y
catkin_make
source devel/setup.bash
roslaunch entfac_fusion_ros colored_pcl.launch
```

Override common params from launch:

```bash
roslaunch entfac_fusion_ros colored_pcl.launch debug:=true output_topic:=/my_colored_cloud
```

## Forestsphere profile

- Launch: `roslaunch entfac_fusion_ros forestsphere.launch`
- Config defaults: `entfac_fusion_ros/config/forestsphere.yaml`
- RViz profile: `entfac_fusion_ros/config/forestsphere.rviz`

## Real-time and memory notes

- PointCloud2 conversion paths are vectorized NumPy (no per-point Python loops).
- LiDAR XYZ/time extraction uses structured array views from message buffers.
- Depth projection caches pixel meshgrids per shape to reduce repeated allocations.
- Use `downsample_factor`, `max_depth_m`, and `pair_max_dt_sec` to control runtime load and reject bad pairs early.
- For high-rate streams, keep IMU and metadata topics local and reliable to avoid correction fallbacks.

## Docs

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

## Tests

```bash
pytest -q
```
