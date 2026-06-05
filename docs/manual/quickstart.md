# Quickstart

## Core (NumPy-only)

```bash
pip install -r requirements.txt
pytest -q
```

## ROS (Noetic)

In a catkin workspace:

```bash
rosdep update
rosdep install --from-paths src --ignore-src -r -y
catkin_make
source devel/setup.bash
roslaunch prune_ros prune.launch
```

Node defaults are split into:

- `prune_ros/config/core.yaml`
- `prune_ros/config/expert.yaml`

Config defaults live in `prune_ros/config/core.yaml` and `prune_ros/config/expert.yaml`.

## Offline time sync (rosbag)

```bash
# Fast skew stats (nearest-neighbor deltas)
python tools/diagnostics/rosbag_time_skew.py /data/*.bag /camera/image /os_cloud_node/points

# Analyze all .bag files under a directory
python tools/diagnostics/rosbag_time_skew.py /data/bags /camera/image /os_cloud_node/points
```

## Build docs (Sphinx)

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```
