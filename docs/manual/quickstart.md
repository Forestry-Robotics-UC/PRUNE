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
roslaunch entfac_fusion_ros colored_pcl.launch
```

Node defaults are split into:

- `entfac_fusion_ros/config/core.yaml`
- `entfac_fusion_ros/config/expert.yaml`

Forestsphere profile:

```bash
roslaunch entfac_fusion_ros forestsphere.launch
```

## Build docs (Sphinx)

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```
