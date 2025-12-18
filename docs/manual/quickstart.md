# Quickstart

## Core (NumPy-only)

Install Python deps:

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
roslaunch entfac_fusion_ros semantic_pcl.launch
```

## Build docs (Sphinx)

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

