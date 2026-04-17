# Rosbag Time Skew Guide

This repository currently ships one offline timing utility:
- `tools/rosbag_time_skew.py`

The script computes nearest-neighbor timestamp deltas between two topics and prints summary stats.

## What It Does
- Reads header timestamps from `topic_a` and `topic_b`.
- For each timestamp in `topic_a`, finds the closest timestamp in `topic_b`.
- Reports:
  - `paired`
  - `mean_delta`
  - `median_delta`
  - `min_delta`
  - `max_delta`
  - `p95_abs_delta`
  - `p99_abs_delta`

Interpretation:
- Positive delta means topic B tends to be later than topic A.
- Negative delta means topic B tends to be earlier than topic A.

## Usage

Single bag:
```bash
python tools/rosbag_time_skew.py /data/run_001.bag /camera/image /os_cloud_node/points
```

Directory of bags (all `*.bag` in the directory are aggregated):
```bash
python tools/rosbag_time_skew.py /data/bags /camera/image /os_cloud_node/points
```

## Container Workflow (Forestsphere)
- Start one dataset container:
  - `docker compose -f docker-compose.forestsphere.yml up -d curt-mini`
- Run the tool in the container:
  - `docker exec -it entfac_curt_mini bash`
  - `python tools/rosbag_time_skew.py /bags /camera/image /os_cloud_node/points`

## Notes and Limits
- Requires ROS Noetic Python environment with `rosbag` available.
- Only messages with `msg.header.stamp` are considered.
- If a topic has no valid header stamps, the script exits with an error.
- This tool is descriptive only; it does not rewrite bag timestamps.
