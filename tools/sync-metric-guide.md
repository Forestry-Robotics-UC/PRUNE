# ROS Bag Sync Metric Guide

## 1) Scope and Intent
This guide explains how to use the tools in `tools/` for camera/LiDAR time alignment, what the metric is actually computing, and how to decide whether a result is acceptable.

Primary script:
- `tools/rosbag_sync_metric.py`

Supporting scripts:
- `tools/run_sync_metric_datasets.py`
- `tools/rosbag_apply_piecewise_offset.py`
- `tools/diagnostics/rosbag_time_skew.py`
- `tools/rosbag_deskew_lidar.py`
- `tools/convert_mcap_to_bag.py`

## 2) Minimal Required Parameters by Tool

### 2.1 `rosbag_sync_metric.py`
Required always:
- `bags`
- `topic_a`
- `topic_b`

Required when geometry/overlay is enabled (`edge: true` or `geom_score_mode: mi|edge_mi`) or `semantic_topic` is set:
- `camera_info`
- `camera_T_lidar`

Required when `topic_b` is Ouster packets (e.g., `/ouster/lidar_packets`):
- Metadata source from one of:
- `packet_metadata_path`
- `packet_metadata_bag`
- metadata topic present in same bags (`packet_metadata_topic`, default `/ouster/metadata`)

Required when `rolling_shutter: true`:
- `readout_time > 0`
- `imu_topic`

Required when `export_ply: true`:
- `camera_info`
- `camera_T_lidar`

### 2.2 `run_sync_metric_datasets.py`
Required:
- valid ICNF and/or Aerodromo config files

### 2.3 `rosbag_apply_piecewise_offset.py`
Required:
- input `bags`
- `--windows-csv`
- `--topics`
- `--out-dir`

### 2.4 `rosbag_time_skew.py`
Required:
- `bags`
- `topic_a`
- `topic_b`

### 2.5 `rosbag_deskew_lidar.py`
Required:
- `bags`
- `lidar_topic`
- `imu_topic`

### 2.6 `convert_mcap_to_bag.py`
Required:
- `src`
- `dst`

## 3) The Science Behind `rosbag_sync_metric.py`

## 3.1 Motion proxy extraction
For each topic, the script builds a time-series motion proxy:
- Image topic: frame-to-frame change magnitude
- PointCloud2 topic: frame-to-frame histogram/range change
- IMU topic: angular velocity magnitude

This turns different modalities into comparable 1D signals over time.

## 3.2 Time-shift sweep
It sweeps `dt` in `[-dt_range, +dt_range]` with `dt_step` and computes correlation:
- `corr(dt) = corrcoef(a(t), b(t + dt))`

Interpretation:
- If `best_dt > 0`, topic B is late relative to A.
- Suggested fix is `-best_dt`.

## 3.3 Geometric consistency (edge / MI)
The geometric term is controlled by `geom_score_mode`:
- `edge`: projected LiDAR depth-edge vs image-edge alignment
- `mi`: normalized mutual information (image intensity vs projected depth)
- `edge_mi`: weighted blend via `geom_edge_weight` / `geom_mi_weight`

MI options:
- `mi_bins`
- `mi_min_samples`
- `mi_log_depth`

## 3.4 Final score
The final score at each `dt` is:
- `score(dt) = w1*corr_norm(dt) + w2*geom(dt) - w3*entropy(dt)`

Where:
- `corr_norm = (corr + 1) / 2`
- `entropy` is optional semantic uncertainty penalty

## 3.5 Peak quality statistics
At best `dt`, the script computes:
- `peak_drop`: how much score drops around the peak (`sharpness`)
- `second_ratio`: second best peak / best peak (`uniqueness`)
- `peaks_over`: number of near-equal peaks (`ambiguity`)

## 3.6 Windowed drift model
With `window_sec > 0` or `window_messages > 0`, best `dt` is computed per window.
This captures offset drift over long runs.

Window filtering can reject low-quality windows using:
- edge-hit at search boundary
- minimum `peak_drop`
- maximum `second_ratio`
- maximum `peaks_over`
- minimum per-window confidence

## 3.7 Confidence model
Global confidence combines:
- peak sharpness
- uniqueness
- stability (MAD across kept windows)
- observability

Segment confidence (when decision mode is segment) combines:
- kept-window ratio
- optional stability
- observability

Per-PCL confidence (when decision mode is `per_pcl`) combines:
- matched-frame coverage
- passed-frame ratio
- median per-frame confidence

Overlay ranking confidence (`overlay_rank_*`) is frame-level:
- geometric mean of geometry alignment (edge/MI), window/global confidence, and timestamp pairing confidence.

## 3.8 Exhaustive per-lidar pairing mode
`overlay_exhaustive_per_lidar` enables the mode you requested:
- for each lidar frame, search every image frame within `+/- overlay_exhaustive_window_sec`
- compute frame confidence for every candidate pair
- keep only the best image for that lidar frame
- export overlay PNG for that best pair

Relevant parameters:
- `overlay_exhaustive_per_lidar: true`
- `overlay_exhaustive_window_sec: 1.0`
- `overlay_exhaustive_max_lidar_frames: 0` (0 means all lidar frames)
- `decision_mode: per_pcl`
- `per_pcl_frame_conf_min`, `per_pcl_min_pass_ratio`, `per_pcl_min_matched_ratio`, `per_pcl_min_matched_frames`
- `per_pcl_window_sec`, `per_pcl_max_lidar_frames`, `per_pcl_image_downsample`
- `overlay_rank_top_k` / `overlay_rank_min_conf` for top-confidence review export
- Note: exhaustive mode currently exports PNGs only (no MP4 assembly).

Example:
```bash
python3 tools/rosbag_sync_metric.py /datasets/ICNF/raw/*.bag \
  /camera/camera/color/image_raw /ouster/points \
  --stamp-source auto --filter-bags --dt-range 0.5 --dt-step 0.001 \
  --out-dir /datasets/ICNF/raw/soa_results_exhaustive \
  --overlay --overlay-exhaustive-per-lidar --overlay-exhaustive-window-sec 1.0 \
  --overlay-exhaustive-max-lidar-frames 0 \
  --overlay-rank-top-k 200 --overlay-rank-min-conf 0.30 --no-progress
```

## 4) What Outputs Mean
When `out_dir` is set, typical outputs are:
- `score_curve.csv` and `score_curve.png`: score vs `dt`
- `report.json`: best dt, gates, confidence, window stats
- `window_offsets.csv`: per-window offsets (`offset_total`)
- `overlays/`: projected LiDAR over camera frames
- `overlays/overlay_ranked_frames.csv`: sorted frame-level confidence list
- `overlays/top_conf/`: top-ranked overlay frames for manual review

When exhaustive mode is enabled:
- per-bag PNGs are written under `overlays/<bag>/exhaustive_per_lidar/`
- `overlay_ranked_frames.csv` still ranks all generated overlays globally
- `exhaustive_pairs.csv` lists `t_img`, `t_lidar`, `dt`, `geom_score`, `edge_score`, `mi_score`, `frame_conf`, and PNG path
- `exhaustive_dt_plot.png` plots `dt` per matched image over time
- `per_pcl_decisions.csv` stores one row per LiDAR frame with matched/pass flags and confidence terms (`geom_score`, `edge_score`, `mi_score`, etc.)

Key console fields:
- `best_dt`: dt maximizing final score
- `suggested_b_offset_delta`: amount to add to topic B timestamps now
- `suggested_b_offset_total`: total topic B offset after combining prior + new estimate

## 5) Acceptance Thresholds (Practical)
These are practical bands aligned with the implemented gates and common outcomes.

Strong (usually safe to apply):
- `confidence >= 0.65`
- `peak_drop >= 0.10`
- `second_ratio <= 0.85`
- `peaks_over == 1`
- if windowed: `window_kept_ratio >= 0.30` and `mad <= 0.008 s`

Usable but review overlays carefully:
- `confidence` in `[0.45, 0.65)`
- `peak_drop` in `[0.05, 0.10)`
- `second_ratio` in `(0.85, 0.95]`
- if windowed: `mad` in `(0.008, 0.020] s`

Weak / ambiguous (do not trust directly):
- `confidence < 0.45`
- `peak_drop < 0.05`
- `second_ratio > 0.95` or `peaks_over > 1`
- best peak stuck near `±dt_range` boundary

For drifting datasets, prioritize segment metrics and `window_offsets.csv` over a single global `best_dt`.

## 6) What You Should Expect Visually
Good overlay:
- LiDAR structures sit on image edges (poles, trunk boundaries, object contours)
- alignment remains stable across many frames, not just isolated moments

Bad overlay:
- consistent left/right or up/down offset
- alignment flips between good and bad in nearby frames
- only one dt variant looks good in very few frames

Use `overlays/top_conf/` first for quick QA, then spot-check random frames for bias.

## 7) Minimal Operating Profiles

Constant offset profile (short clips):
- windowing off or short windows
- decision mode `global` or `auto`

Drifting offset profile (long runs):
- `window_sec` enabled
- `decision_mode: segment`
- `window_filter: true`
- apply `window_offsets.csv` with `rosbag_apply_piecewise_offset.py`

Exhaustive QA profile (per-message pass/fail):
- `overlay_exhaustive_per_lidar: true`
- `decision_mode: per_pcl`
- tune `per_pcl_*` thresholds to your acceptance bar
- for low-memory machines, prefer `per_pcl_image_downsample: 2` and limit `per_pcl_max_lidar_frames`

## 8) Recommended Workflow
1. Run sync metric with windowing and overlays enabled.
2. Inspect `report.json` + `score_curve.png` + `window_offsets.csv`.
3. Review `overlays/top_conf/` and `overlay_ranked_frames.csv`.
4. If stable enough, apply offsets:
- global: use `suggested_b_offset_total`
- drifting: use `rosbag_apply_piecewise_offset.py` with `offset_total`
5. Re-run metric on corrected bag(s): expected `best_dt` near `0`.

## 9) Troubleshooting
No motion series produced:
- wrong topic names or unsupported message types

Low confidence with flat score curve:
- increase `dt_range`
- reduce `dt_step` (e.g., 0.0005)
- increase `edge_samples` / `edge_bags`
- ensure camera and LiDAR both have non-trivial motion

Overlay poor despite decent correlation:
- validate `camera_T_lidar`
- ensure `camera_info` is from the same camera stream
- check timestamp source (`stamp_source: auto` is safest)

Packet input fails:
- provide metadata path/bag or include metadata topic in the same bag

## 10) Lean Configs in This Repository
- `tools/rosbag_sync_metric.icnf.yaml`
- `tools/rosbag_sync_metric.aerodromo.yaml`
- `tools/rosbag_sync_metric.example.yaml`

These were trimmed to essential + high-impact parameters only.
