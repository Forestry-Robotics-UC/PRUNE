# ForestSphere Overlay

This document is the operator manual for the ForestSphere-specific PRUNE workflow.
`README.md` stays generic and reusable; this file owns the ForestSphere CurtMini overlay, runtime assumptions, operator commands, and workflow notes.

## Scope

Use this guide when you are working on the ForestSphere CurtMini pipeline, especially for:

- replaying CurtMini ROS1 bags through the ForestSphere launch stack
- exporting semantic pointcloud bags for CurtMini datasets
- generating UFOMap outputs from semantic pointcloud bags
- running the Airfield semantic relabel/rebuild workflow
- saving intermediate UFOMap snapshots and exporting dense PLY outputs

Current CurtMini dataset names used by the unified export script are:

- `ren_lines`
- `icnf`
- `airfield`

Current CurtMini export modes are:

- `rgb`
- `semantic`
- `ndvi`

Apparatus support is expected later. For now, treat this manual as CurtMini-only.

## Operator Surface

These are the scripts you should treat as the active ForestSphere operator surface.

### Primary tools

- `dataset-compose.yml`
  - Main ForestSphere container entrypoint for replay and processing.
- `scripts/run_semantic_pointcloud_bag_export.sh`
  - Main CurtMini semantic pointcloud export entrypoint.
- `scripts/run_semantic_ufomap.sh`
  - Main UFOMap runner for semantic pointcloud bags.
- `scripts/run_airfield_semantic_full_pipeline.sh`
  - Main Airfield semantic special-case pipeline wrapper.

### Secondary tools

- `scripts/save_semantic_ufomap_snapshot.sh`
  - Manual snapshot helper for an already-running UFOMap job.
- `tools/validation/validate_curtmini_workflow.sh`
  - CurtMini bag/resource validation helper.

This document intentionally focuses on the active operator surface. Legacy wrappers and replaced scripts are not part of the recommended workflow here.

## Runtime Components

### Dataset container + compose

- `dataset-compose.yml`
  - Mounts the ENTFAC source tree, bag roots, X11 socket/cookie, local datasets, and local output folders.
  - `dataset-processing` defaults to `bash` inside the ROS-enabled image.
  - `curt-mini-gui` runs `roslaunch prune_ros forestsphere/curt_mini.launch ...` directly.
- `Docker/entfac-sensor-fusion-noetic.Dockerfile`
  - Builds the Noetic image used for ForestSphere CurtMini workflows.
  - Generates `/entrypoint.sh`, which sources ROS and the built workspace before running the container command.

### ROS launch + helpers

- `prune_ros/launch/forestsphere/curt_mini.launch`
  - Main end-to-end CurtMini replay launch for optional bag playback, coloring, TF publication, optional RViz, and UFOMap.
- `prune_ros/launch/forestsphere/curt_mini_simple.launch`
  - Lightweight CurtMini processing launch used by the export scripts.
- `prune_ros/launch/forestsphere/semantic_pointcloud_ufomap.launch`
  - Dedicated semantic pointcloud -> UFOMap launch.
- `prune_ros/scripts/pose_to_tf.py`
  - Publishes TF from localization topics, optionally using a stamp-source topic to align timing.

## ForestSphere-Owned Configs

### Core CurtMini configs

- `prune_ros/config/forestsphere/icnf_curt_mini.yaml`
- `prune_ros/config/forestsphere/curt_mini.yaml`
- `prune_ros/config/forestsphere/curtmini_debug.rviz`
- `prune_ros/config/forestsphere/curt_mini_realsense_camera_info_720p.txt`
- `prune_ros/config/forestsphere/curt_mini_realsense_camera_info_480p.txt`

### MapIR / NDVI configs

- `prune_ros/config/mapir/curt_mini_mapir3_ocn_camera_info_1920x1440.txt`
- `prune_ros/config/mapir/curt_mini_mapir3_ocn_camera_info_480x360.txt`

### Runtime export presets

These are the current preset configs used by `scripts/run_semantic_pointcloud_bag_export.sh`.

- `output/semantic_pointcloud_bags/configs/ren_lines_curt_rgb.yaml`
- `output/semantic_pointcloud_bags/configs/ren_lines_curt_semantic.yaml`
- `output/semantic_pointcloud_bags/configs/ren_mapir_ndvi_deskew_fast_runtime.yaml`
- `output/semantic_pointcloud_bags/configs/icnf_curt_mini_rgb.yaml`
- `output/semantic_pointcloud_bags/configs/icnf_curt_mini_semantic.yaml`
- `output/semantic_pointcloud_bags/configs/icnf_curt_mini_ndvi.yaml`
- `output/semantic_pointcloud_bags/configs/airfield_curt_mini_rgb.yaml`
- `output/semantic_pointcloud_bags/configs/airfield_curt_mini_semantic.yaml`
- `output/semantic_pointcloud_bags/configs/airfield_curt_mini_ndvi.yaml`

## Required Host Paths

Before running the pipeline, confirm the paths you plan to use are mounted and readable.

Use generic path placeholders in your own commands:

- `/path/to/main.bag`
- `/path/to/localization.bag`
- `/path/to/extra_semantic_or_label_input.bag`
- `/path/to/semantic_pointcloud_bags_root`

Do not rely on hardcoded personal mount points from this document.

## Environment Setup

Use this shell setup before building or running the ForestSphere container workflows.

```bash
export BAGS_PATH=/path/to/bags_root
export ENTFAC_USER=ros
export UBUNTU_APT_MIRROR=http://de.archive.ubuntu.com/ubuntu
export ENTFAC_CPUS=8.0
export ENTFAC_MEM_LIMIT=12g
export ENTFAC_MEMSWAP_LIMIT=12g
```

### Build the dataset image

```bash
docker compose -f dataset-compose.yml build dataset-processing
```

## Quick Start

### GUI / RViz replay

```bash
xhost +si:localuser:$(whoami)
BAGS_DIR=/path/to/bags_root \
DATASET_BAG_PATH=/path/to/main.bag \
DATASET_BAG_PATH_EXTRA=/path/to/localization.bag \
FUSION_RUN_UFOMAP=true \
docker compose -f dataset-compose.yml run --rm curt-mini-gui
```

## Main CurtMini Export Workflow

The main export command is `scripts/run_semantic_pointcloud_bag_export.sh`.

### Design rules

- `--dataset` and `--mode` choose CurtMini defaults only.
- Bag paths remain explicit and operator-controlled.
- Use `--bag-path` for the main bag.
- Use repeatable `--extra-bag-path` for localization bags and any extra semantic/label inputs.
- Use `--bag-paths` only when you already have the complete bag list prepared.

### Supported dataset presets

- `ren_lines`
- `icnf`
- `airfield`

### Supported mode presets

- `rgb`
- `semantic`
- `ndvi`

### General command pattern

```bash
scripts/run_semantic_pointcloud_bag_export.sh \
  --dataset <ren_lines|icnf|airfield> \
  --mode <rgb|semantic|ndvi> \
  --bag-path /path/to/main.bag \
  --extra-bag-path /path/to/localization.bag \
  --extra-bag-path /path/to/extra_semantic_or_label_input.bag
```

### RGB export template

```bash
scripts/run_semantic_pointcloud_bag_export.sh \
  --dataset <ren_lines|icnf|airfield> \
  --mode rgb \
  --bag-path /path/to/main.bag \
  --extra-bag-path /path/to/localization.bag
```

### Semantic export template

```bash
scripts/run_semantic_pointcloud_bag_export.sh \
  --dataset <ren_lines|icnf|airfield> \
  --mode semantic \
  --bag-path /path/to/main.bag \
  --extra-bag-path /path/to/localization.bag \
  --extra-bag-path /path/to/segmentation_or_labels.bag
```

### NDVI export template

```bash
scripts/run_semantic_pointcloud_bag_export.sh \
  --dataset <ren_lines|icnf|airfield> \
  --mode ndvi \
  --bag-path /path/to/main.bag \
  --extra-bag-path /path/to/localization.bag
```

### Output of the export step

The export script writes a new timestamped run directory under:

```bash
output/semantic_pointcloud_bags/
```

Typical contents:

- exported semantic pointcloud bag
- `fusion.log`
- `bagplay.log`
- `record.log`
- `semantic_bag_info.txt`
- `RUN_CONTEXT.txt`

## Main UFOMap Workflow

Use `scripts/run_semantic_ufomap.sh` after you have a semantic pointcloud bag.

### What it does

- replays a semantic pointcloud bag plus localization bag(s)
- launches the dedicated semantic pointcloud UFOMap stack
- saves descriptively named final and snapshot `.um` outputs
- exports normal artifacts through `create_ufomap_artifacts.py`
- runs the integrated dense postprocess for final and snapshot `.um` outputs

### General command pattern

```bash
scripts/run_semantic_ufomap.sh \
  --dataset <ren_lines_curt|icnf_curt|airfield> \
  --mode <rgb|semantic|ndvi> \
  --bags-root /path/to/semantic_pointcloud_bags_root
```

### Output rules

- default behavior creates a fresh timestamped run directory inside `output/semantic_ufomap/<dataset>/<mode>/`
- use `--overwrite` only when you intentionally want to reuse the dataset/mode folder itself
- output names are descriptive and should not use generic `map.um` or `top.png`-style naming anymore

### Naming convention

Final saved map basename:

```text
curtmini_<dataset>_<mode>
```

Examples:

- `curtmini_icnf_curt_rgb.um`
- `curtmini_ren_lines_curt_semantic.ply`
- `curtmini_airfield_ndvi_dense.ply`

Snapshot saved map basename:

```text
T<seconds>_curtmini_<dataset>_<mode>
```

Examples:

- `T460_curtmini_icnf_curt_rgb.um`
- `T920_curtmini_airfield_semantic_dense.ply`
- `T1380_curtmini_ren_lines_curt_ndvi_top.png`

### Example command

```bash
scripts/run_semantic_ufomap.sh \
  --dataset icnf_curt \
  --mode rgb \
  --bags-root /path/to/semantic_pointcloud_bags_root \
  --rate 1.0 \
  --resolution 0.10 \
  --pub-rate 2.0 \
  --max-range 25.0
```

### Main outputs

Default output root:

```bash
output/semantic_ufomap/<dataset>/<mode>/<timestamp>/
```

If `--overwrite` is used:

```bash
output/semantic_ufomap/<dataset>/<mode>/
```

Typical contents:

- `curtmini_<dataset>_<mode>.um`
- `curtmini_<dataset>_<mode>.ply`
- `curtmini_<dataset>_<mode>_dense.ply`
- `curtmini_<dataset>_<mode>_top.png`
- `curtmini_<dataset>_<mode>_front.png`
- `curtmini_<dataset>_<mode>_side.png`
- `curtmini_<dataset>_<mode>_iso.png`
- `run_params.yaml`
- `run_console.log`
- `snapshots/T<sec>/T<sec>_curtmini_<dataset>_<mode>.um`
- `snapshots/T<sec>/T<sec>_curtmini_<dataset>_<mode>.ply`
- `snapshots/T<sec>/T<sec>_curtmini_<dataset>_<mode>_dense.ply`

## Manual Snapshot Workflow

Use this only when a UFOMap run is already active and you want an extra manual snapshot save.

```bash
scripts/save_semantic_ufomap_snapshot.sh \
  --basename T300_curtmini_icnf_curt_rgb \
  --delay-sec 300
```

If you pass `--basename`, choose the same descriptive naming style used by the main UFOMap runner.

## Airfield Semantic Full Pipeline

Keep using this top-level wrapper for now:

```bash
bash scripts/run_airfield_semantic_full_pipeline.sh
```

Use this wrapper when:

- the Airfield semantic labels need preprocessing or topic relabeling
- the semantic bag set spans multiple sessions
- you want the full ForestSphere Airfield semantic flow, not an isolated export call

## Validation Guidance

Current validation helpers that are still relevant:

- `tools/validation/validate_curtmini_workflow.sh`
- `scripts/ufomap/validate_bag_color.py`

### Export validation checklist

1. Confirm the output bag exists.
2. Read `semantic_bag_info.txt`.
3. Check that the `/semantic_pointcloud` message count is reasonable.
4. Check the effective Hz for the workflow you are running.
5. Review `fusion.log` and `bagplay.log` when output rate or content looks wrong.

### UFOMap validation checklist

1. Confirm the descriptive final `.um` file exists.
2. Confirm `run_params.yaml` and `run_console.log` were written.
3. Confirm the descriptive final `.ply` and `_dense.ply` files exist.
4. If snapshots were expected, confirm the snapshot directories contain descriptively named `.um`, `.ply`, and `_dense.ply` files.
5. Inspect the screenshot set when the map content looks suspicious.

## Troubleshooting

### The export run produced no bag

Check:

- `fusion.log`
- `record.log`
- `bagplay.log`
- `RUN_CONTEXT.txt`

Usual causes:

- incorrect bag ordering
- wrong localization topic
- wrong camera info preset
- wrong semantic/NDVI stamp-source topic

### The UFOMap run produced no useful output

Check:

- `run_console.log`
- `run_params.yaml`
- color-validation warnings from `validate_bag_color.py`
- whether the semantic bag actually contains valid `/semantic_pointcloud` data

### Dense outputs are missing

Check:

- that the descriptive final `.um` exists first
- that the UFOMap run completed cleanly
- that the dataset container can run `ufomap_um_to_ply.py`
- that the output directory was not manually modified mid-run

## Current Practical Rules

- Prefer `scripts/run_semantic_pointcloud_bag_export.sh` for CurtMini bag export.
- Prefer `scripts/run_semantic_ufomap.sh` for map generation.
- Prefer `scripts/run_airfield_semantic_full_pipeline.sh` for the Airfield semantic special case.
- Keep bag paths explicit.
- Do not teach or depend on hardcoded personal mount paths in this document.
- Treat this file as the operator manual for the active ForestSphere CurtMini workflow.
