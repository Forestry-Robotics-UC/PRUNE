#!/usr/bin/env python3
import argparse
from pathlib import Path


def _yaml_scalar(value: str) -> str:
    if value == "":
        return '""'
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return str(int(number))
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--semantic-bag", required=True)
    parser.add_argument("--localization-bag", required=True)
    parser.add_argument("--play-rate", required=True)
    parser.add_argument("--rosbag-skip-empty-sec", required=True)
    parser.add_argument("--start-sec", required=True)
    parser.add_argument("--rviz", required=True)
    parser.add_argument("--localization-topic", required=True)
    parser.add_argument("--localization-parent-frame", required=True)
    parser.add_argument("--localization-yaw-offset-deg", required=True)
    parser.add_argument("--localization-use-stamp-source", required=True)
    parser.add_argument("--localization-stamp-source-topic", required=True)
    parser.add_argument("--localization-stamp-source-type", required=True)
    parser.add_argument("--localization-stamp-source-max-age-sec", required=True)
    parser.add_argument("--ufomap-resolution", required=True)
    parser.add_argument("--ufomap-depth-levels", required=True)
    parser.add_argument("--ufomap-num-workers", required=True)
    parser.add_argument("--ufomap-color", required=True)
    parser.add_argument("--ufomap-max-range", required=True)
    parser.add_argument("--ufomap-prob-hit", required=True)
    parser.add_argument("--ufomap-prob-miss", required=True)
    parser.add_argument("--ufomap-pub-rate", required=True)
    parser.add_argument("--ufomap-export-ply", required=True)
    parser.add_argument("--ufomap-export-mesh", required=True)
    parser.add_argument("--ufomap-export-screenshots", required=True)
    args = parser.parse_args()

    params = [
        ("semantic_bag", args.semantic_bag),
        ("localization_bag", args.localization_bag),
        ("play_rate", args.play_rate),
        ("rosbag_skip_empty_sec", args.rosbag_skip_empty_sec),
        ("start_sec", args.start_sec),
        ("rviz", args.rviz),
        ("localization_topic", args.localization_topic),
        ("localization_parent_frame", args.localization_parent_frame),
        ("localization_yaw_offset_deg", args.localization_yaw_offset_deg),
        ("localization_use_stamp_source", args.localization_use_stamp_source),
        ("localization_stamp_source_topic", args.localization_stamp_source_topic),
        ("localization_stamp_source_type", args.localization_stamp_source_type),
        (
            "localization_stamp_source_max_age_sec",
            args.localization_stamp_source_max_age_sec,
        ),
        ("ufomap_resolution", args.ufomap_resolution),
        ("ufomap_depth_levels", args.ufomap_depth_levels),
        ("ufomap_num_workers", args.ufomap_num_workers),
        ("ufomap_color", args.ufomap_color),
        ("ufomap_max_range", args.ufomap_max_range),
        ("ufomap_prob_hit", args.ufomap_prob_hit),
        ("ufomap_prob_miss", args.ufomap_prob_miss),
        ("ufomap_pub_rate", args.ufomap_pub_rate),
        ("ufomap_export_ply", args.ufomap_export_ply),
        ("ufomap_export_mesh", args.ufomap_export_mesh),
        ("ufomap_export_screenshots", args.ufomap_export_screenshots),
    ]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(f"{key}: {_yaml_scalar(value)}\n" for key, value in params))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
