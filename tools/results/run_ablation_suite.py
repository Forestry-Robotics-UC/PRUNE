#!/usr/bin/env python3
"""Generate or execute CurtMini ablation run commands for results collection."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


VARIANTS = {
    "naive": {
        "use_invalid_mask": "false",
        "use_depth_edge_rejection": "false",
        "use_occlusion_gate": "false",
        "projection_patch_size": "1",
        "projection_reject_depth_edges": "false",
        "projection_occlusion_epsilon_m": "0.10",
    },
    "mask_only": {
        "use_invalid_mask": "true",
        "use_depth_edge_rejection": "false",
        "use_occlusion_gate": "false",
        "projection_patch_size": "1",
        "projection_reject_depth_edges": "false",
        "projection_occlusion_epsilon_m": "0.10",
    },
    "mask_edge": {
        "use_invalid_mask": "true",
        "use_depth_edge_rejection": "true",
        "use_occlusion_gate": "false",
        "projection_patch_size": "3",
        "projection_reject_depth_edges": "true",
        "projection_occlusion_epsilon_m": "0.10",
    },
    "full": {
        "use_invalid_mask": "true",
        "use_depth_edge_rejection": "true",
        "use_occlusion_gate": "true",
        "projection_patch_size": "5",
        "projection_reject_depth_edges": "true",
        "projection_occlusion_epsilon_m": "0.10",
    },
}


def _bag_name(path: str) -> str:
    return Path(path).stem


def _command(bag: str, variant: str, results_dir: Path, args) -> str:
    params = {
        "play_bag": "true",
        "bag_path": bag,
        "run_ufomap": "false",
        "rviz": "false",
        "debug": "false",
        "dataset_config": args.base_config,
        "experiment_variant_name": variant,
        "bag_name": _bag_name(bag),
        "results_dir": str(results_dir),
        "enable_metrics_csv": "true",
        "enable_overlay_export": str(args.enable_overlays).lower(),
        "overlay_stride": str(args.overlay_stride),
        "max_overlay_frames": str(args.max_overlay_frames),
        "projection_confidence_min": str(args.projection_confidence_min),
    }
    params.update(VARIANTS[variant])
    parts = ["roslaunch", args.fusion_package, args.launch_file]
    parts.extend(f"{key}:={value}" for key, value in params.items())
    return " ".join(shlex.quote(part) for part in parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bags", nargs="+", required=True)
    parser.add_argument("--variants", nargs="+", default=["naive", "mask_only", "mask_edge", "full"])
    parser.add_argument(
        "--base-config",
        default="$(find entfac_fusion_ros)/config/forestsphere/icnf_curt_mini.yaml",
    )
    parser.add_argument("--fusion-package", default="entfac_fusion_ros")
    parser.add_argument(
        "--launch-file",
        default="forestsphere/curt_mini.launch",
        help="roslaunch file for the fusion pipeline; default targets the ForestSphere CurtMini overlay",
    )
    parser.add_argument("--results-dir", default="results/icist_2026")
    parser.add_argument("--enable-overlays", action="store_true")
    parser.add_argument("--overlay-stride", type=int, default=10)
    parser.add_argument("--max-overlay-frames", type=int, default=50)
    parser.add_argument("--projection-confidence-min", type=float, default=0.0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    unknown = [variant for variant in args.variants if variant not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variants: {', '.join(unknown)}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    commands = [
        _command(bag, variant, results_dir, args)
        for bag in args.bags
        for variant in args.variants
    ]
    script_path = results_dir / "run_all.sh"
    with script_path.open("w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\n")
        handle.write("set -euo pipefail\n\n")
        for command in commands:
            handle.write(command)
            handle.write("\n")
    script_path.chmod(0o755)
    print(f"Wrote {script_path}")

    if args.execute:
        for command in commands:
            subprocess.run(command, shell=True, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
