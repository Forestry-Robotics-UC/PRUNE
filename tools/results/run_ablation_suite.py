#!/usr/bin/env python3
"""Generate or execute ablation run commands for results collection."""

from __future__ import annotations

import argparse
import re
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
    "mask": {
        "use_invalid_mask": "true",
        "use_depth_edge_rejection": "false",
        "use_occlusion_gate": "false",
        "projection_patch_size": "1",
        "projection_reject_depth_edges": "false",
        "projection_occlusion_epsilon_m": "0.10",
    },
    "edge": {
        "use_invalid_mask": "false",
        "use_depth_edge_rejection": "true",
        "use_occlusion_gate": "false",
        "projection_patch_size": "3",
        "projection_reject_depth_edges": "true",
        "projection_occlusion_epsilon_m": "0.10",
    },
    "occlusion": {
        "use_invalid_mask": "false",
        "use_depth_edge_rejection": "false",
        "use_occlusion_gate": "true",
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
    "mask_occlusion": {
        "use_invalid_mask": "true",
        "use_depth_edge_rejection": "false",
        "use_occlusion_gate": "true",
        "projection_patch_size": "1",
        "projection_reject_depth_edges": "false",
        "projection_occlusion_epsilon_m": "0.10",
    },
    "edge_occlusion": {
        "use_invalid_mask": "false",
        "use_depth_edge_rejection": "true",
        "use_occlusion_gate": "true",
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


_SAFE_ROSLAUNCH_TOKEN = re.compile(r"^[A-Za-z0-9_./-]+$")


def _bag_name(path: str) -> str:
    first = path.split()[0]
    return Path(first).stem


def _launch_command_prefix(args) -> str:
    if Path(args.launch_file).is_absolute():
        return " ".join(shlex.quote(part) for part in ["roslaunch", args.launch_file])
    if "/" in args.launch_file:
        for value, label in [
            (args.fusion_package, "--fusion-package"),
            (args.launch_file, "--launch-file"),
        ]:
            if not _SAFE_ROSLAUNCH_TOKEN.match(value):
                raise ValueError(f"{label} contains unsupported shell characters: {value}")
        return f'roslaunch "$(rospack find {args.fusion_package})/launch/{args.launch_file}"'
    return " ".join(
        shlex.quote(part) for part in ["roslaunch", args.fusion_package, args.launch_file]
    )


def _resolve_results_dir(results_dir_arg: str, study_name: str) -> Path:
    """Return an absolute host path for ablation outputs."""
    base = Path(results_dir_arg) if results_dir_arg else Path("results") / study_name
    return base.resolve()


def _command(bag: str, variant: str, results_dir: Path, args) -> str:
    extra_bags = " ".join(args.extra_bags or [])
    params = {
        "play_bag": "true",
        "bag_path": bag,
        "run_ufomap": "false",
        "rviz": "false",
        "debug": "false",
        "rate": str(args.rate),
        "start_sec": str(args.start_sec),
        "dataset_config": args.base_config,
        "experiment_variant_name": variant,
        "bag_name": args.bag_name or _bag_name(bag),
        "results_dir": str(results_dir),
        "enable_metrics_csv": "true",
        "projection_confidence_min": str(args.projection_confidence_min),
    }
    if extra_bags:
        params["localization_bag"] = extra_bags
    if args.duration_sec:
        params["duration_sec"] = str(args.duration_sec)
    params.update(VARIANTS[variant])
    launch = _launch_command_prefix(args)
    launch_args = " ".join(
        shlex.quote(f"{key}:={value}") for key, value in params.items()
    )
    return f"{launch} {launch_args}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bags", nargs="+", required=True)
    parser.add_argument(
        "--extra-bags",
        nargs="*",
        default=[],
        help="Additional bags passed through the launch localization_bag/bag_path_extra arg",
    )
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()))
    parser.add_argument(
        "--base-config",
        default="$(find prune_ros)/config/dataset.yaml",
    )
    parser.add_argument("--fusion-package", default="prune_ros")
    parser.add_argument(
        "--launch-file",
        default="prune.launch",
        help="roslaunch file for the fusion pipeline",
    )
    parser.add_argument(
        "--study-name",
        default="ablation",
        help=(
            "Name of the ablation study output folder under results/ when "
            "--results-dir is not provided explicitly"
        ),
    )
    parser.add_argument(
        "--results-dir",
        default="",
        help=(
            "Explicit results output directory. Defaults to "
            "results/<study-name>."
        ),
    )
    parser.add_argument("--bag-name", default="")
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=0.0,
        help="rosbag play duration in bag seconds; 0 plays to the end",
    )
    parser.add_argument("--projection-confidence-min", type=float, default=0.0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    unknown = [variant for variant in args.variants if variant not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variants: {', '.join(unknown)}")

    results_dir = _resolve_results_dir(args.results_dir, args.study_name)
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
        iterator = commands
        try:
            from tqdm import tqdm

            iterator = tqdm(commands, desc="Running ablations", unit="run")
        except Exception:
            pass
        for command in iterator:
            subprocess.run(command, shell=True, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
