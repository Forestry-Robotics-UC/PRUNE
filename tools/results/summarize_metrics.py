#!/usr/bin/env python3
"""Summarize per-frame experiment metrics into per-run and global CSV files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_ros_package_path() -> None:
    repo = Path(__file__).resolve().parents[2]
    ros_pkg = repo / "prune_ros"
    if str(ros_pkg) not in sys.path:
        sys.path.insert(0, str(ros_pkg))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, help="Experiment results root")
    args = parser.parse_args()

    _add_ros_package_path()
    from prune_ros.experiment_metrics import summarize_results_tree

    summaries = summarize_results_tree(Path(args.results_dir))
    print(f"Wrote summaries for {len(summaries)} runs under {args.results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
