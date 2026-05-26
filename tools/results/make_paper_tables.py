#!/usr/bin/env python3
"""Create results CSV tables from experiment summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_ros_package_path() -> None:
    repo = Path(__file__).resolve().parents[2]
    ros_pkg = repo / "entfac_fusion_ros"
    if str(ros_pkg) not in sys.path:
        sys.path.insert(0, str(ros_pkg))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, help="Experiment results root")
    args = parser.parse_args()

    _add_ros_package_path()
    from entfac_fusion_ros.experiment_metrics import write_results_tables

    write_results_tables(Path(args.results_dir))
    print(f"Wrote results tables under {Path(args.results_dir) / 'paper'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
