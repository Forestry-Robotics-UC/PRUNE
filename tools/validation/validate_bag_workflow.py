#!/usr/bin/env python3
"""Compatibility wrapper for the ICNF results-dir validation report."""

from __future__ import annotations

import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
for parent in _THIS.parents:
    ros_src = parent / "prune_ros"
    if (ros_src / "prune_ros").is_dir():
        for candidate in (parent, ros_src):
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
        break

from tools.validation.icnf_report import main as icnf_main

try:
    import rosbag  # type: ignore
except Exception:  # pragma: no cover
    rosbag = None


def capture_bag_topics(bag_path: str | Path) -> list[str]:
    """Return the sorted topics in a bag file when rosbag is available."""
    if rosbag is None:
        return []
    try:
        bag = rosbag.Bag(str(bag_path), 'r')
    except Exception:
        return []
    try:
        topics = bag.get_type_and_topic_info()[1].keys()
        return sorted(topics)
    finally:
        bag.close()


def main(argv: list[str] | None = None) -> int:
    return icnf_main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
