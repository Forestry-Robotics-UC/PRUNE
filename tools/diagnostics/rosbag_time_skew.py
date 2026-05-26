#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Simple rosbag timestamp skew tool:
#   Finds nearest-neighbor time deltas between two topics and reports stats.

import argparse
import statistics
import sys
from pathlib import Path

import rosbag


def _collect_stamps(bag, topic):
    stamps = []
    for _, msg, _ in bag.read_messages(topics=[topic]):
        if not hasattr(msg, "header") or msg.header is None:
            continue
        stamp = msg.header.stamp
        if stamp is None:
            continue
        stamps.append(stamp.to_sec())
    return stamps


def _nearest_deltas(a, b):
    i = 0
    deltas = []
    for t in a:
        while i + 1 < len(b) and abs(b[i + 1] - t) <= abs(b[i] - t):
            i += 1
        deltas.append(b[i] - t)
    return deltas


def _percentile(vals, pct):
    if not vals:
        return None
    idx = int(round((pct / 100.0) * (len(vals) - 1)))
    return sorted(vals)[max(0, min(idx, len(vals) - 1))]


def main():
    parser = argparse.ArgumentParser(
        description="Estimate timestamp skew between two rosbag topics."
    )
    parser.add_argument("bag_or_dir", help="Path to rosbag file or directory")
    parser.add_argument("topic_a", help="Reference topic (e.g. camera)")
    parser.add_argument("topic_b", help="Comparison topic (e.g. lidar)")
    args = parser.parse_args()

    bag_path = Path(args.bag_or_dir)
    if bag_path.is_dir():
        bags = sorted(bag_path.glob("*.bag"))
        if not bags:
            print(f"No .bag files found in {bag_path}", file=sys.stderr)
            sys.exit(1)
        all_deltas = []
        for bag_file in bags:
            with rosbag.Bag(str(bag_file), "r") as bag:
                a = _collect_stamps(bag, args.topic_a)
                b = _collect_stamps(bag, args.topic_b)
            if not a or not b:
                continue
            all_deltas.extend(_nearest_deltas(a, b))
        deltas = all_deltas
    else:
        with rosbag.Bag(str(bag_path), "r") as bag:
            a = _collect_stamps(bag, args.topic_a)
            b = _collect_stamps(bag, args.topic_b)
        if not a:
            print(f"No stamps found on {args.topic_a}", file=sys.stderr)
            sys.exit(1)
        if not b:
            print(f"No stamps found on {args.topic_b}", file=sys.stderr)
            sys.exit(1)
        deltas = _nearest_deltas(a, b)

    if not deltas:
        print("No paired timestamps found.", file=sys.stderr)
        sys.exit(1)
    abs_deltas = [abs(d) for d in deltas]

    print(f"topic_a={args.topic_a}")
    print(f"topic_b={args.topic_b}")
    print(f"paired={len(deltas)}")
    print(f"mean_delta={statistics.mean(deltas):.6f} s")
    print(f"median_delta={statistics.median(deltas):.6f} s")
    print(f"min_delta={min(deltas):.6f} s")
    print(f"max_delta={max(deltas):.6f} s")
    print(f"p95_abs_delta={_percentile(abs_deltas, 95):.6f} s")
    print(f"p99_abs_delta={_percentile(abs_deltas, 99):.6f} s")


if __name__ == "__main__":
    main()
