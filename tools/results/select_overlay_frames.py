#!/usr/bin/env python3
"""Create a shared selected-frame manifest for overlay panel comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--bag-name", required=True)
    parser.add_argument("--frames", nargs="+", type=int, required=True)
    args = parser.parse_args()

    bag_dir = Path(args.results_dir) / args.bag_name
    bag_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "bag_name": args.bag_name,
        "selected_frame_indices": sorted(set(args.frames)),
    }
    with (bag_dir / "selected_frames.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {bag_dir / 'selected_frames.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
