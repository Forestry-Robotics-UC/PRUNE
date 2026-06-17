#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    if source.resolve() == output.resolve():
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
