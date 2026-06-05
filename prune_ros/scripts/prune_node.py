#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: GNU General Public License v3.0 (GPL-3.0)
# Repository: PRUNE
#
# Description:
#   ROS node entrypoint (kept minimal for roslaunch); implementation lives in prune_ros Python package.

"""ROS node entrypoint for prune_node."""

import sys
from pathlib import Path


def _ensure_pkg_on_path() -> None:
    this = Path(__file__).resolve()
    pkg_root = this.parents[1]
    if pkg_root.is_dir() and str(pkg_root) not in sys.path:
        sys.path.append(str(pkg_root))


try:
    from prune_ros.node.prune_node import main  # noqa: E402
except ModuleNotFoundError:
    _ensure_pkg_on_path()
    from prune_ros.node.prune_node import main  # noqa: E402


if __name__ == "__main__":
    main()
