#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Derived from Semantic SLAM
#
# Original Author:
#   Xuan Zhang
#
# Subsequent Contributions:
#   David Russell
#
# Modified by:
#   Duda Andrada (ENTFAC Sensor Fusion)
#
# Original project:
#   https://github.com/floatlazer/semantic_slam
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: GNU General Public License v3.0 (GPL-3.0)
# Repository: ENTFAC-Sensor-Fusion
#
# Description:
#   ROS node entrypoint (kept minimal for roslaunch); implementation lives in entfac_fusion_ros Python package.

"""ROS node entrypoint for semantic_pcl_node."""

import sys
from pathlib import Path


def _ensure_pkg_on_path() -> None:
    this = Path(__file__).resolve()
    pkg_root = this.parents[1]
    if pkg_root.is_dir() and str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))


_ensure_pkg_on_path()

from entfac_fusion_ros.semantic_pcl_node import main  # noqa: E402


if __name__ == "__main__":
    main()
