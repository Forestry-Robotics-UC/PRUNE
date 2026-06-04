#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ENTFAC Sensor Fusion implementation.
#
# Note:
#   This file was developed specifically for ENTFAC Sensor Fusion.
#   Project-level upstream attribution is documented in README.md.
#
# Modified by:
#   Duda Andrada (ENTFAC Sensor Fusion)
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: GNU General Public License v3.0 (GPL-3.0)
# Repository: ENTFAC-Sensor-Fusion
#
# Description:
#   ROS node entrypoint (kept minimal for roslaunch); implementation lives in entfac_fusion_ros Python package.

"""ROS node entrypoint for prune_node."""

import sys
from pathlib import Path


def _ensure_pkg_on_path() -> None:
    this = Path(__file__).resolve()
    pkg_root = this.parents[1]
    if pkg_root.is_dir() and str(pkg_root) not in sys.path:
        sys.path.append(str(pkg_root))


try:
    from entfac_fusion_ros.prune_node import main  # noqa: E402
except ModuleNotFoundError:
    _ensure_pkg_on_path()
    from entfac_fusion_ros.prune_node import main  # noqa: E402


if __name__ == "__main__":
    main()
