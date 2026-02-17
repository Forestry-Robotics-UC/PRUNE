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
#   Convenience exports for semantic point cloud fusion entry points.

"""Public fusion entry points for ENTFAC Sensor Fusion core.

The functions exported here form part of the v1.0 public API:

- :func:`entfac_fusion_core.colored_pcl.fuse_depth_semantics`
- :func:`entfac_fusion_core.colored_pcl.fuse_lidar_semantics`
"""

from entfac_fusion_core.colored_pcl.fusion import (
    fuse_depth_semantics,
    fuse_lidar_semantics,
)

__all__ = ["fuse_depth_semantics", "fuse_lidar_semantics"]
