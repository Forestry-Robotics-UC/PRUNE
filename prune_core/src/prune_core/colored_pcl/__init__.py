#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: GNU General Public License v3.0 (GPL-3.0)
# Repository: PRUNE
#
# Description:
#   Convenience exports for semantic point cloud fusion entry points.

"""Public fusion entry points for PRUNE core.

The functions exported here form part of the v1.0 public API:

- :func:`prune_core.colored_pcl.fuse_depth_semantics`
- :func:`prune_core.colored_pcl.fuse_lidar_semantics`
"""

from prune_core.colored_pcl.fusion import (
    fuse_depth_semantics,
    fuse_lidar_semantics,
)

__all__ = ["fuse_depth_semantics", "fuse_lidar_semantics"]
