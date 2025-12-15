#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: MIT License (open source, free to modify and redistribute)
# Repository: fruc_ros_utils
#
# Description:
#   Convenience exports for semantic point cloud fusion entry points.

from entfac_fusion_core.semantic_pcl.fusion import (
    fuse_depth_semantics,
    fuse_lidar_semantics,
)

__all__ = ["fuse_depth_semantics", "fuse_lidar_semantics"]
