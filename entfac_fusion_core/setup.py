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
#   setuptools entrypoint for installing the ROS-agnostic fusion core as a
#   catkin Python package.

from setuptools import find_packages, setup


setup(
    name="entfac_fusion_core",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
)
