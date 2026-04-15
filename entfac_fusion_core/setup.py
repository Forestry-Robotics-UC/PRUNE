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
#   setuptools entrypoint for installing the ROS-agnostic fusion core as a
#   catkin Python package.

from setuptools import find_packages, setup


setup(
    name="entfac_fusion_core",
    version="1.0.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
)
