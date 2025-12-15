#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: MIT License (open source, free to modify and redistribute)
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
