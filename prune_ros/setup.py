#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: GNU General Public License v3.0 (GPL-3.0)
# Repository: PRUNE
#
# Description:
#   setuptools entry for installing prune_ros Python modules via catkin.

from setuptools import find_packages, setup


setup(
    name="prune_ros",
    version="1.0.0",
    package_dir={"": "."},
    packages=find_packages(),
)
