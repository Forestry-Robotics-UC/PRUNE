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
#   Sphinx configuration for ENTFAC Sensor Fusion documentation (GitHub Pages ready).

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Make core and ROS python packages importable for autodoc.
sys.path.insert(0, str(PROJECT_ROOT / "entfac_fusion_core" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "entfac_fusion_ros" / "src"))

project = "ENTFAC Sensor Fusion"
author = "Duda Andrada"
copyright = f"{author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Docs should be buildable without ROS installed.
autodoc_mock_imports = [
    "rospy",
    "tf2_ros",
    "message_filters",
    "sensor_msgs",
    "std_srvs",
    "rospkg",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

html_theme = "alabaster"
html_static_path = ["_static"]

master_doc = "index"
root_doc = "index"

