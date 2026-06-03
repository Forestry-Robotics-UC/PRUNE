#!/usr/bin/env python3
"""Tests for grouped colored point-cloud config loaders."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

ROS_SRC = Path(__file__).resolve().parents[2] / "entfac_fusion_ros"
if str(ROS_SRC) not in sys.path:
    sys.path.insert(0, str(ROS_SRC))

from entfac_fusion_ros.colored_pcl.config import (
    load_color_config,
    load_debug_config,
    load_ply_config,
    load_projection_config,
    load_sync_config,
)


class MockLogger:
    def __init__(self):
        self.warnings = []

    def warn(self, *args):
        self.warnings.append(args)


class MockNode:
    def __init__(self, params):
        self._params = dict(params)
        self._param_meta = {}
        self._log = MockLogger()

    def _get_param_bool(self, name, default, description):
        value = bool(self._params.get(name, default))
        self._param_meta[name] = {
            "value": value,
            "source": "test",
            "description": description,
        }
        return value

    def _get_param_int(self, name, default, description):
        value = int(self._params.get(name, default))
        self._param_meta[name] = {
            "value": value,
            "source": "test",
            "description": description,
        }
        return value

    def _get_param_float(self, name, default, description):
        value = float(self._params.get(name, default))
        self._param_meta[name] = {
            "value": value,
            "source": "test",
            "description": description,
        }
        return value

    def _get_param_str(self, name, default, description, *, allow_empty=False):
        raw = self._params.get(name, default)
        value = None if raw is None else str(raw)
        if not allow_empty and value in ("", None):
            value = default
        self._param_meta[name] = {
            "value": value,
            "source": "test",
            "description": description,
        }
        return value

    def _get_color_map(self, name, description):
        raw = self._params.get(name, {1: [2, 3, 4]})
        self._param_meta[name] = {
            "value": raw,
            "source": "test",
            "description": description,
        }
        return raw


class ColoredPclConfigTests(unittest.TestCase):
    def test_sync_config_normalizes_stamp_source(self):
        node = MockNode({
            "~cloud_stamp_source": "  Semantic  ",
            "~sync_slop_sec": 0.25,
        })

        config = load_sync_config(node)

        self.assertEqual(config.cloud_stamp_source, "semantic")
        self.assertEqual(config.sync_slop_sec, 0.25)
        self.assertEqual(node._param_meta["~cloud_stamp_source"]["value"], "  Semantic  ")

    def test_color_config_uses_existing_map_and_meta(self):
        node = MockNode({
            "~colorize_labels": True,
            "~color_map": {"7": [10, 20, 30]},
        })

        config = load_color_config(node)

        self.assertTrue(config.colorize_labels)
        self.assertEqual(config.color_map, {"7": [10, 20, 30]})
        self.assertIn("~color_map", node._param_meta)

    def test_projection_config_rejects_even_patch(self):
        node = MockNode({"~projection_patch_size": 2})

        with self.assertRaises(ValueError):
            load_projection_config(node)

    def test_debug_config_creates_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "debug-out"
            node = MockNode({"~debug_output_dir": str(output_dir)})

            config = load_debug_config(node)

            self.assertEqual(config.debug_output_dir, str(output_dir))
            self.assertTrue(output_dir.exists())

    def test_ply_config_creates_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "ply-out"
            node = MockNode({"~ply_output_dir": str(output_dir)})

            config = load_ply_config(node)

            self.assertEqual(config.ply_output_dir, str(output_dir))
            self.assertTrue(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
