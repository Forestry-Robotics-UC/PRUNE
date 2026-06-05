#!/usr/bin/env python3
"""Unit tests for live parameter tuning."""

import pytest
from prune_ros.runtime import apply_tuning_params, TUNING_PARAMS


class MockNode:
    """Mock prune_node for testing."""

    def __init__(self):
        """Initialize with default tuning parameter values."""
        self.projection_patch_size = 1
        self.projection_confidence_min = 0.5
        self.projection_occlusion_epsilon_m = 0.01
        self.projection_occlusion_radius_px = 5
        self.projection_reject_depth_edges = False
        self.projection_depth_edge_thresh = 0.1
        self.projection_depth_edge_radius_px = 3
        self.debug_project_lidar = False
        self.debug_project_lidar_stride = 5
        self.debug_project_lidar_radius = 2
        self.debug_project_lidar_outline_only = False
        self.tracked_reprojection_fb_thresh_px = 1.0
        self.tracked_reprojection_depth_edge_thresh = 0.05
        self.tracked_reprojection_min_image_edge = 0.1
        self.tracked_reprojection_min_tracks = 50


class TestApplyTuningParams:
    """Test parameter application logic."""

    def test_no_changes(self):
        """Return False when no params change."""
        node = MockNode()

        def get_value(attr, default):
            return default

        changed = apply_tuning_params(node, get_value)

        assert changed is False

    def test_single_param_change(self):
        """Detect and apply single parameter change."""
        node = MockNode()
        logs = []

        def log_fn(msg):
            logs.append(msg)

        def get_value(attr, default):
            if attr == "projection_patch_size":
                return 3
            return default

        changed = apply_tuning_params(node, get_value, log_fn)

        assert changed is True
        assert node.projection_patch_size == 3
        assert len(logs) == 1
        assert "projection_patch_size=3" in logs[0]

    def test_multiple_param_changes(self):
        """Detect and apply multiple parameter changes."""
        node = MockNode()
        logs = []

        def log_fn(msg):
            logs.append(msg)

        def get_value(attr, default):
            if attr == "projection_patch_size":
                return 5
            elif attr == "debug_project_lidar":
                return True
            return default

        changed = apply_tuning_params(node, get_value, log_fn)

        assert changed is True
        assert node.projection_patch_size == 5
        assert node.debug_project_lidar is True
        assert len(logs) == 1
        assert "projection_patch_size=5" in logs[0]
        assert "debug_project_lidar=True" in logs[0]

    def test_invalid_value_skipped(self):
        """Skip parameter when value fails validation."""
        node = MockNode()
        logs = []

        def log_fn(msg):
            logs.append(msg)

        def get_value(attr, default):
            if attr == "projection_patch_size":
                return 2  # Even number (invalid)
            return default

        changed = apply_tuning_params(node, get_value, log_fn)

        assert changed is False
        assert node.projection_patch_size == 1  # Unchanged
        assert len(logs) == 0

    def test_get_value_exception_skipped(self):
        """Skip parameter when get_value raises exception."""
        node = MockNode()

        def get_value(attr, default):
            if attr == "projection_patch_size":
                raise ValueError("Missing param")
            return default

        changed = apply_tuning_params(node, get_value)

        assert changed is False
        assert node.projection_patch_size == 1  # Unchanged

    def test_no_log_function(self):
        """Handle None log_fn gracefully."""
        node = MockNode()

        def get_value(attr, default):
            if attr == "debug_project_lidar":
                return True
            return default

        # Should not raise even without log function
        changed = apply_tuning_params(node, get_value, None)

        assert changed is True
        assert node.debug_project_lidar is True

    def test_float_validation(self):
        """Validate float parameters correctly."""
        node = MockNode()

        def get_value(attr, default):
            if attr == "projection_confidence_min":
                return 1.5  # Out of [0, 1] range
            return default

        changed = apply_tuning_params(node, get_value)

        assert changed is False
        assert node.projection_confidence_min == 0.5  # Unchanged

    def test_bool_validation(self):
        """Validate bool parameters correctly."""
        node = MockNode()

        def get_value(attr, default):
            if attr == "projection_reject_depth_edges":
                return "true"  # String, not bool
            return default

        changed = apply_tuning_params(node, get_value)

        assert changed is False
        assert node.projection_reject_depth_edges is False  # Unchanged

    def test_int_bounds_validation(self):
        """Validate integer bounds."""
        node = MockNode()

        def get_value(attr, default):
            if attr == "projection_patch_size":
                return 0  # Must be >= 1
            elif attr == "debug_project_lidar_stride":
                return 0  # Must be >= 1
            return default

        changed = apply_tuning_params(node, get_value)

        assert changed is False

    def test_tuning_params_completeness(self):
        """Verify all expected parameters are defined."""
        # Ensure TUNING_PARAMS has the expected entries
        param_names = {name for name, _, _ in TUNING_PARAMS}

        expected = {
            "projection_patch_size",
            "projection_confidence_min",
            "projection_occlusion_epsilon_m",
            "projection_occlusion_radius_px",
            "projection_reject_depth_edges",
            "projection_depth_edge_thresh",
            "projection_depth_edge_radius_px",
            "debug_project_lidar",
            "debug_project_lidar_stride",
            "debug_project_lidar_radius",
            "debug_project_lidar_outline_only",
            "tracked_reprojection_fb_thresh_px",
            "tracked_reprojection_depth_edge_thresh",
            "tracked_reprojection_min_image_edge",
            "tracked_reprojection_min_tracks",
        }

        assert param_names == expected

    def test_integration_dynamic_reconfigure_flow(self):
        """Simulate dynamic_reconfigure update flow."""
        node = MockNode()
        logs = []

        def log_fn(msg):
            logs.append(msg)

        # Simulate config dict from dynamic_reconfigure
        config_dict = {
            "projection_patch_size": 3,
            "projection_confidence_min": 0.7,
            "debug_project_lidar": True,
        }

        def get_value(attr, default):
            return config_dict.get(attr, default)

        changed = apply_tuning_params(node, get_value, log_fn)

        assert changed is True
        assert node.projection_patch_size == 3
        assert node.projection_confidence_min == 0.7
        assert node.debug_project_lidar is True

    def test_integration_rospy_get_param_flow(self):
        """Simulate rospy.get_param update flow."""
        node = MockNode()
        logs = []

        def log_fn(msg):
            logs.append(msg)

        # Simulate rospy parameter server
        param_server = {
            "~projection_patch_size": 5,
            "~tracked_reprojection_min_tracks": 100,
        }

        def get_value(attr, default):
            param_name = f"~{attr}"
            if param_name in param_server:
                return param_server[param_name]
            return default

        changed = apply_tuning_params(node, get_value, log_fn)

        assert changed is True
        assert node.projection_patch_size == 5
        assert node.tracked_reprojection_min_tracks == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
