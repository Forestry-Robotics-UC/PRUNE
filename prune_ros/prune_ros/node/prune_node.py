#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: GNU General Public License v3.0 (GPL-3.0)
# Repository: PRUNE
#
# Description:
#   ROS node: converts semantic image + geometry (depth or LiDAR) into semantic PointCloud2.

"""ROS wrapper that converts semantic + geometry into semantic PointCloud2.

This node is part of PRUNE and is explicitly *stateless*:
each callback processes one frame and publishes a semantic measurement for a
separate mapping layer to accumulate over time.

ROS Interface (v1.0)
-------------------

Subscriptions
^^^^^^^^^^^^^

Required:
  - ``~semantic_topic`` (``sensor_msgs/Image``):
    - ``~semantic_input_type=labels``: single-channel label IDs (e.g. ``mono8``,
      ``16UC1``, ``32SC1``).
    - ``~semantic_input_type=rgb``: 3/4-channel colors (e.g. ``rgb8``, ``bgr8``,
      ``rgba8``, ``bgra8``). Colors are passed through to the output when
      ``~colorize_labels`` is enabled.
  - Geometry input:
    - Preferred: ``~depth_input_topic`` (auto-detected):
      - ``sensor_msgs/Image`` → depth mode
      - ``sensor_msgs/PointCloud2`` → lidar mode

Optional:
  - Camera intrinsics source:
    - ``~camera_info`` (``sensor_msgs/CameraInfo``): topic providing intrinsics
      ``K`` and camera frame ID.
    - ``~camera_info_txt`` (``str``): calibration file path. Supports
      ``K: [k00,...,k22]`` / ``camera_matrix.data: [...]`` or keyed
      ``fx/fy/cx/cy`` fields.
    - ``~camera_frame`` (``str``): fallback frame ID used with
      ``~camera_info_txt`` when the file does not include one.
  - ``~confidence_topic`` (``sensor_msgs/Image``): confidence/probability aligned
    with semantic labels (single-channel numeric).
  - ``~projection_invalid_mask_topic`` (``sensor_msgs/Image``): optional
    single-channel invalid mask aligned with ``~semantic_topic``. Invalid pixels
    reject transferred labels/RGB and zero confidence.

Publications
^^^^^^^^^^^^

  - ``semantic_pointcloud`` (``sensor_msgs/PointCloud2``): semantic measurement
    in ``~target_frame`` with fields:
    - ``x, y, z`` (float32)
    - ``label`` (uint16; ``65535`` means unknown/unlabeled)
    - ``confidence`` (float32, optional)
    - ``rgb`` (float32 packed RGB, optional; only when ``~colorize_labels:=true``)

TF / Extrinsics
^^^^^^^^^^^^^^^

Depth mode requires a transform from the depth frame to ``~target_frame``:
  - Static parameter: ``~static_target_T_depth`` (16-element row-major 4x4)
  - Otherwise: TF lookup ``target_frame <- depth_frame`` (resolved once).

LiDAR mode requires:
  - ``~static_camera_T_lidar`` or TF lookup ``camera_frame <- lidar_frame``
  - ``~static_target_T_lidar`` or TF lookup ``target_frame <- lidar_frame``

Services
^^^^^^^^

  - ``~save_ply`` (``std_srvs/Trigger``): write the last published cloud to PLY.
  - ``~set_ply_recording`` (``std_srvs/SetBool``): enable/disable continuous PLY
    recording (written asynchronously).

Failure Behavior
^^^^^^^^^^^^^^^^

  - Invalid configuration raises ``ValueError`` during initialization.
  - Missing TF/extrinsics at runtime logs a warning and skips publishing until
    resolved.
  - Shape/dtype mismatches raise ``ValueError`` to fail fast.
"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import rospy

try:
    from dynamic_reconfigure.server import Server as DynamicReconfigureServer
except Exception:  # noqa: BLE001
    DynamicReconfigureServer = None

try:
    from scipy import ndimage as _scipy_ndimage
except ImportError:
    _scipy_ndimage = None

# Ensure core package is importable when running from a monorepo source tree.
# In a proper catkin workspace, this is handled by PYTHONPATH via devel/setup.bash.
_THIS = Path(__file__).resolve()
for parent in _THIS.parents:
    cand = parent / "prune_core" / "src"
    if (cand / "prune_core").is_dir() and str(cand) not in sys.path:
        sys.path.insert(0, str(cand))
        break

from prune_ros.config import coerce_bool as _coerce_bool
from ..pipelines.camera_model import CameraModel
from ..startup.initializer import NodeInitializer
from prune_ros.config import ParamReader
from ..runtime.logging_ros import NodeLogger, configure_core_logging
from ..runtime.pc2 import semantic_pointcloud_to_msg
from ..startup.startup_builder import PruneStartupBuilder
from ..pipelines.results import LastPcl

try:
    import prune_ros.cfg.PruneTuningConfig as PruneTuningConfig
except Exception:  # noqa: BLE001
    PruneTuningConfig = None


def _rosargv_bool(name: str, default: bool = False) -> bool:
    prefix = f"_{name}:="
    for arg in sys.argv:
        if arg.startswith(prefix):
            return _coerce_bool(arg[len(prefix) :])
    return default


class PruneNode:
    """ROS node bridging topics to the numpy fusion core."""

    def __init__(self):
        self._param_meta: Dict[str, Dict[str, Any]] = {}
        self._node_name = rospy.get_name().lstrip("/")
        self._log = NodeLogger(self._node_name)
        self._params = ParamReader(self)
        self._record_param = self._params.record_param
        self._get_param = self._params.get_param
        self._get_param_str = self._params.get_param_str
        self._get_param_bool = self._params.get_param_bool
        self._get_param_int = self._params.get_param_int
        self._get_param_float = self._params.get_param_float
        self._get_matrix_param = self._params.get_matrix_param
        self._get_color_map = self._params.get_color_map
        self._apply_loaded_config = self._params.apply_loaded_config
        self._load_camera_info_txt = self._params.load_camera_info_txt

        self._initializer = NodeInitializer(self)
        self._initializer.load_initial_params()
        configure_core_logging(self._node_name, debug=self.core_debug)
        self._initializer.load_runtime_config()
        self._initializer.load_runtime_support_params()
        self._initializer.setup_startup_helpers()

        self._initializer.load_camera_info()

        self._startup_builder = PruneStartupBuilder(self)
        self._startup_builder.prepare_runtime_state()
        self._startup_builder.finalize_mode_status()
        startup_components = self._startup_builder.build_components()
        self._stamp_policy = startup_components.stamp_policy
        self._camera_model = startup_components.camera_model
        self._ply_service = startup_components.ply_service
        self._tracked_runtime = startup_components.tracked_runtime
        self._calibration_bridge = startup_components.calibration_bridge
        self._semantic_parser = startup_components.semantic_parser
        self._frame_inputs = startup_components.frame_inputs
        self._runtime_setup.initialize_runtime_state()

        self._runtime_setup.validate_mode_dependent_flags()

        self._runtime_setup.setup_metrics_and_ply()
        self._runtime_setup.setup_projector_and_buffers()
        self._runtime_setup.setup_subsystems()
        self._runtime_setup.setup_ros_runtime(
            DynamicReconfigureServer,
            PruneTuningConfig,
        )
        self._log.info("__init__", "\n%s", self._startup_reporting.render_startup_table())
        self._startup_reporting.log_correction_statuses()
        self._startup_reporting.log_startup_transforms()
        self._log.debug(
            "__init__",
            "Runtime: target_frame=%s camera_frame=%s semantic_input_type=%s colorize_labels=%s include_unlabeled=%s downsample=%d",
            self.target_frame,
            self.camera_frame,
            self.semantic_input_type,
            bool(self.colorize_labels),
            bool(self.include_unlabeled),
            int(self.downsample_factor),
        )
        if self.debug:
            self._startup_reporting.log_param_report()

    def _metadata_callback(self, msg) -> None:
        try:
            key = int(msg.key)
            value = int(msg.value)
        except Exception:  # noqa: BLE001
            return
        self._metadata_latest[key] = (msg.header.stamp, value)

    def _ensure_cv2(self, context: str) -> bool:
        if self._cv2 is not None:
            return True
        try:
            import cv2  # type: ignore
        except Exception as exc:  # noqa: BLE001
            self._log.warn(
                context,
                "OpenCV not available (%s); disabling the requested image-space diagnostic.",
                exc,
            )
            return False
        self._cv2 = cv2
        return True

    def _undistort_array(self, data: np.ndarray, *, interpolation: str) -> np.ndarray:
        return self._camera_model.undistort_array(data, interpolation=interpolation)

    def _lookup_transform(self, target_frame, source_frame, stamp):
        return self._tf_resolver.lookup(target_frame, source_frame, stamp)

    def _lookup_transform_with_stamp(self, target_frame, source_frame, stamp):
        return self._tf_resolver.lookup_with_stamp(target_frame, source_frame, stamp)

    @contextmanager
    def _maybe_profile(self, label):
        if not self.enable_profiling:
            yield
            return
        prof = cProfile.Profile()
        prof.enable()
        try:
            yield
        finally:
            prof.disable()
            s = io.StringIO()
            ps = pstats.Stats(prof, stream=s).sort_stats("tottime")
            ps.print_stats(10)
            self._log.info("_profile", "%s profile:\n%s", label, s.getvalue())

    def _publish_result(self, result) -> None:
        include_rgb = bool(result.debug.get("include_rgb", False))
        rgb_lut = result.debug.get("rgb_lut")
        rgb_values = result.debug.get("rgb_values")
        pcl_msg = semantic_pointcloud_to_msg(result.cloud, result.frame_id, result.stamp, colorize_labels=include_rgb, rgb_lut=rgb_lut, rgb_values=rgb_values)
        publish_t0 = rospy.get_time()
        self.pcl_pub.publish(pcl_msg)
        publish_ms = max(0.0, (rospy.get_time() - publish_t0) * 1000.0)
        self._last_pcl = LastPcl(stamp=result.stamp, points_xyz=result.cloud.points_xyz, labels=result.cloud.labels, confidence=result.cloud.confidence, rgb_packed_float=rgb_values if include_rgb else None)
        if self._ply_recording:
            self._ply_service.enqueue(self._last_pcl)
        self._startup_reporting.emit_status(points=int(result.cloud.points_xyz.shape[0]), callback_sec=result.callback_sec)
        post_publish = result.debug.get("post_publish")
        if callable(post_publish):
            post_publish(publish_ms)

    def _depth_callback(self, sem_msg, depth_msg, conf_msg=None, invalid_mask_msg=None):
        with self._maybe_profile("depth_callback"):
            result = self._depth_pipeline.process(sem_msg, depth_msg, conf_msg, invalid_mask_msg)
            if result is not None:
                self._publish_result(result)

    def _lidar_callback(self, sem_msg, lidar_msg, conf_msg=None, invalid_mask_msg=None):
        with self._maybe_profile("lidar_callback"):
            result = self._lidar_pipeline.process(sem_msg, lidar_msg, conf_msg, invalid_mask_msg)
            if result is not None:
                self._publish_result(result)


def main():
    import threading
    log_level = rospy.DEBUG if _rosargv_bool("debug", False) else rospy.INFO
    rospy.init_node("prune_node", log_level=log_level)
    PruneNode()
    num_threads = int(rospy.get_param("~spin_threads", 1))
    if num_threads > 1:
        threads = [threading.Thread(target=rospy.spin) for _ in range(num_threads - 1)]
        for t in threads:
            t.daemon = True
            t.start()
    rospy.spin()


if __name__ == "__main__":
    main()
