"""Camera intrinsics and undistortion helpers for prune."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import time
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo

from entfac_fusion_ros.prune.params import load_camera_info_txt as _load_camera_info_txt_helper


@dataclass
class CameraInfoData:
    intrinsics: np.ndarray
    intrinsics_raw: np.ndarray
    frame_id: str
    distortion: Optional[np.ndarray]
    distortion_model: str
    image_size: Tuple[int, int]
    source: str


class CameraModel:
    def __init__(self, node: Any, logger: Any):
        self._node = node
        self._log = logger
        self._cv2 = None
        self._undistort_map1 = None
        self._undistort_map2 = None
        self._undistort_active = False
        self._undistort_status = "disabled"

    @property
    def intrinsics(self) -> np.ndarray:
        return self._node.intrinsics

    @property
    def intrinsics_raw(self) -> np.ndarray:
        return self._node.intrinsics_raw

    @property
    def frame_id(self) -> str:
        return self._node.camera_frame

    @property
    def active(self) -> bool:
        return self._undistort_active

    def load(self) -> CameraInfoData:
        if self._node.camera_info_txt:
            (
                intrinsics,
                distortion,
                distortion_model,
                camera_info_size,
                frame_id_from_file,
            ) = _load_camera_info_txt_helper(self._node, self._node.camera_info_txt)
            self._node.intrinsics = np.asarray(intrinsics, dtype=float)
            self._node.intrinsics_raw = self._node.intrinsics.copy()
            self._node._camera_distortion = distortion
            self._node._camera_distortion_model = distortion_model
            self._node._camera_info_size = camera_info_size
            self._node.camera_frame = frame_id_from_file or self._node.camera_frame_param
            if not self._node.camera_frame:
                raise ValueError(
                    "~camera_info_txt requires a camera frame. Add frame_id/camera_frame in the file or set ~camera_frame."
                )
            self._log.info(
                "__init__",
                "Using camera intrinsics from ~camera_info_txt=%s (topic ~camera_info=%s is ignored for intrinsics).",
                self._node.camera_info_txt,
                self._node.camera_info_topic,
            )
        else:
            cam_info = self._wait_for_camera_info()
            self._node.intrinsics = np.asarray(cam_info.K, dtype=float).reshape(3, 3)
            self._node.camera_frame = cam_info.header.frame_id
            self._node.intrinsics_raw = self._node.intrinsics.copy()
            self._node._camera_distortion = np.asarray(cam_info.D, dtype=float) if cam_info.D else None
            self._node._camera_distortion_model = (cam_info.distortion_model or "").strip().lower()
            self._node._camera_info_size = (int(cam_info.height), int(cam_info.width))
            self._node._camera_info_source = f"topic:{self._node.camera_info_topic}"

        self._maybe_init_undistort()
        return CameraInfoData(
            intrinsics=self._node.intrinsics,
            intrinsics_raw=self._node.intrinsics_raw,
            frame_id=self._node.camera_frame,
            distortion=self._node._camera_distortion,
            distortion_model=self._node._camera_distortion_model,
            image_size=self._node._camera_info_size,
            source=getattr(self._node, "_camera_info_source", ""),
        )

    def _wait_for_camera_info(self) -> CameraInfo:
        self._log.debug("__init__", "Waiting for CameraInfo on topic=%s", self._node.camera_info_topic)
        cam_info = None
        next_warn = time.time() + 5.0
        while cam_info is None and not rospy.is_shutdown():
            try:
                cam_info = rospy.wait_for_message(self._node.camera_info_topic, CameraInfo, timeout=1.0)
            except rospy.ROSException:
                cam_info = None
            if cam_info is None and time.time() >= next_warn:
                self._log.warn("__init__", "Waiting for CameraInfo on %s (check topic name and bag/driver)", self._node.camera_info_topic)
                next_warn = time.time() + 5.0
        if cam_info is None:
            raise rospy.ROSInterruptException("Shutdown while waiting for CameraInfo")
        return cam_info

    def _ensure_cv2(self) -> bool:
        if self._cv2 is not None:
            return True
        try:
            import cv2  # type: ignore
        except ImportError:
            return False
        self._cv2 = cv2
        return True

    def _maybe_init_undistort(self) -> None:
        if self._node.mode != "lidar" or not self._node.undistort_semantic:
            self._undistort_status = "disabled"
            return
        if self._node._camera_distortion is None or not self._node._camera_distortion.size:
            self._undistort_status = "disabled (CameraInfo has no distortion)"
            return
        if np.allclose(self._node._camera_distortion, 0.0):
            self._undistort_status = "disabled (zero distortion coefficients)"
            return
        if not self._ensure_cv2():
            self._undistort_status = "disabled (OpenCV unavailable)"
            return
        h, w = self._node._camera_info_size
        if h <= 0 or w <= 0:
            self._undistort_status = "disabled (invalid CameraInfo image size)"
            return
        model = self._node._camera_distortion_model
        k_mat = np.asarray(self._node.intrinsics_raw, dtype=float)
        d_vec = np.asarray(self._node._camera_distortion, dtype=float).reshape(-1)
        if model in ("plumb_bob", "rational_polynomial", ""):
            new_k, _ = self._cv2.getOptimalNewCameraMatrix(k_mat, d_vec, (w, h), float(self._node.undistort_alpha))
            map1, map2 = self._cv2.initUndistortRectifyMap(k_mat, d_vec, None, new_k, (w, h), self._cv2.CV_32FC1)
        elif model == "equidistant":
            new_k = self._cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(k_mat, d_vec, (w, h), np.eye(3), balance=float(self._node.undistort_alpha))
            map1, map2 = self._cv2.fisheye.initUndistortRectifyMap(k_mat, d_vec, np.eye(3), new_k, (w, h), self._cv2.CV_32FC1)
        else:
            self._undistort_status = f"disabled (unsupported model {model})"
            return
        self._undistort_map1 = map1
        self._undistort_map2 = map2
        self._undistort_active = True
        self._undistort_status = "active"
        self._node.intrinsics = np.asarray(new_k, dtype=float)

    def undistort_array(self, data: np.ndarray, *, interpolation: str) -> np.ndarray:
        if not self._undistort_active or self._cv2 is None:
            return data
        h, w = self._node._camera_info_size
        if data.shape[0] != h or data.shape[1] != w:
            return data
        interp = self._cv2.INTER_NEAREST if interpolation == "nearest" else self._cv2.INTER_LINEAR
        orig_dtype = data.dtype
        work = data if data.dtype in (np.uint8, np.uint16, np.float32) else data.astype(np.float32)
        remapped = self._cv2.remap(work, self._undistort_map1, self._undistort_map2, interp)
        if remapped.dtype != orig_dtype:
            remapped = remapped.astype(orig_dtype)
        return remapped
