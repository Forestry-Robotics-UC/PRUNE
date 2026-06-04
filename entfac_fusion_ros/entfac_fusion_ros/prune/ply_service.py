"""PLY recording helpers for prune."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import rospy
from std_srvs.srv import SetBool, SetBoolResponse, Trigger, TriggerResponse

from entfac_fusion_ros.ply import PlyJob, PlyWriterThread
from entfac_fusion_core.transforms.se3 import transform_points

from .results import LastPcl


class PlyRecordingService:
    def __init__(self, node: Any, logger: Any):
        self._node = node
        self._log = logger
        self._writer = PlyWriterThread(queue_size=2)
        self._recording = False
        self._queue_warned_at = 0.0
        self._seq = 0

    def setup(self) -> None:
        Path(self._node.ply_output_dir).mkdir(parents=True, exist_ok=True)
        self._recording = bool(self._node.ply_recording_enable)
        if self._recording:
            self._writer.start()
            self._log.info("_setup_ply_runtime", "PLY recording enabled at startup (output_dir=%s)", self._node.ply_output_dir)
        self._node._srv_set_record = rospy.Service("~set_ply_recording", SetBool, self.handle_set_recording)
        self._node._srv_save_ply = rospy.Service("~save_ply", Trigger, self.handle_save_ply)

    def set_recording(self, enable: bool) -> None:
        self._recording = bool(enable)
        if self._recording:
            self._writer.start()

    def handle_set_recording(self, req: SetBool.Request) -> SetBoolResponse:
        enable = bool(req.data)
        self.set_recording(enable)
        if enable:
            self._log.info("_srv_set_ply_recording", "PLY recording enabled (output_dir=%s)", self._node.ply_output_dir)
        else:
            self._log.info("_srv_set_ply_recording", "PLY recording disabled")
        return SetBoolResponse(success=True, message=str(enable))

    def next_path(self, stamp: rospy.Time) -> Path:
        t_ns = int(stamp.to_nsec()) if hasattr(stamp, "to_nsec") else int(stamp.to_sec() * 1e9)
        self._seq += 1
        return Path(self._node.ply_output_dir) / f"prune_{t_ns}_{self._seq:06d}.ply"

    def enqueue(self, last: LastPcl) -> bool:
        points_xyz = last.points_xyz
        if self._node.ply_target_frame and self._node.ply_target_frame != self._node.target_frame:
            mat = self._node._lookup_transform(self._node.ply_target_frame, self._node.target_frame, last.stamp)
            if mat is None and self._node.ply_tf_use_latest:
                mat, tf_stamp = self._node._lookup_transform_with_stamp(self._node.ply_target_frame, self._node.target_frame, rospy.Time(0))
                if mat is None:
                    self._log.warn("_enqueue_ply", "PLY transform unavailable (%s -> %s); skipping write", self._node.target_frame, self._node.ply_target_frame)
                    return False
                delta = abs((tf_stamp - last.stamp).to_sec())
                if delta > self._node.ply_tf_tolerance_sec:
                    self._log.warn("_enqueue_ply", "PLY latest TF too far from cloud stamp (dt=%.6fs tol=%.6fs); skipping write", delta, self._node.ply_tf_tolerance_sec)
                    return False
            elif mat is None:
                self._log.warn("_enqueue_ply", "PLY transform unavailable (%s -> %s); skipping write", self._node.target_frame, self._node.ply_target_frame)
                return False
            points_xyz = transform_points(mat, points_xyz)
        self._writer.start()
        job = PlyJob(path=self.next_path(last.stamp), points_xyz=points_xyz, labels=last.labels, confidence=last.confidence, rgb_packed_float=last.rgb_packed_float)
        ok = self._writer.enqueue(job)
        if not ok:
            now = rospy.get_time()
            if now - self._queue_warned_at > 1.0:
                self._log.warn("_enqueue_ply", "PLY writer queue is full; dropping frames. Consider lowering publish rate or increasing queue_size.")
                self._queue_warned_at = now
        return ok

    def handle_save_ply(self, req: Trigger.Request) -> TriggerResponse:
        last = self._node._last_pcl
        if last is None:
            return TriggerResponse(success=False, message="No point cloud published yet")
        ok = self.enqueue(last)
        if ok:
            return TriggerResponse(success=True, message="enqueued")
        return TriggerResponse(success=False, message="enqueue failed")

