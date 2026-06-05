"""Pipeline building blocks for PRUNE."""

from importlib import import_module

_EXPORTS = {
    "CameraModel": (".camera_model", "CameraModel"),
    "DepthFusionPipeline": (".depth_pipeline", "DepthFusionPipeline"),
    "FrameInputPreparer": (".frame_inputs", "FrameInputPreparer"),
    "LidarFusionPipeline": (".lidar_pipeline", "LidarFusionPipeline"),
    "OnlineCalibrationBridge": (".online_calibration_bridge", "OnlineCalibrationBridge"),
    "PlyRecordingService": (".ply_service", "PlyRecordingService"),
    "LastPcl": (".results", "LastPcl"),
    "PipelineResult": (".results", "PipelineResult"),
    "SemanticInputs": (".results", "SemanticInputs"),
    "PruneRosIo": (".ros_io", "PruneRosIo"),
    "SemanticInputParser": (".semantic_inputs", "SemanticInputParser"),
    "StampPolicy": (".sync_policy", "StampPolicy"),
    "TransformResolver": (".tf_resolver", "TransformResolver"),
    "TrackedReprojectionRuntime": (".tracked_reprojection_runtime", "TrackedReprojectionRuntime"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + __all__)
