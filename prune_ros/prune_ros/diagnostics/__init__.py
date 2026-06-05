"""Diagnostics and metrics helpers for PRUNE."""

from importlib import import_module

_EXPORTS = {
    "DiagnosticsOrchestrator": (".diagnostics", "DiagnosticsOrchestrator"),
    "FrameMetrics": (".experiment_metrics", "FrameMetrics"),
    "MetricsCsvLogger": (".experiment_metrics", "MetricsCsvLogger"),
    "summarize_metrics_file": (".experiment_metrics", "summarize_metrics_file"),
    "write_results_tables": (".experiment_metrics", "write_results_tables"),
    "MetricsReporter": (".metrics_reporting", "MetricsReporter"),
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
