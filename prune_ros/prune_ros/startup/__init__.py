"""Startup assembly helpers for PRUNE."""

from importlib import import_module

_EXPORTS = {
    "StartupBootstrap": (".bootstrap", "StartupBootstrap"),
    "NodeInitializer": (".initializer", "NodeInitializer"),
    "build_projector_params": (".runtime_builders", "build_projector_params"),
    "build_debug_pub_params": (".runtime_builders", "build_debug_pub_params"),
    "build_debug_publisher": (".runtime_builders", "build_debug_publisher"),
    "RuntimeSetup": (".runtime_setup", "RuntimeSetup"),
    "PruneStartupBuilder": (".startup_builder", "PruneStartupBuilder"),
    "StartupReporter": (".startup_reporting", "StartupReporter"),
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
