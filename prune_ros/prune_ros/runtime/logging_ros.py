#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: GNU General Public License v3.0 (GPL-3.0)
# Repository: PRUNE
#
# Description:
#   Route Python logging through rospy with a node/method tag for clean ROS logs.

"""ROS logging utilities."""

from __future__ import annotations

import logging
import threading
from typing import Optional

import rospy


class RospyLogHandler(logging.Handler):
    """Forward Python logging records into rospy logging with node prefix."""

    def __init__(self, node_name: str):
        super().__init__()
        self._node_name = str(node_name).lstrip("/")
        self._local = threading.local()

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(self._local, "in_emit", False):
            return
        self._local.in_emit = True
        try:
            msg = self.format(record)
            tag = f"{self._node_name}:{record.name}.{record.funcName}"
            full = f"[{tag}] {msg}"
            if record.levelno >= logging.ERROR:
                rospy.logerr(full)
            elif record.levelno >= logging.WARNING:
                rospy.logwarn(full)
            elif record.levelno >= logging.INFO:
                rospy.loginfo(full)
            else:
                rospy.logdebug(full)
        finally:
            self._local.in_emit = False


def configure_core_logging(node_name: str, *, debug: bool) -> None:
    """Route prune_core logs through rospy with node context."""
    logger = logging.getLogger("prune_core")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    handler = RospyLogHandler(node_name)
    handler.setLevel(logging.DEBUG if debug else logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))

    for existing in list(logger.handlers):
        if isinstance(existing, RospyLogHandler):
            logger.removeHandler(existing)
    logger.addHandler(handler)
    logger.propagate = False


class NodeLogger:
    """Small helper to ensure node/method tags exist on ROS logs."""

    def __init__(self, node_name: str):
        self._node_name = str(node_name).lstrip("/")

    def debug(self, method: str, msg: str, *args) -> None:
        tag = f"{self._node_name}.{method}" if method else self._node_name
        rospy.logdebug(f"[{tag}] {str(msg)}", *args)

    def info(self, method: str, msg: str, *args) -> None:
        tag = f"{self._node_name}.{method}" if method else self._node_name
        rospy.loginfo(f"[{tag}] {str(msg)}", *args)

    def warn(self, method: str, msg: str, *args) -> None:
        tag = f"{self._node_name}.{method}" if method else self._node_name
        rospy.logwarn(f"[{tag}] {str(msg)}", *args)

    def error(self, method: str, msg: str, *args) -> None:
        tag = f"{self._node_name}.{method}" if method else self._node_name
        rospy.logerr(f"[{tag}] {str(msg)}", *args)

