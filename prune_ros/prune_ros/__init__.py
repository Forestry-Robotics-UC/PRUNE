#!/usr/bin/env python3
"""ROS1 Noetic helpers and nodes for PRUNE."""


def main():
    """Run the PRUNE ROS node entrypoint."""
    from prune_ros.node.prune_node import main as _main

    return _main()


__all__ = ["main"]
