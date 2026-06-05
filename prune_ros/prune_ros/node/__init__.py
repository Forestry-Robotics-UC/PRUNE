"""ROS node entrypoints for PRUNE."""

from .prune_node import PruneNode, main

ColoredPclNode = PruneNode

__all__ = ["PruneNode", "ColoredPclNode", "main"]
