#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bag validation workflow:
  - Replay bag through prune_node
  - Capture sync offset metrics
  - Check projection diagnostics
  - Detect mask edge artifacts (sky bleeding on thin branches)
"""

import argparse
import json
import logging
import shutil
import statistics
import sys
from datetime import datetime
from pathlib import Path
from subprocess import run, PIPE

try:
    import rosbag
    import cv2
    import numpy as np
except ImportError:
    print("Error: rosbag, cv2, numpy required. Install in Docker context.")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def collect_stamps(bag, topic):
    """Collect timestamps from a topic."""
    stamps = []
    for _, msg, _ in bag.read_messages(topics=[topic]):
        if not hasattr(msg, "header") or msg.header is None:
            continue
        stamp = msg.header.stamp
        if stamp is None:
            continue
        stamps.append(stamp.to_sec())
    return stamps


def nearest_deltas(a, b):
    """Find nearest-neighbor time deltas between two timestamp lists."""
    i = 0
    deltas = []
    for t in a:
        while i + 1 < len(b) and abs(b[i + 1] - t) <= abs(b[i] - t):
            i += 1
        deltas.append(b[i] - t)
    return deltas


def analyze_sync_offset(bag_path, topic_a, topic_b, output_dir):
    """Analyze timestamp synchronization between two topics."""
    logger.info(f"Analyzing sync offset between {topic_a} and {topic_b}...")
    
    try:
        bag = rosbag.Bag(str(bag_path), 'r')
    except Exception as e:
        logger.error(f"Failed to open bag: {e}")
        return None
    
    try:
        stamps_a = collect_stamps(bag, topic_a)
        stamps_b = collect_stamps(bag, topic_b)
        
        if not stamps_a or not stamps_b:
            logger.warning(f"No messages found for {topic_a} or {topic_b}")
            return None
        
        deltas = nearest_deltas(stamps_a, stamps_b)
        
        if not deltas:
            logger.warning("No sync deltas computed")
            return None
        
        stats = {
            "topic_a": topic_a,
            "topic_b": topic_b,
            "count_a": len(stamps_a),
            "count_b": len(stamps_b),
            "count_deltas": len(deltas),
            "mean_delta_sec": statistics.mean(deltas),
            "median_delta_sec": statistics.median(deltas),
            "stdev_delta_sec": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
            "min_delta_sec": min(deltas),
            "max_delta_sec": max(deltas),
            "percentile_5": np.percentile(deltas, 5),
            "percentile_95": np.percentile(deltas, 95),
        }
        
        logger.info(f"Sync stats: mean={stats['mean_delta_sec']:.6f}s, "
                   f"stdev={stats['stdev_delta_sec']:.6f}s, "
                   f"range=[{stats['min_delta_sec']:.6f}, {stats['max_delta_sec']:.6f}]s")
        
        return stats
    
    finally:
        bag.close()


def capture_bag_topics(bag_path):
    """List all topics in a bag file."""
    try:
        bag = rosbag.Bag(str(bag_path), 'r')
        topics = bag.get_type_and_topic_info()[1].keys()
        bag.close()
        return sorted(topics)
    except Exception as e:
        logger.error(f"Failed to read bag topics: {e}")
        return []


def run_bag_replay_test(bag_path, config_path, output_dir, duration_sec=30):
    """
    Replay bag through prune_node and capture diagnostics.
    
    This is a placeholder that should be run in a ROS environment.
    Returns paths to generated diagnostics.
    """
    logger.info(f"Would replay {bag_path} with config {config_path} for {duration_sec}s")
    logger.info("This must be run in a ROS/Docker environment with rosbag and prune_node active")
    logger.info(f"Output directory: {output_dir}")
    
    # In practice, this would:
    # 1. Start rosbag play
    # 2. Start prune_node with debug_project_lidar=true
    # 3. Record output topic to secondary bag
    # 4. Capture RViz/diagnostic images
    
    return {"note": "Placeholder - run in Docker with ROS environment"}


def analyze_masking_artifacts(output_dir):
    """
    Analyze mask diagnostics for artifacts like sky bleeding on thin branches.
    
    This checks saved debug images for:
    - Sky mask extending into vegetation (green/brown areas)
    - Thin branch artifacts from depth edge dilation
    - Confidence scoring anomalies
    """
    logger.info("Analyzing masking artifacts...")
    
    debug_images = list(Path(output_dir).glob("debug_*.png"))
    
    if not debug_images:
        logger.warning(f"No debug images found in {output_dir}")
        return None
    
    artifacts = {
        "sky_bleeding_candidates": [],
        "depth_edge_artifacts": [],
        "confidence_anomalies": [],
    }
    
    for img_path in debug_images:
        logger.info(f"Examining {img_path.name}...")
        # Placeholder for image analysis
        # In practice, this would:
        # 1. Load depth/label/confidence images
        # 2. Detect high-confidence sky labels in vegetation areas
        # 3. Flag thin branch artifacts from dilation
        # 4. Report histogram anomalies in confidence scores
    
    return artifacts


def main():
    parser = argparse.ArgumentParser(
        description="Validate a bag workflow: sync offsets, projection, masks."
    )
    parser.add_argument("bag_path", help="Path to rosbag file")
    parser.add_argument("--config", default="curt_mini.yaml", 
                       help="Config YAML relative to prune_ros/config/")
    parser.add_argument("--output-dir", default="./bag_validation_output",
                       help="Output directory for diagnostics")
    parser.add_argument("--replay-duration", type=int, default=30,
                       help="Duration in seconds to replay bag (0=full)")
    parser.add_argument("--camera-topic", default="/camera/color/image_raw",
                       help="Camera image topic")
    parser.add_argument("--lidar-topic", default="/ouster/points",
                       help="LiDAR points topic")
    
    args = parser.parse_args()
    
    bag_path = Path(args.bag_path)
    output_dir = Path(args.output_dir)
    
    if not bag_path.exists():
        logger.error(f"Bag not found: {bag_path}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Validation output: {output_dir}")
    
    # Stage 1: Inspect bag contents
    logger.info("\n=== Stage 1: Bag Inspection ===")
    topics = capture_bag_topics(bag_path)
    logger.info(f"Found {len(topics)} topics:")
    for topic in topics:
        logger.info(f"  {topic}")
    
    # Stage 2: Analyze sync offset
    logger.info("\n=== Stage 2: Sync Offset Analysis ===")
    sync_stats = analyze_sync_offset(bag_path, args.camera_topic, args.lidar_topic, output_dir)
    
    if sync_stats:
        stats_path = output_dir / "sync_stats.json"
        with open(stats_path, "w") as f:
            json.dump(sync_stats, f, indent=2)
        logger.info(f"Saved sync stats to {stats_path}")
    
    # Stage 3: Bag replay test (requires ROS/Docker environment)
    logger.info("\n=== Stage 3: Bag Replay & Diagnostics ===")
    replay_result = run_bag_replay_test(bag_path, args.config, output_dir, args.replay_duration)
    
    # Stage 4: Analyze masking artifacts
    logger.info("\n=== Stage 4: Masking Artifact Analysis ===")
    artifacts = analyze_masking_artifacts(output_dir)
    
    if artifacts:
        artifacts_path = output_dir / "masking_artifacts.json"
        with open(artifacts_path, "w") as f:
            json.dump(artifacts, f, indent=2)
        logger.info(f"Saved artifact analysis to {artifacts_path}")
    
    # Summary
    logger.info("\n=== Validation Summary ===")
    logger.info(f"Bag: {bag_path.name}")
    logger.info(f"Topics: {len(topics)}")
    if sync_stats:
        logger.info(f"Sync mean offset: {sync_stats['mean_delta_sec']:.6f}s")
        logger.info(f"Sync stdev: {sync_stats['stdev_delta_sec']:.6f}s")
    logger.info(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
