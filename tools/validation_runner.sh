#!/bin/bash
# Bag validation runner for Docker environment
# Usage: docker-compose run --rm sensor-fusion-ros /bags/validation_runner.sh <bag_path>

set -e

BAG_PATH="${1:?Usage: $0 <bag_path>}"
CONFIG="${2:-curt_mini.yaml}"
OUTPUT_BASE="${3:-/tmp/bag_validation}"
DURATION_SEC="${4:-60}"

if [[ ! -f "$BAG_PATH" ]]; then
    echo "Error: Bag not found: $BAG_PATH"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_BASE}/${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "====== Bag Validation Workflow ======"
echo "Bag: $BAG_PATH"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "Duration: ${DURATION_SEC}s"
echo "========================================"

# Stage 1: Pre-flight checks
echo ""
echo "[Stage 1] Pre-flight checks..."
rosbag info "$BAG_PATH" > "$OUTPUT_DIR/bag_info.txt" 2>&1 || true
python3 -c "import sys; sys.path.insert(0, '/home/ros/ws/src/entfac_fusion'); from tools.validation.validate_bag_workflow import capture_bag_topics; topics = capture_bag_topics('$BAG_PATH'); [print(t) for t in topics]" > "$OUTPUT_DIR/topics.txt" 2>&1 || true

# Stage 2: Sync offset analysis
echo ""
echo "[Stage 2] Analyzing sync offset..."
python3 /home/ros/ws/src/entfac_fusion/tools/validation/validate_bag_workflow.py \
    "$BAG_PATH" \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --replay-duration "$DURATION_SEC" \
    2>&1 | tee -a "$OUTPUT_DIR/validation.log"

# Stage 3: Prepare ROS environment
echo ""
echo "[Stage 3] Starting ROS and prune_node..."
source /home/ros/ws/devel/setup.bash

# Create temporary ROS master
roscore &
ROSCORE_PID=$!
sleep 2

# Start bag replay in background
echo "[Stage 3a] Starting bag replay..."
(
  sleep 1
  rosbag play "$BAG_PATH" \
      --clock \
      --duration "$DURATION_SEC" \
      2>&1 | tee -a "$OUTPUT_DIR/rosbag_play.log" &
  ROSBAG_PID=$!
  wait $ROSBAG_PID
) &

# Start prune_node with debug enabled
echo "[Stage 3b] Starting prune_node..."
rosrun entfac_fusion_ros prune_node.py _debug:=true \
    2>&1 | tee -a "$OUTPUT_DIR/prune_node.log" &
NODE_PID=$!

# Let it run for specified duration
sleep "$((DURATION_SEC + 2))"

# Cleanup
echo ""
echo "[Stage 4] Cleanup..."
kill $NODE_PID 2>/dev/null || true
sleep 1
pkill -f rosbag || true
kill $ROSCORE_PID 2>/dev/null || true

# Stage 5: Collect outputs
echo ""
echo "[Stage 5] Collecting results..."
cp "$OUTPUT_DIR"/*.log "$OUTPUT_DIR/" 2>/dev/null || true

echo ""
echo "====== Validation Complete ======"
echo "Results: $OUTPUT_DIR"
echo "  - bag_info.txt: Topic summary"
echo "  - topics.txt: Full topic list"
echo "  - sync_stats.json: Timestamp offset analysis"
echo "  - validation.log: Validation script output"
echo "  - rosbag_play.log: Bag replay log"
echo "  - prune_node.log: Node output with debug info"
