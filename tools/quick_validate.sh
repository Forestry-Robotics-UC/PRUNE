#!/bin/bash
# Simplified CurtMini validation - runs in single Docker session
# Usage: ./quick_validate.sh [duration_sec]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DURATION_SEC="${1:-300}"
OUTPUT_DIR="/tmp/bag_validation_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║           CurtMini Quick Validation                            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Repo: $REPO_ROOT"
echo "  Duration: ${DURATION_SEC}s"
echo "  Output: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

# Determine compose command
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
else
    COMPOSE_CMD="docker compose"
fi

echo "Starting Docker session (from $REPO_ROOT)..."
echo ""

# Change to repo root and run everything in a single container session
cd "$REPO_ROOT"

$COMPOSE_CMD run --rm \
    -v "$OUTPUT_DIR:/output:rw" \
    -v "/mnt/t7_shield:/bags:ro" \
    sensor-fusion-ros \
    bash -c "
    set -e
    
    echo '[INFO] Building workspace...'
    cd /home/ros/ws
    catkin build --no-status 2>&1 | grep -E 'Successful|Failed|Runtime' || true
    
    echo '[INFO] Sourcing setup...'
    source /home/ros/ws/devel/setup.bash
    
    echo '[INFO] Analyzing slim bag...'
    rosbag info /bags/ICNF/icnf_ikalibr_order_003_010_slim.bag > /output/bag_info.txt 2>&1
    
    echo '[INFO] Computing sync offsets...'
    python3 /home/ros/ws/src/entfac_fusion/tools/validate_bag_workflow.py \\
        /bags/ICNF/icnf_ikalibr_order_003_010_slim.bag \\
        --config curt_mini.yaml \\
        --output-dir /output \\
        --replay-duration $DURATION_SEC \\
        2>&1 | tee /output/validation.log
    
    echo ''
    echo '[SUCCESS] Validation complete'
"

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    Results Ready                               ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Output: $OUTPUT_DIR"
echo ""

if [[ -f "$OUTPUT_DIR/sync_stats.json" ]]; then
    echo "📊 Sync Analysis:"
    python3 << 'EOF'
import json
try:
    with open("$OUTPUT_DIR/sync_stats.json") as f:
        stats = json.load(f)
        print(f"  Camera-LiDAR offset: {stats['mean_delta_sec']:+.6f}s (±{stats['stdev_delta_sec']:.6f}s)")
        print(f"  Range: [{stats['min_delta_sec']:+.6f}, {stats['max_delta_sec']:+.6f}]s")
        if abs(stats['mean_delta_sec']) < 0.005 and stats['stdev_delta_sec'] < 0.01:
            print("  ✓ GOOD - timestamps well synchronized")
        else:
            print("  ⚠ May need offset tuning")
except Exception as e:
    print(f"  Could not parse results: {e}")
EOF
    echo ""
fi

ls -lh "$OUTPUT_DIR" | tail -n +2 | awk '{printf "  %-40s %8s\n", $9, $5}'
echo ""

echo "✅ Next:"
echo "  1. Check bag_info.txt for topic structure"
echo "  2. Review sync_stats.json for offset quality"
echo "  3. If offset good: proceed to UFOMap ingestion"
echo "  4. Otherwise: adjust semantic_time_offset_sec in curt_mini.yaml"
echo ""
