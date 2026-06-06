#!/bin/bash
# Main bag validation orchestrator
# Runs the full CurtMini bag workflow with diagnostics
# Usage: ./validate_curtmini_workflow.sh [--bag BAG_PATH] [--duration SECS] [--check-resources]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
BAG_PATH=""
DURATION_SEC=60
CHECK_RESOURCES=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --bag)
            BAG_PATH="$2"
            shift 2
            ;;
        --duration)
            DURATION_SEC="$2"
            shift 2
            ;;
        --check-resources)
            CHECK_RESOURCES=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Defaults
if [[ -z "$BAG_PATH" ]]; then
    BAG_PATH="/mnt/t7_shield/ICNF/ICNF_curt_localization_50hz.bag"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║           CurtMini Bag Validation Workflow                     ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Bag: $BAG_PATH"
echo "  Duration: ${DURATION_SEC}s"
echo "  Repo: $REPO_ROOT"
echo ""

# Check bag exists
if [[ ! -f "$BAG_PATH" ]]; then
    echo "❌ Error: Bag not found: $BAG_PATH"
    exit 1
fi

# Check resources if requested
if [[ "$CHECK_RESOURCES" == "true" ]]; then
    echo "📊 System Resources:"
    echo "  RAM:"
    free -h | head -2 | tail -1 | awk '{print "    Total: " $2 ", Available: " $7}'
    echo "  GPU:"
    nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null | \
        awk -F',' '{printf "    Total: %d MB, Free: %d MB\n", $1, $2}' || echo "    Not available"
    echo ""
fi

# Check Docker
echo "🐳 Docker environment:"
if ! command -v docker &> /dev/null; then
    echo "  ❌ docker not found"
    exit 1
fi
echo "  ✓ docker available"

# Check if we can run docker-compose
if ! command -v docker-compose &> /dev/null; then
    echo "  ⚠ docker-compose not found, will try 'docker compose'"
fi

# Build image if needed
echo ""
echo "🔨 Building Docker image..."
cd "$REPO_ROOT"

if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
else
    COMPOSE_CMD="docker compose"
fi

$COMPOSE_CMD build --quiet sensor-fusion-ros 2>&1 | grep -v "Step\|Using cache\|Setting" || true

echo "✓ Image ready"

# Run validation
echo ""
echo "🎬 Starting validation workflow..."
echo ""

OUTPUT_BASE="/tmp/bag_validation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_BASE"

echo "[Stage 1] Analyzing bag structure..."
rosbag info "$BAG_PATH" > "$OUTPUT_BASE/bag_info.txt" 2>&1 || {
    echo "⚠ rosbag-tool not available on host, proceeding with Docker-based analysis"
}

echo "[Stage 2] Building Docker workspace..."
$COMPOSE_CMD run --rm sensor-fusion-ros \
    bash -c "
        cd /home/ros/ws
        catkin build --no-status --summary 2>&1 | tail -20
    " 2>&1 | tee -a "$OUTPUT_BASE/docker_build.log"

echo ""
echo "[Stage 3] Running sync analysis in Docker..."
$COMPOSE_CMD run --rm \
    -v "$OUTPUT_BASE:/output:rw" \
    -v "/mnt/t7_shield:/bags:ro" \
    sensor-fusion-ros \
    bash -c "
        source /home/ros/ws/devel/setup.bash
        mkdir -p /output
        
        echo '=== Bag Validation ===' > /output/validation.log
        date >> /output/validation.log
        echo '' >> /output/validation.log
        
        # Analyze bag structure
        echo '[Stage 1] Bag Info' >> /output/validation.log
        rosbag info /bags/ICNF/icnf_ikalibr_order_003_010_slim.bag >> /output/bag_info.txt 2>&1 || true
        
        # Run sync analysis if Python script is available
        if [[ -f /home/ros/ws/src/entfac_fusion/tools/validation/validate_bag_workflow.py ]]; then
            echo '[Stage 2] Sync Offset Analysis' >> /output/validation.log
            python3 /home/ros/ws/src/entfac_fusion/tools/validation/validate_bag_workflow.py \\
                /bags/ICNF/icnf_ikalibr_order_003_010_slim.bag \\
                --config curt_mini.yaml \\
                --output-dir /output \\
                --replay-duration 300 >> /output/validation.log 2>&1
        else
            echo 'Sync analysis script not found' >> /output/validation.log
        fi
        
        date >> /output/validation.log
    " 2>&1 | tee -a "$OUTPUT_BASE/docker_build.log"

# Summary
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    Validation Complete                         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Results saved to: $OUTPUT_BASE"
echo ""
echo "📋 Generated files:"
ls -lh "$OUTPUT_BASE"/ 2>/dev/null | tail -n +2 | awk '{printf "  %-40s %6s\n", $9, $5}' || true
echo ""

if [[ -f "$OUTPUT_BASE/sync_stats.json" ]]; then
    echo "📊 Sync Offset Results:"
    python3 -c "
import json
with open('$OUTPUT_BASE/sync_stats.json') as f:
    stats = json.load(f)
    print(f'  Mean offset: {stats[\"mean_delta_sec\"]:+.6f}s')
    print(f'  Stdev: {stats[\"stdev_delta_sec\"]:.6f}s')
    print(f'  Range: [{stats[\"min_delta_sec\"]:+.6f}, {stats[\"max_delta_sec\"]:+.6f}]s')
" 2>/dev/null || true
    echo ""
fi

if [[ -f "$OUTPUT_BASE/bag_info.txt" ]]; then
    echo "📦 Bag Contents:"
    head -20 "$OUTPUT_BASE/bag_info.txt" | grep -E "^path:|^duration|^start|messages" || true
    echo ""
fi

echo "✅ Next steps:"
echo "  1. Review bag_info.txt for topic structure"
echo "  2. Check sync_stats.json for timestamp offset quality"
echo "  3. If offset mean ±5ms: proceed to UFOMap ingestion"
echo "  4. If offset large: adjust semantic_time_offset_sec in curt_mini.yaml"
echo ""
