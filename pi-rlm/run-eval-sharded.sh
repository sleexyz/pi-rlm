#!/bin/bash
# Launch 10 parallel shards of the ARC-AGI-2 eval run
# Each shard gets ~12 tasks (120 tasks / 10 shards, round-robin)
#
# Usage: ./run-eval-sharded.sh
# Monitor: tail -f logs/arc-eval-shard-*.log
# Check progress: grep "pass@2:" logs/arc-eval-shard-*.log | wc -l

set -e

NUM_SHARDS=10
LOGS_DIR="logs"
mkdir -p "$LOGS_DIR"

echo "Launching $NUM_SHARDS shards for ARC-AGI-2 evaluation split..."
echo "  Model: claude-opus-4-6"
echo "  Thinking: high"
echo "  Attempts: 2 (pass@2)"
echo "  Max agents: 10"
echo ""

PIDS=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  LOG_FILE="$LOGS_DIR/arc-eval-shard-${i}.log"
  echo "Starting shard $i → $LOG_FILE"
  nohup bun src/arc-runner.ts \
    --split evaluation \
    --all --count 120 \
    --thinking high \
    --shard "$i" --num-shards "$NUM_SHARDS" \
    --log \
    > "$LOG_FILE" 2>&1 &
  PIDS+=($!)
done

echo ""
echo "All shards launched. PIDs: ${PIDS[*]}"
echo ""
echo "Monitor progress:"
echo "  grep 'pass@2:' logs/arc-eval-shard-*.log | wc -l    # completed tasks"
echo "  grep 'CORRECT' logs/arc-eval-shard-*.log | grep 'pass@2' | wc -l  # correct tasks"
echo "  tail -f logs/arc-eval-shard-*.log                    # live output"
echo ""
echo "Merge results when done:"
echo "  bun src/merge-results.ts results/arc/*shard*.json"
