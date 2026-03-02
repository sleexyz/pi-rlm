#!/bin/bash
# Retry the 41 failed ARC-AGI-2 tasks with fixed max_tokens (128k vs 32k)
# Uses --task-ids-file to load specific tasks instead of the full split
#
# Usage: ./run-retry-failed.sh
# Monitor: tail -f logs/arc-retry-shard-*.log
# Check progress: grep "pass@2:" logs/arc-retry-shard-*.log | wc -l

set -e

NUM_SHARDS=5
LOGS_DIR="logs"
mkdir -p "$LOGS_DIR"

echo "Launching $NUM_SHARDS shards for ARC-AGI-2 retry of failed tasks..."
echo "  Tasks file: failed-tasks.txt (41 tasks)"
echo "  Model: claude-opus-4-6"
echo "  Thinking: high"
echo "  Attempts: 2 (pass@2)"
echo "  Max agents: 10"
echo ""

PIDS=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  LOG_FILE="$LOGS_DIR/arc-retry-shard-${i}.log"
  echo "Starting shard $i → $LOG_FILE"
  nohup bun src/arc-runner.ts \
    --split evaluation \
    --task-ids-file failed-tasks.txt \
    --label retry \
    --thinking high \
    --max-agents 10 \
    --shard "$i" --num-shards "$NUM_SHARDS" \
    --log \
    > "$LOG_FILE" 2>&1 &
  PIDS+=($!)
done

echo ""
echo "All shards launched. PIDs: ${PIDS[*]}"
echo ""
echo "Monitor progress:"
echo "  grep 'pass@2:' logs/arc-retry-shard-*.log | wc -l    # completed tasks"
echo "  grep 'CORRECT' logs/arc-retry-shard-*.log | grep 'pass@2' | wc -l  # correct tasks"
echo "  tail -f logs/arc-retry-shard-*.log                    # live output"
echo ""
echo "Merge results when done:"
echo "  # 1. Merge retry shards"
echo "  bun src/merge-results.ts results/arc/*retry-shard*.json"
echo "  # 2. Combine baseline + retry (retry overrides baseline for the 41 tasks)"
echo "  bun src/merge-results.ts results/arc/2026-02-28T17-56-02-573Z-merged.json results/arc/*retry-merged*.json"
