"""Merge per-shard eval results into a single run.json.

Usage:
    modal volume get sdpo-arc-data eval-runs/<run-name> runs/<run-name>
    python rl/merge_shards.py runs/<run-name>
"""

import json
import os
import sys
from glob import glob


def merge_shards(run_dir: str):
    shard_files = sorted(glob(os.path.join(run_dir, "shard-*.json")))
    if not shard_files:
        print(f"No shard-*.json files found in {run_dir}")
        sys.exit(1)

    all_results = []
    config = None

    for path in shard_files:
        with open(path) as f:
            data = json.load(f)
        if config is None:
            config = data.get("config", {})
        all_results.extend(data.get("results", []))

    # Deduplicate by taskId (in case of overlap)
    seen = set()
    unique_results = []
    for r in all_results:
        if r["taskId"] not in seen:
            seen.add(r["taskId"])
            unique_results.append(r)

    correct = sum(1 for r in unique_results if r.get("correct"))
    total = len(unique_results)
    total_tokens = sum(r.get("tokens", 0) for r in unique_results)
    total_time_ms = sum(r.get("timeMs", 0) for r in unique_results)

    run_json = {
        "name": os.path.basename(run_dir),
        "startedAt": config.get("startedAt", ""),
        "config": config,
        "results": unique_results,
        "summary": {
            "correct": correct,
            "total": total,
            "pct": correct / total if total > 0 else 0,
            "tokens": total_tokens,
            "timeMs": total_time_ms,
        },
    }

    out_path = os.path.join(run_dir, "run.json")
    with open(out_path, "w") as f:
        json.dump(run_json, f, indent=2)

    print(f"Merged {len(shard_files)} shards: {correct}/{total} correct ({correct/total*100:.1f}%)")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <run-dir>")
        sys.exit(1)
    merge_shards(sys.argv[1])
