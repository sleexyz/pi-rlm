# ARC-AGI-2 Domain

## Resuming runs

The runner supports `--resume <session-id>` to continue an interrupted run.

**Requirements:**
- Logs are at `logs/arc-2/<session-id>/agent-0.jsonl`

**Example:**
```
bun domains/arc-agi-2/src/runner.ts --resume 20260304_024307 --log --stream
```

The resume automatically reads `taskId` and `split` from session metadata. Override with `--task-id` or `--split` if needed.
