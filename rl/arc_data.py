"""
ARC-AGI-2 data loader for verl SDPO training.

Loads ARC tasks and produces parquet files in verl's expected format.
Eval prompt generation is handled by Bun (agent-runner.ts).
generate_prompt_via_ts is still used for training data prep (parquet creation).
"""

import json
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERATE_PROMPT_SCRIPT = PROJECT_ROOT / "domains" / "arc-agi-2" / "src" / "generate-prompt.ts"
DEFAULT_DATA_DIR = PROJECT_ROOT / "downloads" / "ARC-AGI-2" / "data"


def load_arc_tasks(split: str, data_dir: str | None = None) -> list[dict]:
    """Load ARC tasks from a directory.

    Returns list of {"id": str, "task": {train: [...], test: [...]}}
    """
    if data_dir is None:
        data_dir = str(DEFAULT_DATA_DIR)
    task_dir = os.path.join(data_dir, split)
    tasks = []
    for fname in sorted(os.listdir(task_dir)):
        if not fname.endswith(".json"):
            continue
        task_id = fname.replace(".json", "")
        with open(os.path.join(task_dir, fname)) as f:
            task = json.load(f)
        tasks.append({"id": task_id, "task": task})
    return tasks


def generate_prompt_via_ts(task: dict) -> str:
    """Generate system prompt by calling the existing TS function via bun.

    This ensures the system prompt matches exactly what the inference runner uses.
    """
    result = subprocess.run(
        ["bun", "run", str(GENERATE_PROMPT_SCRIPT)],
        input=json.dumps(task).encode(),
        capture_output=True,
        cwd=str(PROJECT_ROOT),
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"generate-prompt.ts failed: {result.stderr.decode()}")
    return result.stdout.decode()


def prepare_verl_dataset(tasks: list[dict]) -> "pandas.DataFrame":
    """Convert ARC tasks to verl's expected parquet format.

    Each row has:
    - data_source: "arc-agi-2"
    - prompt: [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_msg}]
    - ability: "code"
    - reward_model: {"style": "code", "ground_truth": expected_output}
    - extra_info: {"task_id": ..., "interaction_kwargs": {"name": "arc", "task_id": ...}}
    """
    import pandas as pd

    # System prompt is task-independent — generate once
    system_prompt = generate_prompt_via_ts(tasks[0]["task"])

    rows = []
    for item in tasks:
        task_id = item["id"]
        task = item["task"]

        # User message: present the task data
        user_msg = _format_user_message(task)

        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        rows.append({
            "data_source": "arc-agi-2",
            "prompt": prompt,
            "ability": "code",
            "reward_model": {
                "style": "code",
                "ground_truth": json.dumps(task["test"][0]["output"]),
            },
            "extra_info": {
                "task_id": task_id,
                "interaction_kwargs": {
                    "name": "arc",
                    "task_id": task_id,
                    "task": task,
                },
            },
        })

    return pd.DataFrame(rows)


def _format_user_message(task: dict) -> str:
    """Format the initial user message presenting the ARC task."""
    parts = ["Here is the ARC task to solve:\n"]

    for i, ex in enumerate(task["train"]):
        parts.append(f"### Training Example {i + 1}")
        parts.append(f"Input ({len(ex['input'])}x{len(ex['input'][0])}):")
        parts.append("```")
        parts.append(_grid_to_str(ex["input"]))
        parts.append("```")
        parts.append(f"Output ({len(ex['output'])}x{len(ex['output'][0])}):")
        parts.append("```")
        parts.append(_grid_to_str(ex["output"]))
        parts.append("```\n")

    for i, ex in enumerate(task["test"]):
        parts.append(f"### Test Input {i + 1}")
        parts.append(f"Input ({len(ex['input'])}x{len(ex['input'][0])}):")
        parts.append("```")
        parts.append(_grid_to_str(ex["input"]))
        parts.append("```\n")

    parts.append("Analyze the training examples, discover the transformation rule, implement `transform(grid)`, and call `submit(transform)` when it achieves accuracy=1.0 on all training examples.")

    return "\n".join(parts)


def _grid_to_str(grid: list[list[int]]) -> str:
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def _to_large(field: "pyarrow.Field") -> "pyarrow.Field":
    """Convert Arrow field to large types (avoids 32-bit offset overflow)."""
    import pyarrow as pa
    t = field.type
    if pa.types.is_string(t):
        return pa.field(field.name, pa.large_string(), field.nullable, field.metadata)
    if pa.types.is_binary(t):
        return pa.field(field.name, pa.large_binary(), field.nullable, field.metadata)
    if pa.types.is_list(t):
        return pa.field(field.name, pa.large_list(_to_large(pa.field("item", t.value_type)).type),
                        field.nullable, field.metadata)
    if pa.types.is_struct(t):
        return pa.field(field.name,
            pa.struct([_to_large(pa.field(f.name, f.type, f.nullable, f.metadata)) for f in t]),
            field.nullable, field.metadata)
    return field


def save_parquet(df: "pandas.DataFrame", path: str):
    """Save DataFrame to parquet with verl-compatible settings.

    Uses LargeString/LargeList types and small row groups to match
    kirby's preprocess.py format (avoids 32-bit offset overflow).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(df)
    large_schema = pa.schema([_to_large(pa.field(f.name, f.type, f.nullable, f.metadata)) for f in table.schema])
    table = table.cast(large_schema)
    writer = None
    try:
        for start in range(0, len(table), 32):
            chunk = table.slice(start, min(32, len(table) - start))
            if writer is None:
                writer = pq.ParquetWriter(path, chunk.schema, compression="zstd")
            writer.write_table(chunk)
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="training")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=None)
    args = parser.parse_args()

    tasks = load_arc_tasks(args.split, args.data_dir)
    if args.count:
        tasks = tasks[: args.count]
    print(f"Loaded {len(tasks)} tasks from {args.split}")

    df = prepare_verl_dataset(tasks)
    save_parquet(df, args.output)
    print(f"Saved {len(df)} rows to {args.output}")
