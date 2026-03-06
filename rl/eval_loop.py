"""Baseline evaluation loop for ARC-AGI-2.

Uses the same components as the verl SDPO training loop:
- vllm.LLM (in-process engine, not HTTP API)
- AutoTokenizer with apply_chat_template(tools=, enable_thinking=False)
- ArcInteraction (same class as training)

Zero code skew by design. Uses tool call format (not code blocks).
"""

import asyncio
import json
import time
from datetime import datetime, timezone

from .arc_data import _format_user_message
from .arc_interaction import ArcInteraction
from .parser import parse_tool_call
from .trace_logger import TraceLogger, write_run_json

# Tool definition for the eval REPL — passed to apply_chat_template
EVAL_TOOL = [{
    "type": "function",
    "function": {
        "name": "eval",
        "description": "Execute JavaScript code in the sandboxed REPL. Variables persist across calls. Use console.log() to see output. Call submit(transform) when done.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "JavaScript code to execute",
                },
            },
            "required": ["code"],
        },
    },
}]


def evaluate_tasks(
    model_name: str,
    tasks: list[dict],
    system_prompt: str,
    run_dir: str,
    run_name: str,
    max_turns: int = 15,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 8192,
    max_model_len: int = 32768,
    gpu_memory_utilization: float = 0.9,
    # Dependency injection for testing (None = create from model_name)
    llm=None,
    tokenizer=None,
    sampling_params=None,
) -> dict:
    """Run eval loop matching verl's GENERATING -> INTERACTING cycle.

    Returns config dict suitable for run.json.
    """
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if llm is None:
        import vllm
        llm = vllm.LLM(
            model=model_name,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )

    if sampling_params is None:
        import vllm
        sampling_params = vllm.SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    # Create interaction (same class as training)
    task_map = {t["id"]: t["task"] for t in tasks}
    interaction = ArcInteraction(tasks=task_map)

    config = {
        "model": model_name,
        "split": "eval",
        "maxTurns": max_turns,
        "temperature": temperature,
        "topP": top_p,
        "maxTokens": max_tokens,
        "startedAt": datetime.now(timezone.utc).isoformat(),
    }
    results = []
    loop = asyncio.new_event_loop()

    try:
        for i, item in enumerate(tasks):
            task_id = item["id"]
            task = item["task"]
            t0 = time.time()
            total_tokens = 0

            # Build initial messages (same as verl)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _format_user_message(task)},
            ]

            # Start interaction
            loop.run_until_complete(
                interaction.start_interaction(instance_id=task_id, task_id=task_id, task=task)
            )

            # Create trace logger
            session_id = f"{task_id}_a0"
            trace = TraceLogger(run_dir, session_id, metadata={
                "taskId": task_id,
                "model": model_name,
                "attempt": 0,
                "run": run_name,
            })
            trace.log_message("user", messages[1]["content"])

            # Multi-turn loop (matches verl's GENERATING -> INTERACTING)
            reward = 0.0
            submitted = False
            turns = 0

            for turn in range(max_turns):
                # GENERATING: tokenize + generate + decode (matches verl exactly)
                token_ids = tokenizer.apply_chat_template(
                    messages,
                    tools=EVAL_TOOL,
                    add_generation_prompt=True,
                    tokenize=True,
                    enable_thinking=False,
                )

                # Guard: skip if conversation exceeds model context
                if len(token_ids) > max_model_len:
                    print(f"  [!] Context overflow ({len(token_ids)} > {max_model_len}), terminating task")
                    break

                outputs = llm.generate(
                    [{"prompt_token_ids": token_ids}],
                    sampling_params=sampling_params,
                )
                output_ids = outputs[0].outputs[0].token_ids
                text = tokenizer.decode(output_ids, skip_special_tokens=True)
                total_tokens += len(token_ids) + len(output_ids)

                # Parse tool call from model output
                code = parse_tool_call(text)
                if code:
                    # Log as structured tool_use content block
                    # Extract any text before the <tool_call> tag
                    tc_idx = text.find("<tool_call>")
                    preamble = text[:tc_idx].strip() if tc_idx > 0 else ""
                    content_blocks = []
                    if preamble:
                        content_blocks.append({"type": "text", "text": preamble})
                    content_blocks.append({"type": "tool_use", "name": "eval", "arguments": {"code": code}})
                    trace.log_message("assistant", content_blocks)

                    # Build tool_calls message for chat template
                    messages.append({
                        "role": "assistant",
                        "content": preamble,
                        "tool_calls": [{"type": "function", "function": {"name": "eval", "arguments": json.dumps({"code": code})}}],
                    })
                else:
                    # No tool call — log raw text
                    trace.log_message("assistant", text)
                    messages.append({"role": "assistant", "content": text})

                # INTERACTING: call interaction (same as verl)
                should_terminate, feedback, rwd, metrics = loop.run_until_complete(
                    interaction.generate_response(task_id, messages)
                )
                turns += 1

                # Log feedback as tool result
                trace.log_message("tool", feedback)

                if should_terminate:
                    reward = rwd
                    submitted = metrics.get("submitted", False)
                    break

                # Tool result message for chat template
                messages.append({"role": "tool", "name": "eval", "content": feedback})

            # Finalize
            loop.run_until_complete(interaction.finalize_interaction(task_id))
            elapsed_ms = int((time.time() - t0) * 1000)
            trace.close(usage={"totalTokens": total_tokens})

            correct = reward > 0
            result = {
                "taskId": task_id,
                "correct": correct,
                "attempts": [{"correct": correct, "tokens": total_tokens, "timeMs": elapsed_ms}],
                "tokens": total_tokens,
                "timeMs": elapsed_ms,
                "reward": reward,
                "submitted": submitted,
                "turns": turns,
            }
            results.append(result)

            # Write incremental run.json
            write_run_json(run_dir, run_name, config, results)

            c = sum(1 for r in results if r["correct"])
            print(f"[{i+1}/{len(tasks)}] {task_id}: {'CORRECT' if correct else 'WRONG'} "
                  f"(turns={turns}, tokens={total_tokens}, {elapsed_ms}ms) "
                  f"-- running: {c}/{len(results)}")

    finally:
        loop.close()
        interaction.pool.close_all()

    return config
