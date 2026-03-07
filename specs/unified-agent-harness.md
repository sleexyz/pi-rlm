# Unified Agent Harness

## Problem

Three separate code paths exist today:

1. **`runner.ts`** (TS) — full pi-rlm orchestrator, sub-agents, thinking, native tool calls. Used for Claude Opus eval.
2. **`eval_loop.py`** (Python) — bare vLLM loop, no orchestrator, no sub-agents, manually-constructed messages. Used for Qwen/rlm eval.
3. **verl training** — verl's own rollout engine, similar to eval_loop.py but with batched generation + SDPO gradient computation.

These diverge in: system prompt, multi-turn logic, message construction, code extraction, and capabilities.

**Goal:** One agent harness for eval and training. Bun/pi-rlm owns all agent logic. Python owns RL gradients.

## Key Insight

verl needs per-token log-probs (top-k, k=20–100) for SDPO. vLLM's OpenAI-compatible API already returns these via `top_logprobs` parameter — no extra forward pass needed. So Bun can drive inference for both eval and training through the same HTTP interface.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Python Harness                                      │
│                                                      │
│  ┌─────────────┐    ┌──────────────────────────────┐ │
│  │ eval mode   │    │ train mode                   │ │
│  │             │    │                              │ │
│  │ for task:   │    │ for task:                    │ │
│  │   spawn bun │    │   spawn bun                  │ │
│  │   collect   │    │   collect trajectory +       │ │
│  │   results   │    │   top-k logprobs + reward    │ │
│  │             │    │   feed to SDPO               │ │
│  └─────────────┘    └──────────────────────────────┘ │
│                                                      │
│  vLLM HTTP Server (shared)                           │
│  ┌────────────────────────────────────┐              │
│  │ /v1/chat/completions              │              │
│  │ --max-logprobs 100                │              │
│  │ Continuous batching across all     │              │
│  │ concurrent Bun subprocess requests │              │
│  └────────────────────────────────────┘              │
└───────────────────┬──────────────────────────────────┘
                    │ HTTP (localhost)
┌───────────────────▼──────────────────────────────────┐
│  Bun Subprocesses (one per rollout)                  │
│                                                      │
│  agent-runner.ts                                     │
│  ├─ createVllmModel({ baseUrl })                     │
│  ├─ createArcAdapter(task)                           │
│  ├─ Orchestrator (multi-turn loop)                   │
│  │  ├─ createCodeBlockStreamFn()                     │
│  │  ├─ generateCodeBlockSystemPrompt()               │
│  │  ├─ EvalRuntime (persistent JS sandbox)           │
│  │  └─ Sub-agents (spawnAgent) → same vLLM server    │
│  ├─ SessionDir (trace logging)                       │
│  └─ Output: JSON to stdout                           │
│     { reward, trajectory, logprobs, ... }            │
└──────────────────────────────────────────────────────┘
```

**Same `agent-runner.ts` for both eval and training.** The only difference is what Python does with the output:
- **Eval:** log results, write run.json
- **Training:** feed trajectory + logprobs to SDPO optimizer

## vLLM Server Setup

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B \
  --port 8000 \
  --max-model-len 32768 \
  --max-logprobs 100 \
  --gpu-memory-utilization 0.9
```

Bun requests include:
```json
{
  "model": "Qwen/Qwen3-8B",
  "messages": [...],
  "temperature": 0.6,
  "top_p": 0.95,
  "max_tokens": 8192,
  "logprobs": true,
  "top_logprobs": 100
}
```

vLLM's continuous batching handles concurrent requests from multiple Bun subprocesses efficiently — same GPU utilization as verl's in-process batching.

## agent-runner.ts Interface

### Invocation

```bash
echo '$TASK_JSON' | bun run domains/arc-agi-2/src/agent-runner.ts \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen3-8B \
  --task-id 007bbfb7 \
  --max-turns 15 \
  --thinking off \
  --timeout 300000 \
  --top-logprobs 100 \
  --run-dir /data/eval-runs/my-run \
  --session-id 007bbfb7_a0
```

### Output (JSON to stdout)

```json
{
  "taskId": "007bbfb7",
  "reward": 1.0,
  "submitted": true,
  "correct": true,
  "turns": 5,
  "tokens": 12480,
  "timeMs": 45000,
  "trajectory": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "logprobs": [
      {"token": "Let", "logprob": -0.12, "top_logprobs": [{"token": "Let", "logprob": -0.12}, {"token": "I", "logprob": -2.3}, ...]},
      {"token": " me", "logprob": -0.05, "top_logprobs": [...]},
      ...
    ]},
    {"role": "user", "content": "42\n"},
    {"role": "assistant", "content": "...", "logprobs": [...]},
    ...
  ]
}
```

Only assistant messages have `logprobs` (those are the generated tokens). System/user messages don't.

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--base-url` | required | vLLM OpenAI-compat endpoint |
| `--model` | required | Model name/path |
| `--task-id` | required | ARC task ID |
| `--task-from-stdin` | false | Read task JSON from stdin |
| `--max-turns` | 15 | Max generate→interact cycles |
| `--thinking` | off | Thinking level: off, low, medium, high |
| `--timeout` | 300000 | Wall-clock timeout in ms (5 min) |
| `--top-logprobs` | 0 | Top-k logprobs per token (0 = disabled, for eval) |
| `--max-agents` | 10 | Max total agents (root + sub-agents) |
| `--run-dir` | none | Directory for session traces |
| `--session-id` | auto | Session ID for trace files |
| `--num-attempts` | 1 | Attempts per task (pass@K) |
| `--temperature` | 0.6 | Sampling temperature |
| `--top-p` | 0.95 | Top-p sampling |

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success (submitted or max turns reached) |
| 1 | Error (crash, timeout, vLLM unreachable) |

Stderr for progress/debug. Stdout reserved for JSON result.

## Implementation

### agent-runner.ts (new file)

Thin wrapper around existing Orchestrator:

```typescript
#!/usr/bin/env bun
import { Orchestrator, createVllmModel, createCodeBlockStreamFn,
         generateCodeBlockSystemPrompt, SessionDir } from "pi-rlm";
import { createArcAdapter } from "./adapter.js";
import { loadTask } from "./task-loader.js";
import { accuracy } from "./grid-helpers.js";

// Parse args, read task from stdin
const task = JSON.parse(await Bun.stdin.text());
const model = createVllmModel({ id: modelName, baseUrl });
const adapter = createArcAdapter(task);

const orchestrator = new Orchestrator({
  model, adapter,
  thinkingLevel,
  maxAgents,
  streamFn: createCodeBlockStreamFn(),
  generateSystemPrompt: generateCodeBlockSystemPrompt,
  sessionDir: runDir ? new SessionDir({ logDir, sessionId }) : undefined,
  // TODO: pass top_logprobs through to model requests
});

const t0 = Date.now();
const submitted = await orchestrator.run(TASK_PROMPT);

// Score
const predicted = typeof submitted === "function"
  ? submitted(task.test[0].input) : null;
const correct = predicted !== null && accuracy(predicted, task.test[0].output) === 1.0;

// Collect trajectory with logprobs from agent's message history
const trajectory = orchestrator.getAgent().state.messages.map(msg => ({
  role: msg.role,
  content: msg.content,
  ...(msg.logprobs ? { logprobs: msg.logprobs } : {}),
}));

const result = {
  taskId, reward: correct ? 1.0 : 0.0,
  submitted: submitted != null, correct,
  turns: orchestrator.getRuntime().turnCount,
  tokens: orchestrator.getUsageTracker().totalUsage().totalTokens,
  timeMs: Date.now() - t0,
  trajectory,
};

// JSON to stdout
process.stdout.write(JSON.stringify(result));
```

### Logprobs Passthrough

`createVllmModel` creates a model that speaks the OpenAI API. We need to:

1. **Pass `logprobs` + `top_logprobs` through** in the request options. This likely requires a small change to `pi-rlm`'s `streamSimple` or `createVllmModel` to forward these parameters.

2. **Capture logprobs from the response** and attach them to the agent's message history. The OpenAI response includes `choices[0].logprobs.content[*].top_logprobs`. These need to flow through `createCodeBlockStreamFn` and land on the stored assistant message.

This is the main pi-rlm change needed. Everything else (orchestrator, adapter, eval runtime, code block parsing) works as-is.

### eval_loop.py (rewritten)

Becomes a thin dispatcher:

```python
def evaluate_tasks(model_name, tasks, run_dir, run_name,
                   vllm_base_url, max_turns=15, ...):
    """Spawn agent-runner.ts per task, collect results."""
    results = []
    for i, item in enumerate(tasks):
        result = _run_agent(
            task_id=item["id"],
            task=item["task"],
            base_url=vllm_base_url,
            model=model_name,
            run_dir=run_dir,
            max_turns=max_turns,
            timeout=300,
        )
        results.append(result)
        write_run_json(run_dir, run_name, config, results)
        print(f"[{i+1}/{len(tasks)}] {result['taskId']}: "
              f"{'CORRECT' if result['correct'] else 'WRONG'}")


def _run_agent(task_id, task, base_url, model, run_dir,
               max_turns, timeout, top_logprobs=0):
    """Spawn one agent-runner.ts subprocess."""
    cmd = [
        "bun", "run", str(AGENT_RUNNER_PATH),
        "--base-url", base_url,
        "--model", model,
        "--task-id", task_id,
        "--task-from-stdin",
        "--max-turns", str(max_turns),
        "--timeout", str(timeout * 1000),
    ]
    if top_logprobs > 0:
        cmd += ["--top-logprobs", str(top_logprobs)]
    if run_dir:
        cmd += ["--run-dir", run_dir, "--session-id", f"{task_id}_a0"]

    result = subprocess.run(
        cmd, input=json.dumps(task).encode(),
        capture_output=True, timeout=timeout + 30,
    )
    if result.returncode != 0:
        return {"taskId": task_id, "correct": False, "error": result.stderr.decode()[:500]}
    return json.loads(result.stdout)
```

### arc_interaction.py (rewritten for verl)

Same subprocess interface, but verl calls it per-rollout:

```python
class ArcInteraction:
    """verl Interaction that delegates to agent-runner.ts.

    Spawns one Bun subprocess per rollout. The subprocess drives the
    full multi-turn agent loop via HTTP to vLLM. Returns trajectory
    with top-k logprobs for SDPO.
    """

    def __init__(self, config):
        self.vllm_base_url = config.get("vllm_base_url", "http://localhost:8000/v1")
        self.model_name = config.get("model_name")
        self.top_logprobs = config.get("top_logprobs", 100)

    async def run_rollout(self, task_id, task, **kwargs):
        """Run full agent loop, return trajectory + logprobs + reward."""
        result = subprocess.run(
            ["bun", "run", str(AGENT_RUNNER_PATH),
             "--base-url", self.vllm_base_url,
             "--model", self.model_name,
             "--task-id", task_id,
             "--task-from-stdin",
             "--top-logprobs", str(self.top_logprobs)],
            input=json.dumps(task).encode(),
            capture_output=True,
            timeout=300,
        )
        return json.loads(result.stdout)
```

**Note:** This changes verl's calling convention. Instead of verl calling `start_interaction` → `generate_response` (per turn) → `finalize_interaction`, verl calls `run_rollout` once and gets everything back. This requires verl config changes to use the "full rollout" interaction mode instead of the per-turn mode.

### modal_app.py changes

```python
@app.function(gpu="A100:1", ...)
def evaluate(model, dataset, count, ...):
    # Start vLLM as HTTP server
    server = start_vllm_server(model, port=8000, max_logprobs=100)
    wait_for_server(server, port=8000)

    # Load tasks
    tasks = load_tasks(dataset, count)

    # Run eval (spawns agent-runner.ts per task)
    evaluate_tasks(
        model_name=model,
        tasks=tasks,
        run_dir=run_dir,
        run_name=run_name,
        vllm_base_url="http://localhost:8000/v1",
    )

    server.terminate()
```

## File Changes

### New Files

| File | Description |
|------|-------------|
| `domains/arc-agi-2/src/agent-runner.ts` | Single-task CLI: orchestrator + JSON output |

### Modified Files

| File | Change |
|------|--------|
| `pi-rlm` (streamSimple / createVllmModel) | Pass through `top_logprobs`, capture logprobs in response |
| `rl/eval_loop.py` | Rewritten: thin subprocess dispatcher |
| `rl/arc_interaction.py` | Rewritten: single `run_rollout` call instead of per-turn |
| `rl/modal_app.py` | Start vLLM as HTTP server |
| `rl/arc_data.py` | Remove `generate_prompt_via_ts()` (Bun owns prompt) |

### Deleted Files

| File | Reason |
|------|--------|
| `rl/parser.py` | Code extraction in Bun (pi-rlm does this) |
| `rl/js_sandbox.py` | Sandbox in Bun (EvalRuntime directly) |

## Single Source of Truth

| Concern | Owner |
|---------|-------|
| System prompt | Bun: `generateCodeBlockSystemPrompt` + adapter |
| Message formatting | Bun: adapter + orchestrator |
| Code extraction | Bun: `createCodeBlockStreamFn` |
| Tool execution | Bun: `EvalRuntime` + adapter scope |
| Grid helpers | `domains/arc-agi-2/src/grid-helpers.ts` |
| Trace format | Bun: `SessionDir` / `AgentLogger` |
| Inference | vLLM HTTP server (shared by all paths) |
| RL gradients | Python: verl SDPO (consumes trajectory + logprobs) |

## Migration Plan

### Phase 1: agent-runner.ts + eval

1. Add logprobs passthrough to `pi-rlm` (`createVllmModel` / `streamSimple`)
2. Create `agent-runner.ts` — single-task CLI
3. Test locally: vLLM server + agent-runner against one task
4. Rewrite `eval_loop.py` as subprocess dispatcher
5. Update `modal_app.py` to start vLLM HTTP server
6. Delete `parser.py`, `js_sandbox.py`
7. Verify: Qwen3-8B + rlm-qwen3-8b sanity evals work

### Phase 2: Training integration

1. Rewrite `arc_interaction.py` — `run_rollout` spawns agent-runner.ts
2. Update verl config for full-rollout interaction mode
3. Verify: training works, rewards correct, SDPO gets logprobs
4. Compare training curves with old per-turn interaction

### Phase 3: Optimizations

1. Concurrent rollouts: spawn N Bun processes in parallel, vLLM batches automatically
2. Warm subprocess pool: keep Bun processes alive across tasks
3. Sub-agent inference: hits same vLLM server (already works)

## Decisions Log

| Question | Decision | Rationale |
|----------|----------|-----------|
| One or two Bun entry points? | One (`agent-runner.ts`) | vLLM HTTP returns logprobs — no need for separate training path |
| Sub-agents during training? | Yes, reward to main agent only | Sub-agents help; their tokens are "free" for SDPO |
| Who owns prompt construction? | Bun | Single source of truth; Python just consumes trajectory |
| Who owns tokenization? | vLLM server (via chat template) | Bun sends messages, vLLM tokenizes server-side |
| Thinking mode? | Supported via `--thinking` flag | Useful for Qwen3 |
| Timeout? | 5 min per task | Generous for sub-agents, bounded |
| Logprobs? | top-k via `--top-logprobs` (k=20–100) | SDPO needs them; vLLM returns them natively |
| Batching? | vLLM continuous batching handles concurrent subprocess requests | No batch efficiency loss vs in-process |
