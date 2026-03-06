# Multi-Turn SDPO for ARC-AGI-2

## Overview

Train Qwen3-8B to solve ARC-AGI-2 tasks using multi-turn Self-Distillation Policy Optimization (SDPO), running on Modal via the existing verl + SDPO infrastructure from kirby.

We already have a working SDPO training pipeline in `~/projects/kirby/sdpo_modal.py` that uses the verl framework with top-K KL divergence (default K=100, over the teacher's most probable tokens per position, with optional tail bucket for remaining probability mass), EMA teacher, and importance-sampling correction. The verl SDPO fork already supports multi-turn rollouts via `tool_agent_loop.py`. We need to plug in an ARC environment and ARC data.

We also have a complete ARC evaluation environment in TypeScript: `EvalRuntime` (persistent JS execution), `createArcAdapter()` (scope setup with grid helpers + task data), `generateCodeBlockSystemPrompt()` (system prompt), and a runner that already supports `--provider vllm --format block`. **The RL training environment must reuse this stack** to guarantee zero execution skew between training and inference.

## Why verl, not custom

The user_interactions paper (Kleine Buening et al.) proves that the logprob-advantage policy gradient is an unbiased one-sample estimator of the full KL distillation gradient (Lemma B.1). They chose the simpler form for convenience. But with expensive multi-turn trajectories (few samples per step), we want the lower-variance top-K KL — which verl already computes.

| | Top-K KL (verl, K=100) | Logprob advantage (user_interactions) |
|---|---|---|
| Expected gradient | Same (approx) | Same (unbiased) |
| Variance per sample | Low (averages over 100 tokens + tail) | High (one-sample REINFORCE) |
| Signal per position | ~100x richer | 1x |
| Implementation | Already done (kirby) | Would need to build |

## What verl/SDPO already provides

From `~/projects/kirby/downloads/SDPO/`:

- **Multi-turn rollout engine** (`verl/experimental/agent_loop/tool_agent_loop.py`): State machine — GENERATING → PROCESSING_TOOLS → INTERACTING → repeat. Tracks `response_ids`, `response_logprobs`, `response_mask`, `turn_scores` per turn.
- **Interaction interface** (`verl/interactions/base.py`): Extension point for custom environments. Returns `(should_terminate, feedback, reward, metrics)`.
- **SDPO loss** (`verl/trainer/ppo/core_algos.py:compute_self_distillation_loss`): Full-logit KL divergence with configurable alpha (forward KL, reverse KL, or JSD). Top-K approximation.
- **Teacher model**: EMA-updated teacher for stable distillation targets.
- **Rollout correction**: Per-token importance sampling with clipping.
- **GRPO advantage estimator**: Vectorized, supports multi-turn via response_mask.
- **Reprompt template** (`SelfDistillationConfig`): Configurable feedback injection format.
- **vLLM integration**: Rollout generation, weight sync, LoRA support.
- **Modal deployment** (`sdpo_modal.py`): Volumes, GPU config, image building, WandB.

## What pi-rlm/domains already provides

The existing TypeScript stack that the RL environment **must reuse**:

- **`EvalRuntime`** (`pi-rlm/src/eval-runtime.ts`): Persistent JavaScript execution context using `node:vm`. Declaration scanning, async IIFE wrapping, console capture, timeout, error handling. Variables survive across eval calls within a session.
- **`createArcAdapter()`** (`domains/arc-agi-2/src/adapter.ts`): Single source of truth for what the model sees in its execution scope — all grid helpers, task data (trainingExamples, testInputs), scoring functions (accuracy, softAccuracy).
- **`parseReplBlocks()`** (`pi-rlm/src/repl-stream.ts`): Parses ```js/```repl code blocks from model text.
- **`generateCodeBlockSystemPrompt()`** (`pi-rlm/src/repl-stream.ts`): System prompt for code-block format (code execution env, variable persistence, sub-agents, submit protocol).
- **`toolResultToUserMessage()`** format: `"Code executed:\n```js\n{code}\n```\n\nREPL output:\n{output}"`.
- **Grid helpers** (`domains/arc-agi-2/src/grid-helpers.ts`): 20+ functions (rotate, flip, crop, paste, tile, connectedComponents, accuracy, etc.).
- **ARC prompts** (`domains/arc-agi-2/src/prompt.ts`): ARC_PREMISE, ARC_REFERENCE, ARC_SUB_AGENT_PROMPT.
- **ARC runner** (`domains/arc-agi-2/src/runner.ts`): End-to-end evaluation with `--provider vllm --format block`.

## What we need to build

### 1. Sandbox Server (`domains/arc-agi-2/src/sandbox-server.ts`)

A long-lived TypeScript process that wraps `EvalRuntime` + `createArcAdapter()`. JSON-line protocol over stdin/stdout.

- Imports `EvalRuntime` from `pi-rlm/src/eval-runtime.ts`
- Imports `createArcAdapter` from `./adapter.ts`
- On init: creates adapter, creates EvalRuntime with adapter scope + submit()
- On eval: calls `runtime.eval(code)`, returns result with stdout/stderr/submitted/error
- Persistent context across evals within a session (variables survive)
- ~50 lines of new code

### 2. Python Sandbox Wrapper (`rl/js_sandbox.py`)

Manages long-lived `bun` subprocesses — one per rollout instance.

- `SandboxSession`: wraps a single bun subprocess, sends JSON-line commands, reads results
- `SandboxPool`: manages sessions by instance_id, creates/destroys as needed
- Timeout and error handling
- ~80 lines of new code

### 3. ARC Interaction (`rl/arc_interaction.py`)

Implements verl's `Interaction` interface. The bridge between verl's rollout engine and the TS execution environment.

```python
class ArcInteraction(Interaction):
    def __init__(self, tasks):
        self.tasks = tasks
        self.pool = SandboxPool()

    async def generate_response(self, instance_id, messages, **kwargs):
        code_blocks = parse_code_blocks(messages[-1]["content"])
        session = self.pool.get_or_create(instance_id, self.tasks[instance_id])

        for block in code_blocks:
            result = session.eval(block)

        reward = 1.0 if result.submitted and accuracy == 1.0 else 0.0
        should_terminate = result.submitted is not None or turn_count >= max_turns
        # Format matches toolResultToUserMessage from pi-rlm
        feedback = f"Code executed:\n```js\n{code}\n```\n\nREPL output:\n{output}"

        return should_terminate, feedback, reward, {"accuracy": reward}
```

~100 lines of new code.

### 4. Code-Block Parser (`rl/parser.py`)

Port of `parseReplBlocks` to Python. Same regex: ` ```(?:repl|js)\s*\n(.*?)\n``` `.
~10 lines.

### 5. ARC Data Loader (`rl/arc_data.py`)

Load ARC-AGI-2 tasks into verl's expected data format. System prompt generated by calling `generateCodeBlockSystemPrompt()` via bun at data-prep time — NOT ported to Python.

### 6. Configuration (`rl/sdpo_arc.yaml`)

Adapted from kirby's SDPO config. Key changes for multi-turn ARC:
```yaml
actor_rollout_ref:
  actor:
    self_distillation:
      max_reprompt_len: 32768
  rollout:
    n: 4
    max_new_tokens: 8192
data:
  train_batch_size: 8
trainer:
  total_epochs: 30
  test_freq: 3
```

### 7. Modal Deployment (`rl/modal_app.py`)

Adapted from `sdpo_modal.py`. Same image base + Bun + TS source files (pi-rlm/src/, domains/arc-agi-2/src/).

## Architecture

```
                        Modal (4x A100-80GB)
  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │  verl Framework                                             │
  │  ├─ Actor (policy model, LoRA)                             │
  │  ├─ Rollout (vLLM, multi-turn via tool_agent_loop)         │
  │  ├─ Teacher (EMA-updated model)                            │
  │  └─ Trainer (SDPO loss, GRPO advantages)                   │
  │                                                             │
  │  ARC Environment (NEW — thin glue over existing TS stack)   │
  │  ├─ ArcInteraction (Python) — verl Interaction impl        │
  │  ├─ SandboxPool (Python) — manages bun subprocesses        │
  │  └─ sandbox-server.ts — reuses EvalRuntime + createAdapter │
  │       ├─ imports pi-rlm/src/eval-runtime.ts                │
  │       ├─ imports domains/arc-agi-2/src/adapter.ts          │
  │       └─ imports domains/arc-agi-2/src/grid-helpers.ts     │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘
```

The multi-turn loop per rollout:
```
verl rollout worker:
  1. Generate assistant response (vLLM)
  2. → ArcInteraction.generate_response()
     a. Parse ```js blocks (Python regex — port of parseReplBlocks)
     b. Send code to long-lived bun subprocess (sandbox-server.ts)
     c. sandbox-server.ts: EvalRuntime.eval(code) with persistent context
     d. Return result (stdout, stderr, submitted, error)
     e. Compute reward (accuracy of submitted vs expected)
     f. Format feedback (matching toolResultToUserMessage format)
  3. If not terminated: append feedback, goto 1
  4. If terminated: return trajectory with per-turn logprobs + rewards
```

## Data Flow

```
ARC Task JSON
    │
    ▼
Data Prep (bun generates system prompts via generateCodeBlockSystemPrompt)
    │
    ▼
verl Data Loader (parquet with prompts)
    │
    ▼
verl Rollout (tool_agent_loop state machine)
  ├─ GENERATING: vLLM generates response with logprobs
  ├─ PROCESSING_TOOLS: ArcInteraction parses ```js blocks
  ├─ INTERACTING: sandbox-server.ts executes via EvalRuntime, returns feedback + reward
  └─ (repeat until submit or max_turns)
    │
    ▼
verl Teacher Forward Pass
  └─ EMA teacher computes logprobs with feedback-conditioned reprompt
    │
    ▼
SDPO Loss (top-K KL, K=100, IS correction)
    │
    ▼
Gradient Step → Weight Update → EMA Update
    │
    ▼
Next Epoch
```

## Hyperparameters

Based on kirby's SDPO config, adjusted for multi-turn ARC:

| Parameter | Value | Source |
|-----------|-------|--------|
| Model | Qwen/Qwen3-8B | kirby |
| Alpha | 1.0 (reverse KL) | kirby sdpo_modal.py |
| Top-K | 100 | verl SDPO default (with tail bucket) |
| EMA rate | 0.01 | kirby |
| Learning rate | 1e-6 | kirby sdpo_modal.py |
| Train batch size | 8 | reduced from 32 (multi-turn is longer) |
| Rollouts per task | 4 | reduced from 8 |
| Max turns | 15 | rlm-explore convention |
| Max tokens per turn | 8192 | |
| Context window | 32768 | |
| IS clip | 2.0 | SDPO config |
| Eval frequency | every 3 epochs | kirby |
| Total epochs | 30 | kirby |
| GPUs | 4x A100-80GB | kirby |

## File Structure

```
domains/arc-agi-2/src/
├── sandbox-server.ts    # NEW: Long-lived process wrapping EvalRuntime + adapter
├── generate-prompt.ts   # NEW: One-shot prompt generator for data prep
├── adapter.ts           # EXISTING: createArcAdapter — scope source of truth
├── grid-helpers.ts      # EXISTING: all grid helper functions
├── prompt.ts            # EXISTING: ARC_PREMISE, ARC_REFERENCE
└── runner.ts            # EXISTING: evaluation harness

rl/
├── arc_interaction.py   # ArcInteraction — verl Interaction implementation
├── js_sandbox.py        # SandboxSession + SandboxPool — subprocess management
├── parser.py            # parse_code_blocks — port of parseReplBlocks
├── arc_data.py          # ARC task loading → verl parquet format
├── sdpo_arc.yaml        # verl/SDPO config for ARC
├── modal_app.py         # Modal deployment (adapted from kirby/sdpo_modal.py)
├── tests/
│   ├── test_js_sandbox.py
│   ├── test_parser.py
│   ├── test_arc_interaction.py
│   └── test_arc_data.py
└── README.md
```

## Why reuse TS, not reimplement

| Concern | Risk if reimplemented | How reuse solves it |
|---------|----------------------|---------------------|
| Scope setup | Someone adds a grid helper to adapter.ts, forgets RL sandbox | sandbox-server.ts imports adapter.ts — automatic |
| Variable persistence | Model writes `const x = ...` in turn 1, uses `x` in turn 2. Naive sandbox loses state | EvalRuntime already handles this with persistent vm.Context |
| Declaration scanning | `let`, `const`, `var`, `function` declarations must survive async IIFE wrapping | EvalRuntime's `scanDeclarations` + `__persist` already handles this |
| Execution semantics | Async IIFE wrapping, implicit return values, SyntaxError retry | EvalRuntime handles all of this (~200 lines of battle-tested code) |
| Feedback format | Model expects "Code executed: ... REPL output: ..." | Match toolResultToUserMessage format exactly |
| System prompt | Training prompt must match eval prompt | Generated from same TS function at data-prep time |

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| EvalRuntime import path from sandbox-server.ts | Use relative imports. Bun resolves TS natively. Test standalone first. |
| Subprocess overhead (many rollouts × many turns) | JSON-line over stdio is fast. One process per rollout, not per eval. Pool manages lifecycle. |
| verl tool_agent_loop doesn't fit ARC code-block format | The state machine is generic. ArcInteraction handles parsing. Worst case: subclass the agent loop. |
| Bun + TS source files on Modal | Install Bun in image, copy TS source dirs. One-time setup. |
| Multi-turn context exceeds 32K | Cap at 15 turns. verl's `max_reprompt_len` handles truncation. |
| Qwen3-8B too weak for ARC | Goal is improvement over zero-shot, not matching Claude. Even 1-2% improvement validates the method. |
| verl API changes between versions | Pin verl version in Modal image. Use same image as kirby. |

## Success Criteria

1. **Zero execution skew**: sandbox-server.ts uses identical EvalRuntime + createArcAdapter as inference runner
2. **ArcInteraction works**: Parses code blocks, delegates to TS subprocess, returns rewards/feedback correctly
3. **Variable persistence**: Multi-turn rollouts maintain state across turns (tested)
4. **verl trains**: SDPO loop completes 30 epochs without divergence on ARC data
5. **Signal quality**: kl_positive_frac > 0.5, adv_abs_mean in reasonable range
6. **Improvement**: pass@1 on eval split improves over zero-shot baseline
7. **Interop**: Trained model can be evaluated via `--provider vllm --format block` in the TS runner
