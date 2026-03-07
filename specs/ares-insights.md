# Insights from ARES (Agentic Research & Evaluation Suite)

Deep dive comparison of [withmartian/ares](https://github.com/withmartian/ares) against our rlm-explore architecture. ARES is an RL-first framework for training and evaluating LLM agents (primarily coding agents). It implements an async version of DeepMind's `dm_env` specification.

**Date:** 2026-03-06
**Branch:** sdpo-arc

---

## Table of Contents

1. [Architecture Comparison](#1-architecture-comparison)
2. [Component-by-Component Mapping](#2-component-by-component-mapping)
3. [The Queue-Mediated Pattern](#3-the-queue-mediated-pattern-ares-core-innovation)
4. [The HTTP Proxy Pattern](#4-the-http-proxy-pattern-ares-proxy)
5. [Protocol-Oriented Design](#5-protocol-oriented-design)
6. [Environment & Episode Semantics](#6-environment--episode-semantics)
7. [Converter Pattern for Multi-API Support](#7-converter-pattern-for-multi-api-support)
8. [Parallel Evaluation](#8-parallel-evaluation)
9. [Testing & Mocking](#9-testing--mocking)
10. [Registry & Preset System](#10-registry--preset-system)
11. [StatTracker & Instrumentation](#11-stattracker--instrumentation)
12. [Resource Cleanup (Janitor Pattern)](#12-resource-cleanup-janitor-pattern)
13. [What We Do Better](#13-what-we-do-better)
14. [Priority Recommendations](#14-priority-recommendations)

---

## 1. Architecture Comparison

### ARES: The RL Gym Abstraction

ARES treats the entire agent interaction as a reinforcement learning environment:

```
Observation = LLMRequest   (what the agent wants to say to the LLM)
Action      = LLMResponse  (what the LLM actually responds)
Reward      = float        (did the task succeed? e.g. 1.0 if tests pass)
```

The central loop looks like a standard gym:

```python
async with ares.make("sbv-mswea") as env:
    ts = await env.reset()          # spins up container, starts agent, blocks until first LLM request
    while not ts.last():
        action = await agent(ts.observation)  # observation IS the LLMRequest
        ts = await env.step(action)           # action IS the LLMResponse
    print(ts.reward)
```

The agent (e.g. MiniSWEAgent) writes normal code — makes LLM calls, parses responses, runs bash commands. It has no idea it's inside an RL loop. The magic is in the **queue-mediated LLM client** that transparently intercepts its LLM calls.

### rlm-explore: Subprocess + Direct LLM

We have a **process boundary** instead of a queue boundary:

```
Python (eval_loop.py)
  └─ spawns Bun subprocess (agent-runner.ts) per task
       └─ creates Orchestrator → Agent makes LLM calls directly to vLLM
       └─ outputs JSON result to stdout
```

The agent talks **directly** to the LLM. Python has no ability to intercept, inspect, or modify LLM calls in-flight. Python is purely a dispatcher that collects results.

### Key Structural Difference

| Aspect | ARES | rlm-explore |
|--------|------|-------------|
| **LLM call control** | Environment intercepts via queue | Agent calls LLM directly |
| **Agent isolation** | Same Python process (async task) | Separate Bun subprocess |
| **Sandbox** | Docker/Daytona containers (bash exec) | In-process V8 sandbox (EvalRuntime) |
| **Reward** | Read from `/reward.txt` in container after tests | Compare submitted transform output to expected grid |
| **Language** | All Python + Go proxy | Python dispatcher + TypeScript agent |

---

## 2. Component-by-Component Mapping

| ARES Component | File(s) | rlm-explore Analog | File(s) | Notes |
|---|---|---|---|---|
| `Environment` protocol (`reset`/`step`) | `environments/base.py`, `code_env.py` | `eval_loop.py` + `agent-runner.ts` | `rl/eval_loop.py`, `domains/arc-agi-2/src/agent-runner.ts` | We don't have a gym-style interface |
| `CodeAgent` protocol | `code_agents/code_agent_base.py` | `Orchestrator` + `Adapter` | `pi-rlm/src/orchestrator.ts`, `pi-rlm/src/domain-adapter.ts` | Our Orchestrator is more opinionated |
| `QueueMediatedLLMClient` | `llms/queue_mediated_client.py` | **No analog** | — | We have no LLM interception layer |
| `Container` protocol | `containers/containers.py`, `docker.py`, `daytona.py` | `EvalRuntime` | `pi-rlm/src/eval-runtime.ts` | They run bash in containers; we run JS in V8 |
| `CodeAgentFactory` | `code_agents/code_agent_base.py` | `createArcAdapter()` | `domains/arc-agi-2/src/adapter.ts` | Their factory injects container+llm; our adapter is simpler |
| `LLMRequest` / `LLMResponse` | `llms/request.py`, `response.py` | pi-agent-core types | `@mariozechner/pi-agent-core` | Theirs is a unified canonical format across APIs |
| `ChatCompletionCompatibleLLMClient` | `llms/chat_completions_compatible.py` | `createCodeBlockStreamFn` | `pi-rlm/src/repl-stream.ts` | Both wrap OpenAI-compatible APIs |
| API Converters | `llms/openai_chat_converter.py`, `anthropic_converter.py` | **No analog** | — | We only support OpenAI format |
| `Registry` / `Presets` | `registry.py`, `presets.py` | CLI args | `agent-runner.ts` CLI flags | They have composable presets; we hardcode |
| `StatTracker` protocol | `experiment_tracking/stat_tracker.py` | `UsageTracker` | `pi-rlm/src/usage-tracker.ts` | Theirs is pluggable (null/logging/tensorboard); ours is fixed |
| `ares-proxy` (Go HTTP proxy) | `ares-proxy/` | **No analog** | — | Enables out-of-process agents via HTTP interception |
| `Janitor` (atexit cleanup) | `environments/code_env.py` | **No analog** | — | We don't manage long-lived containers |
| `MockContainer` / `MockLLMClient` | `testing/mock_container.py`, `mock_llm.py` | **No analog** | — | We lack testing mocks |
| `accounting.py` (cost tracking) | `llms/accounting.py` | `UsageTracker` (tokens only) | `pi-rlm/src/usage-tracker.ts` | Theirs tracks USD cost per model from API |
| `TwentyQuestionsEnvironment` | `environments/twenty_questions.py` | **No analog** | — | Shows the protocol works for non-code domains too |

---

## 3. The Queue-Mediated Pattern (ARES Core Innovation)

This is the single most important pattern in ARES. It enables the RL abstraction by intercepting LLM calls without agents needing to be RL-aware.

### How It Works

```python
# ares/llms/queue_mediated_client.py
@dataclass(frozen=True)
class QueueMediatedLLMClient(LLMClient):
    q: asyncio.Queue[ValueAndFuture[LLMRequest, LLMResponse]]

    async def __call__(self, req: LLMRequest) -> LLMResponse:
        future = asyncio.Future[LLMResponse]()
        await self.q.put(ValueAndFuture(value=req, future=future))
        return await future  # BLOCKS HERE until environment resolves the future
```

```python
# ares/async_utils.py
@dataclass(frozen=True)
class ValueAndFuture[ValType, FutureType]:
    value: ValType                    # The LLM request
    future: asyncio.Future[FutureType]  # Will hold the LLM response
```

### The Full Flow

```
┌──────────────────────────────────────────────────────────────┐
│ CODE AGENT (writes natural code, unaware of RL)              │
│                                                              │
│   response = await llm_client(request)   ← normal call      │
│                                                              │
└────────────────────────┬─────────────────────────────────────┘
                         │ internally...
                         ▼
┌──────────────────────────────────────────────────────────────┐
│ QUEUE-MEDIATED LLM CLIENT                                    │
│                                                              │
│   future = asyncio.Future()                                  │
│   await queue.put(ValueAndFuture(request, future))           │
│   return await future   ← BLOCKS until resolved              │
│                                                              │
└────────────────────────┬─────────────────────────────────────┘
                         │ request appears on queue
                         ▼
┌──────────────────────────────────────────────────────────────┐
│ ENVIRONMENT (owns the RL loop)                               │
│                                                              │
│   request, future = await queue.get()                        │
│   # request is now an OBSERVATION                            │
│   # Environment returns it to the RL trainer                 │
│   #                                                          │
│   # ... RL trainer computes action (calls real LLM, etc) ... │
│   #                                                          │
│   future.set_result(llm_response)  ← UNBLOCKS agent         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│ CODE AGENT (continues naturally with the response)           │
│                                                              │
│   response = await llm_client(request)   ← returns here     │
│   # agent parses response, runs commands, loops...           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Why This Matters for Us

For SDPO/RL training (Phase 2), we need to:

1. **Intercept the agent's LLM requests** to collect token-level logprobs from the policy model
2. **Potentially modify responses** (e.g., for rejection sampling, KL-constrained generation)
3. **Run the same agent trajectory with different LLM backends** (vLLM for training, API for eval)
4. **Replay trajectories** with different temperatures/sampling params

Right now our agent talks directly to vLLM. To switch backends or intercept calls, we'd need to modify `agent-runner.ts` itself. The queue-mediated pattern makes this a configuration change.

### How We Could Adopt It

**Option A: TypeScript queue-mediated pattern.** Wrap our `StreamFn` in a queue-mediated version. The `StreamFn` interface is already close — it's a function `(model, context, options) => AsyncIterable<chunk>`. We could create a `QueueMediatedStreamFn` that yields control to an external loop. However, this requires the controller to be in the same Bun process.

**Option B: HTTP proxy (see next section).** More practical for our subprocess architecture. The agent points at a local proxy instead of vLLM. The proxy captures requests, forwards to vLLM, captures responses, and records everything. This is what ARES's Go proxy does.

**Option C: Hybrid.** Use the HTTP proxy for training (where we need interception) and direct calls for eval (where speed matters).

---

## 4. The HTTP Proxy Pattern (ares-proxy)

ARES includes a Go HTTP proxy (`ares-proxy/`) that enables the queue-mediated pattern **across process boundaries**.

### Architecture

```
Agent (any language/process)
    │
    │ POST /v1/chat/completions  (OpenAI-compatible, BLOCKS)
    ▼
┌─────────────────────────────────┐
│         ares-proxy (Go)         │
│                                 │
│  broker.SubmitRequest():        │
│    1. Generate UUID             │
│    2. Create buffered chan[1]    │
│    3. Add to requestQueue       │
│    4. Wait on chan (blocks)      │
│                                 │
│  Three endpoints:               │
│    POST /v1/chat/completions    │
│    GET  /poll                   │
│    POST /respond                │
└───────┬─────────────┬───────────┘
        │             │
  GET /poll      POST /respond
        │             │
        ▼             │
┌─────────────────────────────────┐
│    Environment / Controller     │
│                                 │
│  1. Poll pending requests       │
│  2. Inspect/modify request      │
│  3. Forward to real LLM         │
│  4. Capture logprobs            │
│  5. Respond with result         │
└─────────────────────────────────┘
```

### Key Implementation Details

**Broker (`broker.go`, 123 lines):**
- `pendingRequests: map[string]chan json.RawMessage` — maps request ID to response channel
- `requestQueue: []PendingRequest` — FIFO queue of requests waiting to be polled
- Buffered channels (capacity 1) prevent blocking on response delivery
- All shared state protected by `sync.Mutex`
- Three termination paths: timeout, context cancellation, normal response
- All paths clean up both the map AND the queue (no memory leaks)

**Exactly-once semantics:**
- `PollRequests()` atomically copies and clears the queue
- Once polled, requests never reappear
- Stale/timed-out requests can't be responded to
- Tests verify this rigorously (8 tests including concurrent requests, timeouts, cancellation)

**Configuration:**
- `PORT` (default 8080)
- `TIMEOUT_MINUTES` (default 15)

### Why This Is Relevant for Us

For our Phase 2 training pipeline, the proxy approach is **the path of least resistance**:

1. **No TypeScript changes needed.** `agent-runner.ts` already points at `--base-url`. Just point it at the proxy instead of vLLM directly.
2. **Full request/response capture.** The proxy sees every LLM call — we can log full trajectories with token-level detail.
3. **Logprob injection.** The proxy can add `logprobs: N` to every request before forwarding to vLLM, even if the agent didn't request them.
4. **Response modification.** For RL training, we could modify responses (e.g., inject thinking tokens, apply KL penalties at the token level).
5. **Language-agnostic.** Works with any agent that speaks OpenAI-compatible HTTP.

### Implementation Sketch for Our Use Case

We don't need the full poll/respond pattern (that's for when the controller is a separate process). For training, a simpler **transparent proxy** would suffice:

```python
# Simplified proxy for training (Python, not Go)
class TrainingProxy:
    """HTTP proxy that sits between agent-runner and vLLM.
    Captures full request/response pairs with logprobs."""

    def __init__(self, vllm_url: str, logprobs: int = 100):
        self.vllm_url = vllm_url
        self.logprobs = logprobs
        self.trajectory: list[dict] = []

    async def handle_completions(self, request):
        body = await request.json()
        # Inject logprobs
        body["logprobs"] = True
        body["top_logprobs"] = self.logprobs
        # Forward to vLLM
        response = await forward_to_vllm(self.vllm_url, body)
        # Capture trajectory
        self.trajectory.append({"request": body, "response": response})
        return response
```

This gives us everything we need for SDPO without touching TypeScript.

---

## 5. Protocol-Oriented Design

### ARES's Approach

Everything in ARES is a `Protocol` (structural interface, not inheritance):

```python
class Environment(Protocol):
    async def reset() -> TimeStep: ...
    async def step(action) -> TimeStep: ...
    async def close() -> None: ...

class CodeAgent(Protocol):
    async def run(task: str) -> None: ...

class Container(Protocol):
    async def start(env) -> None: ...
    async def exec_run(command, workdir, env, timeout_s) -> ExecResult: ...
    async def upload_files(...) -> None: ...
    def stop_and_remove() -> None: ...  # sync for atexit

class LLMClient(Protocol):
    async def __call__(request: LLMRequest) -> LLMResponse: ...

class ContainerFactory(Protocol):
    @classmethod
    def from_image(image, name, resources, default_workdir) -> Container: ...
    @classmethod
    def from_dockerfile(dockerfile_path, ...) -> Container: ...

class CodeAgentFactory[T: CodeAgent](Protocol):
    def __call__(*, container: Container, llm_client: LLMClient) -> T: ...

class StatTracker(Protocol):
    @contextmanager
    def timeit(name: str) -> Generator: ...
    def scalar(name: str, value: float) -> None: ...
```

This enables plug-and-play composition:

```python
# Swap container backend
env = ares.make("sbv-mswea", container_factory=DockerContainer)   # local
env = ares.make("sbv-mswea", container_factory=DaytonaContainer)  # cloud
env = ares.make("sbv-mswea", container_factory=MockContainer)     # test

# Swap agent
ares.register_preset("sbv-custom", HarborSpec(code_agent_factory=MyAgent))

# Swap tracker
env = ares.make("sbv-mswea", tracker=TensorboardStatTracker(writer))
env = ares.make("sbv-mswea", tracker=NullStatTracker())
```

### Our Current State

Our `Adapter` interface is the closest analog:

```typescript
interface Adapter {
    scope: Record<string, unknown>;      // sandbox variables
    reference: string;                    // domain reference doc
    onEvalResult?: () => string | undefined;
    defaultSubAgentPrompt?: string;
}
```

But the Orchestrator hardwires:
- The eval runtime (always `EvalRuntime`, always JS)
- The stream function (always OpenAI-compatible)
- The model (always a `Model` from pi-ai)
- The agent loop (always pi-agent-core `Agent`)

### Takeaway

Not urgent — we're single-domain (ARC) right now. But if we add SWE-bench or other domains, extracting `Container`, `Sandbox`, and `LLMBackend` protocols would help. The factory pattern (`CodeAgentFactory`) is particularly useful — it lets you compose "which agent + which container + which LLM" as a configuration, not code changes.

---

## 6. Environment & Episode Semantics

### ARES's TimeStep

```python
class TimeStep[ObservationType, RewardType, DiscountType](NamedTuple):
    step_type: StepType    # "FIRST" | "MID" | "LAST"
    reward: RewardType | None
    discount: DiscountType | None
    observation: ObservationType
```

**Semantics:**
- `step_type = "FIRST"`: Initial timestep from `reset()`. reward=None, discount=None.
- `step_type = "MID"`: Intermediate. reward=0.0 (no step reward), discount=1.0 (continue).
- `step_type = "LAST"`: Terminal. Two sub-cases:
  - `discount = 0.0`: True terminal (agent submitted/completed). Should NOT bootstrap.
  - `discount = 1.0`: Truncation (step limit hit). SHOULD bootstrap value estimate.

This distinction between terminal and truncation is **critical for RL training** — it tells the value function whether to estimate future returns or not.

### Our Current State

```typescript
// agent-runner.ts output
{
    taskId: string,
    reward: 0.0 | 1.0,
    submitted: boolean,
    correct: boolean,
    turns: number,
    tokens: number,
    timeMs: number,
    trajectory: [...],
    attempts: [...]
}
```

We have no `discount` or `step_type`. We can't distinguish:
- Agent submitted a wrong answer (terminal, discount=0)
- Agent hit maxTurns without submitting (truncation, discount=1)
- Agent timed out (truncation, discount=1)
- Agent errored (terminal, discount=0)

### Recommendation

Add a `termination_reason` field:

```typescript
const result = {
    // ... existing fields ...
    terminationReason: "submitted" | "max_turns" | "timeout" | "error",
    // For RL: discount = (terminationReason === "submitted" || terminationReason === "error") ? 0.0 : 1.0
};
```

This is a tiny change with significant RL implications.

### ARES's Reward Computation

ARES computes reward by:
1. Uploading test scripts to the container
2. Running bash tests inside the container
3. Reading `/reward.txt` (plain float) or `/reward.json` (single-key dict → float) from the container

Our approach is more direct: the submitted `transform` function is applied to the test input and compared to the expected output. This is simpler and faster (no container filesystem I/O).

---

## 7. Converter Pattern for Multi-API Support

### ARES's Approach

ARES defines a canonical `LLMRequest` format and provides bidirectional converters for three API formats:

```python
# Canonical format
@dataclass(frozen=True)
class LLMRequest:
    messages: list[Message]
    system_prompt: str | None
    max_output_tokens: int | None
    temperature: float | None      # stored in OpenAI range [0, 2]
    top_p: float | None
    tools: list[Tool] | None
    tool_choice: str | dict | None
    stop_sequences: list[str] | None
    top_k: int | None              # Claude-only
    service_tier: str | None
    metadata: dict | None
    stream: bool

# Converters
openai_chat_converter.to_external(request)      # → OpenAI Chat Completions dict
openai_chat_converter.from_external(kwargs)      # ← OpenAI Chat Completions dict
anthropic_converter.to_external(request)         # → Anthropic Messages dict
anthropic_converter.from_external(kwargs)        # ← Anthropic Messages dict
openai_responses_converter.to_external(request)  # → OpenAI Responses dict
openai_responses_converter.from_external(kwargs) # ← OpenAI Responses dict
```

Each converter handles API-specific quirks:
- **Temperature range:** OpenAI [0, 2] vs Claude [0, 1] — automatic conversion (`claude_temp = openai_temp / 2.0`)
- **Message role alternation:** Claude requires strict user/assistant alternation; converter enforces this
- **Tool format:** Different schema shapes across APIs
- **Stop sequences:** OpenAI limits to 4; converter truncates with warning
- **`strict` parameter:** Controls whether information loss raises errors or warnings

### Our Current State

We're locked to OpenAI-compatible APIs via `createCodeBlockStreamFn` in pi-rlm. The `StreamFn` interface abstracts the call, but the payload format is always OpenAI chat completions.

### Takeaway

Low priority for now — vLLM speaks OpenAI format, which covers our training use case. But if we want to eval against Claude or other APIs, we'd need converters. ARES's approach of a canonical format with bidirectional converters is clean and well-tested.

---

## 8. Parallel Evaluation

### ARES's Approach

Example 03 (`03_parallel_eval_with_api.py`) shows the pattern:

```python
@dataclass(frozen=True)
class Args:
    num_parallel_workers: int = 20

async def main():
    num_tasks = ares.info(args.preset_name).num_tasks
    sem = asyncio.Semaphore(args.num_parallel_workers)

    async def evaluate_with_semaphore(task_idx):
        async with sem:
            return await evaluate_single_task(task_idx)

    tasks = [evaluate_with_semaphore(i) for i in range(num_tasks)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Fault tolerance: exceptions in individual tasks don't crash the whole run
    total_successes = sum(
        r.reward for r in results
        if isinstance(r, ares.TimeStep) and r.reward is not None
    )
```

Key patterns:
- **Semaphore** limits concurrent containers/GPU memory
- **`asyncio.gather(..., return_exceptions=True)`** for fault tolerance
- Each task creates its own environment instance (independent lifecycle)
- `EvaluationDashboard` provides real-time TUI with task status, stats, logs

### Our Current State

`eval_loop.py` runs tasks **sequentially**:

```python
for i, item in enumerate(tasks):
    result = _run_agent(task_id=..., task=..., ...)
    results.append(result)
```

On Modal with a single A100 GPU running vLLM, we can only generate one completion at a time anyway... but vLLM supports batched inference. If we send multiple requests concurrently, vLLM batches them internally for higher throughput.

### Recommendation: Add Parallel Dispatch

```python
import asyncio
import subprocess
from concurrent.futures import ThreadPoolExecutor

async def evaluate_tasks_parallel(tasks, max_workers=4, **kwargs):
    sem = asyncio.Semaphore(max_workers)

    async def run_one(item):
        async with sem:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                executor, _run_agent, item["id"], item["task"], **kwargs
            )

    results = await asyncio.gather(
        *[run_one(item) for item in tasks],
        return_exceptions=True
    )
    return results
```

**Expected impact:** 3-5x throughput improvement. vLLM's continuous batching means concurrent requests share GPU compute efficiently. Each subprocess is mostly waiting on LLM generation, so the Bun process overhead is minimal.

**Important:** The `_run_agent` subprocess model is already well-suited for parallelism — each task is fully independent (separate Bun process, separate Orchestrator instance). We just need to dispatch them concurrently instead of sequentially.

---

## 9. Testing & Mocking

### ARES's Approach

ARES provides mock implementations that conform to the same protocols:

```python
# testing/mock_container.py
@dataclass
class MockContainer(Container):
    exec_commands: list[str] = field(default_factory=list)
    exec_responses: dict[str, ExecResult] = field(default_factory=dict)
    exec_handler: Callable[[str], ExecResult] | None = None

    async def exec_run(self, command, **kwargs):
        self.exec_commands.append(command)
        if self.exec_handler:
            return self.exec_handler(command)
        if command in self.exec_responses:
            return self.exec_responses[command]
        return ExecResult(output="", exit_code=0)

# testing/mock_llm.py
@dataclass
class MockLLMClient:
    responses: list[str] = field(default_factory=list)
    response_handler: Callable[[LLMRequest], str] | None = None
    call_count: int = 0

    async def __call__(self, req: LLMRequest) -> LLMResponse:
        self.call_count += 1
        # Response priority: handler > cycle through responses > default
        ...
```

Tests are colocated with source (`*_test.py` naming) and use these mocks:

```python
# Can test agent behavior without Docker or LLM API
container = MockContainer(exec_responses={"pwd": ExecResult("/testbed", 0)})
llm = MockLLMClient(responses=["```bash\nls\n```", "```bash\nsubmit\n```"])
agent = MiniSWECodeAgent(container=container, llm_client=llm)
await agent.run("Fix the bug in main.py")
assert llm.call_count == 2
assert "ls" in container.exec_commands
```

### Our Current State

We have no mocks. Testing the Orchestrator requires a running vLLM server. The test files in `pi-rlm/test/` exist but are limited by this constraint.

### Recommendation

Create a `MockStreamFn` that returns canned responses:

```typescript
function createMockStreamFn(responses: string[]): StreamFn {
    let callIndex = 0;
    return async function* (model, context, options) {
        const response = responses[callIndex % responses.length];
        callIndex++;
        yield { type: "text", text: response };
    };
}

// Usage in tests:
const streamFn = createMockStreamFn([
    "```javascript\nfunction transform(grid) { return grid; }\n```",
    "```javascript\nsubmit(transform)\n```",
]);
const orchestrator = new Orchestrator({ model, adapter, streamFn });
const result = await orchestrator.run("Solve the task");
```

This would enable fast, deterministic testing of the Orchestrator and Adapter without any external dependencies.

---

## 10. Registry & Preset System

### ARES's Approach

A composable registry with selector syntax:

```python
# Preset registration
ares.register_preset("sbv-mswea", HarborSpec(
    ds_spec=swe_bench_verified_spec,
    code_agent_factory=MiniSWECodeAgent,
))
ares.register_preset("sbv-terminus2", HarborSpec(
    ds_spec=swe_bench_verified_spec,
    code_agent_factory=Terminus2Agent,
))

# Task selectors
env = ares.make("sbv-mswea:0")       # single task at index 0
env = ares.make("sbv-mswea:0:10")    # slice: tasks 0-9
env = ares.make("sbv-mswea:5:")      # slice: tasks 5 to end
env = ares.make("sbv-mswea@2/8")     # shard 2 of 8 (for distributed eval)

# Introspection
ares.info()                           # list all presets
ares.info("sbv-mswea")               # → EnvironmentInfo(name, description, num_tasks)
ares.list_presets()                   # human-readable listing
```

The `ShardSelector` is particularly clever for distributed evaluation:

```python
class ShardSelector:
    def select(self, tasks):
        start = round(self.shard_index * len(tasks) / self.total_shards)
        end = round((self.shard_index + 1) * len(tasks) / self.total_shards)
        return tasks[start:end]
```

### Our Current State

We use CLI args:

```bash
modal run rl/modal_app.py::evaluate --model Qwen/Qwen3-8B --dataset ARC-AGI-1 --count 5
```

Task selection is via `--count` (take first N) or nothing (all tasks).

### Recommendation

Adopt the selector syntax for our eval pipeline:

```bash
# Current
--dataset ARC-AGI-1 --count 5

# Proposed
--tasks "ARC-AGI-1:0:5"         # first 5 tasks
--tasks "ARC-AGI-1:42"          # single task
--tasks "ARC-AGI-1@0/4"         # shard 0 of 4 (for multi-GPU distributed eval)
```

This is a small parsing change with good ergonomics, especially when we scale to multi-node evaluation.

---

## 11. StatTracker & Instrumentation

### ARES's Approach

A `StatTracker` protocol with pluggable implementations:

```python
class StatTracker(Protocol):
    @contextmanager
    def timeit(self, name: str) -> Generator: ...
    def scalar(self, name: str, value: float) -> None: ...

# Implementations:
class NullStatTracker:     # no-op (default, zero overhead)
class LoggingStatTracker:  # logs percentiles every 60s
class TensorboardStatTracker:  # writes histograms to TensorBoard
```

Usage is non-intrusive:

```python
with tracker.timeit("llm_call"):
    response = await llm(request)

tracker.scalar("reward", reward)
tracker.scalar("episode_length", step_count)
```

Both `LoggingStatTracker` and `TensorboardStatTracker` use the same accumulate-then-flush pattern:
- Accumulate values in `defaultdict[str, list[float]]` during the period
- Every 60 seconds, compute percentiles (logging) or histograms (tensorboard)
- Clear for next period

### Our Current State

We have `UsageTracker` that tracks token counts per agent. It's not pluggable, not instrumented for timing, and doesn't integrate with any visualization backend.

We also have ad-hoc `Date.now()` timing in `agent-runner.ts`.

### Recommendation

If we adopt parallel evaluation, a stat tracker becomes more valuable — we'll want to see distributions of latencies, token counts, and rewards across tasks. The protocol pattern (null by default, opt into logging/tensorboard) is clean and low-cost to implement.

---

## 12. Resource Cleanup (Janitor Pattern)

### ARES's Approach

A global singleton `_Janitor` that ensures container cleanup even on abnormal termination:

```python
class _Janitor:
    def __init__(self):
        self._environment_by_id: dict[int, CodeEnvironment] = {}
        atexit.register(self._sync_cleanup)

    def register(self, env: CodeEnvironment):
        self._environment_by_id[id(env)] = env

    def _sync_cleanup(self):
        """Called by atexit — must be synchronous."""
        for key in list(self._environment_by_id.keys()):
            env = self._environment_by_id[key]
            container = getattr(env, "_container", None)
            if container is not None:
                container.stop_and_remove()  # sync method
            del self._environment_by_id[key]

_ENVIRONMENT_JANITOR = _Janitor()  # global singleton
```

Key details:
- `Container.stop_and_remove()` is **synchronous** (atexit handlers can't be async)
- Uses `getattr` for safe access (env might be partially initialized)
- Environments register on `__init__` and `__aenter__`, unregister on `__aexit__`
- Iterates a copy of keys (modifying dict during iteration is unsafe)

### Our Situation

We don't manage containers — our Bun subprocesses are ephemeral and die when the parent Python process dies. But if we ever run long-lived containers (e.g., for SWE-bench style eval with persistent Docker containers), this pattern would be essential.

For now, the takeaway is: **always have a sync cleanup path for resources that outlive the process.**

---

## 13. What We Do Better

### 13.1 In-Process JS Sandbox

Our `EvalRuntime` runs code in the same V8 isolate as the agent. This is **10-100x faster** than Docker exec per command. For ARC (many small code evaluations per episode), this is a massive advantage.

ARES runs bash commands via `container.exec_run()` which involves:
1. Docker/Daytona API call
2. Shell startup
3. Command execution
4. Output capture and transfer

Our EvalRuntime does:
1. `eval()` in the same V8 context

### 13.2 Submit-Gated Reward

Our `submit()` function returns the actual transform function as a JS closure. We score it by applying it to the test input and comparing to the expected output. This is more precise than reading a file from a container filesystem.

### 13.3 Streaming with Logprobs

We already support `--top-logprobs` for collecting token-level data during eval. ARES's `LLMResponse` only captures `Usage` (token counts) and `TextData` (content) — no logprobs. For RL training, logprobs at the token level are essential.

### 13.4 Multi-Attempt with Early Exit

Our `--num-attempts` with `if (correct) break` is a simple pass@k pattern:

```typescript
for (let a = 0; a < numAttempts; a++) {
    // ... run attempt ...
    if (correct) break;  // no need for more attempts
}
```

ARES doesn't have this built in — each environment instance is a single episode.

### 13.5 Session Resumption

Our Orchestrator supports `resume()` to continue from an aborted session by restoring conversation state from a prior agent log file. ARES has no resumption mechanism.

---

## 14. Priority Recommendations

| Priority | Action | Effort | Impact | Details |
|----------|--------|--------|--------|---------|
| **1** | Parallel task dispatch in `eval_loop.py` | Small | 3-5x eval throughput | Use `ThreadPoolExecutor` or `asyncio` to run multiple `_run_agent()` calls concurrently. vLLM batches concurrent requests automatically. See [Section 8](#8-parallel-evaluation). |
| **2** | Add `terminationReason` to result JSON | Tiny | Correct RL semantics | Distinguish submitted/max_turns/timeout/error. Needed for proper discount calculation in RL training. See [Section 6](#6-environment--episode-semantics). |
| **3** | Build HTTP proxy for training | Medium | Enables SDPO pipeline | Sit between agent-runner and vLLM. Capture full request/response pairs, inject logprobs, record trajectories. No TypeScript changes needed. See [Section 4](#4-the-http-proxy-pattern-ares-proxy). |
| **4** | Gate submit on test accuracy | Small | Fix "submit without testing" | Make `submit()` reject with "Test your function first" if no training examples have been evaluated. Addresses the turns=2 issue from eval runs. |
| **5** | Create `MockStreamFn` for testing | Small | Faster dev iteration | Enable deterministic Orchestrator testing without vLLM. See [Section 9](#9-testing--mocking). |
| **6** | Task selector syntax | Small | Cleaner CLI ergonomics | `--tasks "ARC-AGI-1:0:5"` instead of `--dataset ARC-AGI-1 --count 5`. Supports index, slice, shard. See [Section 10](#10-registry--preset-system). |
| **7** | Pluggable StatTracker | Small | Better observability | Null by default, opt into logging/tensorboard. Useful when running parallel eval at scale. See [Section 11](#11-stattracker--instrumentation). |

### Strategic Summary

ARES was designed for RL training from day one (observations/actions/rewards). We designed for eval. The gap is **LLM call interception** — to do RL training (Phase 2), we need some mechanism to capture and control the agent's LLM interactions.

The **HTTP proxy approach** (Priority 3) is the path of least resistance:
- Requires zero TypeScript changes
- `agent-runner.ts` already accepts `--base-url` — just point it at the proxy
- Proxy captures everything needed for SDPO (full trajectories, logprobs, token-level data)
- Can be incrementally adopted (use direct calls for eval, proxy for training)

The queue-mediated pattern is more elegant but would require significant refactoring of our subprocess architecture. The proxy gives us 90% of the benefit for 10% of the effort.
