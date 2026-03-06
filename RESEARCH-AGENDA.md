# Test-Time Parallel Evolutionary Search with Small Language Models

**Goal**: Make 8B models competitive with frontier models on ARC-AGI through RL training (SDPO), test-time evolutionary search, and meta-optimization.

**Core thesis**: The large model advantage is per-sample quality. The small model advantage is you can run 500 of them for the same cost. We attack both sides: SDPO training improves per-sample quality, evolution provides test-time search. Combined, a trained 8B + massive parallel search can match frontier models at 50-100x lower cost.

**Hardware trend alignment**: Inference is getting cheaper and more parallelizable. Single-sample quality improvements are plateauing. We want to be on the right side of that curve.


## Background & Prior Work

### Darwinian Evolver (Imbue)
- Evolutionary framework: population of solutions, sigmoid + novelty parent selection, LLM-powered mutation
- ARC-AGI-2 results with frontier models (Opus 4.6, Gemini 3 Pro, GPT-5.2)
- Key property: tolerant of noisy mutators — works even at 20% mutation success rate
- Repo: `downloads/darwinian_evolver/`
- Blog: https://imbue.com/research/2026-02-27-arc-agi-2-evolution/

### Recursive Models for Long-Horizon Reasoning (Yang et al. 2026)
- arxiv:2603.02112
- Recursion (call/return) exponentially reduces required context vs standard autoregressive models
- A fine-tuned Qwen 2.5-3B with recursion beats GPT-4o, LLaMA 3.3-70B, Qwen3-235B on SAT
- Multi-agent sophistication adds zero computational power over simple call/return
- Implication: don't scale the model, scale the recursion depth

### SOAR (Self-improving Operators for Automated program Refinements)
- Evolutionary program synthesis + hindsight self-distillation
- 14B model hits 42.75% on ARC-AGI, beating GPT-4.1 one-shot
- Key insight: failed programs reframed as correct programs for synthetic tasks — densifies training signal
- https://julienp.netlify.app/posts/soar/

### SDPO / Self-Distillation
- Information asymmetry: self-teacher (sees answer) vs self-student (doesn't)
- Per-token distribution signal across full vocabulary (not just scalar reward per trajectory)
- Addresses the sparse reward problem in RL (see Dwarkesh/bits-per-sample)
- Papers: SDFT (arxiv:2601.19897), SDPO (self-distillation.github.io/SDPO)

### Absolute Zero
- Self-play RL with zero human-curated data
- Model proposes tasks optimized for learnability + learns to solve them
- SOTA on math and coding without gold labels
- arxiv:2505.03335

### Multi-Turn SDPO (verl)
- Self-distillation policy optimization with top-K KL divergence (K=100, tail bucket)
- EMA teacher, importance sampling correction, GRPO advantage estimation
- Multi-turn rollout engine with tool/code execution environments
- Already working on LiveCodeBench via kirby; adapting to ARC via `PROMPT-sdpo.md`
- Infrastructure: verl framework, Modal (4xA100-80GB), vLLM for rollouts
- Papers: SDPO (self-distillation.github.io/SDPO)


## Two-Phase Strategy: Train Then Search

The research has two complementary pillars that compound:

```
Pillar 1: TRAIN — Multi-Turn SDPO (improves per-sample quality)
  Train Qwen3-8B to solve ARC via self-distillation RL
  Model learns from its own rollouts with dense per-token signal
  Infrastructure: verl + JS sandbox + ArcInteraction + Modal
  Outcome: a competent 8B ARC solver (target: 15-30% on AGI-1)

Pillar 2: SEARCH — Test-Time Evolutionary Search (amplifies via parallel search)
  Plug the SDPO-trained 8B into the darwinian evolver as mutator
  Trade per-mutation quality for massive search volume
  Evolution selects from the trained model's improved solution space
  Outcome: 8B + search matches frontier models (target: 40-55% on AGI-1)
```

**Why this order matters**: Evolution amplifies the base model's capability. A raw 8B + evolution might reach 10-15%. An SDPO-trained 8B + evolution could reach 40-55%. The SDPO investment makes every evolution dollar go further.

**The layers still apply**, but the stack shifts:

```
Layer 3: Meta-Search
  Evolve the harness itself — prompts, affordances, hyperparameters
  Fitness = Layer 1 convergence speed / solve rate across task suite
  Runs: rarely (expensive), high leverage

Layer 2: SDPO Training  [WAS: SDFT on Opus traces]
  Multi-turn SDPO via verl — model learns from own rollouts
  Dense per-token signal via top-K KL divergence
  Runs: offline training rounds on Modal (4xA100-80GB)

Layer 1: Search (Test-Time Evolutionary Search)
  SDPO-trained 8B + darwinian evolver, massively parallel, per-task
  Trade per-mutation quality for search volume
  Runs: at test time, per task
```

Build bottom-up: SDPO training first (Pillar 1), then evaluate, then add evolution (Pillar 2). Each layer makes the layer below it better.

```
Layer 3: Meta-Search
  Evolve the harness itself — prompts, affordances, hyperparameters
  Fitness = Layer 1 convergence speed / solve rate across task suite
  Runs: rarely (expensive), high leverage

Layer 2: Learn (RL / Distillation)
  Train the 8B to be a better mutator over time
  SDFT on Opus mutation traces -> SFT on own successful mutations -> SDPO
  Runs: periodically, between evaluation rounds

Layer 1: Search (Test-Time Evolutionary Search)
  8B model + darwinian evolver, massively parallel, per-task
  Trade per-mutation quality for search volume (20 parents x 100 iterations)
  Runs: at test time, per task
```

Each layer is the training signal for the layer above it. Each layer makes the layer below it cheaper or better. The stack compounds.

Build and validate bottom-up. But before all three layers: establish the 8B's raw capability (Phase -1). Then Layer 1 works or it doesn't. Layer 2 is "collect winners, fine-tune, measure." Layer 3 only matters once 1 and 2 are working.


## Evolutionary Search Roles (from Imbue)

The evolver uses 4 distinct LLM roles at 3 capability tiers, plus 3 algorithmic roles:

| Role | Thinking | Description | 8B feasibility |
|---|---|---|---|
| Mutator | HIGH | Read failure diffs, propose code fix | Core target for SDFT |
| Crossover | HIGH | Synthesize insights from multiple parents | Harder, lower priority |
| Transfer Scorer | MEDIUM | Judge if solution generalizes to test inputs | Drop or replace with heuristic |
| Simplicity Scorer | LOW | Count branches, literals, hardcoded colors | Replace with AST analysis |
| Code Executor | none | Run transform(), check output | Already algorithmic |
| Parent Selector | none | Sigmoid + novelty weighted sampling | Already algorithmic |
| Verification | none | Check mutation changed output | Already algorithmic |

**Optimization strategy**: Eliminate 2 of 4 LLM roles (simplicity -> heuristic, transfer -> drop). Focus SDFT budget on the Mutator role. Crossover is stretch goal.


## Research Plan

### Phase -2: Multi-Turn SDPO Training (2-3 weeks)

**Goal**: Train Qwen3-8B to solve ARC tasks via multi-turn SDPO using verl on Modal.

**This is the foundational investment.** Everything downstream — evaluation, evolution, meta-search — is amplified by a better base model. A raw 8B might score 5% on ARC-AGI-1. An SDPO-trained 8B might score 20-30%. That difference is the difference between evolution having nothing to work with vs having a strong base to amplify.

**Full spec**: `PROMPT-sdpo.md` — this is the active build.

**What we're building**:
1. **JS Sandbox** (`rl/js_sandbox.py`) — Bun subprocess that executes ```js blocks with grid helpers
2. **Code-Block Parser** (`rl/parser.py`) — port of `parseReplBlocks` from TypeScript
3. **ARC System Prompt** (`rl/system_prompt.py`) — ported from TypeScript ARC_PREMISE
4. **ArcInteraction** (`rl/arc_interaction.py`) — bridges verl's rollout engine to ARC environment
5. **ARC Data Loader** (`rl/arc_data.py`) — ARC tasks to verl parquet format
6. **verl Config** (`rl/sdpo_arc.yaml`) — SDPO hyperparameters for ARC
7. **Modal Deployment** (`rl/modal_app.py`) — 4xA100-80GB training runs

**What verl already provides (DO NOT REBUILD)**:
- Multi-turn rollout engine (state machine: GENERATING → PROCESSING_TOOLS → INTERACTING)
- SDPO loss (top-K KL divergence, K=100, tail bucket, EMA teacher, IS correction)
- GRPO advantage estimation
- vLLM integration (rollouts + weight sync)
- Checkpointing, WandB logging

**Training recipe**:
- Model: Qwen3-8B
- Data: ARC-AGI-2 training split (1000 tasks)
- Rollouts: 4 per task, multi-turn (model writes code, gets execution feedback, iterates)
- Reward: 1.0 if submitted answer matches ground truth, 0.0 otherwise
- Self-distillation: top-K KL divergence between EMA teacher and current policy
- Infrastructure: Modal, 4xA100-80GB, ~30 epochs

**Success criteria**:
- Training completes without divergence (loss finite, kl_positive_frac in [0.2, 0.8])
- pass@1 on eval split improves over zero-shot baseline (any improvement counts)
- Trained model evaluable via existing TypeScript runner (`--provider vllm --format block`)

**Reusable infrastructure for later phases**:
- JS sandbox → reusable for Phase -1 evaluation and evolver code execution
- Code-block parser → reusable everywhere
- ARC data loader → reusable for evolver experiments
- Modal vLLM serving → reusable for Phase -1 scale runs


### Phase -1: Capability Frontier — What can the 8B do on ARC? (2-3 days)

**Question**: What is the unaided capability of the 8B on ARC tasks — both the raw base model AND the SDPO-trained model? Where does each break? How much did SDPO help?

**Why this comes after SDPO**: We evaluate two models back-to-back — raw Qwen3-8B and SDPO-trained Qwen3-8B — on the same tasks. The delta is the SDPO training signal. The absolute numbers tell us what evolution has to work with.

**Infrastructure — already built**:
- pi-rlm supports `--provider vllm --base-url http://localhost:8000/v1`
- `--format block` uses ```js code blocks instead of tool calling (critical — 8B models can't do tool use reliably)
- `--thinking` supports Qwen-style thinking via `thinkingFormat: "qwen"` in `createVllmModel()`
- ARC-AGI-1 data at `downloads/ARC-AGI-1/data/` — 400 training + 400 evaluation tasks, same JSON format as ARC-AGI-2
- `--data-dir` flag on the runner switches between ARC-AGI-1 and ARC-AGI-2
- Task loader's `selectDevSet()` filters to small-grid tasks (max dim < 10) for fast iteration

**Experiment matrix**:

| Variable | Values | Why |
|---|---|---|
| Model | Qwen3-8B (raw), Qwen3-8B (SDPO-trained) | A/B comparison; measures SDPO lift |
| Format | `block` (primary), `tool` (test if works) | 8B can't do tool calling — block is the real path |
| Thinking | `off`, `medium`, `high` | Qwen thinking support via vLLM; does it help? |
| Sub-agents | `--max-agents 1` (solo), `--max-agents 5` | Does delegation help or hurt at 8B? |
| Dataset | ARC-AGI-1 training (small grids first) | Easier than AGI-2; 400 tasks; answers available |
| Prompt | Full ARC_PREMISE, simplified version | 183-line Opus prompt may overwhelm 8B |

**Staged execution**:

**Stage A: Smoke test — raw vs trained (2 hours)**
```bash
# Serve raw model
vllm serve Qwen/Qwen3-8B --port 8000

# Run 5 small-grid tasks, solo agent
bun domains/arc-agi-2/src/runner.ts \
  --data-dir downloads/ARC-AGI-1/data \
  --provider vllm --model Qwen3-8B \
  --format block --thinking off \
  --max-agents 1 --count 5 --split training \
  --stream --name qwen3-8b-raw-smoke

# Swap to SDPO-trained checkpoint, same 5 tasks
vllm serve <sdpo-checkpoint-path> --port 8000

bun domains/arc-agi-2/src/runner.ts \
  --data-dir downloads/ARC-AGI-1/data \
  --provider vllm --model qwen3-8b-sdpo \
  --format block --thinking off \
  --max-agents 1 --count 5 --split training \
  --stream --name qwen3-8b-sdpo-smoke
```
Answers: Does each model produce valid JS? Call `submit()`? Use grid helpers? Iterate on failures? How does trained compare to raw?

**Stage B: Thinking ablation (half day)**
Run 20 small-grid ARC-AGI-1 training tasks at each thinking level (`off`, `medium`, `high`), solo agent. Compare:
- Code validity rate (does `transform()` compile?)
- Partial solve rate (soft_accuracy > 0 on any training example)
- Full solve rate (all training examples correct + test correct)

**Stage C: Sub-agent ablation (half day)**
Same 20 tasks, best thinking level from Stage B. Compare `--max-agents 1` vs `--max-agents 5`:
- Does the 8B generate coherent sub-agent tasks?
- Do sub-agents return useful results?
- Or does delegation just waste tokens?

**Stage D: Scale run — both models (1 day)**
Best configuration from A-C. Run BOTH raw and SDPO-trained on full ARC-AGI-1 training split (400 tasks), pass@2:
```bash
# Raw model
bun domains/arc-agi-2/src/runner.ts \
  --data-dir downloads/ARC-AGI-1/data \
  --provider vllm --model Qwen3-8B \
  --format block --thinking <best> \
  --max-agents <best> --split training --all \
  --num-attempts 2 --name qwen3-8b-raw-agi1-training

# SDPO-trained model (same config)
bun domains/arc-agi-2/src/runner.ts \
  --data-dir downloads/ARC-AGI-1/data \
  --provider vllm --model qwen3-8b-sdpo \
  --format block --thinking <best> \
  --max-agents <best> --split training --all \
  --num-attempts 2 --name qwen3-8b-sdpo-agi1-training
```

The comparison between these two runs is the key deliverable of Phase -1.

**Stage E: Prompt simplification (if needed)**
If Stage D solve rate < 5%, the full ARC_PREMISE is likely overwhelming the 8B. Create a stripped-down prompt:
- Remove sub-agent examples and delegation guidance
- Shorten analysis instructions
- Keep: grid helper reference, scoring functions, submit() instructions
- Re-run Stage D with simplified prompt

**Key metrics**:

| Metric | What it tells us |
|---|---|
| Code validity rate | Can the 8B write JS at all in this harness? |
| Partial solve rate (soft_accuracy > 0.5) | Can it get the right general shape? |
| Full solve rate (pass@1) | Raw single-attempt capability |
| Full solve rate (pass@2) | Does a second attempt help? (diversity signal) |
| Avg turns per task | Does it iterate or give up? |
| Token usage per task | Throughput planning for later phases |
| Failure mode distribution | Where to focus SDFT / evolution |

**Failure mode taxonomy** (classify every non-solved task):
1. **Code invalid** — JS doesn't parse or `transform()` not defined
2. **Runtime error** — code runs but throws (TypeError, undefined, etc.)
3. **Wrong shape** — output grid has wrong dimensions
4. **Wrong content** — right shape, wrong values (partial credit via soft_accuracy)
5. **Didn't submit** — model never called `submit()` (ran out of turns or got stuck)
6. **Hallucinated success** — model says "correct!" but answer is wrong

The distribution across these categories tells us exactly what to target:
- Mostly (1-2)? → Code generation is the bottleneck. SDFT on code repair.
- Mostly (3-4)? → Hypothesis formation is weak but code-gen works. Evolution can search over hypotheses.
- Mostly (5)? → Harness/prompt issue. Model doesn't understand the protocol.
- Mostly (6)? → Model can't self-evaluate. Evolver's external evaluation fixes this for free.

**Comparison points**:
- Opus on ARC-AGI-2 eval: 65.8% (79/120) — our existing result
- SOAR 14B on ARC-AGI-1: 42.75% — best published small-model result
- Published 8B baselines on ARC-AGI-1 (if any — search literature)

**Go/no-go for subsequent phases** (using SDPO-trained model numbers):
- **> 20% solve rate**: Excellent. SDPO worked. Evolution can amplify to 40%+.
- **10-20% solve rate**: Good. SDPO helped. Evolution + mutation-specific tuning (Phase 2) needed.
- **< 10% solve rate (but > raw baseline)**: SDPO signal exists but weak. Check if more training epochs help. May need Opus trace distillation (Phase 2) to supplement.
- **= raw baseline (no SDPO lift)**: SDPO training failed. Debug: reward signal too sparse? ArcInteraction bug? Wrong hyperparameters? Don't proceed to evolution until this is fixed.
- **0% for raw model, code-invalid**: The 8B can't write valid JS at all in this harness. Consider Python instead of JS, or move to 14B.

**Deliverables**:
- `runs/qwen7b-agi1-training/run.json` — full results with per-task accuracy
- Failure mode breakdown (counts per category)
- Best configuration (model, thinking, agents, prompt)
- Session traces for successful solves (`runs/<name>/sessions/`) — these become early SFT candidates
- Decision: proceed to Phase 0 (evolver) or iterate on prompt/harness first


### Phase 0: Baseline — Can an 8B mutate at all? (1-2 days)

**Question**: What is the baseline mutation success rate of an off-the-shelf 8B on ARC tasks using the evolver?

**Method**:
1. Serve Qwen2.5-Coder-7B-Instruct via vLLM
2. Point the existing evolver at it via OpenAI-compatible endpoint
3. Run on 10-20 ARC tasks, 8-16 iterations each
4. Measure: mutation success rate, solve rate, cost per task

**Key metrics**:
- Mutation success rate (does the code compile and change output?)
- Score improvement rate (does the mutation improve soft_score?)
- Solve rate vs Opus baseline on the same tasks

**Go/no-go**: With the SDPO-trained model, we expect mutation success rate >15%. If <5%, the SDPO training didn't transfer to the mutation task — go to Phase 2 (mutation-specific tuning). If >25%, skip Phase 2 entirely and go to Phase 3 (scale search).

**Models to test**:
- Qwen3-8B SDPO-trained (primary — from Phase -2)
- Qwen3-8B raw (control — measures SDPO lift for the mutation task specifically)

**Prompt variants to test**:
- Original Imbue prompt (as control)
- Simplified prompt (shorter, more direct, no "natural language instructions" requirement)
- Minimal prompt (just problem + code + diffs, no meta-instructions)


### Phase 1: Framework Adaptation (3-5 days)

**Goal**: Create a small-model-optimized Problem implementation for the darwinian evolver.

**Tasks**:
1. Write `SmallModelArcProblem` — new Problem subclass targeting 8B models
2. Replace `_prompt_llm()` with a clean provider abstraction (OpenAI-compatible endpoint)
3. Drop transfer scoring (LLM-as-judge, requires frontier model)
4. Replace simplicity scoring with AST-based heuristic (zero LLM cost)
5. Simplify mutation prompt for lower-capability models
6. Add JSON parsing fallbacks (regex extraction, retry with simpler prompt)
7. Tune concurrency for local inference (batch size, parallelism)

**Evaluation scoring (simplified)**:
```
score = correctness_score  (soft_score across training examples)
```

No transfer score, no LLM-based simplicity score. Just: does the code produce the right output?

**Concurrency model for local inference**:
- Run vLLM server separately (handles batching/parallelism)
- Evolver makes HTTP requests (ThreadPoolExecutor is fine)
- Can scale to 50+ concurrent requests against vLLM

**Deliverable**: A working evolver configuration that runs ARC tasks against a local 8B model with reasonable cost and latency.


### Phase 2: Mutation-Specific Fine-Tuning (1-2 weeks)

**Goal**: Specialize the SDPO-trained 8B for the evolver's mutation role — reading failure diffs and proposing targeted code fixes.

**Why this is still needed**: Phase -2 (SDPO) trains the model to solve ARC from scratch. The evolver's mutation task is different: you start with an existing (broken) solution and improve it based on specific failure feedback. These are overlapping but distinct skills.

**Two complementary data sources**:

1. **Evolver mutation traces** (from Phase 0/1 runs):
   - `(problem, current_code, failure_diffs, learning_log) → improved_code`
   - Filter for mutations that improved the score
   - Both Opus traces (if available from Imbue) and SDPO-8B traces (from our Phase 0 runs)

2. **SOAR-style hindsight reframing**:
   - For failed mutations, find what task the failed code *does* solve
   - Densifies training signal — every run produces data, not just winners

**Training recipe**:
1. SFT on successful mutation traces (off-policy from Opus + on-policy from 8B)
2. Measure mutation success rate improvement in evolver
3. Optional: DPO on (winning_mutation, losing_mutation) pairs from same parent

**This phase is optional if Phase 0 shows the SDPO-trained model already has a high enough mutation success rate (>25%).** SDPO's multi-turn training may have already taught the model to iterate on failures.


### Phase 3: Scale Search — Exploit Cost Advantage (1 week)

**Goal**: Use the 8B's cost advantage to run evolutionary search at scales impossible with frontier models.

**The math**:
- Opus: ~$5/M input, $25/M output. ~$2-5 per task.
- 8B local: ~$0.01-0.05 per task (100-500x cheaper)
- Same budget ($100) = 50 tasks with Opus vs 5000+ tasks with 8B

**Experiments**:
1. **Population scaling**: 2 parents (Imbue default) vs 10 vs 50 parents per iteration
2. **Iteration scaling**: 16 iterations (Imbue default) vs 100 vs 500 iterations
3. **Attempt scaling**: pass@2 (Imbue default) vs pass@10 vs pass@50
4. **Crossover**: Does crossover help more at scale? (more diverse population to draw from)
5. **Learning log**: Does ancestor/neighborhood learning log matter more with more iterations?

**Key question**: What's the Pareto frontier of (population_size x iterations x attempts) for a fixed compute budget?

**Hypothesis**: The 8B + massive search will match Opus + light search on a meaningful fraction of ARC tasks — specifically the ones where the bottleneck is search breadth, not per-sample insight.


### Phase 4: SDPO for Evolution — RL on the Mutation Task (2-4 weeks)

**Goal**: Adapt the verl SDPO pipeline to optimize the 8B specifically for the evolver mutation task.

**Key insight**: Phase -2 trains the model to solve ARC from scratch. Phase 4 trains it to improve existing solutions given failure feedback — the evolver's core loop. We can reuse the entire verl infrastructure but swap the environment:

**New environment: `EvolverMutationInteraction`**
- Input: (problem, current_code, failure_diffs) — what the evolver mutator sees
- Model generates: reflection + improved code
- Reward: score_improvement (new_score - parent_score)
- Multi-turn: model can analyze diffs, test hypotheses, then propose a fix

This is the same verl SDPO pipeline from Phase -2 with a different `Interaction` class. The training infra (Modal, vLLM, checkpointing) is already built.

**Why this may be unnecessary**: If the Phase -2 SDPO model already achieves >25% mutation success rate in the evolver (Phase 0), the marginal value of mutation-specific RL may not justify the cost. Measure first.

**When to pursue**: If Phase 0 shows the SDPO model's mutation success rate is <15%, or if Phase 3 scale experiments hit a clear ceiling that per-mutation quality (not search volume) would fix.


### Phase 5: Meta-Search — Evolve the Evolver (2-4 weeks)

**Goal**: Use evolution to optimize the evolutionary search harness itself.

**What to evolve**:
- Mutation prompt template (the 50+ line instruction set)
- Available affordances (which grid helpers to expose)
- Feedback format (how failure diffs are presented)
- Decomposition heuristics ("when shapes differ, try X")
- Hyperparameters (population size, iteration count, aggressive_fraction, batch_size)

**Architecture** (outer/inner loop):
```
Outer loop (darwinian evolver):
  Organism = {mutation_prompt, affordance_set, hyperparameters}
  Fitness = run inner-loop evolver on K tasks with this config
          -> measure: solve_rate / cost

Inner loop (existing per-task evolver):
  Uses outer organism's prompt/affordances
  Evolves per-task transform() functions
  Reports: iterations_to_solve, solve_rate, total_cost
```

**Pre-validation experiment** (before committing to full meta-search):
1. Hand-craft 5-6 candidate configs (different prompts, affordances, hyperparams)
2. Run per-task evolver with each config on 20 tasks
3. Measure variance in solve rate across configs
4. If configs produce significantly different results -> search space has structure -> meta-search is worthwhile
5. If all configs perform ~same -> bottleneck is elsewhere -> skip this phase

**Cost control**: Each inner-loop evaluation is cheap (8B model). But outer loop needs N tasks x M iterations per fitness evaluation. Budget carefully.


### Phase 6: Recursive Decomposition (speculative, 4+ weeks)

**Goal**: Combine recursive models (call/return) with evolutionary search.

**The gap**: The recursion paper shows recursion exponentially reduces context requirements, but the right recursive decomposition for ARC is unknown. SAT has DPLL. ARC has... what?

**Approach**: Use evolution to discover recursive decomposition patterns for ARC.
- Organism = a set of call/return decomposition rules
- e.g., "call: decompose grid into connected components, solve each"
- e.g., "call: identify symmetry type, then call: apply symmetry to missing region"
- Fitness = solve rate on ARC tasks using these patterns with the 8B model

**Heterogeneous model selection** (from recursion paper Section 6.2):
- The 8B learns to call Opus for hard subtasks and handle routine mutations itself
- The model learns its own capability boundary
- Cost = dominated by Opus calls, but those are rare (only for genuinely hard cases)

**This phase is speculative.** Only pursue if Phases 1-4 show that search breadth alone is insufficient and the bottleneck is per-task decomposition quality.


## Key Metrics

| Metric | Phase -2 (SDPO) | Phase -1 | Phase 0 | Phase 3 Target |
|---|---|---|---|---|
| ARC-AGI-1 solve rate (raw 8B) | — | measure | — | — |
| ARC-AGI-1 solve rate (SDPO 8B) | pass@1 improves | measure | — | — |
| SDPO training lift | any improvement | quantify | — | — |
| Mutation success rate (evolver) | — | — | >15% (SDPO model) | >30% |
| ARC-AGI-1 solve rate (evolver) | — | — | measure | >45% |
| ARC-AGI-2 solve rate (evolver) | — | — | — | >35% |
| Cost per task | N/A (training) | measure | ~$0.05 | ~$0.05 |

**North star**: Achieve >50% ARC-AGI-2 solve rate with an 8B model at <$0.10 per task.

**Progression**: SDPO training first (Pillar 1), then evaluate raw vs trained (Phase -1), then evolutionary search (Pillar 2). ARC-AGI-1 for development, ARC-AGI-2 for final comparison.

**Key comparison**: SOAR achieved 42.75% on ARC-AGI-1 with a 14B model + evolution + hindsight self-distillation. We're targeting comparable results with 8B + SDPO + evolution.


## Technical Infrastructure

### Model Serving
- **vLLM** for local inference (OpenAI-compatible API, handles batching)
- Candidate models: Qwen2.5-Coder-7B-Instruct, DeepSeek-Coder-V2-Lite
- GPU requirement: 1x A100 or 2x RTX 4090 for 8B model at reasonable throughput

### Evolver Framework
- Use Imbue's darwinian_evolver core (population.py, evolver.py, evolve_problem_loop.py)
- Write new SmallModelArcProblem (Organism, Evaluator, Mutator subclasses)
- Logging: JSONL (already built in), format for SFT data collection

### Training
- Fine-tuning: LoRA on 8B model (fits on single GPU)
- Data: mutation traces from evolver runs (Opus and 8B)
- Framework: torchtune / axolotl / OpenRLHF (for RL phases)

### Evaluation
- **Phase -1**: ARC-AGI-1 training split (400 tasks, answers available) — establish 8B capability floor
- **Phase 0+**: ARC-AGI-1 training/eval splits for development, ARC-AGI-2 eval split (120 tasks) for final comparison
- Pass@2 scoring (standard ARC metric)
- Cost tracking per task
- Trace export: `bun pi-rlm/src/trace-export.ts sharegpt` for SFT data collection from successful solves


## Open Questions

1. **What's the minimum model capability for viable mutation?** Is 8B enough, or do we need 14B? The SOAR result (14B, 42.75%) suggests 14B works. We're betting 8B + better training can match.

2. **Does transfer scoring matter?** Imbue uses it (7% of score weight). If we drop it, do we just need more iterations to compensate? Or does it provide signal that raw correctness can't?

3. **How much does prompt matter vs model capability?** Phase 0 tests this — if a simplified prompt works nearly as well, the model capability is the bottleneck. If the original complex prompt works much better, prompt engineering is high leverage.

4. **Does the SOAR hindsight trick work for mutation traces?** Reframing failed mutations as correct for synthetic tasks could 5-10x our training data. Worth testing early.

5. **Where does the 8B ceiling actually hit?** Some ARC tasks require genuine creative insight — perceiving a novel abstract pattern from 3 examples. Can an 8B do this at all, even with perfect search? Or is there a class of tasks where only frontier models can form the right hypothesis?

6. **Is the evolver's population dynamics optimal for small models?** Imbue tuned for frontier models (2 parents, 16 iterations, high mutation quality). The optimal regime for 8B might look completely different (50 parents, 500 iterations, low mutation quality but massive search).


## References

- Darwinian Evolver: https://github.com/imbue-ai/darwinian_evolver/
- Recursive Models for Long-Horizon Reasoning: arxiv:2603.02112
- SOAR: https://julienp.netlify.app/posts/soar/
- SDPO: https://self-distillation.github.io/SDPO
- SDFT: arxiv:2601.19897
- Absolute Zero: arxiv:2505.03335
- Darwin Goedel Machines: arxiv:2505.22954
- AlphaEvolve: arxiv:2506.13131
- Molecular Structure of Thought: arxiv:2601.06002
- GEPA/OptimizeAnything: https://gepa-ai.github.io/gepa/
- RLM (PrimeIntellect): https://www.primeintellect.ai/blog/rlm
