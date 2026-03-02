#!/usr/bin/env node
/**
 * ARC-AGI-2 evaluation harness for pi-rlm.
 *
 * Usage:
 *   bun pi-rlm/src/arc-runner.ts [options]
 *
 * Options:
 *   --data-dir <path>      Path to ARC-AGI-2 data dir (default: ../downloads/ARC-AGI-2/data)
 *   --split <name>         training or evaluation (default: training)
 *   --count <n>            Number of tasks to run (default: 5)
 *   --task-id <id>         Run a specific task by ID
 *   --model <model>        Model name (default: claude-opus-4-6)
 *   --thinking <level>     Thinking level: off, low, medium, high, xhigh (default: xhigh)
 *   --num-attempts <n>     Attempts per task for pass@K scoring (default: 2)
 *   --max-agents <n>       Max total agents per attempt (default: 10)
 *   --task-ids-file <path> Run specific tasks listed in a file (one ID per line)
 *   --label <name>         Label suffix for result files (e.g. "retry")
 *   --shard <n>            Shard index for parallel runs (0-based)
 *   --num-shards <n>       Total number of shards
 *   --debug                Enable debug TUI
 *   --log                  Enable session logging
 */

import { getModel, streamSimple } from "@mariozechner/pi-ai";
import type { ThinkingLevel } from "@mariozechner/pi-agent-core";
import { Orchestrator } from "./orchestrator.js";
import { OrchestratorTUI } from "./tui.js";
import { createOAuthResolver } from "./auth.js";
import { SessionLogger } from "./session-logger.js";
import { createArcAdapter } from "./arc/adapter.js";
import { loadTask, loadTasksFromDir, selectDevSet } from "./arc/task-loader.js";
import { accuracy } from "./arc/grid-helpers.js";
import type { ArcGrid, ArcAttempt, ArcResult } from "./arc/types.js";
import type { TaggedAgentEvent } from "./event-tagging.js";
import { writeFileSync, mkdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

// ── Parse CLI args ──

const args = process.argv.slice(2);

function getArg(name: string, fallback: string): string {
	const idx = args.indexOf(name);
	if (idx === -1 || idx + 1 >= args.length) return fallback;
	return args[idx + 1];
}

const dataDir = getArg("--data-dir", join(process.cwd(), "../downloads/ARC-AGI-2/data"));
const split = getArg("--split", "training");
const count = parseInt(getArg("--count", "5"), 10);
const taskId = getArg("--task-id", "");
const modelName = getArg("--model", "claude-opus-4-6");
const thinkingLevel = getArg("--thinking", "high") as ThinkingLevel;
const numAttempts = parseInt(getArg("--num-attempts", "2"), 10);
const maxAgents = parseInt(getArg("--max-agents", "10"), 10);
const shard = parseInt(getArg("--shard", "-1"), 10);
const numShards = parseInt(getArg("--num-shards", "1"), 10);
const taskIdsFile = getArg("--task-ids-file", "");
const label = getArg("--label", "");
const allTasks = args.includes("--all");
const debug = args.includes("--debug");
const logEnabled = args.includes("--log");

// ── Load tasks ──

const splitDir = join(dataDir, split);

interface TaskEntry {
	id: string;
	task: ReturnType<typeof loadTask>;
}

let tasks: TaskEntry[];
if (taskIdsFile) {
	const ids = readFileSync(taskIdsFile, "utf-8").split("\n").map(s => s.trim()).filter(Boolean);
	tasks = ids.map(id => ({ id, task: loadTask(join(splitDir, `${id}.json`)) }));
} else if (taskId) {
	const task = loadTask(join(splitDir, `${taskId}.json`));
	tasks = [{ id: taskId, task }];
} else if (allTasks) {
	tasks = loadTasksFromDir(splitDir, count);
} else {
	tasks = selectDevSet(splitDir, count);
}

// Apply sharding (round-robin to distribute difficulty evenly)
if (shard >= 0 && numShards > 1) {
	tasks = tasks.filter((_, i) => i % numShards === shard);
}

console.log(`ARC-AGI-2 Evaluation`);
console.log(`  Split: ${split}`);
console.log(`  Tasks: ${tasks.length}${shard >= 0 ? ` (shard ${shard}/${numShards})` : ""}`);
console.log(`  Model: ${modelName}`);
console.log(`  Thinking: ${thinkingLevel}`);
console.log(`  Attempts: ${numAttempts} (pass@${numAttempts})`);
console.log(`  Max agents: ${maxAgents}`);
console.log(`  Task IDs: ${tasks.map((t) => t.id).join(", ")}`);
console.log();

// ── Run tasks ──

const model = getModel("anthropic", modelName as any);
const getApiKey = process.env.ANTHROPIC_API_KEY ? undefined : createOAuthResolver();

// Override max_tokens to 128k (default caps at 32k, which exhausts budget
// before the model can emit tool calls when extended thinking is enabled)
const maxTokensStreamFn: typeof streamSimple = (model, context, options) => {
	return streamSimple(model, context, { ...options, maxTokens: 128000 });
};

const results: ArcResult[] = [];

for (let i = 0; i < tasks.length; i++) {
	const { id, task } = tasks[i];
	console.log(`[${i + 1}/${tasks.length}] Task ${id}...`);

	const expected = task.test[0].output;
	const attempts: ArcAttempt[] = [];

	for (let a = 0; a < numAttempts; a++) {
		console.log(`  Attempt ${a + 1}/${numAttempts}...`);

		const adapter = createArcAdapter(task);

		const sessionLogger = logEnabled ? new SessionLogger({ logDir: `./logs/arc`, metadata: { taskId: id, split, model: modelName, attempt: a } }) : undefined;
		if (sessionLogger) sessionLogger.start();

		let tui: OrchestratorTUI | null = null;

		const orchestrator = new Orchestrator({
			model,
			adapter,
			thinkingLevel,
			maxAgents,
			getApiKey,
			streamFn: maxTokensStreamFn,
			onEvent: (event) => {
				if (tui) tui.handleEvent(event);
				if (sessionLogger) sessionLogger.logEvent(event as TaggedAgentEvent);
			},
		});

		if (debug) {
			tui = new OrchestratorTUI({
				debug: true,
				usageTracker: orchestrator.getUsageTracker(),
			});
			tui.start();
		}

		const startTime = Date.now();
		let resolvedValue: unknown;
		try {
			resolvedValue = await orchestrator.run(
				"Analyze the training examples, discover the transformation rule, write a `transform(grid)` function, test it on ALL training examples until accuracy=1.0, then resolve with your answer for the test input.",
			);
		} catch (err) {
			console.error(`    Error: ${err}`);
		}
		const elapsed = Date.now() - startTime;

		if (tui) {
			await new Promise((resolve) => setTimeout(resolve, 100));
			tui.stop();
		}

		if (sessionLogger) {
			sessionLogger.close(orchestrator.getUsageTracker());
		}

		const predicted = (resolvedValue as ArcGrid | undefined) ?? null;
		const usage = orchestrator.getUsageTracker().totalUsage();
		const correct = predicted !== null && accuracy(predicted, expected) === 1.0;

		const attempt: ArcAttempt = {
			predicted: predicted ?? [],
			correct,
			cost: usage.totalCost,
			tokens: usage.totalTokens,
			timeMs: elapsed,
		};
		attempts.push(attempt);

		const status = correct ? "CORRECT" : "WRONG";
		console.log(`    ${status} | cost=$${usage.totalCost.toFixed(4)} | tokens=${usage.totalTokens.toLocaleString()} | time=${(elapsed / 1000).toFixed(1)}s`);

		if (correct) break; // no need for more attempts
	}

	const taskCorrect = attempts.some((a) => a.correct);
	const taskCost = attempts.reduce((sum, a) => sum + a.cost, 0);
	const taskTokens = attempts.reduce((sum, a) => sum + a.tokens, 0);
	const taskTime = attempts.reduce((sum, a) => sum + a.timeMs, 0);

	const result: ArcResult = {
		taskId: id,
		correct: taskCorrect,
		attempts,
		expected,
		cost: taskCost,
		tokens: taskTokens,
		timeMs: taskTime,
	};
	results.push(result);

	const taskStatus = taskCorrect ? "CORRECT" : "WRONG";
	console.log(`  pass@${numAttempts}: ${taskStatus} | cost=$${taskCost.toFixed(4)} | tokens=${taskTokens.toLocaleString()} | time=${(taskTime / 1000).toFixed(1)}s`);
	console.log();
}

// ── Summary ──

const correctCount = results.filter((r) => r.correct).length;
const totalCost = results.reduce((sum, r) => sum + r.cost, 0);
const totalTokens = results.reduce((sum, r) => sum + r.tokens, 0);
const totalTime = results.reduce((sum, r) => sum + r.timeMs, 0);

console.log("=".repeat(60));
console.log(`Results Summary (pass@${numAttempts})`);
console.log("=".repeat(60));
console.log();

for (const r of results) {
	console.log(`  ${r.taskId}: ${r.correct ? "CORRECT" : "WRONG   "} | $${r.cost.toFixed(4)} | ${(r.timeMs / 1000).toFixed(1)}s`);
}

console.log();
console.log(`Score: ${correctCount}/${results.length} correct (${((correctCount / results.length) * 100).toFixed(1)}%) [pass@${numAttempts}]`);
console.log(`Total cost: $${totalCost.toFixed(4)}`);
console.log(`Total tokens: ${totalTokens.toLocaleString()}`);
console.log(`Total time: ${(totalTime / 1000).toFixed(1)}s`);

// ── Save results ──

const resultsDir = join(process.cwd(), "results/arc");
mkdirSync(resultsDir, { recursive: true });

const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
const labelSuffix = label ? `-${label}` : "";
const shardSuffix = shard >= 0 ? `-shard${shard}` : "";
const resultsPath = join(resultsDir, `${timestamp}${labelSuffix}${shardSuffix}.json`);

writeFileSync(
	resultsPath,
	JSON.stringify(
		{
			config: { split, model: modelName, thinking: thinkingLevel, numAttempts, maxAgents, count: tasks.length, ...(shard >= 0 ? { shard, numShards } : {}) },
			score: { correct: correctCount, total: results.length, pct: correctCount / results.length },
			totals: { cost: totalCost, tokens: totalTokens, timeMs: totalTime },
			results,
		},
		null,
		2,
	),
);
console.log(`\nResults saved: ${resultsPath}`);
