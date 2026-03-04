#!/usr/bin/env node
/**
 * ARC-AGI-2 evaluation harness for pi-rlm.
 *
 * Usage:
 *   bun domains/arc-agi-2/src/runner.ts [options]
 *
 * Options:
 *   --data-dir <path>      Path to ARC-AGI-2 data dir (default: ../downloads/ARC-AGI-2/data)
 *   --log-dir <path>       Log directory relative to repo root (default: logs/arc-2)
 *   --results-dir <path>   Results directory relative to repo root (default: results/arc-2)
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
 *   --stream               Stream agent output to stdout
 *   --log                  Enable session logging
 */

import { getModel, streamSimple } from "@mariozechner/pi-ai";
import type { AssistantMessageEvent } from "@mariozechner/pi-ai";
import type { ThinkingLevel } from "@mariozechner/pi-agent-core";
import { Orchestrator, OrchestratorTUI, createOAuthResolver, SessionDir } from "pi-rlm";
import type { TaggedAgentEvent } from "pi-rlm";
import { createArcAdapter } from "./adapter.js";
import { loadTask, loadTasksFromDir, selectDevSet } from "./task-loader.js";
import { accuracy } from "./grid-helpers.js";
import type { ArcGrid, ArcAttempt, ArcResult } from "./types.js";
import { writeFileSync, mkdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

// ── Stream handler ──

const STREAM_COLORS = {
	reset: "\x1b[0m",
	dim: "\x1b[2m",
	cyan: "\x1b[36m",
	yellow: "\x1b[33m",
	green: "\x1b[32m",
	red: "\x1b[31m",
	blue: "\x1b[34m",
	magenta: "\x1b[35m",
};

function handleStreamEvent(event: TaggedAgentEvent): void {
	const depth = event._depth ?? 0;
	const gutter = STREAM_COLORS.dim + "│ ".repeat(depth) + STREAM_COLORS.reset;

	switch (event.type) {
		case "agent_start": {
			const label = event._label ?? "Agent";
			const id = (event as any)._agentId ?? "";
			process.stderr.write(`${gutter}${STREAM_COLORS.dim}┌─ #${id} ${label} ${"─".repeat(Math.max(0, 45 - label.length))}${STREAM_COLORS.reset}\n`);
			break;
		}

		case "message_update": {
			const ame = (event as any).assistantMessageEvent as AssistantMessageEvent | undefined;
			if (!ame) break;
			if (ame.type === "text_delta") {
				process.stderr.write(ame.delta);
			} else if (ame.type === "thinking_delta") {
				process.stderr.write(STREAM_COLORS.cyan + ame.delta + STREAM_COLORS.reset);
			}
			break;
		}

		case "message_end":
			process.stderr.write("\n");
			break;

		case "tool_execution_start": {
			const code = (event as any).args?.code;
			if (code) {
				const preview = code.split("\n").slice(0, 5).join("\n");
				const more = code.split("\n").length > 5 ? `\n${STREAM_COLORS.dim}  ... (${code.split("\n").length - 5} more lines)${STREAM_COLORS.reset}` : "";
				process.stderr.write(`${gutter}${STREAM_COLORS.yellow}EVAL ${preview}${more}${STREAM_COLORS.reset}\n`);
			}
			break;
		}

		case "tool_execution_end": {
			const r = (event as any).result;
			const text = r?.content?.filter((c: any) => c.type === "text").map((c: any) => c.text?.trim()).join("\n") ?? "";
			if (text) {
				for (const line of text.split("\n")) {
					if (line.startsWith("ERROR:")) process.stderr.write(`${gutter}${STREAM_COLORS.red}  ${line}${STREAM_COLORS.reset}\n`);
					else if (/accuracy|correct|→/i.test(line)) process.stderr.write(`${gutter}${STREAM_COLORS.green}  ${line}${STREAM_COLORS.reset}\n`);
					else process.stderr.write(`${gutter}${STREAM_COLORS.dim}  ${line}${STREAM_COLORS.reset}\n`);
				}
			}
			break;
		}

		case "agent_end":
			process.stderr.write(`${gutter}${STREAM_COLORS.dim}└${"─".repeat(55)}${STREAM_COLORS.reset}\n`);
			break;
	}
}

// ── Parse CLI args ──

const args = process.argv.slice(2);

function getArg(name: string, fallback: string): string {
	const idx = args.indexOf(name);
	if (idx === -1 || idx + 1 >= args.length) return fallback;
	return args[idx + 1];
}

const repoRoot = join(import.meta.dirname, "../../..");
const dataDir = getArg("--data-dir", join(import.meta.dirname, "../../../downloads/ARC-AGI-2/data"));
const logDir = getArg("--log-dir", "logs/arc-2");
const resultsDir = getArg("--results-dir", "results/arc-2");
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
const stream = args.includes("--stream");
const logEnabled = args.includes("--log");

// Resolve log/results dirs relative to repo root
const resolvedLogDir = join(repoRoot, logDir);
const resolvedResultsDir = join(repoRoot, resultsDir);

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

		const sessionDir = logEnabled ? new SessionDir({ logDir: resolvedLogDir, metadata: { taskId: id, split, model: modelName, attempt: a } }) : undefined;

		let tui: OrchestratorTUI | null = null;

		const orchestrator = new Orchestrator({
			model,
			adapter,
			thinkingLevel,
			maxAgents,
			getApiKey,
			streamFn: maxTokensStreamFn,
			sessionDir,
			onEvent: (event) => {
				if (tui) tui.handleEvent(event);
				if (stream) handleStreamEvent(event as TaggedAgentEvent);
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
		let runError: string | undefined;
		try {
			resolvedValue = await orchestrator.run(
				"Analyze the training examples, discover the transformation rule, write a `transform(grid)` function, test it on ALL training examples until accuracy=1.0, then resolve with the transform function.",
			);
		} catch (err) {
			runError = String(err);
			console.error(`    Error: ${err}`);
		}
		const elapsed = Date.now() - startTime;

		if (tui) {
			await new Promise((resolve) => setTimeout(resolve, 100));
			tui.stop();
		}

		if (orchestrator.getAgentLogger()) {
			orchestrator.getAgentLogger()!.close(orchestrator.getUsageTracker());
		}

		// Resolved value is the transform function. Apply it to the test input for scoring.
		let predicted: ArcGrid | null = null;
		if (typeof resolvedValue === "function") {
			try {
				predicted = (resolvedValue as (grid: ArcGrid) => ArcGrid)(task.test[0].input);
			} catch {
				// transform threw — treat as wrong
			}
		}
		const usage = orchestrator.getUsageTracker().totalUsage();
		const correct = predicted !== null && accuracy(predicted, expected) === 1.0;
		const failed = runError !== undefined || resolvedValue === undefined;

		const attempt: ArcAttempt = {
			predicted: predicted ?? [],
			correct,
			...(failed && !correct ? { failed: true, error: runError } : {}),
			cost: usage.totalCost,
			tokens: usage.totalTokens,
			timeMs: elapsed,
		};
		attempts.push(attempt);

		const status = correct ? "CORRECT" : failed ? "FAILED" : "WRONG";
		console.log(`    ${status} | cost=$${usage.totalCost.toFixed(4)} | tokens=${usage.totalTokens.toLocaleString()} | time=${(elapsed / 1000).toFixed(1)}s`);

		if (correct) break; // no need for more attempts
	}

	const taskCorrect = attempts.some((a) => a.correct);
	const taskFailed = !taskCorrect && attempts.every((a) => a.failed);
	const taskCost = attempts.reduce((sum, a) => sum + a.cost, 0);
	const taskTokens = attempts.reduce((sum, a) => sum + a.tokens, 0);
	const taskTime = attempts.reduce((sum, a) => sum + a.timeMs, 0);

	const result: ArcResult = {
		taskId: id,
		correct: taskCorrect,
		...(taskFailed ? { failed: true } : {}),
		attempts,
		expected,
		cost: taskCost,
		tokens: taskTokens,
		timeMs: taskTime,
	};
	results.push(result);

	const taskStatus = taskCorrect ? "CORRECT" : taskFailed ? "FAILED" : "WRONG";
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
	console.log(`  ${r.taskId}: ${r.correct ? "CORRECT" : r.failed ? "FAILED " : "WRONG  "} | $${r.cost.toFixed(4)} | ${(r.timeMs / 1000).toFixed(1)}s`);
}

console.log();
console.log(`Score: ${correctCount}/${results.length} correct (${((correctCount / results.length) * 100).toFixed(1)}%) [pass@${numAttempts}]`);
console.log(`Total cost: $${totalCost.toFixed(4)}`);
console.log(`Total tokens: ${totalTokens.toLocaleString()}`);
console.log(`Total time: ${(totalTime / 1000).toFixed(1)}s`);

// ── Save results ──

mkdirSync(resolvedResultsDir, { recursive: true });

const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
const labelSuffix = label ? `-${label}` : "";
const shardSuffix = shard >= 0 ? `-shard${shard}` : "";
const resultsPath = join(resolvedResultsDir, `${timestamp}${labelSuffix}${shardSuffix}.json`);

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
