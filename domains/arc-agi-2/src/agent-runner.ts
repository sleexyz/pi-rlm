#!/usr/bin/env bun
/**
 * Single-task ARC agent runner for eval and training pipelines.
 *
 * Reads a task from stdin, runs the pi-rlm Orchestrator against a vLLM server,
 * scores the result, and outputs JSON to stdout. All progress/debug goes to stderr.
 *
 * Usage:
 *   echo '$TASK_JSON' | bun run domains/arc-agi-2/src/agent-runner.ts \
 *     --base-url http://localhost:8000/v1 --model Qwen/Qwen3-8B --task-id 007bbfb7 \
 *     --task-from-stdin [--max-turns 15] [--timeout 300000] [--top-logprobs 0]
 */

import type { ThinkingLevel } from "@mariozechner/pi-agent-core";
import type { StreamFn } from "@mariozechner/pi-agent-core";
import {
	Orchestrator,
	SessionDir,
	createCodeBlockStreamFn,
	generateCodeBlockSystemPrompt,
	createVllmModel,
} from "pi-rlm";
import { createArcAdapter } from "./adapter.js";
import { accuracy } from "./grid-helpers.js";
import type { ArcGrid, ArcTask } from "./types.js";
import { readFileSync } from "node:fs";

// ── Parse CLI args ──

const args = process.argv.slice(2);

function getArg(name: string, fallback: string): string {
	const idx = args.indexOf(name);
	if (idx === -1 || idx + 1 >= args.length) return fallback;
	return args[idx + 1];
}

const baseUrl = getArg("--base-url", "");
const modelName = getArg("--model", "");
const taskId = getArg("--task-id", "");
const maxTurns = parseInt(getArg("--max-turns", "15"), 10);
const thinkingLevel = getArg("--thinking", "off") as ThinkingLevel;
const timeout = parseInt(getArg("--timeout", "300000"), 10);
const topLogprobs = parseInt(getArg("--top-logprobs", "0"), 10);
const maxAgents = parseInt(getArg("--max-agents", "10"), 10);
const runDir = getArg("--run-dir", "");
const sessionId = getArg("--session-id", "");
const numAttempts = parseInt(getArg("--num-attempts", "1"), 10);
const temperature = parseFloat(getArg("--temperature", "0.6"));
const topP = parseFloat(getArg("--top-p", "0.95"));
const taskFromStdin = args.includes("--task-from-stdin");

if (!baseUrl || !modelName || !taskId) {
	process.stderr.write("Required: --base-url, --model, --task-id\n");
	process.exit(1);
}

// ── Read task from stdin ──

let task: ArcTask;
if (taskFromStdin) {
	const input = readFileSync(0, "utf-8");
	task = JSON.parse(input);
} else {
	process.stderr.write("--task-from-stdin is required\n");
	process.exit(1);
}

// ── Set up model ──

const model = createVllmModel({ id: modelName, baseUrl });

// ── Stream function with optional logprobs ──

const baseStreamFn = createCodeBlockStreamFn({ topLogprobs });
const streamFn: StreamFn = (m, ctx, opts) => {
	return baseStreamFn(m, ctx, {
		...opts,
		temperature,
		maxTokens: 8192,
		onPayload: (payload: any) => {
			payload.top_p = topP;
		},
	});
};

// ── Run attempts ──

interface AttemptResult {
	predicted: ArcGrid;
	correct: boolean;
	failed?: boolean;
	error?: string;
	tokens: number;
	timeMs: number;
	turns: number;
	trajectory?: unknown[];
}

const expected = task.test[0].output;
const attempts: AttemptResult[] = [];

for (let a = 0; a < numAttempts; a++) {
	process.stderr.write(`[agent-runner] Attempt ${a + 1}/${numAttempts} for ${taskId}\n`);

	const adapter = createArcAdapter(task);

	const sessionDir = runDir
		? new SessionDir({
				logDir: runDir + "/sessions",
				sessionId: sessionId || `${taskId}_a${a}`,
				metadata: { taskId, model: modelName, attempt: a },
			})
		: undefined;

	const orchestrator = new Orchestrator({
		model,
		adapter,
		thinkingLevel,
		maxAgents,
		streamFn,
		sessionDir,
		generateSystemPrompt: generateCodeBlockSystemPrompt,
	});

	const t0 = Date.now();
	let submittedValue: unknown;
	let runError: string | undefined;

	try {
		const result = await Promise.race([
			orchestrator.run(
				"Analyze the training examples, discover the transformation rule, write a `transform(grid)` function, test it on ALL training examples until accuracy=1.0, then submit the transform function.",
			),
			new Promise<never>((_, reject) => setTimeout(() => reject(new Error("Timeout")), timeout)),
		]);
		submittedValue = result;
	} catch (err) {
		runError = String(err);
		process.stderr.write(`[agent-runner] Error: ${runError}\n`);
	}

	const elapsed = Date.now() - t0;

	// Close logger
	if (orchestrator.getAgentLogger()) {
		orchestrator.getAgentLogger()!.close(orchestrator.getUsageTracker());
	}

	// Score: apply submitted transform to test input
	let predicted: ArcGrid | null = null;
	if (typeof submittedValue === "function") {
		try {
			predicted = (submittedValue as (grid: ArcGrid) => ArcGrid)(task.test[0].input);
		} catch {
			// transform threw — treat as wrong
		}
	}

	const usage = orchestrator.getUsageTracker().totalUsage();
	const correct = predicted !== null && accuracy(predicted, expected) === 1.0;

	// Collect trajectory from agent message history
	const trajectory = orchestrator.getAgent().state.messages.map((msg: any) => ({
		role: msg.role,
		content:
			msg.role === "assistant"
				? msg.content
						.filter((c: any) => c.type === "text")
						.map((c: any) => c.text)
						.join("")
				: typeof msg.content === "string"
					? msg.content
					: msg.content
							?.filter((c: any) => c.type === "text")
							.map((c: any) => c.text)
							.join(""),
		...(msg.logprobs ? { logprobs: msg.logprobs } : {}),
	}));

	const attempt: AttemptResult = {
		predicted: predicted ?? [],
		correct,
		...(runError ? { failed: true, error: runError } : {}),
		tokens: usage.totalTokens,
		timeMs: elapsed,
		turns: trajectory.filter((m: any) => m.role === "assistant").length,
		trajectory,
	};
	attempts.push(attempt);

	process.stderr.write(
		`[agent-runner] ${correct ? "CORRECT" : "WRONG"} | tokens=${usage.totalTokens} | time=${(elapsed / 1000).toFixed(1)}s\n`,
	);

	if (correct) break; // no need for more attempts
}

// ── Build result ──

const bestAttempt = attempts.find((a) => a.correct) ?? attempts[attempts.length - 1];
const totalTokens = attempts.reduce((sum, a) => sum + a.tokens, 0);
const totalTime = attempts.reduce((sum, a) => sum + a.timeMs, 0);

const result = {
	taskId,
	reward: bestAttempt.correct ? 1.0 : 0.0,
	submitted: bestAttempt.predicted.length > 0,
	correct: bestAttempt.correct,
	turns: bestAttempt.turns,
	tokens: totalTokens,
	timeMs: totalTime,
	trajectory: bestAttempt.trajectory,
	attempts: attempts.map((a) => ({
		correct: a.correct,
		...(a.failed ? { failed: true, error: a.error } : {}),
		tokens: a.tokens,
		timeMs: a.timeMs,
	})),
};

// ── JSON to stdout ──

process.stdout.write(JSON.stringify(result));
