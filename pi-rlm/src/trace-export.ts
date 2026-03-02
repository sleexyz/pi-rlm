#!/usr/bin/env node
/**
 * Trace exporter: SessionLogger JSONL + Results JSON → sparse-v1.0 Rollout JSONL → ShareGPT JSONL
 *
 * Usage:
 *   bun pi-rlm/src/trace-export.ts rollout --sessions <dir> --results <file> --out <file>
 *   bun pi-rlm/src/trace-export.ts sharegpt --rollouts <file> --out <file> [--filter correct|all|wrong]
 *   bun pi-rlm/src/trace-export.ts sharegpt --sessions <dir> --results <file> --out <file> [--filter correct|all|wrong]
 */

import { readFileSync, writeFileSync, readdirSync } from "node:fs";
import { join, resolve } from "node:path";

// ── Types ──

interface Step {
	step_id: number;
	timestamp: string;
	source: "user" | "assistant";
	message: string;
	reasoning_content: string | null;
	code_executed: string | null;
	execution_result: string | null;
	depth: number;
	parent_step_id: number | null;
	metrics: {
		prompt_tokens: number | null;
		completion_tokens: number | null;
		cost_usd: number | null;
	} | null;
}

interface Rollout {
	schema_version: "sparse-v1.0";
	session_id: string;
	agent: {
		name: string;
		model_name: string;
		version: string;
	};
	task_id: string;
	dataset: string;
	system_prompt: string;
	steps: Step[];
	n_turns: number;
	elapsed_s: number;
	reward: number;
	correct: boolean;
	cost: number;
	tokens: number;
}

interface ShareGPTRow {
	conversations: { role: string; content: string }[];
	task_id: string;
	correct: boolean;
	reward: number;
	model: string;
	dataset: string;
	n_turns: number;
	cost: number;
}

// ── Session JSONL Parser ──

interface SessionEvent {
	type: string;
	ts?: number;
	sessionId?: string;
	taskId?: string;
	split?: string;
	model?: string;
	attempt?: number;
	agentId?: number;
	depth?: number;
	label?: string;
	systemPrompt?: string;
	userMessage?: string;
	content?: ContentBlock[];
	usage?: { input: number; output: number; totalTokens: number; cost: number };
	toolName?: string;
	code?: string;
	result?: { content?: { type: string; text: string }[]; details?: Record<string, unknown> };
}

interface ContentBlock {
	type: string;
	text?: string;
	thinking?: string;
	thinkingSignature?: string;
	id?: string;
	name?: string;
	arguments?: { code?: string };
}

function parseSessionJsonl(filePath: string): SessionEvent[] {
	const lines = readFileSync(filePath, "utf-8").split("\n").filter(Boolean);
	return lines.map((line) => JSON.parse(line) as SessionEvent);
}

// ── Rollout Converter ──

function sessionToRollout(
	events: SessionEvent[],
	attemptResult: { correct: boolean; cost: number; tokens: number } | null,
): Rollout | null {
	const sessionStart = events.find((e) => e.type === "session_start");
	const sessionEnd = events.find((e) => e.type === "session_end");
	const agentStart = events.find((e) => e.type === "agent_start" && e.depth === 0);

	if (!sessionStart || !agentStart) {
		console.warn(`[trace-export] Skipping: missing session_start or agent_start`);
		return null;
	}

	const sessionId = sessionStart.sessionId ?? "unknown";
	const taskId = sessionStart.taskId ?? "unknown";
	const split = sessionStart.split ?? "unknown";
	const model = sessionStart.model ?? "unknown";

	const dataset = split === "evaluation" ? "agi2-eval" : split === "training" ? "agi2-train" : split;

	const steps: Step[] = [];
	let stepId = 0;

	// Step 1: Initial user message (system prompt + task)
	stepId++;
	steps.push({
		step_id: stepId,
		timestamp: new Date((sessionStart.ts ?? 0) * 1000).toISOString(),
		source: "user",
		message: agentStart.userMessage ?? "",
		reasoning_content: null,
		code_executed: null,
		execution_result: null,
		depth: 0,
		parent_step_id: null,
		metrics: null,
	});

	// Walk through events, extracting assistant turns and tool results
	// Key insight: message_end events WITH usage = assistant messages
	// tool_execution_end = user messages (tool results)
	let lastAssistantStepId: number | null = null;

	for (let i = 0; i < events.length; i++) {
		const e = events[i];

		if (e.type === "message_end" && e.usage && e.content) {
			// Assistant message — extract thinking, text, and tool calls
			let reasoning = "";
			let text = "";
			let code: string | null = null;

			for (const block of e.content) {
				if (block.type === "thinking" && block.thinking) {
					reasoning += (reasoning ? "\n" : "") + block.thinking;
				} else if (block.type === "text" && block.text) {
					text += (text ? "\n" : "") + block.text;
				} else if ((block.type === "toolCall" || block.type === "tool_use") && block.arguments?.code) {
					code = block.arguments.code;
				}
			}

			stepId++;
			const step: Step = {
				step_id: stepId,
				timestamp: new Date((e.ts ?? 0) * 1000).toISOString(),
				source: "assistant",
				message: text,
				reasoning_content: reasoning || null,
				code_executed: code,
				execution_result: null,
				depth: e.depth ?? 0,
				parent_step_id: e.depth && e.depth > 0 ? (lastAssistantStepId ?? null) : null,
				metrics: {
					prompt_tokens: e.usage.input,
					completion_tokens: e.usage.output,
					cost_usd: e.usage.cost,
				},
			};
			steps.push(step);
			if ((e.depth ?? 0) === 0) {
				lastAssistantStepId = stepId;
			}
		} else if (e.type === "tool_execution_end" && e.result) {
			// Tool result → user step
			const resultText =
				e.result.content?.map((c) => c.text).join("\n") ?? "";

			stepId++;
			steps.push({
				step_id: stepId,
				timestamp: new Date((e.ts ?? 0) * 1000).toISOString(),
				source: "user",
				message: resultText,
				reasoning_content: null,
				code_executed: null,
				execution_result: resultText,
				depth: e.depth ?? 0,
				parent_step_id: null,
				metrics: null,
			});
		}
	}

	// Count root assistant turns
	const nTurns = steps.filter((s) => s.source === "assistant" && s.depth === 0).length;

	// Elapsed time
	const startTs = sessionStart.ts ?? 0;
	const endTs = sessionEnd?.ts ?? startTs;
	const elapsedS = endTs - startTs;

	// Reward from results
	const correct = attemptResult?.correct ?? false;
	const reward = correct ? 1.0 : 0.0;

	return {
		schema_version: "sparse-v1.0",
		session_id: sessionId,
		agent: {
			name: "arc-rlm",
			model_name: model,
			version: "1.0",
		},
		task_id: taskId,
		dataset,
		system_prompt: agentStart.systemPrompt ?? "",
		steps,
		n_turns: nTurns,
		elapsed_s: elapsedS,
		reward,
		correct,
		cost: attemptResult?.cost ?? 0,
		tokens: attemptResult?.tokens ?? 0,
	};
}

// ── Results JSON Loader ──

interface ResultsFile {
	config: Record<string, unknown>;
	score: { correct: number; total: number; pct: number };
	totals: { cost: number; tokens: number; timeMs: number };
	results: TaskResult[];
}

interface TaskResult {
	taskId: string;
	correct: boolean;
	attempts: AttemptResult[];
	expected: unknown;
	cost: number;
	tokens: number;
	timeMs: number;
}

interface AttemptResult {
	predicted: unknown;
	correct: boolean;
	cost: number;
	tokens: number;
	timeMs: number;
}

function loadResults(filePath: string): ResultsFile {
	return JSON.parse(readFileSync(filePath, "utf-8")) as ResultsFile;
}

// ── Join sessions to results ──

function findAttemptResult(
	results: ResultsFile,
	taskId: string,
	attempt: number,
): AttemptResult | null {
	const task = results.results.find((r) => r.taskId === taskId);
	if (!task) return null;
	return task.attempts[attempt] ?? null;
}

// ── ShareGPT Exporter ──

function rolloutToShareGPT(rollout: Rollout): ShareGPTRow {
	const conversations: { role: string; content: string }[] = [];

	// System prompt
	if (rollout.system_prompt) {
		conversations.push({ role: "system", content: rollout.system_prompt });
	}

	// Walk root-level steps only (depth=0)
	for (const step of rollout.steps) {
		if (step.depth !== 0) continue;

		if (step.source === "user") {
			// User messages: initial prompt or tool execution results
			if (step.execution_result !== null) {
				conversations.push({ role: "user", content: step.execution_result });
			} else {
				conversations.push({ role: "user", content: step.message });
			}
		} else if (step.source === "assistant") {
			const parts: string[] = [];

			// Thinking tokens in <think> tags
			if (step.reasoning_content) {
				parts.push(`<think>${step.reasoning_content}</think>`);
			}

			// Text message
			if (step.message) {
				parts.push(step.message);
			}

			// Code in fenced JS blocks
			if (step.code_executed) {
				parts.push("```javascript\n" + step.code_executed + "\n```");
			}

			conversations.push({
				role: "assistant",
				content: parts.join("\n"),
			});
		}
	}

	return {
		conversations,
		task_id: rollout.task_id,
		correct: rollout.correct,
		reward: rollout.reward,
		model: rollout.agent.model_name,
		dataset: rollout.dataset,
		n_turns: rollout.n_turns,
		cost: rollout.cost,
	};
}

// ── Pipeline ──

function processSessionDir(
	sessionsDir: string,
	results: ResultsFile,
): Rollout[] {
	const files = readdirSync(sessionsDir).filter((f) => f.endsWith(".jsonl")).sort();
	const rollouts: Rollout[] = [];

	for (const file of files) {
		const filePath = join(sessionsDir, file);
		try {
			const events = parseSessionJsonl(filePath);
			const sessionStart = events.find((e) => e.type === "session_start");
			if (!sessionStart) {
				console.warn(`[trace-export] Skipping ${file}: no session_start`);
				continue;
			}

			const taskId = sessionStart.taskId ?? "";
			const attempt = sessionStart.attempt ?? 0;

			const attemptResult = findAttemptResult(results, taskId, attempt);
			if (!attemptResult) {
				console.warn(`[trace-export] No result for ${taskId} attempt ${attempt}, skipping`);
				continue;
			}

			const rollout = sessionToRollout(events, attemptResult);
			if (rollout) {
				rollouts.push(rollout);
			}
		} catch (err) {
			console.warn(`[trace-export] Error processing ${file}:`, err);
		}
	}

	return rollouts;
}

function writeJsonl(filePath: string, objects: unknown[]): void {
	const content = objects.map((o) => JSON.stringify(o)).join("\n") + "\n";
	writeFileSync(filePath, content);
}

// ── CLI ──

function printUsage(): void {
	console.log(`Usage:
  bun src/trace-export.ts rollout --sessions <dir> --results <file> --out <file>
  bun src/trace-export.ts sharegpt --sessions <dir> --results <file> --out <file> [--filter correct|all|wrong]
  bun src/trace-export.ts sharegpt --rollouts <file> --out <file> [--filter correct|all|wrong]`);
}

function main(): void {
	const args = process.argv.slice(2);
	const subcommand = args[0];

	if (!subcommand || subcommand === "--help" || subcommand === "-h") {
		printUsage();
		process.exit(0);
	}

	function getArg(name: string, fallback = ""): string {
		const idx = args.indexOf(name);
		if (idx === -1 || idx + 1 >= args.length) return fallback;
		return args[idx + 1];
	}

	if (subcommand === "rollout") {
		const sessionsDir = resolve(getArg("--sessions"));
		const resultsFile = resolve(getArg("--results"));
		const outFile = resolve(getArg("--out", "rollouts.jsonl"));

		if (!sessionsDir || !resultsFile) {
			console.error("Missing --sessions or --results");
			printUsage();
			process.exit(1);
		}

		const results = loadResults(resultsFile);
		const rollouts = processSessionDir(sessionsDir, results);

		writeJsonl(outFile, rollouts);
		printSummary(rollouts, "Rollout");

	} else if (subcommand === "sharegpt") {
		const filter = getArg("--filter", "correct") as "correct" | "all" | "wrong";
		const outFile = resolve(getArg("--out", "sharegpt.jsonl"));
		const rolloutsFile = getArg("--rollouts");

		let rollouts: Rollout[];

		if (rolloutsFile) {
			const lines = readFileSync(resolve(rolloutsFile), "utf-8").split("\n").filter(Boolean);
			rollouts = lines.map((l) => JSON.parse(l) as Rollout);
		} else {
			const sessionsDir = resolve(getArg("--sessions"));
			const resultsFile = resolve(getArg("--results"));

			if (!sessionsDir || !resultsFile) {
				console.error("Missing --sessions/--results or --rollouts");
				printUsage();
				process.exit(1);
			}

			const results = loadResults(resultsFile);
			rollouts = processSessionDir(sessionsDir, results);
		}

		// Apply filter
		let filtered = rollouts;
		if (filter === "correct") {
			filtered = rollouts.filter((r) => r.correct);
		} else if (filter === "wrong") {
			filtered = rollouts.filter((r) => !r.correct);
		}

		// Convert to ShareGPT
		const rows = filtered.map((r) => rolloutToShareGPT(r));

		writeJsonl(outFile, rows);
		printSummary(rollouts, "ShareGPT", filter, rows.length);

		// Print first conversation to stdout for inspection if small output
		if (rows.length > 0 && rows.length <= 5) {
			console.log("\n--- First conversation preview ---");
			const first = rows[0];
			for (const msg of first.conversations) {
				const preview = msg.content.length > 300 ? msg.content.slice(0, 300) + "..." : msg.content;
				console.log(`[${msg.role}] ${preview}`);
				console.log();
			}
		}

	} else {
		console.error(`Unknown subcommand: ${subcommand}`);
		printUsage();
		process.exit(1);
	}
}

function printSummary(
	rollouts: Rollout[],
	format: string,
	filter?: string,
	exported?: number,
): void {
	const correct = rollouts.filter((r) => r.correct).length;
	const totalTurns = rollouts.reduce((sum, r) => sum + r.n_turns, 0);
	const totalCost = rollouts.reduce((sum, r) => sum + r.cost, 0);

	console.log(`\n=== ${format} Export Summary ===`);
	console.log(`Sessions processed: ${rollouts.length}`);
	console.log(`Correct: ${correct}/${rollouts.length}`);
	if (exported !== undefined && filter) {
		console.log(`Exported (filter=${filter}): ${exported}`);
	}
	console.log(`Total turns: ${totalTurns}`);
	console.log(`Total cost: $${totalCost.toFixed(2)}`);
}

main();
