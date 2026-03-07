/**
 * Training/eval sandbox agent — single source of truth for ARC feedback format.
 *
 * Wraps EvalRuntime + ARC adapter scope + parseReplBlocks + extractThinking
 * over a JSON-line stdin/stdout protocol. One process per rollout instance.
 *
 * Protocol:
 *   → {"type":"init","task":{train:[...],test:[...]}}
 *   ← {"type":"ready","systemPrompt":"...","userMessage":"..."}
 *   → {"type":"step","assistantText":"..."}
 *   ← {"type":"observation","feedback":"...","reward":0.0,"done":false,"info":{...}}
 *   → {"type":"close"}
 */

import { EvalRuntime } from "../../../pi-rlm/src/eval-runtime.ts";
import {
	parseReplBlocks,
	extractThinking,
	generateCodeBlockSystemPrompt,
} from "../../../pi-rlm/src/repl-stream.ts";
import * as helpers from "./grid-helpers.ts";
import { ARC_PREMISE, ARC_REFERENCE, ARC_SUB_AGENT_PROMPT } from "./prompt.ts";
import type { ArcTask } from "./types.ts";

let runtime: EvalRuntime | null = null;
let submitted: unknown = undefined;
let hasSubmitted = false;
let task: ArcTask | null = null;
let turns = 0;

function respond(obj: Record<string, unknown>) {
	process.stdout.write(JSON.stringify(obj) + "\n");
}

/** Format eval result output — matches eval-tool.ts execute() exactly. */
function formatOutput(result: {
	stdout: string;
	returnValue?: unknown;
	error: string | null;
}): string {
	let output = "";

	if (result.stdout) {
		output += result.stdout;
	}

	if (result.returnValue !== undefined) {
		const formatted = formatReturnValue(result.returnValue);
		if (formatted) {
			if (output) output += "\n";
			output += `→ ${formatted}`;
		}
	}

	if (result.error) {
		if (output) output += "\n";
		output += `ERROR: ${result.error}`;
	}

	if (!output) {
		output = "(no output)";
	}

	if (output.length > 20_000) {
		output =
			output.slice(0, 20_000) +
			`\n... (truncated, ${output.length} total chars)`;
	}

	return output;
}

function formatReturnValue(value: unknown): string {
	if (value === undefined || value === null) return "";
	if (typeof value === "string") return value;
	try {
		return JSON.stringify(value, null, 2);
	} catch {
		return String(value);
	}
}

/** Format feedback — matches toolResultToUserMessage() in repl-stream.ts. */
function formatFeedback(code: string, output: string): string {
	return `Code executed:\n\`\`\`javascript\n${code}\n\`\`\`\n\nREPL output:\n${output}`;
}

function computeReward(): number {
	if (!hasSubmitted || !task) return 0.0;
	const expected = task.test[0].output;
	if (typeof submitted === "function") {
		try {
			const predicted = (submitted as (g: number[][]) => number[][])(
				task.test[0].input,
			);
			return helpers.accuracy(predicted, expected) === 1.0 ? 1.0 : 0.0;
		} catch {
			return 0.0;
		}
	}
	// Direct value submission
	return helpers.accuracy(submitted as number[][], expected) === 1.0
		? 1.0
		: 0.0;
}

async function handleMessage(msg: {
	type: string;
	task?: ArcTask;
	assistantText?: string;
}) {
	if (msg.type === "init") {
		task = msg.task!;
		submitted = undefined;
		hasSubmitted = false;
		turns = 0;

		runtime = new EvalRuntime({
			trainingExamples: task.train.map((ex) => ({
				input: ex.input,
				output: ex.output,
			})),
			testInputs: task.test.map((ex) => ex.input),
			renderGrid: helpers.renderGrid,
			gridShape: helpers.gridShape,
			makeGrid: helpers.makeGrid,
			copyGrid: helpers.copyGrid,
			gridsEqual: helpers.gridsEqual,
			rotate90: helpers.rotate90,
			rotate180: helpers.rotate180,
			rotate270: helpers.rotate270,
			flipH: helpers.flipH,
			flipV: helpers.flipV,
			transpose: helpers.transpose,
			crop: helpers.crop,
			paste: helpers.paste,
			tile: helpers.tile,
			findColor: helpers.findColor,
			colorCounts: helpers.colorCounts,
			replaceColor: helpers.replaceColor,
			uniqueColors: helpers.uniqueColors,
			connectedComponents: helpers.connectedComponents,
			accuracy: helpers.accuracy,
			softAccuracy: helpers.softAccuracy,
			submit: (value: unknown) => {
				submitted = value;
				hasSubmitted = true;
				return value;
			},
		});

		const adapter = {
			name: "ARC-AGI-2",
			premise: ARC_PREMISE,
			reference: ARC_REFERENCE,
			defaultSubAgentPrompt: ARC_SUB_AGENT_PROMPT,
			scope: {},
		};

		const systemPrompt = generateCodeBlockSystemPrompt(adapter);
		const userMessage =
			"Analyze the training examples, discover the transformation rule, write a `transform(grid)` function, test it on ALL training examples until accuracy=1.0, then submit the transform function.";

		respond({ type: "ready", systemPrompt, userMessage });
	} else if (msg.type === "step") {
		if (!runtime || !task) {
			respond({ type: "error", error: "Not initialized" });
			return;
		}

		turns++;
		const assistantText = msg.assistantText ?? "";

		// Step 1: strip <think> blocks
		const [, cleanText] = extractThinking(assistantText);

		// Step 2: parse code blocks
		const segments = parseReplBlocks(cleanText);
		const codeSegments = segments.filter((s) => s.type === "code");

		if (codeSegments.length === 0) {
			respond({
				type: "observation",
				feedback:
					"No code found. Write a ```javascript block to execute code.",
				reward: 0.0,
				done: false,
				info: { turns, submitted: false },
			});
			return;
		}

		// Execute only the first code block (enforce one-block-per-turn)
		const code = (codeSegments[0] as { type: "code"; code: string }).code;
		const result = await runtime.eval(code, 30_000);
		const output = formatOutput(result);
		let feedback = formatFeedback(code, output);

		if (codeSegments.length > 1) {
			feedback += `\n\n⚠️ ${codeSegments.length - 1} additional code block(s) were skipped. Write ONE code block per response, then wait for the result.`;
		}

		// Check submission
		const done = hasSubmitted;
		const reward = done ? computeReward() : 0.0;

		respond({
			type: "observation",
			feedback,
			reward,
			done,
			info: { turns, submitted: hasSubmitted },
		});
	} else if (msg.type === "close") {
		process.exit(0);
	}
}

// Read JSON lines from stdin
const decoder = new TextDecoder();
let buffer = "";

process.stdin.on("data", async (chunk: Buffer) => {
	buffer += decoder.decode(chunk, { stream: true });
	const lines = buffer.split("\n");
	buffer = lines.pop()!;
	for (const line of lines) {
		if (!line.trim()) continue;
		try {
			const msg = JSON.parse(line);
			await handleMessage(msg);
		} catch (err) {
			respond({ type: "error", error: String(err) });
		}
	}
});

process.stdin.on("end", () => process.exit(0));
