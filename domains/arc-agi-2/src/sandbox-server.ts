/**
 * Long-lived sandbox server for RL training.
 *
 * Wraps EvalRuntime + createArcAdapter over a JSON-line stdin/stdout protocol.
 * One process per rollout instance — variables persist across eval calls.
 *
 * Protocol:
 *   → {"type":"init","task":{train:[...],test:[...]}}
 *   ← {"type":"ready"}
 *   → {"type":"eval","code":"..."}
 *   ← {"type":"result","stdout":"...","stderr":"...","returnValue":...,"error":...,"submitted":...}
 *   → {"type":"shutdown"}
 */

import { EvalRuntime } from "../../../pi-rlm/src/eval-runtime.ts";
import * as helpers from "./grid-helpers.ts";
import type { ArcTask } from "./types.ts";

let runtime: EvalRuntime | null = null;
let submitted: unknown = undefined;
let hasSubmitted = false;

function respond(obj: Record<string, unknown>) {
	process.stdout.write(JSON.stringify(obj) + "\n");
}

async function handleMessage(msg: { type: string; task?: ArcTask; code?: string }) {
	if (msg.type === "init") {
		const task = msg.task!;
		// Reset submission state
		submitted = undefined;
		hasSubmitted = false;
		// Create runtime with scope matching createArcAdapter — inlined to avoid
		// importing adapter.ts which has `import type { Adapter } from "pi-rlm"`
		// that bun can't resolve on Modal without node_modules.
		runtime = new EvalRuntime({
			// Data
			trainingExamples: task.train.map((ex) => ({ input: ex.input, output: ex.output })),
			testInputs: task.test.map((ex) => ex.input),
			// Grid helpers
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
			// Submission
			submit: (value: unknown) => {
				submitted = value;
				hasSubmitted = true;
				return value;
			},
		});
		respond({ type: "ready" });
	} else if (msg.type === "eval") {
		if (!runtime) {
			respond({ type: "result", stdout: "", stderr: "", returnValue: null, error: "Not initialized", submitted: null });
			return;
		}
		// Reset submission for this eval call
		submitted = undefined;
		hasSubmitted = false;
		const result = await runtime.eval(msg.code!, 30_000);
		respond({
			type: "result",
			stdout: result.stdout,
			stderr: result.stderr,
			returnValue: result.returnValue ?? null,
			error: result.error,
			submitted: hasSubmitted ? submitted : null,
		});
	} else if (msg.type === "shutdown") {
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
