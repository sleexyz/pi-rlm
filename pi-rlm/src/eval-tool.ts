import { Type, type Static } from "@sinclair/typebox";
import type { AgentTool, AgentToolResult } from "@mariozechner/pi-agent-core";
import { EvalRuntime } from "./eval-runtime.js";
import type { EvalResult } from "./types.js";

const EvalParams = Type.Object({
	code: Type.String({ description: "JavaScript/TypeScript code to execute in the persistent eval environment." }),
});

type EvalParams = Static<typeof EvalParams>;

export interface EvalToolOptions {
	maxOutputChars?: number;
	/** Called after each eval with the result, before returning to the LLM. Returns optional status text to append. */
	onResult?: (result: EvalResult) => string | undefined;
}

/**
 * Creates an AgentTool that wraps an EvalRuntime.
 * Single tool named "eval" with a `{ code: string }` parameter.
 */
export function createEvalTool(
	runtime: EvalRuntime,
	options: EvalToolOptions = {},
): AgentTool<typeof EvalParams, EvalResult> {
	const maxOutputChars = options.maxOutputChars ?? 20_000;

	return {
		name: "eval",
		description:
			"Execute JavaScript code in a persistent environment. " +
			"Variables declared with `var` persist across calls. " +
			"Top-level `await` is supported. " +
			"Use `console.log()` for output. " +
			"All injected scope objects (spawnAgent, memories, domain objects) are available as globals.",
		label: "Eval",
		parameters: EvalParams,
		execute: async (
			_toolCallId: string,
			params: EvalParams,
		): Promise<AgentToolResult<EvalResult>> => {
			const result = await runtime.eval(params.code);

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

			// Append optional status line from callback
			if (options.onResult) {
				const status = options.onResult(result);
				if (status) {
					if (output) output += "\n";
					output += status;
				}
			}

			if (!output) {
				output = "(no output)";
			}

			// Truncate if needed
			if (output.length > maxOutputChars) {
				output = output.slice(0, maxOutputChars) + `\n... (truncated, ${output.length} total chars)`;
			}

			return {
				content: [{ type: "text", text: output }],
				details: result,
			};
		},
	};
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
