/**
 * Code-block stream wrapper — uses ```js blocks instead of formal tool calling.
 *
 * This module handles the full round-trip:
 *
 * 1. **Outbound (context → LLM):** Transforms conversation history so the model
 *    sees text with ```js blocks + user messages with "REPL output: ..." results,
 *    instead of tool_calls + tool results.
 * 2. **Inbound (LLM → agent loop):** Parses ```js blocks from the model's
 *    text response and re-emits them as synthetic toolcall events.
 *
 * Works with any model/provider (Claude, OpenAI, vLLM, etc.) — uses the generic
 * streamSimple which dispatches based on the model's API type.
 *
 * The rest of the stack (agent loop, eval tool, orchestrator, logging, viewer)
 * sees standard pi-agent-core events and works unchanged.
 */

import {
	createAssistantMessageEventStream,
	streamSimple,
} from "@mariozechner/pi-ai";
import type {
	AssistantMessage,
	Context,
	Message,
	Model,
	SimpleStreamOptions,
	ToolCall,
	ToolResultMessage,
	UserMessage,
} from "@mariozechner/pi-ai";
import type { StreamFn } from "@mariozechner/pi-agent-core";
import type { Adapter } from "./domain-adapter.js";

// ── Parsing ──────────────────────────────────────────────────────────────────

interface TextSegment {
	type: "text";
	text: string;
}
interface CodeSegment {
	type: "code";
	code: string;
}
type Segment = TextSegment | CodeSegment;

/** Parse text into segments: alternating prose and ```js/```repl code blocks. */
export function parseReplBlocks(text: string): Segment[] {
	const segments: Segment[] = [];
	const pattern = /```(?:repl|js)\s*\n(.*?)\n```/gs;
	let lastIndex = 0;

	for (const match of text.matchAll(pattern)) {
		const matchStart = match.index!;
		if (matchStart > lastIndex) {
			const before = text.slice(lastIndex, matchStart).trim();
			if (before) {
				segments.push({ type: "text", text: before });
			}
		}
		segments.push({ type: "code", code: match[1].trim() });
		lastIndex = matchStart + match[0].length;
	}

	if (lastIndex < text.length) {
		const after = text.slice(lastIndex).trim();
		if (after) {
			segments.push({ type: "text", text: after });
		}
	}

	return segments;
}

// ── Conversation history transformation ──────────────────────────────────────

/**
 * Reconstruct an assistant message as plain text with ```js blocks.
 * The agent loop stores: [TextContent, ToolCall, TextContent, ToolCall, ...]
 * We convert back to: "prose\n```js\ncode\n```\nprose\n```js\ncode\n```"
 */
function assistantToReplText(msg: AssistantMessage): string {
	const parts: string[] = [];
	for (const block of msg.content) {
		if (block.type === "text") {
			parts.push(block.text);
		} else if (block.type === "toolCall" && block.name === "eval") {
			const code = block.arguments?.code ?? "";
			parts.push("```js\n" + code + "\n```");
		}
		// Skip thinking blocks — model doesn't need to see its own thinking
	}
	return parts.join("\n\n");
}

/**
 * Convert a tool result into a user message:
 * "Code executed:\n```js\n<code>\n```\n\nREPL output:\n<result>"
 */
function toolResultToUserMessage(
	result: ToolResultMessage,
	codeMap: Map<string, string>,
): UserMessage {
	const code = codeMap.get(result.toolCallId) ?? "(unknown code)";
	const output = result.content
		.filter((c) => c.type === "text")
		.map((c) => c.text)
		.join("\n");

	return {
		role: "user",
		content:
			`Code executed:\n\`\`\`js\n${code}\n\`\`\`\n\nREPL output:\n${output || "(no output)"}`,
		timestamp: result.timestamp,
	};
}

/**
 * Transform the message history from pi-agent-core format to code-block text format.
 *
 * Input:  [User, Assistant(text+toolCalls), ToolResult, User, Assistant, ...]
 * Output: [User, Assistant(text with ```js), User(REPL output), User, Assistant, ...]
 */
export function transformMessagesForRepl(messages: Message[]): Message[] {
	// First pass: build a map of toolCallId → code from assistant messages
	const codeMap = new Map<string, string>();
	for (const msg of messages) {
		if (msg.role === "assistant") {
			for (const block of msg.content) {
				if (block.type === "toolCall" && block.name === "eval") {
					codeMap.set(block.id, block.arguments?.code ?? "");
				}
			}
		}
	}

	// Second pass: transform messages
	const result: Message[] = [];
	for (const msg of messages) {
		if (msg.role === "assistant") {
			const text = assistantToReplText(msg);
			if (text.trim()) {
				result.push({
					role: "assistant",
					content: [{ type: "text", text }],
					api: msg.api,
					provider: msg.provider,
					model: msg.model,
					usage: msg.usage,
					stopReason: msg.stopReason,
					timestamp: msg.timestamp,
				});
			}
		} else if (msg.role === "toolResult") {
			result.push(toolResultToUserMessage(msg, codeMap));
		} else {
			// User messages pass through unchanged
			result.push(msg);
		}
	}

	return result;
}

// ── Stream function ──────────────────────────────────────────────────────────

let toolCallCounter = 0;
function genToolCallId(): string {
	return `repl_${Date.now()}_${toolCallCounter++}`;
}

export interface CodeBlockStreamOptions {
	/** Number of top logprobs per token to request (0 = disabled). Only works with vLLM. */
	topLogprobs?: number;
}

/**
 * Direct fetch to a vLLM/OpenAI-compatible server, returning full text + logprobs.
 * Used when topLogprobs > 0 (training mode). Non-streaming since we collect everything anyway.
 */
async function fetchWithLogprobs(
	model: Model<any>,
	context: Context,
	options: SimpleStreamOptions | undefined,
	topLogprobs: number,
): Promise<{ text: string; usage: AssistantMessage["usage"]; logprobs: unknown[] }> {
	const messages: { role: string; content: string }[] = [];
	if (context.systemPrompt) {
		messages.push({ role: "system", content: context.systemPrompt });
	}
	for (const msg of context.messages) {
		if (msg.role === "user") {
			const text = typeof msg.content === "string"
				? msg.content
				: msg.content.filter((c) => c.type === "text").map((c) => (c as { text: string }).text).join("");
			messages.push({ role: "user", content: text });
		} else if (msg.role === "assistant") {
			const text = msg.content.filter((c) => c.type === "text").map((c) => (c as { text: string }).text).join("");
			messages.push({ role: "assistant", content: text });
		}
	}

	const body: Record<string, unknown> = {
		model: model.id,
		messages,
		temperature: options?.temperature,
		max_tokens: options?.maxTokens ?? model.maxTokens,
		logprobs: true,
		top_logprobs: topLogprobs,
		stream: false,
	};

	// Let onPayload inject extra params (e.g. top_p)
	options?.onPayload?.(body);
	// Ensure logprobs params aren't overridden
	body.logprobs = true;
	body.top_logprobs = topLogprobs;
	body.stream = false;

	const response = await fetch(`${model.baseUrl}/chat/completions`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(body),
		signal: options?.signal,
	});

	if (!response.ok) {
		const errorText = await response.text();
		throw new Error(`vLLM API error (${response.status}): ${errorText}`);
	}

	const data = (await response.json()) as {
		choices: Array<{ message: { content?: string }; logprobs?: { content?: unknown[] } }>;
		usage?: { prompt_tokens?: number; completion_tokens?: number };
	};
	const choice = data.choices[0];
	const promptTokens = data.usage?.prompt_tokens ?? 0;
	const completionTokens = data.usage?.completion_tokens ?? 0;

	return {
		text: choice.message.content ?? "",
		usage: {
			input: promptTokens,
			output: completionTokens,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: promptTokens + completionTokens,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		logprobs: choice.logprobs?.content ?? [],
	};
}

/**
 * Create a StreamFn that uses ```js code blocks instead of tool calling.
 *
 * Works with any model/provider — uses the generic streamSimple which
 * dispatches to the right provider (Anthropic, OpenAI, etc.) based on
 * the model's API type. This lets you A/B test tool calling vs code blocks
 * on the same model.
 *
 * Handles the full round-trip:
 * - Outbound: strips tools, transforms history to code-block text format
 * - Inbound: parses ```js blocks into synthetic toolcall events
 *
 * When `topLogprobs > 0`, bypasses streamSimple and calls the vLLM API directly
 * to capture per-token logprobs. The logprobs are attached to the AssistantMessage
 * as an extra `logprobs` property.
 *
 * Usage:
 *   const streamFn = createCodeBlockStreamFn();
 *   new Orchestrator({ model, streamFn, generateSystemPrompt: generateCodeBlockSystemPrompt, ... });
 */
export function createCodeBlockStreamFn(opts?: CodeBlockStreamOptions): StreamFn {
	const topLogprobs = opts?.topLogprobs ?? 0;

	return (model: Model<any>, context: Context, options?: SimpleStreamOptions) => {
		// Transform context: strip tools + convert history to code-block text format
		const transformedContext: Context = {
			systemPrompt: context.systemPrompt,
			messages: transformMessagesForRepl(context.messages),
			tools: undefined, // model uses text blocks, not function calling
		};

		const output = createAssistantMessageEventStream();

		(async () => {
			try {
				let fullText = "";
				let usage: AssistantMessage["usage"];
				let logprobsData: unknown[] | undefined;

				if (topLogprobs > 0) {
					// Direct API call to capture logprobs (training mode)
					const result = await fetchWithLogprobs(model, transformedContext, options, topLogprobs);
					fullText = result.text;
					usage = result.usage;
					logprobsData = result.logprobs;
				} else {
					// Use generic streamSimple — works with any provider
					const upstream = streamSimple(model, transformedContext, options);

					let finalMessage: AssistantMessage | null = null;

					for await (const event of upstream) {
						if (event.type === "text_delta") {
							fullText += event.delta;
						} else if (event.type === "done") {
							finalMessage = event.message;
						} else if (event.type === "error") {
							output.push(event);
							output.end();
							return;
						}
					}

					if (!finalMessage) {
						throw new Error("No final message from upstream");
					}
					usage = finalMessage.usage;
				}

				// Parse text into segments
				const segments = parseReplBlocks(fullText);

				// Build output message with proper content blocks
				const outMessage: AssistantMessage = {
					role: "assistant",
					content: [],
					api: model.api,
					provider: model.provider,
					model: model.id,
					usage,
					stopReason: segments.some((s) => s.type === "code") ? "toolUse" : "stop",
					timestamp: Date.now(),
				};

				// Attach logprobs as extra property (survives in agent's message history)
				if (logprobsData && logprobsData.length > 0) {
					(outMessage as any).logprobs = logprobsData;
				}

				output.push({ type: "start", partial: outMessage });

				let contentIndex = 0;
				for (const segment of segments) {
					if (segment.type === "text") {
						const textContent = { type: "text" as const, text: segment.text };
						outMessage.content.push(textContent);
						output.push({ type: "text_start", contentIndex, partial: outMessage });
						output.push({ type: "text_delta", contentIndex, delta: segment.text, partial: outMessage });
						output.push({ type: "text_end", contentIndex, content: segment.text, partial: outMessage });
						contentIndex++;
					} else {
						const toolCall: ToolCall = {
							type: "toolCall",
							id: genToolCallId(),
							name: "eval",
							arguments: { code: segment.code },
						};
						outMessage.content.push(toolCall);
						output.push({ type: "toolcall_start", contentIndex, partial: outMessage });
						output.push({
							type: "toolcall_delta",
							contentIndex,
							delta: JSON.stringify(toolCall.arguments),
							partial: outMessage,
						});
						output.push({ type: "toolcall_end", contentIndex, toolCall, partial: outMessage });
						contentIndex++;
					}
				}

				output.push({
					type: "done",
					reason: outMessage.stopReason as "stop" | "toolUse",
					message: outMessage,
				});
				output.end();
			} catch (error) {
				const errorMessage: AssistantMessage = {
					role: "assistant",
					content: [],
					api: model.api,
					provider: model.provider,
					model: model.id,
					usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
					stopReason: "error",
					errorMessage: error instanceof Error ? error.message : String(error),
					timestamp: Date.now(),
				};
				output.push({ type: "error", reason: "error", error: errorMessage });
				output.end();
			}
		})();

		return output;
	};
}

// ── System prompt ────────────────────────────────────────────────────────────

/**
 * Generate a system prompt for code-block format.
 *
 * Same domain content as the standard generateSystemPrompt(), but instructs
 * the model to use ```js code blocks instead of the eval tool.
 */
export function generateCodeBlockSystemPrompt(domain: Adapter): string {
	return `${domain.premise}

## Code Execution Environment

You have a persistent JavaScript REPL. To execute code, wrap it in a \`\`\`js code block:

\`\`\`js
console.log("hello");
\`\`\`

After each code block, you'll see the execution result. You can then write more code blocks to continue.

### Variable Persistence
All declarations (\`let\`, \`const\`, \`var\`) persist across code blocks. You can re-declare variables freely — the latest value wins.

### Top-level await
You can use \`await\` directly in your code.

### Console output
Use \`console.log()\` to print values. Output is captured and returned to you.

### Important
- Write ONE code block at a time, then wait for the result before continuing.
- Do NOT put multiple \`\`\`js blocks in a single response — write one, see the result, then write the next.

## Sub-agents

\`spawnAgent(systemPrompt?)\` creates a sub-agent. Returns synchronously — the async work happens in \`.call()\`.
- No args: auto-configured with ${domain.name} context and reference.
- With custom prompt: uses exactly what you pass. Include \`DOMAIN_REFERENCE\` for domain context.

\`agent.call(task, objects?)\` sends a task to the agent. Returns the submitted value.

\`\`\`js
// One-step (preferred for one-off tasks)
const result = await spawnAgent().call("Do something", { data });

// Parallel exploration
const [r1, r2] = await Promise.all([
  spawnAgent().call("Hypothesis A", { data }),
  spawnAgent().call("Hypothesis B", { data }),
]);
\`\`\`

Sub-agents have access to \`spawnAgent\` for further delegation.

**submit(value)** — signals completion and returns a value.

## ${domain.name} Reference

${domain.reference}`;
}

// ── Aliases (backwards compat) ───────────────────────────────────────────────

/** @deprecated Use createCodeBlockStreamFn */
export const createReplStreamFn = createCodeBlockStreamFn;

/** @deprecated Use generateCodeBlockSystemPrompt */
export const generateReplSystemPrompt = generateCodeBlockSystemPrompt;

// ── Model factory ────────────────────────────────────────────────────────────

/**
 * Create a Model object for a vLLM-served model.
 */
export function createVllmModel(options: {
	id: string;
	name?: string;
	baseUrl: string;
	contextWindow?: number;
	maxTokens?: number;
}): Model<"openai-completions"> {
	return {
		id: options.id,
		name: options.name ?? options.id,
		api: "openai-completions",
		provider: "openai", // vLLM speaks OpenAI protocol
		baseUrl: options.baseUrl,
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: options.contextWindow ?? 32768,
		maxTokens: options.maxTokens ?? 8192,
		compat: {
			supportsStore: false,
			supportsDeveloperRole: false,
			supportsReasoningEffort: false,
			maxTokensField: "max_tokens",
			thinkingFormat: "qwen",
		},
	};
}
