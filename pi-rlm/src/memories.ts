import { Agent, type AgentEvent, type ThinkingLevel } from "@mariozechner/pi-agent-core";
import type { Model } from "@mariozechner/pi-ai";
import type { Memory } from "./types.js";
import { EvalRuntime } from "./eval-runtime.js";
import { createEvalTool } from "./eval-tool.js";
import { createTaggedOnEvent } from "./event-tagging.js";

export class MemoryQueryError extends Error {
	constructor(message: string) {
		super(message);
		this.name = "MemoryQueryError";
	}
}

const MEMORY_AGENT_PREMISE =
	"You retrieve information from a shared `memories` object. " +
	"You can call any of its methods: `memories.stack`, `memories.get(i)`, " +
	"`memories.add(summary, details)`, `memories.evict(i)`, `memories.summaries()`. " +
	"Answer the user's query by examining the memories. " +
	"If you cannot answer the query from the available memories, " +
	"call `resolve(new MemoryQueryError('reason'))`. " +
	"Otherwise, call `resolve(answer)` with the answer matching the requested return type.";

export interface MemoriesOptions {
	/** Event listener for the memory query agent */
	onEvent?: (event: AgentEvent) => void;
	/** Resolves an API key dynamically for each LLM call (e.g. OAuth token refresh). */
	getApiKey?: (provider: string) => Promise<string | undefined> | string | undefined;
}

/**
 * Shared memory database with LLM-powered natural language query.
 * Direct port of the Python Memories class from Agentica.
 */
export class Memories {
	readonly stack: Memory[] = [];
	private memoryAgent: Agent | null = null;
	private memoryRuntime: EvalRuntime | null = null;
	private lastSeen: number = 0;
	private model: Model<any>;
	private thinkingLevel: ThinkingLevel;
	private options: MemoriesOptions;

	constructor(model: Model<any>, thinkingLevel: ThinkingLevel = "off", options: MemoriesOptions = {}) {
		this.model = model;
		this.thinkingLevel = thinkingLevel;
		this.options = options;
	}

	/** Append an insight to the memory stack. */
	add(summary: string, details: string): void {
		this.stack.push({
			summary,
			details,
			timestamp: new Date(),
		});
	}

	/** Short summary of every stored memory, for a quick glance at what's known. */
	summaries(): string[] {
		return this.stack.map((m, i) => `[${i}] ${m.summary}`);
	}

	/** Retrieve a memory by index (supports negative indices). */
	get(i: number): Memory {
		const idx = i < 0 ? this.stack.length + i : i;
		if (idx < 0 || idx >= this.stack.length) {
			throw new RangeError(`Memory index ${i} out of range (0..${this.stack.length - 1})`);
		}
		return this.stack[idx];
	}

	/** Remove a memory by index (defaults to last). */
	evict(i: number = -1): Memory {
		const idx = i < 0 ? this.stack.length + i : i;
		if (idx < 0 || idx >= this.stack.length) {
			throw new RangeError(`Memory index ${i} out of range (0..${this.stack.length - 1})`);
		}
		return this.stack.splice(idx, 1)[0];
	}

	/**
	 * Natural language query over the memories using a dedicated LLM agent.
	 *
	 * @param question - The question to ask about the memories
	 * @returns The answer from the memory agent
	 * @throws MemoryQueryError if the query can't be answered
	 */
	async query<T = unknown>(question: string): Promise<T> {
		this.ensureMemoryAgent();

		const newCount = this.stack.length - this.lastSeen;
		this.lastSeen = this.stack.length;

		let preamble = "";
		if (newCount > 0) {
			preamble = `(${newCount} new memor${newCount === 1 ? "y" : "ies"} since your last query.) `;
		}

		// Track submit value
		let submitted = false;
		let submittedValue: unknown = undefined;
		this.memoryRuntime!.injectScope({
			submit: (value: unknown) => {
				if (submitted) return;
				submitted = true;
				submittedValue = value;
			},
			memories: this,
			stack: this.stack,
		});

		await this.memoryAgent!.prompt(
			`${preamble}Answer the following query by examining the memories.\n\nQuery: ${question}`,
		);

		if (!submitted) {
			throw new MemoryQueryError("Memory agent did not submit a response");
		}

		if (submittedValue instanceof MemoryQueryError) {
			throw submittedValue;
		}

		return submittedValue as T;
	}

	private ensureMemoryAgent(): void {
		if (this.memoryAgent) return;

		this.memoryRuntime = new EvalRuntime({
			memories: this,
			stack: this.stack,
			MemoryQueryError,
		});

		const evalTool = createEvalTool(this.memoryRuntime);

		this.memoryAgent = new Agent({
			initialState: {
				systemPrompt: MEMORY_AGENT_PREMISE,
				model: this.model,
				thinkingLevel: this.thinkingLevel,
				tools: [evalTool],
			},
			getApiKey: this.options.getApiKey,
		});

		// Wrap onEvent with Memory label for debug rendering
		const tagged = createTaggedOnEvent(this.options.onEvent, {
			depth: 1,
			label: "Memory",
			systemPrompt: MEMORY_AGENT_PREMISE,
		});

		if (tagged.handler) {
			this.memoryAgent.subscribe(tagged.handler);
		}
	}
}
