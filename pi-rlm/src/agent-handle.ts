import { Agent, type AgentEvent, type StreamFn, type ThinkingLevel } from "@mariozechner/pi-agent-core";
import type { Model } from "@mariozechner/pi-ai";
import { EvalRuntime } from "./eval-runtime.js";
import { createEvalTool, type EvalToolOptions } from "./eval-tool.js";
import { createTaggedOnEvent } from "./event-tagging.js";

export interface AgentHandleOptions {
	evalToolOptions?: EvalToolOptions;
	/** Event listener for agent lifecycle events */
	onEvent?: (event: AgentEvent) => void;
	/** Depth level for debug event tagging. */
	eventDepth?: number;
	/** Resolves an API key dynamically for each LLM call (e.g. OAuth token refresh). */
	getApiKey?: (provider: string) => Promise<string | undefined> | string | undefined;
	/** Custom stream function to override default streaming behavior (e.g. inject maxTokens). */
	streamFn?: StreamFn;
}

/**
 * AgentHandle wraps a pi-mono Agent with an EvalRuntime + eval tool.
 * Provides the spawn/call pattern from the Agentica SDK.
 */
export class AgentHandle {
	readonly agent: Agent;
	readonly runtime: EvalRuntime;
	private resolved = false;
	private resolvedValue: unknown = undefined;
	private taggedOnEvent: ReturnType<typeof createTaggedOnEvent>;

	constructor(
		model: Model<any>,
		thinkingLevel: ThinkingLevel,
		systemPrompt: string,
		scope: Record<string, unknown>,
		options: AgentHandleOptions = {},
	) {
		// Wrap onEvent with depth/label metadata
		const depth = options.eventDepth ?? 0;
		this.taggedOnEvent = createTaggedOnEvent(options.onEvent, {
			depth,
			label: "Sub-agent",
			systemPrompt,
		});

		// Create eval runtime with scope + resolve function
		this.runtime = new EvalRuntime({
			...scope,
			resolve: (value: unknown) => {
				if (this.resolved) return; // no-op after first resolve
				this.resolved = true;
				this.resolvedValue = value;
			},
		});

		// Create the eval tool
		const evalTool = createEvalTool(this.runtime, {
			...options.evalToolOptions,
			onResult: (result) => {
				// Chain the user's onResult if provided
				let status = options.evalToolOptions?.onResult?.(result);

				// After resolve, signal the LLM to stop
				if (this.resolved) {
					const resolveMsg = "Resolved. Do not make further eval calls.";
					status = status ? `${status}\n${resolveMsg}` : resolveMsg;
				}

				return status;
			},
		});

		// Create the pi-mono Agent
		this.agent = new Agent({
			initialState: {
				systemPrompt,
				model,
				thinkingLevel,
				tools: [evalTool],
			},
			getApiKey: options.getApiKey,
			streamFn: options.streamFn,
		});

		// Subscribe to tagged events if listener provided
		if (this.taggedOnEvent.handler) {
			this.agent.subscribe(this.taggedOnEvent.handler);
		}
	}

	/**
	 * Call this agent with a task. Injects objects into the eval scope,
	 * builds a user message, and runs the agent loop.
	 *
	 * @param task - The task description for the agent
	 * @param objects - Objects to inject into the eval scope for this call
	 * @returns The resolved value, or undefined if the agent stopped without resolving
	 */
	async call<T = unknown>(
		task: string,
		objects: Record<string, unknown> = {},
	): Promise<T | undefined> {
		// Inject objects into eval scope
		if (Object.keys(objects).length > 0) {
			this.runtime.injectScope(objects);
		}

		// Build user message describing available objects
		let message = task;
		const objectNames = Object.keys(objects);
		if (objectNames.length > 0) {
			message += `\n\nThe following objects are available in your eval scope: ${objectNames.map((n) => `\`${n}\``).join(", ")}`;
		}

		// Reset resolve state for this call
		this.resolved = false;
		this.resolvedValue = undefined;

		// Set user message on tagged event wrapper so agent_start includes it
		this.taggedOnEvent.setUserMessage(message);

		// Run the agent loop
		await this.agent.prompt(message);

		return this.resolvedValue as T | undefined;
	}
}

export interface SpawnAgentOptions {
	/** Maximum agent spawn depth to prevent infinite recursion. Default: 5 */
	maxDepth?: number;
	/** Current depth (internal, set by recursive spawns) */
	currentDepth?: number;
	/** Maximum total agents across all depths (flat pool). */
	maxAgents?: number;
	/** Shared counter for flat agent pool (internal, same object passed to all levels). */
	agentCounter?: { count: number };
	/** Eval tool options passed to all spawned agents */
	evalToolOptions?: EvalToolOptions;
	/** Event listener for agent lifecycle events */
	onEvent?: (event: AgentEvent) => void;
	/** Depth level for debug event tagging. */
	eventDepth?: number;
	/** Resolves an API key dynamically for each LLM call (e.g. OAuth token refresh). */
	getApiKey?: (provider: string) => Promise<string | undefined> | string | undefined;
	/** Custom stream function to override default streaming behavior (e.g. inject maxTokens). */
	streamFn?: StreamFn;
}

/**
 * Creates a `spawnAgent` function that can be injected into eval scopes.
 * The returned function creates AgentHandles with the given model and base scope.
 * The spawnAgent function itself is included in the scope, enabling recursive spawning.
 */
export function createSpawnAgent(
	model: Model<any>,
	thinkingLevel: ThinkingLevel,
	baseScope: Record<string, unknown>,
	options: SpawnAgentOptions = {},
): (systemPrompt?: string) => Promise<AgentHandle> {
	const maxDepth = options.maxDepth ?? 5;
	const currentDepth = options.currentDepth ?? 0;
	const { agentCounter } = options;

	const eventDepth = options.eventDepth ?? 0;

	const spawnAgent = async (systemPrompt?: string): Promise<AgentHandle> => {
		if (currentDepth >= maxDepth) {
			throw new Error(
				`Maximum agent spawn depth (${maxDepth}) reached. Cannot spawn more nested agents.`,
			);
		}

		if (agentCounter && options.maxAgents != null && agentCounter.count >= options.maxAgents) {
			throw new Error(
				`Maximum total number of agents (${options.maxAgents}) reached. Cannot spawn more agents.`,
			);
		}

		if (agentCounter) {
			agentCounter.count++;
		}

		// Create a child spawnAgent with incremented depth (same counter object shared)
		const childSpawnAgent = createSpawnAgent(model, thinkingLevel, baseScope, {
			...options,
			currentDepth: currentDepth + 1,
			eventDepth: eventDepth + 1,
		});

		// Build scope for the new agent: base scope + spawnAgent itself
		const scope: Record<string, unknown> = {
			...baseScope,
			spawnAgent: childSpawnAgent,
		};

		return new AgentHandle(
			model,
			thinkingLevel,
			systemPrompt || "You are a helpful sub-agent. Use the eval tool to execute code and accomplish your task. Call resolve(value) when done.",
			scope,
			{
				evalToolOptions: options.evalToolOptions,
				onEvent: options.onEvent,
				eventDepth,
				getApiKey: options.getApiKey,
				streamFn: options.streamFn,
			},
		);
	};

	return spawnAgent;
}
