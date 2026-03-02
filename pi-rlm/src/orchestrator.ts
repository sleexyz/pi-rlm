import { Agent, type AgentEvent, type StreamFn, type ThinkingLevel } from "@mariozechner/pi-agent-core";
import type { AssistantMessage, Model } from "@mariozechner/pi-ai";
import type { DomainAdapter } from "./domain-adapter.js";
import { EvalRuntime } from "./eval-runtime.js";
import { createEvalTool, type EvalToolOptions } from "./eval-tool.js";
import { createSpawnAgent } from "./agent-handle.js";
import { Memories } from "./memories.js";
import { generateSystemPrompt } from "./system-prompt.js";
import { createTaggedOnEvent, type TaggedAgentEvent } from "./event-tagging.js";
import { UsageTracker } from "./usage-tracker.js";
import { ActionHistory, createTrackedAction } from "./action-history.js";
import { createBudgetedAction } from "./budgeted-action.js";

export interface OrchestratorOptions {
	/** Model for the main orchestrator agent. */
	model: Model<any>;
	/** Model for sub-agents. Defaults to the main model. */
	subAgentModel?: Model<any>;
	/** Thinking level for all agents. Default: "off" */
	thinkingLevel?: ThinkingLevel;
	/** Thinking level for sub-agents. Defaults to thinkingLevel. */
	subAgentThinkingLevel?: ThinkingLevel;
	/** Domain adapter providing scope, prompts, and reference. */
	adapter: DomainAdapter;
	/** Maximum eval output characters. Default: 20000 */
	maxOutputChars?: number;
	/** Maximum agent spawn depth. Default: 5 */
	maxSpawnDepth?: number;
	/** Maximum total agents across all depths (flat pool). Default: unlimited */
	maxAgents?: number;
	/** Event listener for all agent events. */
	onEvent?: (event: AgentEvent) => void;
	/** Resolves an API key dynamically for each LLM call (e.g. OAuth token refresh). */
	getApiKey?: (provider: string) => Promise<string | undefined> | string | undefined;
	/** Custom stream function to override default streaming behavior (e.g. inject maxTokens). */
	streamFn?: StreamFn;
}

/**
 * Main orchestrator that ties everything together.
 *
 * Creates an eval runtime with full scope (spawnAgent, memories, domain objects),
 * wraps it as the single eval tool on a pi-mono Agent, and runs the agent loop.
 */
export class Orchestrator {
	private agent: Agent;
	private runtime: EvalRuntime;
	private memories: Memories;
	private adapter: DomainAdapter;
	private resolved = false;
	private resolvedValue: unknown = undefined;
	private rejected = false;
	private rejectionReason: string | undefined = undefined;
	private options: OrchestratorOptions;
	private taggedOnEvent: ReturnType<typeof createTaggedOnEvent>;
	private usageTracker: UsageTracker;

	constructor(options: OrchestratorOptions) {
		this.options = options;
		this.adapter = options.adapter;
		this.usageTracker = new UsageTracker();

		const thinkingLevel = options.thinkingLevel ?? "off";
		const subAgentModel = options.subAgentModel ?? options.model;
		const subAgentThinkingLevel = options.subAgentThinkingLevel ?? thinkingLevel;

		// Generate system prompt early so we can attach it to tagged events
		const systemPrompt = this.adapter.generateSystemPrompt?.() ?? generateSystemPrompt(this.adapter);

		// Wrap onEvent with depth=0 metadata for the orchestrator.
		// Also intercept tagged events for usage tracking.
		const wrappedOnEvent = options.onEvent
			? (event: AgentEvent) => {
					// Track usage from tagged message_end events
					if (event.type === "message_end") {
						const taggedEvent = event as TaggedAgentEvent;
						const msg = event.message as AssistantMessage;
						if (msg?.usage && taggedEvent._agentId != null) {
							this.usageTracker.recordUsage(taggedEvent._agentId, msg.usage);
						}
					}
					options.onEvent!(event);
				}
			: undefined;

		const tagged = createTaggedOnEvent(wrappedOnEvent, {
			depth: 0,
			label: "Orchestrator",
			systemPrompt,
		});
		this.taggedOnEvent = tagged;

		// Create shared memories (wrappedOnEvent so memory agents also track usage)
		this.memories = new Memories(subAgentModel, subAgentThinkingLevel, {
			onEvent: wrappedOnEvent,
			getApiKey: options.getApiKey,
		});

		// Get domain scope
		const domainScope = this.adapter.getScope();

		// Action history — wrap domain actions if actionNames specified
		const actionHistory = new ActionHistory();
		const trackedScope: Record<string, unknown> = { ...domainScope };
		if (this.adapter.actionNames) {
			for (const name of this.adapter.actionNames) {
				if (typeof trackedScope[name] === "function") {
					trackedScope[name] = createTrackedAction(
						trackedScope[name] as (...args: any[]) => any,
						name,
						actionHistory,
					);
				}
			}
		}

		// Build base scope for sub-agents (without spawnAgent — that's added by createSpawnAgent)
		const baseScope: Record<string, unknown> = {
			...trackedScope,
			memories: this.memories,
			DOMAIN_REFERENCE: this.adapter.reference,
		};

		// Create spawnAgent factory
		const agentCounter = options.maxAgents != null ? { count: 0 } : undefined;
		const spawnAgent = createSpawnAgent(subAgentModel, subAgentThinkingLevel, baseScope, {
			maxDepth: options.maxSpawnDepth ?? 5,
			maxAgents: options.maxAgents,
			agentCounter,
			evalToolOptions: {
				maxOutputChars: options.maxOutputChars,
				onResult: this.adapter.getStatus
					? () => this.adapter.getStatus!()
					: undefined,
			},
			onEvent: wrappedOnEvent,
			eventDepth: 1,
			getApiKey: options.getApiKey,
			streamFn: options.streamFn,
		});

		// Build the full orchestrator scope
		const orchestratorScope: Record<string, unknown> = {
			...baseScope,
			spawnAgent: spawnAgent,
			createBudgetedAction,
			history: actionHistory,
			resolve: (value: unknown) => {
				if (this.resolved || this.rejected) return;
				this.resolved = true;
				this.resolvedValue = value;
			},
			reject: (reason?: string) => {
				if (this.resolved || this.rejected) return;
				this.rejected = true;
				this.rejectionReason = reason ?? "Rejected without reason";
			},
		};

		// Create eval runtime
		this.runtime = new EvalRuntime(orchestratorScope);

		// Build eval tool options
		const evalToolOptions: EvalToolOptions = {
			maxOutputChars: options.maxOutputChars,
			onResult: (result) => {
				const parts: string[] = [];

				// Domain status
				if (this.adapter.getStatus) {
					const status = this.adapter.getStatus();
					if (status) parts.push(status);
				}

				// resolve/reject notification
				if (this.resolved) {
					parts.push("Resolved. Do not make further eval calls.");
				} else if (this.rejected) {
					parts.push(`Rejected: ${this.rejectionReason}. Do not make further eval calls.`);
				}

				return parts.length > 0 ? parts.join("\n") : undefined;
			},
		};

		// Create the eval tool
		const evalTool = createEvalTool(this.runtime, evalToolOptions);

		// Create the orchestrator agent
		this.agent = new Agent({
			initialState: {
				systemPrompt,
				model: options.model,
				thinkingLevel,
				tools: [evalTool],
			},
			getApiKey: options.getApiKey,
			streamFn: options.streamFn,
		});

		if (tagged.handler) {
			this.agent.subscribe(tagged.handler);
		}
	}

	/**
	 * Run the orchestrator with a task.
	 *
	 * @param task - The task description
	 * @param initialObjects - Optional objects to inject into scope before running
	 * @returns The resolved value, or undefined if rejected/not resolved
	 */
	async run(task: string, initialObjects?: Record<string, unknown>): Promise<unknown> {
		// Inject initial objects if provided
		if (initialObjects) {
			this.runtime.injectScope(initialObjects);
		}

		// Reset resolve/reject state
		this.resolved = false;
		this.resolvedValue = undefined;
		this.rejected = false;
		this.rejectionReason = undefined;

		// Build the user message
		let message = task;
		if (initialObjects) {
			const names = Object.keys(initialObjects);
			if (names.length > 0) {
				message += `\n\nThe following objects are available in your eval scope: ${names.map((n) => `\`${n}\``).join(", ")}`;
			}
		}

		// Set user message on tagged event wrapper so agent_start includes it
		this.taggedOnEvent.setUserMessage(message);

		// Run the agent loop
		await this.agent.prompt(message);

		if (this.rejected) {
			throw new Error(`Agent rejected: ${this.rejectionReason}`);
		}

		return this.resolvedValue;
	}

	/** Get the underlying pi-mono Agent for direct event access. */
	getAgent(): Agent {
		return this.agent;
	}

	/** Get the shared memories database. */
	getMemories(): Memories {
		return this.memories;
	}

	/** Get the eval runtime for direct scope inspection. */
	getRuntime(): EvalRuntime {
		return this.runtime;
	}

	/** Get the usage tracker for token/cost reporting. */
	getUsageTracker(): UsageTracker {
		return this.usageTracker;
	}
}
