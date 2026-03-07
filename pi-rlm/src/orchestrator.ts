import { Agent, type AgentEvent, type StreamFn, type ThinkingLevel } from "@mariozechner/pi-agent-core";
import type { AssistantMessage, Model } from "@mariozechner/pi-ai";
import type { Adapter } from "./domain-adapter.js";
import { EvalRuntime } from "./eval-runtime.js";
import { createEvalTool, type EvalToolOptions } from "./eval-tool.js";
import { createSpawnAgent } from "./agent-handle.js";
import { generateSystemPrompt } from "./system-prompt.js";
import { createTaggedOnEvent, type TaggedAgentEvent } from "./event-tagging.js";
import { UsageTracker } from "./usage-tracker.js";
import type { SessionDir } from "./session-dir.js";
import { AgentLogger } from "./agent-logger.js";

export interface OrchestratorOptions {
	/** Model for the main orchestrator agent. */
	model: Model<any>;
	/** Model for sub-agents. Defaults to the main model. */
	subAgentModel?: Model<any>;
	/** Thinking level for all agents. Default: "off" */
	thinkingLevel?: ThinkingLevel;
	/** Thinking level for sub-agents. Defaults to thinkingLevel. */
	subAgentThinkingLevel?: ThinkingLevel;
	/** Adapter providing scope, prompts, and reference. */
	adapter: Adapter;
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
	/** Session directory for per-agent message logging. */
	sessionDir?: SessionDir;
	/** Custom system prompt generator (e.g. generateReplSystemPrompt for code-block models). */
	generateSystemPrompt?: (adapter: Adapter) => string;
}

/**
 * Thin shell that wires an Adapter into an eval-loop agent.
 *
 * Takes domain.scope, adds spawnAgent + submit + DOMAIN_REFERENCE,
 * assembles the system prompt from domain fields, and runs the agent loop.
 */
export class Orchestrator {
	private agent: Agent;
	private runtime: EvalRuntime;
	private adapter: Adapter;
	private resolved = false;
	private resolvedValue: unknown = undefined;
	private taggedOnEvent: ReturnType<typeof createTaggedOnEvent>;
	private usageTracker: UsageTracker;
	private agentLogger?: AgentLogger;

	constructor(options: OrchestratorOptions) {
		this.adapter = options.adapter;
		this.usageTracker = new UsageTracker();

		const thinkingLevel = options.thinkingLevel ?? "off";
		const subAgentModel = options.subAgentModel ?? options.model;
		const subAgentThinkingLevel = options.subAgentThinkingLevel ?? thinkingLevel;

		// Assemble system prompt from domain fields
		const promptGenerator = options.generateSystemPrompt ?? generateSystemPrompt;
		const systemPrompt = promptGenerator(this.adapter);

		// Wrap onEvent with depth=0 metadata + usage tracking
		const wrappedOnEvent = options.onEvent
			? (event: AgentEvent) => {
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

		// Base scope = domain scope + DOMAIN_REFERENCE
		const baseScope: Record<string, unknown> = {
			...this.adapter.scope,
			DOMAIN_REFERENCE: this.adapter.reference,
		};

		// Create root agent logger if sessionDir provided
		if (options.sessionDir) {
			this.agentLogger = options.sessionDir.createAgentLogger({ role: "root" });
		}

		// Create spawnAgent factory
		const agentCounter = options.maxAgents != null ? { count: 0 } : undefined;
		const spawnAgent = createSpawnAgent(subAgentModel, subAgentThinkingLevel, baseScope, {
			maxDepth: options.maxSpawnDepth ?? 5,
			maxAgents: options.maxAgents,
			agentCounter,
			evalToolOptions: {
				maxOutputChars: options.maxOutputChars,
				onResult: this.adapter.onEvalResult
					? () => this.adapter.onEvalResult!()
					: undefined,
			},
			onEvent: wrappedOnEvent,
			eventDepth: 1,
			getApiKey: options.getApiKey,
			streamFn: options.streamFn,
			defaultSubAgentPrompt: this.adapter.defaultSubAgentPrompt,
			domainReference: this.adapter.reference,
			sessionDir: options.sessionDir,
		});

		// Full scope: domain + framework primitives
		const fullScope: Record<string, unknown> = {
			...baseScope,
			spawnAgent,
			submit: (value: unknown) => {
				if (this.resolved) return;
				this.resolved = true;
				this.resolvedValue = value;
				// Stop the agent loop — value has been submitted, no more turns needed
				this.agent.abort();
			},
		};

		// Create eval runtime + tool
		this.runtime = new EvalRuntime(fullScope);

		const evalToolOptions: EvalToolOptions = {
			maxOutputChars: options.maxOutputChars,
			onResult: () => {
				const parts: string[] = [];

				if (this.adapter.onEvalResult) {
					const status = this.adapter.onEvalResult();
					if (status) parts.push(status);
				}

				if (this.resolved) {
					parts.push("Submitted. Do not make further eval calls.");
				}

				return parts.length > 0 ? parts.join("\n") : undefined;
			},
		};

		const evalTool = createEvalTool(this.runtime, evalToolOptions);

		// Create the agent
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

		// Incremental message logging: snapshot after each turn
		if (this.agentLogger) {
			const logger = this.agentLogger;
			this.agent.subscribe((event) => {
				if (event.type === "turn_end") {
					logger.snapshotMessages(this.agent.state.messages);
				}
			});
		}
	}

	/**
	 * Run the orchestrator with a task.
	 *
	 * @param task - The task description
	 * @param initialObjects - Optional objects to inject into scope before running
	 * @returns The submitted value, or undefined if not submitted
	 */
	async run(task: string, initialObjects?: Record<string, unknown>): Promise<unknown> {
		if (initialObjects) {
			this.runtime.injectScope(initialObjects);
		}

		this.resolved = false;
		this.resolvedValue = undefined;

		let message = task;
		if (initialObjects) {
			const names = Object.keys(initialObjects);
			if (names.length > 0) {
				message += `\n\nThe following objects are available in your eval scope: ${names.map((n) => `\`${n}\``).join(", ")}`;
			}
		}

		this.taggedOnEvent.setUserMessage(message);
		await this.agent.prompt(message);

		// Snapshot new messages for logging
		if (this.agentLogger) {
			this.agentLogger.snapshotMessages(this.agent.state.messages);
		}

		return this.resolvedValue;
	}

	/** Get the underlying pi-mono Agent for direct event access. */
	getAgent(): Agent {
		return this.agent;
	}

	/** Get the eval runtime for direct scope inspection. */
	getRuntime(): EvalRuntime {
		return this.runtime;
	}

	/** Get the usage tracker for token/cost reporting. */
	getUsageTracker(): UsageTracker {
		return this.usageTracker;
	}

	/** Get the root agent logger (if session logging is enabled). */
	getAgentLogger(): AgentLogger | undefined {
		return this.agentLogger;
	}

	/**
	 * Resume from an aborted session by restoring conversation state
	 * from a prior agent log file.
	 *
	 * @param agentLogPath - Path to agent-0.jsonl from the aborted session
	 * @param initialObjects - Optional objects to inject into scope
	 * @returns The submitted value, or undefined if not submitted
	 */
	async resume(agentLogPath: string, initialObjects?: Record<string, unknown>): Promise<unknown> {
		const messages = AgentLogger.loadMessages(agentLogPath);

		if (initialObjects) {
			this.runtime.injectScope(initialObjects);
		}

		this.resolved = false;
		this.resolvedValue = undefined;

		// Restore prior conversation state
		this.agent.replaceMessages(messages);

		// Update logger so it doesn't re-log restored messages
		if (this.agentLogger) {
			this.agentLogger.snapshotMessages(messages);
		}

		const resumeMessage =
			"Your previous session was interrupted. Your eval scope has been reset — re-define any variables or functions you need. Continue working on the task.";

		this.taggedOnEvent.setUserMessage(resumeMessage);
		await this.agent.prompt(resumeMessage);

		// Snapshot new messages for logging
		if (this.agentLogger) {
			this.agentLogger.snapshotMessages(this.agent.state.messages);
		}

		return this.resolvedValue;
	}
}
