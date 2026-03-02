import { mkdirSync, appendFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { AssistantMessage } from "@mariozechner/pi-ai";
import type { TaggedAgentEvent } from "./event-tagging.js";
import type { UsageTracker } from "./usage-tracker.js";

export interface SessionLoggerOptions {
	/** Log directory. Default: "./logs" */
	logDir?: string;
	/** Session ID. Default: auto-generated timestamp. */
	sessionId?: string;
	/** Arbitrary metadata spread into the session_start event. */
	metadata?: Record<string, unknown>;
}

/**
 * Logs tagged agent events to a JSONL file for post-hoc analysis.
 */
export class SessionLogger {
	private filePath: string;
	private sessionId: string;
	private metadata: Record<string, unknown>;
	private started = false;

	constructor(options: SessionLoggerOptions = {}) {
		const logDir = options.logDir ?? "./logs";
		this.sessionId = options.sessionId ?? generateSessionId();
		this.metadata = options.metadata ?? {};
		this.filePath = join(logDir, `${this.sessionId}.jsonl`);

		// Ensure log directory exists
		mkdirSync(logDir, { recursive: true });
	}

	/** Start logging (creates file, writes header). */
	start(): void {
		if (this.started) return;
		this.started = true;

		writeFileSync(this.filePath, "");
		this.writeLine({
			type: "session_start",
			sessionId: this.sessionId,
			ts: now(),
			...this.metadata,
		});
	}

	/** Log a tagged event. */
	logEvent(event: TaggedAgentEvent): void {
		if (!this.started) return;

		const base = {
			type: event.type,
			agentId: event._agentId,
			depth: event._depth,
			ts: now(),
		};

		switch (event.type) {
			case "agent_start":
				this.writeLine({
					...base,
					label: event._label,
					...(event._systemPrompt ? { systemPrompt: event._systemPrompt } : {}),
					...(event._userMessage ? { userMessage: event._userMessage } : {}),
				});
				break;

			case "message_start":
			case "message_update":
				this.writeLine(base);
				break;

			case "message_end": {
				const msg = event.message as AssistantMessage;
				const usage = msg?.usage;
				const content = msg?.content;
				this.writeLine({
					...base,
					...(content ? { content } : {}),
					...(usage
						? {
								usage: {
									input: usage.input,
									output: usage.output,
									totalTokens: usage.totalTokens,
									cost: usage.cost.total,
								},
							}
						: {}),
				});
				break;
			}

			case "tool_execution_start":
				this.writeLine({
					...base,
					toolName: event.toolName,
					...(event.args?.code ? { code: event.args.code } : {}),
				});
				break;

			case "tool_execution_end":
				this.writeLine({
					...base,
					...(event.result != null ? { result: event.result } : {}),
				});
				break;

			case "agent_end":
				this.writeLine(base);
				break;

			default:
				// turn_start, turn_end, tool_execution_update — log type only
				this.writeLine(base);
				break;
		}
	}

	/** Close the log file. Returns the file path. */
	close(usageTracker?: UsageTracker): string {
		if (!this.started) {
			this.start();
		}

		const endLine: Record<string, unknown> = {
			type: "session_end",
			ts: now(),
		};

		if (usageTracker) {
			const total = usageTracker.totalUsage();
			endLine.usage = {
				totalTokens: total.totalTokens,
				totalCost: total.totalCost,
			};
		}

		this.writeLine(endLine);
		return this.filePath;
	}

	/** Path to the current log file. */
	get logPath(): string {
		return this.filePath;
	}

	private writeLine(obj: Record<string, unknown>): void {
		appendFileSync(this.filePath, JSON.stringify(obj) + "\n");
	}
}

function generateSessionId(): string {
	const d = new Date();
	const pad = (n: number, w = 2) => String(n).padStart(w, "0");
	return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
}

function now(): number {
	return Math.floor(Date.now() / 1000);
}
