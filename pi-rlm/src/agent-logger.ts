import { mkdirSync, appendFileSync, writeFileSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type { AgentMessage } from "@mariozechner/pi-agent-core";
import type { UsageTracker } from "./usage-tracker.js";

export interface AgentLoggerOptions {
	/** Directory to write the log file in. */
	logDir: string;
	/** Agent index (0 = root, 1+ = sub-agents). */
	agentIndex: number;
	/** Arbitrary metadata written into the session header. */
	metadata?: Record<string, unknown>;
}

/**
 * Logs AgentMessage entries for a single agent to a JSONL file.
 *
 * Format:
 *   Line 1: {"type":"session","version":1,"agentIndex":N,"timestamp":"...","metadata":{...}}
 *   Lines 2+: {"type":"message","ts":<unix_seconds>,"message":<AgentMessage>}
 *   Last line: {"type":"session_end","ts":...,"usage":{...}}
 */
export class AgentLogger {
	private filePath: string;
	private lastLoggedIndex = 0;

	constructor(options: AgentLoggerOptions) {
		mkdirSync(options.logDir, { recursive: true });
		this.filePath = join(options.logDir, `agent-${options.agentIndex}.jsonl`);

		// Write session header
		writeFileSync(this.filePath, "");
		this.writeLine({
			type: "session",
			version: 1,
			agentIndex: options.agentIndex,
			timestamp: new Date().toISOString(),
			metadata: options.metadata ?? {},
		});
	}

	/**
	 * Snapshot new messages from the agent's state.messages array.
	 * Call after each agent.prompt() returns.
	 */
	snapshotMessages(messages: AgentMessage[]): void {
		const newMessages = messages.slice(this.lastLoggedIndex);
		for (const message of newMessages) {
			this.writeLine({
				type: "message",
				ts: now(),
				message,
			});
		}
		this.lastLoggedIndex = messages.length;
	}

	/** Close the log file with optional usage summary. */
	close(usageTracker?: UsageTracker): void {
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
	}

	/** Path to the log file. */
	get logPath(): string {
		return this.filePath;
	}

	/**
	 * Load AgentMessage entries from a JSONL log file.
	 * Returns all messages from `type: "message"` entries, skipping
	 * session headers and session_end lines.
	 */
	static loadMessages(filePath: string): AgentMessage[] {
		const content = readFileSync(filePath, "utf-8");
		const messages: AgentMessage[] = [];
		for (const line of content.split("\n")) {
			if (!line) continue;
			const entry = JSON.parse(line);
			if (entry.type === "message") {
				messages.push(entry.message as AgentMessage);
			}
		}
		return messages;
	}

	/**
	 * Load the session header metadata from a JSONL log file.
	 * Returns the metadata object from the first `type: "session"` entry.
	 */
	static loadSessionMetadata(filePath: string): Record<string, unknown> {
		const content = readFileSync(filePath, "utf-8");
		for (const line of content.split("\n")) {
			if (!line) continue;
			const entry = JSON.parse(line);
			if (entry.type === "session") {
				return (entry.metadata ?? {}) as Record<string, unknown>;
			}
		}
		return {};
	}

	private writeLine(obj: Record<string, unknown>): void {
		appendFileSync(this.filePath, JSON.stringify(obj) + "\n");
	}
}

function now(): number {
	return Math.floor(Date.now() / 1000);
}
