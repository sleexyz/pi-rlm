import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { AgentLogger } from "./agent-logger.js";

export interface SessionDirOptions {
	/** Base log directory (e.g. "logs/arc-2"). */
	logDir: string;
	/** Session ID. Default: auto-generated timestamp. */
	sessionId?: string;
	/** Metadata included in every agent's session header. */
	metadata?: Record<string, unknown>;
}

/**
 * Manages a session directory containing per-agent JSONL log files.
 *
 * Directory structure:
 *   {logDir}/{sessionId}/
 *     agent-0.jsonl    ← root agent
 *     agent-1.jsonl    ← first sub-agent
 *     agent-2.jsonl    ← etc.
 */
export class SessionDir {
	readonly dirPath: string;
	readonly sessionId: string;
	private nextIndex = 0;
	private metadata: Record<string, unknown>;

	constructor(options: SessionDirOptions) {
		this.sessionId = options.sessionId ?? generateSessionId();
		this.metadata = options.metadata ?? {};
		this.dirPath = join(options.logDir, this.sessionId);
		mkdirSync(this.dirPath, { recursive: true });
	}

	/** Returns the next agent index (0, 1, 2, ...). */
	nextAgentIndex(): number {
		return this.nextIndex++;
	}

	/** Creates an AgentLogger for the next agent. */
	createAgentLogger(extraMetadata?: Record<string, unknown>): AgentLogger {
		const agentIndex = this.nextAgentIndex();
		return new AgentLogger({
			logDir: this.dirPath,
			agentIndex,
			metadata: { ...this.metadata, ...extraMetadata },
		});
	}
}

function generateSessionId(): string {
	const d = new Date();
	const pad = (n: number, w = 2) => String(n).padStart(w, "0");
	return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
}
