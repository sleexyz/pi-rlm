#!/usr/bin/env bun
/**
 * ARC Trace Viewer — API server for session logs and results.
 *
 * Usage: bun domains/arc-agi-2/src/viewer.ts [--port <n>]
 *
 * Scans logs/<domain>/ and results/<domain>/ at the repo root for all domains.
 * Serves JSON APIs consumed by the Vite frontend in domains/arc-agi-2/viewer/.
 */

import { readdirSync, readFileSync, existsSync, watch, openSync, readSync, closeSync, statSync, mkdirSync, type FSWatcher } from "node:fs";
import { join, basename } from "node:path";
import { createServer } from "node:http";
import { WebSocketServer, type WebSocket } from "ws";

// ── CLI args ────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const portIdx = args.indexOf("--port");
const port = portIdx !== -1 && args[portIdx + 1]
	? parseInt(args[portIdx + 1], 10)
	: process.env.PORT ? parseInt(process.env.PORT, 10) : 3334;

// Resolve relative to repo root — scan all subdirs of logs/ and results/
const repoRoot = join(import.meta.dirname, "../../..");
const LOGS_ROOT = join(repoRoot, "logs");
const RESULTS_ROOT = join(repoRoot, "results");

/** List subdirectories of a root dir (each subdir is a domain). */
function listDomains(root: string): string[] {
	if (!existsSync(root)) return [];
	return readdirSync(root, { withFileTypes: true })
		.filter((d) => d.isDirectory())
		.map((d) => d.name)
		.sort();
}

// ── Data helpers ────────────────────────────────────────────────────

interface SessionSummary {
	sessionId: string;
	filename: string;
	domain: string;
	ts: number;
	taskId?: string;
	split?: string;
	model?: string;
	turns: number;
	usage?: { totalTokens: number; totalCost: number };
	active?: boolean;
}

function loadAnnotations(logsDir: string): Record<string, { model?: string; taskId?: string }> {
	const annoPath = join(logsDir, "annotations.json");
	if (!existsSync(annoPath)) return {};
	try {
		const data = JSON.parse(readFileSync(annoPath, "utf-8"));
		return data.sessions ?? {};
	} catch {
		return {};
	}
}

function listSessions(): SessionSummary[] {
	const summaries: SessionSummary[] = [];

	for (const domain of listDomains(LOGS_ROOT)) {
		const logsDir = join(LOGS_ROOT, domain);
		const annotations = loadAnnotations(logsDir);

		// New format: session directories containing agent-*.jsonl files
		const entries = readdirSync(logsDir, { withFileTypes: true });
		for (const entry of entries) {
			if (entry.isDirectory()) {
				const sessionDir = join(logsDir, entry.name);
				const agent0Path = join(sessionDir, "agent-0.jsonl");
				if (!existsSync(agent0Path)) continue;

				try {
					const raw = readFileSync(agent0Path, "utf-8");
					const lines = raw.split("\n").filter((l) => l.trim());
					if (lines.length === 0) continue;

					const header = JSON.parse(lines[0]);
					const last = JSON.parse(lines[lines.length - 1]);
					const meta = header.metadata ?? {};

					// Count turns: each "user" role message = one turn
					let turns = 0;
					for (const line of lines) {
						try {
							const entry = JSON.parse(line);
							if (entry.type === "message" && entry.message?.role === "user") turns++;
						} catch { /* skip */ }
					}

					const sid = entry.name;
					const anno = annotations[sid];
					const summary: SessionSummary = {
						sessionId: sid,
						filename: entry.name,
						domain,
						ts: header.timestamp ? Math.floor(new Date(header.timestamp).getTime() / 1000) : 0,
						taskId: meta.taskId ?? anno?.taskId,
						split: meta.split,
						model: meta.model ?? anno?.model,
						turns,
					};

					if (last.type === "session_end" && last.usage) {
						summary.usage = {
							totalTokens: last.usage.totalTokens ?? 0,
							totalCost: last.usage.totalCost ?? 0,
						};
					}

					if (fileWatcher?.isActive(sid)) {
						summary.active = true;
					}

					summaries.push(summary);
				} catch {
					// skip malformed
				}
				continue;
			}

			// Legacy format: flat JSONL files
			if (entry.isFile() && entry.name.endsWith(".jsonl")) {
				try {
					const raw = readFileSync(join(logsDir, entry.name), "utf-8");
					const lines = raw.split("\n").filter((l) => l.trim());
					if (lines.length === 0) continue;

					const first = JSON.parse(lines[0]);
					const last = JSON.parse(lines[lines.length - 1]);

					let turns = 0;
					for (const line of lines) {
						if (line.includes('"turn_start"')) turns++;
					}

					const sid = first.sessionId ?? basename(entry.name, ".jsonl");
					const anno = annotations[sid];
					const summary: SessionSummary = {
						sessionId: sid,
						filename: entry.name,
						domain,
						ts: first.ts ?? 0,
						taskId: first.taskId ?? anno?.taskId,
						split: first.split,
						model: first.model ?? anno?.model,
						turns,
					};

					if (last.type === "session_end" && last.usage) {
						summary.usage = {
							totalTokens: last.usage.totalTokens ?? 0,
							totalCost: last.usage.totalCost ?? 0,
						};
					}

					if (fileWatcher?.isActive(sid)) {
						summary.active = true;
					}

					summaries.push(summary);
				} catch {
					// skip malformed files
				}
			}
		}
	}

	// Sort all sessions by timestamp descending
	summaries.sort((a, b) => b.ts - a.ts);
	return summaries;
}

const RENDERABLE_TYPES = new Set([
	"session_start",
	"agent_start",
	"turn_start",
	"message_update",
	"message_end",
	"tool_execution_start",
	"tool_execution_end",
	"agent_end",
	"session_end",
]);

/**
 * Convert per-agent message JSONL files into the event format the frontend expects.
 * Maps AgentMessage entries to the same event structure used by the old SessionLogger.
 */
function convertMessagesToEvents(sessionDir: string): object[] {
	const events: object[] = [];

	// Find all agent-*.jsonl files
	const agentFiles = readdirSync(sessionDir)
		.filter((f) => /^agent-\d+\.jsonl$/.test(f))
		.sort((a, b) => {
			const ai = parseInt(a.match(/\d+/)![0]);
			const bi = parseInt(b.match(/\d+/)![0]);
			return ai - bi;
		});

	for (const file of agentFiles) {
		const agentIndex = parseInt(file.match(/\d+/)![0]);
		const depth = agentIndex === 0 ? 0 : 1;
		const raw = readFileSync(join(sessionDir, file), "utf-8");
		const lines = raw.split("\n").filter((l) => l.trim());

		let header: any = null;
		let turnCount = 0;

		for (const line of lines) {
			let entry;
			try {
				entry = JSON.parse(line);
			} catch {
				continue;
			}

			if (entry.type === "session") {
				header = entry;
				// Emit session_start for agent-0 only
				if (agentIndex === 0) {
					events.push({
						type: "session_start",
						sessionId: basename(sessionDir),
						ts: header.timestamp ? Math.floor(new Date(header.timestamp).getTime() / 1000) : 0,
						...header.metadata,
					});
				}
				// Emit agent_start
				events.push({
					type: "agent_start",
					agentId: agentIndex,
					depth,
					label: agentIndex === 0 ? "Orchestrator" : "Sub-agent",
					ts: header.timestamp ? Math.floor(new Date(header.timestamp).getTime() / 1000) : 0,
				});
				continue;
			}

			if (entry.type === "message") {
				const msg = entry.message;
				if (!msg) continue;

				if (msg.role === "user") {
					turnCount++;
					if (turnCount > 1) {
						events.push({ type: "turn_start", depth, ts: entry.ts });
					}
					// First user message on agent-0 includes system prompt in agent_start
					if (turnCount === 1 && agentIndex === 0) {
						// Retroactively add systemPrompt and userMessage to the agent_start event
						const agentStart = events.find((e: any) => e.type === "agent_start" && e.agentId === agentIndex) as any;
						if (agentStart) {
							// Find system prompt from the agent state — it's not in the message log
							// The user message content is in the message
							const userContent = typeof msg.content === "string" ? msg.content : msg.content?.map((c: any) => c.text).join("\n") ?? "";
							agentStart.userMessage = userContent;
						}
					} else if (turnCount === 1 && agentIndex > 0) {
						const agentStart = events.find((e: any) => e.type === "agent_start" && e.agentId === agentIndex) as any;
						if (agentStart) {
							const userContent = typeof msg.content === "string" ? msg.content : msg.content?.map((c: any) => c.text).join("\n") ?? "";
							agentStart.userMessage = userContent;
						}
					}
					continue;
				}

				if (msg.role === "assistant") {
					// Emit message_end with content blocks (renders thinking, text, and tool_use)
					events.push({
						type: "message_end",
						agentId: agentIndex,
						depth,
						ts: entry.ts,
						content: msg.content,
						...(msg.usage ? {
							usage: {
								input: msg.usage.input,
								output: msg.usage.output,
								totalTokens: msg.usage.totalTokens,
								cost: msg.usage.cost?.total,
							},
						} : {}),
					});

					// Also emit tool_execution_start for each tool_use block
					// (the frontend renders these as EVAL code blocks)
					if (Array.isArray(msg.content)) {
						for (const block of msg.content) {
							if (block.type === "tool_use" || block.type === "toolCall") {
								const code = block.arguments?.code ?? block.input?.code ?? "";
								if (code) {
									events.push({
										type: "tool_execution_start",
										agentId: agentIndex,
										depth,
										ts: entry.ts,
										toolName: block.name ?? "eval",
										code,
									});
								}
							}
						}
					}
					continue;
				}

				if (msg.role === "toolResult") {
					// Emit tool_execution_end for each tool result
					events.push({
						type: "tool_execution_end",
						agentId: agentIndex,
						depth,
						ts: entry.ts,
						result: { content: msg.content },
					});
					continue;
				}
			}

			if (entry.type === "session_end") {
				events.push({
					type: "agent_end",
					agentId: agentIndex,
					depth,
					ts: entry.ts,
				});
				if (agentIndex === 0) {
					events.push({
						type: "session_end",
						ts: entry.ts,
						usage: entry.usage,
					});
				}
			}
		}
	}

	return events;
}

function loadSession(id: string): object[] {
	// Search across all domain dirs
	for (const domain of listDomains(LOGS_ROOT)) {
		// New format: session directory
		const sessionDir = join(LOGS_ROOT, domain, id);
		if (existsSync(sessionDir) && statSync(sessionDir).isDirectory()) {
			const agent0 = join(sessionDir, "agent-0.jsonl");
			if (existsSync(agent0)) {
				return convertMessagesToEvents(sessionDir);
			}
		}

		// Legacy format: flat JSONL file
		const filePath = join(LOGS_ROOT, domain, `${id}.jsonl`);
		if (!existsSync(filePath)) continue;

		const raw = readFileSync(filePath, "utf-8");
		const events: object[] = [];

		for (const line of raw.split("\n")) {
			if (!line.trim()) continue;
			try {
				const evt = JSON.parse(line);
				if (RENDERABLE_TYPES.has(evt.type)) {
					events.push(evt);
				}
			} catch {
				// skip
			}
		}

		return events;
	}

	return [];
}

interface ResultFile {
	filename: string;
	domain: string;
	config: Record<string, unknown>;
	score: { correct: number; total: number; pct: number };
	totals: { cost: number; tokens: number; timeMs: number };
	results: Array<{
		taskId: string;
		correct: boolean;
		failed?: boolean;
		cost: number;
		tokens: number;
		timeMs: number;
	}>;
}

function listResults(): ResultFile[] {
	const results: ResultFile[] = [];

	for (const domain of listDomains(RESULTS_ROOT)) {
		const resultsDir = join(RESULTS_ROOT, domain);
		const files = readdirSync(resultsDir).filter((f) => f.endsWith(".json")).sort().reverse();

		for (const file of files) {
			try {
				const raw = readFileSync(join(resultsDir, file), "utf-8");
				const data = JSON.parse(raw);
				results.push({
					filename: file,
					domain,
					config: data.config ?? {},
					score: data.score ?? { correct: 0, total: 0, pct: 0 },
					totals: data.totals ?? { cost: 0, tokens: 0, timeMs: 0 },
					results: (data.results ?? []).map((r: Record<string, unknown>) => ({
						taskId: r.taskId,
						correct: r.correct,
						failed: r.failed,
						cost: r.cost,
						tokens: r.tokens,
						timeMs: r.timeMs,
					})),
				});
			} catch {
				// skip
			}
		}
	}

	return results;
}

// ── Server ──────────────────────────────────────────────────────────

function jsonResponse(res: import("node:http").ServerResponse, data: unknown): void {
	const body = JSON.stringify(data);
	res.writeHead(200, {
		"Content-Type": "application/json; charset=utf-8",
		"Access-Control-Allow-Origin": "*",
	});
	res.end(body);
}

const server = createServer((req, res) => {
	const url = new URL(req.url ?? "/", `http://localhost:${port}`);
	const path = url.pathname;

	if (path === "/api/sessions") {
		jsonResponse(res, listSessions());
		return;
	}

	if (path.startsWith("/api/session/")) {
		const id = decodeURIComponent(path.slice("/api/session/".length));
		jsonResponse(res, loadSession(id));
		return;
	}

	if (path === "/api/active-sessions") {
		jsonResponse(res, { sessionIds: fileWatcher ? [...fileWatcher.activeSessions()] : [] });
		return;
	}

	if (path === "/api/results") {
		jsonResponse(res, listResults());
		return;
	}

	if (path === "/api/debug") {
		jsonResponse(res, {
			watcherActive: fileWatcher ? [...fileWatcher.activeSessions()] : [],
			wsClients: wss.clients.size,
			watchedDomains: listDomains(LOGS_ROOT),
		});
		return;
	}

	res.writeHead(404, { "Content-Type": "text/plain" });
	res.end("Not found");
});

// ── FileWatcher — detects active sessions and streams new events ────

const STALE_THRESHOLD_MS = 2 * 60 * 1000; // 2 minutes
const BUFFER_CLEANUP_MS = 30 * 60 * 1000; // 30 minutes after session_end
const POLL_INTERVAL_MS = 250;

interface AgentFileState {
	offset: number;
	pollTimer: ReturnType<typeof setInterval>;
	turnCount: number;
	agentIndex: number;
}

interface ActiveSession {
	agentFiles: Map<string, AgentFileState>;
	lastEventTs: number;
	sessionDir: string | null; // null for legacy flat files
	dirWatcher: FSWatcher | null; // watches session dir for new agent-*.jsonl
}

class FileWatcher {
	/** Per-session event buffer — persists after session ends for replay on connect */
	private eventBuffers = new Map<string, string[]>();
	private bufferEndTimes = new Map<string, number>(); // sessionId → timestamp when session ended

	/** Per-client subscriptions — which sessions each client wants events for */
	private subscriptions = new Map<WebSocket, Set<string>>();

	/** Active sessions being watched */
	private activeSessions_ = new Map<string, ActiveSession>();
	private dirWatchers: FSWatcher[] = [];
	private wss: WebSocketServer;

	constructor(wss: WebSocketServer) {
		this.wss = wss;
	}

	// ── Subscription management ─────────────────────────────────────

	/** Handle incoming client messages for subscribe/unsubscribe */
	handleClientMessage(ws: WebSocket, msg: any): void {
		switch (msg.type) {
			case "subscribe": {
				const subs = this.getOrCreateSubs(ws);
				const sid = msg.sessionId as string;
				subs.add(sid);
				// Replay buffered events for this session
				this.replayBuffer(ws, sid);
				break;
			}
			case "unsubscribe": {
				const subs = this.subscriptions.get(ws);
				if (subs) subs.delete(msg.sessionId);
				break;
			}
			case "subscribe_all": {
				const subs = this.getOrCreateSubs(ws);
				subs.add("*");
				// Replay all active session buffers
				for (const sid of this.activeSessions_.keys()) {
					this.replayBuffer(ws, sid);
				}
				break;
			}
			case "unsubscribe_all": {
				const subs = this.subscriptions.get(ws);
				if (subs) {
					subs.delete("*");
					subs.clear();
				}
				break;
			}
			case "replay": {
				this.handleReplay(ws, msg.sessionId, msg.speed ?? 1);
				break;
			}
		}
	}

	/** Remove client subscriptions on disconnect */
	removeClient(ws: WebSocket): void {
		this.subscriptions.delete(ws);
	}

	private getOrCreateSubs(ws: WebSocket): Set<string> {
		let subs = this.subscriptions.get(ws);
		if (!subs) {
			subs = new Set();
			this.subscriptions.set(ws, subs);
		}
		return subs;
	}

	/** Replay buffered events for a session to a single client */
	private replayBuffer(ws: WebSocket, sessionId: string): void {
		const buffer = this.eventBuffers.get(sessionId);
		if (!buffer || buffer.length === 0) return;

		if (ws.readyState !== 1) return;
		ws.send(JSON.stringify({ type: "replay_start", sessionId, count: buffer.length }));
		for (const msg of buffer) {
			ws.send(msg);
		}
		ws.send(JSON.stringify({ type: "replay_end", sessionId }));
	}

	/** Replay a completed session with timing delays */
	private async handleReplay(ws: WebSocket, sessionId: string, speed: number): Promise<void> {
		const events = loadSession(sessionId);
		if (events.length === 0) return;

		if (ws.readyState !== 1) return;
		ws.send(JSON.stringify({ type: "replay_start", sessionId, count: events.length }));

		let prevTs = 0;
		for (const evt of events) {
			if (ws.readyState !== 1) return;
			const evtAny = evt as any;
			const ts = evtAny.ts ?? 0;
			if (prevTs > 0 && ts > prevTs) {
				const deltaMs = ((ts - prevTs) * 1000) / speed;
				const cappedMs = Math.min(deltaMs, 5000);
				if (cappedMs > 10) {
					await new Promise((r) => setTimeout(r, cappedMs));
				}
			}
			if (ts) prevTs = ts;
			ws.send(JSON.stringify({ type: "event", sessionId, event: evt }));
		}

		ws.send(JSON.stringify({ type: "replay_complete", sessionId }));
	}

	// ── Event push (replaces broadcast) ─────────────────────────────

	/** Push an event to the session's buffer and send to subscribed clients */
	private pushEvent(sessionId: string, data: unknown): void {
		const msg = JSON.stringify(data);

		// Append to session buffer
		let buffer = this.eventBuffers.get(sessionId);
		if (!buffer) {
			buffer = [];
			this.eventBuffers.set(sessionId, buffer);
		}
		buffer.push(msg);

		// Send to subscribed clients only
		for (const [client, subs] of this.subscriptions) {
			if (client.readyState !== 1) continue;
			if (subs.has(sessionId) || subs.has("*")) {
				client.send(msg);
			}
		}
	}

	/** Broadcast a control message to ALL clients (bypasses subscription filter) */
	private broadcastControl(data: unknown): void {
		const msg = JSON.stringify(data);
		for (const client of this.wss.clients) {
			if (client.readyState === 1) {
				client.send(msg);
			}
		}
	}

	// ── Session scanning ────────────────────────────────────────────

	scanForActiveSessions(): void {
		for (const domain of listDomains(LOGS_ROOT)) {
			const logsDir = join(LOGS_ROOT, domain);
			const entries = readdirSync(logsDir, { withFileTypes: true });

			for (const entry of entries) {
				if (entry.isDirectory()) {
					const sessionDir = join(logsDir, entry.name);
					const agent0Path = join(sessionDir, "agent-0.jsonl");
					if (!existsSync(agent0Path)) continue;

					try {
						const raw = readFileSync(agent0Path, "utf-8");
						const lines = raw.split("\n").filter((l) => l.trim());
						if (lines.length === 0) continue;
						const last = JSON.parse(lines[lines.length - 1]);
						if (last.type === "session_end") continue;

						const lastTs: number = last.ts ?? 0;
						if ((Date.now() / 1000) - lastTs > STALE_THRESHOLD_MS / 1000) continue;

						this.watchSessionDir(entry.name, sessionDir, true);
					} catch {
						// skip
					}
					continue;
				}

				// Legacy format: flat JSONL files
				if (entry.isFile() && entry.name.endsWith(".jsonl")) {
					const filePath = join(logsDir, entry.name);
					try {
						const raw = readFileSync(filePath, "utf-8");
						const lines = raw.split("\n").filter((l) => l.trim());
						if (lines.length === 0) continue;
						const last = JSON.parse(lines[lines.length - 1]);
						if (last.type === "session_end") continue;

						const lastTs: number = last.ts ?? 0;
						if ((Date.now() / 1000) - lastTs > STALE_THRESHOLD_MS / 1000) continue;

						const sid = basename(entry.name, ".jsonl");
						const stat = statSync(filePath);
						this.watchLegacyFile(sid, filePath, stat.size, lastTs);
					} catch {
						// skip
					}
				}
			}
		}
	}

	startStaleCheck(): void {
		setInterval(() => {
			const nowSec = Date.now() / 1000;

			// Check for stale active sessions
			for (const [sid, session] of this.activeSessions_) {
				const ageSec = nowSec - session.lastEventTs;
				if (ageSec * 1000 > STALE_THRESHOLD_MS) {
					this.stopWatching(sid);
					this.broadcastControl({ type: "session_ended", sessionId: sid });
					console.log(`Session stale (no events for ${Math.round(ageSec)}s): ${sid}`);
				}
			}

			// Clean up old event buffers (30min after session_end)
			const nowMs = Date.now();
			for (const [sid, endTime] of this.bufferEndTimes) {
				if (nowMs - endTime > BUFFER_CLEANUP_MS) {
					this.eventBuffers.delete(sid);
					this.bufferEndTimes.delete(sid);
					console.log(`Cleaned up buffer for ended session: ${sid}`);
				}
			}
		}, 30_000);
	}

	// ── Multi-agent session watching ────────────────────────────────

	/** Watch a session directory for all agent-*.jsonl files */
	private watchSessionDir(sid: string, sessionDir: string, resumeFromEnd = false): void {
		if (this.activeSessions_.has(sid)) return;
		const agent0Path = join(sessionDir, "agent-0.jsonl");
		if (!existsSync(agent0Path)) return;

		let firstTs = Date.now() / 1000;
		try {
			const head = readFileSync(agent0Path, "utf-8").split("\n")[0];
			if (head) {
				const parsed = JSON.parse(head);
				firstTs = parsed.timestamp ? Math.floor(new Date(parsed.timestamp).getTime() / 1000) : firstTs;
			}
		} catch { /* use fallback */ }

		const session: ActiveSession = {
			agentFiles: new Map(),
			lastEventTs: firstTs,
			sessionDir,
			dirWatcher: null,
		};
		this.activeSessions_.set(sid, session);

		// Scan for all existing agent-*.jsonl files
		const agentFiles = readdirSync(sessionDir)
			.filter((f) => /^agent-\d+\.jsonl$/.test(f))
			.sort();

		for (const file of agentFiles) {
			const filePath = join(sessionDir, file);
			const agentIndex = parseInt(file.match(/\d+/)![0]);
			const offset = resumeFromEnd ? statSync(filePath).size : 0;
			this.watchAgentFile(sid, filePath, agentIndex, offset);
		}

		// Watch the session directory for new agent-*.jsonl files appearing
		try {
			session.dirWatcher = watch(sessionDir, (_, filename) => {
				if (!filename || !/^agent-\d+\.jsonl$/.test(filename)) return;
				const filePath = join(sessionDir, filename);
				if (session.agentFiles.has(filePath)) return;
				if (!existsSync(filePath)) return;
				const agentIndex = parseInt(filename.match(/\d+/)![0]);
				console.log(`[watch] New agent file detected: ${filename} in session ${sid}`);
				this.watchAgentFile(sid, filePath, agentIndex, 0);
			});
		} catch { /* session dir may vanish */ }

		console.log(`Watching active session: ${sid} (${agentFiles.length} agent file(s))`);
	}

	/** Start polling a single agent-*.jsonl file */
	private watchAgentFile(sessionId: string, filePath: string, agentIndex: number, offset: number): void {
		const session = this.activeSessions_.get(sessionId);
		if (!session || session.agentFiles.has(filePath)) return;

		const state: AgentFileState = {
			offset,
			pollTimer: setInterval(() => this.readNewAgentLines(sessionId, filePath), POLL_INTERVAL_MS),
			turnCount: 0,
			agentIndex,
		};
		session.agentFiles.set(filePath, state);
	}

	/** Read new lines from an agent file and convert+push events */
	private readNewAgentLines(sessionId: string, filePath: string): void {
		const session = this.activeSessions_.get(sessionId);
		if (!session) return;
		const agentState = session.agentFiles.get(filePath);
		if (!agentState) return;

		let stat;
		try {
			stat = statSync(filePath);
		} catch {
			return;
		}
		if (stat.size <= agentState.offset) return;

		const bytesToRead = stat.size - agentState.offset;
		const buf = Buffer.alloc(bytesToRead);
		let fd: number | undefined;
		try {
			fd = openSync(filePath, "r");
			readSync(fd, buf, 0, bytesToRead, agentState.offset);
		} catch {
			return;
		} finally {
			if (fd !== undefined) closeSync(fd);
		}
		agentState.offset = stat.size;

		const chunk = buf.toString("utf-8");
		const lines = chunk.split("\n").filter((l) => l.trim());

		for (const line of lines) {
			try {
				const parsed = JSON.parse(line);
				if (parsed.ts) session.lastEventTs = parsed.ts;
				this.convertAndPush(sessionId, parsed, agentState);
			} catch {
				// skip malformed lines
			}
		}
	}

	/** Watch a legacy flat JSONL file */
	private watchLegacyFile(sessionId: string, filePath: string, offset: number, lastEventTs: number): void {
		if (this.activeSessions_.has(sessionId)) return;

		const session: ActiveSession = {
			agentFiles: new Map(),
			lastEventTs,
			sessionDir: null,
			dirWatcher: null,
		};
		this.activeSessions_.set(sessionId, session);

		const state: AgentFileState = {
			offset,
			pollTimer: setInterval(() => this.readNewLegacyLines(sessionId, filePath), POLL_INTERVAL_MS),
			turnCount: 0,
			agentIndex: -1, // legacy marker
		};
		session.agentFiles.set(filePath, state);
		console.log(`Watching active session (legacy): ${sessionId}`);
	}

	/** Read new lines from a legacy flat JSONL file */
	private readNewLegacyLines(sessionId: string, filePath: string): void {
		const session = this.activeSessions_.get(sessionId);
		if (!session) return;
		const state = session.agentFiles.get(filePath);
		if (!state) return;

		let stat;
		try {
			stat = statSync(filePath);
		} catch {
			return;
		}
		if (stat.size <= state.offset) return;

		const bytesToRead = stat.size - state.offset;
		const buf = Buffer.alloc(bytesToRead);
		let fd: number | undefined;
		try {
			fd = openSync(filePath, "r");
			readSync(fd, buf, 0, bytesToRead, state.offset);
		} catch {
			return;
		} finally {
			if (fd !== undefined) closeSync(fd);
		}
		state.offset = stat.size;

		const chunk = buf.toString("utf-8");
		const lines = chunk.split("\n").filter((l) => l.trim());

		for (const line of lines) {
			try {
				const parsed = JSON.parse(line);
				if (parsed.ts) session.lastEventTs = parsed.ts;

				if (RENDERABLE_TYPES.has(parsed.type)) {
					this.pushEvent(sessionId, { type: "event", sessionId, event: parsed });
				}
				if (parsed.type === "session_end") {
					this.stopWatching(sessionId);
					this.broadcastControl({ type: "session_ended", sessionId });
				}
			} catch {
				// skip
			}
		}
	}

	// ── Convert agent JSONL to viewer events ────────────────────────

	private convertAndPush(
		sessionId: string,
		entry: any,
		agentState: AgentFileState,
	): void {
		const agentIndex = agentState.agentIndex;
		const depth = agentIndex === 0 ? 0 : 1;

		if (entry.type === "session") {
			const meta = entry.metadata ?? {};
			const ts = entry.timestamp ? Math.floor(new Date(entry.timestamp).getTime() / 1000) : 0;

			// Only emit session_start for agent-0
			if (agentIndex === 0) {
				this.pushEvent(sessionId, {
					type: "event", sessionId,
					event: { type: "session_start", sessionId, ts, ...meta },
				});
			}
			this.pushEvent(sessionId, {
				type: "event", sessionId,
				event: {
					type: "agent_start",
					agentId: agentIndex,
					depth,
					label: agentIndex === 0 ? "Orchestrator" : `Sub-agent #${agentIndex}`,
					ts,
				},
			});
			return;
		}

		if (entry.type === "message") {
			const msg = entry.message;
			if (!msg) return;

			if (msg.role === "user") {
				agentState.turnCount++;
				if (agentState.turnCount > 1) {
					this.pushEvent(sessionId, {
						type: "event", sessionId,
						event: { type: "turn_start", depth, ts: entry.ts },
					});
				}
				return;
			}

			if (msg.role === "assistant") {
				this.pushEvent(sessionId, {
					type: "event", sessionId,
					event: {
						type: "message_end",
						agentId: agentIndex,
						depth,
						ts: entry.ts,
						content: msg.content,
						...(msg.usage ? {
							usage: {
								input: msg.usage.input,
								output: msg.usage.output,
								totalTokens: msg.usage.totalTokens,
								cost: msg.usage.cost?.total,
							},
						} : {}),
					},
				});

				if (Array.isArray(msg.content)) {
					for (const block of msg.content) {
						if (block.type === "tool_use" || block.type === "toolCall") {
							const code = block.arguments?.code ?? block.input?.code ?? "";
							if (code) {
								this.pushEvent(sessionId, {
									type: "event", sessionId,
									event: {
										type: "tool_execution_start",
										agentId: agentIndex,
										depth,
										ts: entry.ts,
										toolName: block.name ?? "eval",
										code,
									},
								});
							}
						}
					}
				}
				return;
			}

			if (msg.role === "toolResult") {
				this.pushEvent(sessionId, {
					type: "event", sessionId,
					event: {
						type: "tool_execution_end",
						agentId: agentIndex,
						depth,
						ts: entry.ts,
						result: { content: msg.content },
					},
				});
				return;
			}
		}

		if (entry.type === "session_end") {
			this.pushEvent(sessionId, {
				type: "event", sessionId,
				event: { type: "agent_end", agentId: agentIndex, depth, ts: entry.ts },
			});
			// Only end the session when agent-0 ends
			if (agentIndex === 0) {
				this.pushEvent(sessionId, {
					type: "event", sessionId,
					event: { type: "session_end", ts: entry.ts, usage: entry.usage },
				});
				this.bufferEndTimes.set(sessionId, Date.now());
				this.stopWatching(sessionId);
				this.broadcastControl({ type: "session_ended", sessionId });
			}
		}
	}

	// ── Directory watching ──────────────────────────────────────────

	private watchDomainDir(logsDir: string): void {
		console.log(`[watch] Watching domain dir: ${logsDir}`);
		const watcher = watch(logsDir, (eventType, filename) => {
			if (!filename) return;
			const fullPath = join(logsDir, filename);

			// New format: directory with agent-*.jsonl
			if (existsSync(fullPath) && statSync(fullPath).isDirectory()) {
				console.log(`[watch] New session dir detected: ${filename}`);
				const innerWatcher = watch(fullPath, (_, innerFile) => {
					if (innerFile === "agent-0.jsonl") {
						if (!this.activeSessions_.has(filename)) {
							this.watchSessionDir(filename, fullPath);
							this.broadcastControl({ type: "session_active", sessionId: filename });
						}
					}
				});
				this.dirWatchers.push(innerWatcher);
				// Check immediately
				if (!this.activeSessions_.has(filename) && existsSync(join(fullPath, "agent-0.jsonl"))) {
					this.watchSessionDir(filename, fullPath);
					this.broadcastControl({ type: "session_active", sessionId: filename });
				}
				return;
			}

			// Legacy format
			if (!filename.endsWith(".jsonl")) return;
			const sid = basename(filename, ".jsonl");
			if (this.activeSessions_.has(sid)) return;
			if (!existsSync(fullPath)) return;

			let firstTs = Date.now() / 1000;
			try {
				const head = readFileSync(fullPath, "utf-8").split("\n")[0];
				if (head) firstTs = JSON.parse(head).ts ?? firstTs;
			} catch { /* use fallback */ }

			this.watchLegacyFile(sid, fullPath, 0, firstTs);
			this.broadcastControl({ type: "session_active", sessionId: sid });
		});
		this.dirWatchers.push(watcher);
	}

	watchDirectories(): void {
		for (const domain of listDomains(LOGS_ROOT)) {
			this.watchDomainDir(join(LOGS_ROOT, domain));
		}

		if (existsSync(LOGS_ROOT)) {
			const rootWatcher = watch(LOGS_ROOT, (eventType, filename) => {
				if (!filename) return;
				const newDir = join(LOGS_ROOT, filename);
				if (existsSync(newDir) && statSync(newDir).isDirectory()) {
					this.watchDomainDir(newDir);
				}
			});
			this.dirWatchers.push(rootWatcher);
		}
	}

	// ── Helpers ─────────────────────────────────────────────────────

	isActive(sessionId: string): boolean {
		return this.activeSessions_.has(sessionId);
	}

	activeSessions(): string[] {
		return [...this.activeSessions_.keys()];
	}

	private stopWatching(sessionId: string): void {
		const session = this.activeSessions_.get(sessionId);
		if (!session) return;
		for (const [, state] of session.agentFiles) {
			clearInterval(state.pollTimer);
		}
		if (session.dirWatcher) session.dirWatcher.close();
		this.activeSessions_.delete(sessionId);
		console.log(`Session ended: ${sessionId}`);
	}
}

// ── WebSocket server ────────────────────────────────────────────────

const wss = new WebSocketServer({ noServer: true });

let fileWatcher: FileWatcher | null = null;

wss.on("connection", (ws: WebSocket) => {
	if (fileWatcher) {
		ws.send(JSON.stringify({
			type: "active_sessions",
			sessionIds: fileWatcher.activeSessions(),
		}));
	}

	ws.on("message", (raw) => {
		try {
			const msg = JSON.parse(raw.toString());
			fileWatcher?.handleClientMessage(ws, msg);
		} catch { /* ignore */ }
	});

	ws.on("close", () => {
		fileWatcher?.removeClient(ws);
	});
});

server.on("upgrade", (req, socket, head) => {
	const url = new URL(req.url ?? "/", `http://localhost:${port}`);
	if (url.pathname === "/ws") {
		wss.handleUpgrade(req, socket, head, (ws) => {
			wss.emit("connection", ws, req);
		});
	} else {
		socket.destroy();
	}
});

server.listen(port, () => {
	console.log(`ARC Viewer API running on http://localhost:${port}`);
	console.log(`  Scanning: ${LOGS_ROOT}/*/  and  ${RESULTS_ROOT}/*/`);
	console.log(`  Domains: ${listDomains(LOGS_ROOT).join(", ") || "(none yet)"}`);

	fileWatcher = new FileWatcher(wss);
	fileWatcher.scanForActiveSessions();
	fileWatcher.watchDirectories();
	fileWatcher.startStaleCheck();
});
