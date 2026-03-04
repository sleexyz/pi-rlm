#!/usr/bin/env bun
/**
 * ARC Trace Viewer — API server for run data and session traces.
 *
 * Usage: bun domains/arc-agi-2/src/viewer.ts [--port <n>]
 *
 * Scans runs/ at the repo root for run directories containing run.json.
 * Serves JSON APIs consumed by the frontend in domains/arc-agi-2/viewer/.
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

const repoRoot = join(import.meta.dirname, "../../..");
const RUNS_ROOT = join(repoRoot, "runs");

// ── Data helpers ────────────────────────────────────────────────────

function listRuns(): object[] {
	if (!existsSync(RUNS_ROOT)) return [];
	const runs: any[] = [];
	const entries = readdirSync(RUNS_ROOT, { withFileTypes: true });
	for (const entry of entries) {
		if (!entry.isDirectory()) continue;
		const runJsonPath = join(RUNS_ROOT, entry.name, "run.json");
		if (!existsSync(runJsonPath)) continue;
		try {
			const data = JSON.parse(readFileSync(runJsonPath, "utf-8"));
			// Annotate with active session info
			data._activeSessions = fileWatcher?.activeSessionsInRun(data.name) ?? [];
			runs.push(data);
		} catch {
			// skip malformed
		}
	}
	// Sort by startedAt descending
	runs.sort((a: any, b: any) => {
		const ta = a.startedAt ? new Date(a.startedAt).getTime() : 0;
		const tb = b.startedAt ? new Date(b.startedAt).getTime() : 0;
		return tb - ta;
	});
	return runs;
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
					// First user message — attach to agent_start
					if (turnCount === 1) {
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

/**
 * Load a session trace by ID.
 * ID format: "<runName>/<sessionDirName>" e.g. "stellar-meadow/007bbfb7_a0"
 * Falls back to searching all runs if no "/" in the ID.
 */
function loadSession(id: string): object[] {
	// Composite ID: runName/sessionDirName
	const slashIdx = id.indexOf("/");
	if (slashIdx !== -1) {
		const runName = id.substring(0, slashIdx);
		const sessionDirName = id.substring(slashIdx + 1);
		const sessionDir = join(RUNS_ROOT, runName, "sessions", sessionDirName);
		if (existsSync(sessionDir) && statSync(sessionDir).isDirectory()) {
			const agent0 = join(sessionDir, "agent-0.jsonl");
			if (existsSync(agent0)) {
				return convertMessagesToEvents(sessionDir);
			}
		}
	}

	// Fallback: search all runs for a matching session dir name
	if (existsSync(RUNS_ROOT)) {
		for (const runEntry of readdirSync(RUNS_ROOT, { withFileTypes: true })) {
			if (!runEntry.isDirectory()) continue;
			const sessionsDir = join(RUNS_ROOT, runEntry.name, "sessions");
			if (!existsSync(sessionsDir)) continue;
			const sessionDir = join(sessionsDir, id);
			if (existsSync(sessionDir) && statSync(sessionDir).isDirectory()) {
				const agent0 = join(sessionDir, "agent-0.jsonl");
				if (existsSync(agent0)) {
					return convertMessagesToEvents(sessionDir);
				}
			}
		}
	}

	return [];
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

	if (path === "/api/runs") {
		jsonResponse(res, listRuns());
		return;
	}

	if (path.startsWith("/api/session/")) {
		const id = decodeURIComponent(path.slice("/api/session/".length));
		jsonResponse(res, loadSession(id));
		return;
	}

	if (path === "/api/active-sessions") {
		jsonResponse(res, { sessionIds: fileWatcher ? fileWatcher.activeSessions() : [] });
		return;
	}

	if (path === "/api/debug") {
		jsonResponse(res, {
			watcherActive: fileWatcher ? fileWatcher.activeSessions() : [],
			wsClients: wss.clients.size,
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
	sessionDir: string;
	dirWatcher: FSWatcher | null; // watches session dir for new agent-*.jsonl
	runName: string;
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
		if (!existsSync(RUNS_ROOT)) return;

		for (const runEntry of readdirSync(RUNS_ROOT, { withFileTypes: true })) {
			if (!runEntry.isDirectory()) continue;
			const sessionsDir = join(RUNS_ROOT, runEntry.name, "sessions");
			if (!existsSync(sessionsDir)) continue;

			for (const sessionEntry of readdirSync(sessionsDir, { withFileTypes: true })) {
				if (!sessionEntry.isDirectory()) continue;
				const sessionDir = join(sessionsDir, sessionEntry.name);
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

					const sid = `${runEntry.name}/${sessionEntry.name}`;
					this.watchSessionDir(sid, sessionDir, runEntry.name, true);
				} catch {
					// skip
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
					this.broadcastControl({ type: "session_ended", sessionId: sid, runName: session.runName });
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
	private watchSessionDir(sid: string, sessionDir: string, runName: string, resumeFromEnd = false): void {
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
			runName,
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
			const session = this.activeSessions_.get(sessionId);
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
				const runName = session?.runName ?? "";
				this.stopWatching(sessionId);
				this.broadcastControl({ type: "session_ended", sessionId, runName });
			}
		}
	}

	// ── Directory watching ──────────────────────────────────────────

	/** Watch a run's sessions directory for new session subdirs */
	private watchRunSessionsDir(runName: string, sessionsDir: string): void {
		console.log(`[watch] Watching run sessions dir: ${sessionsDir}`);
		const watcher = watch(sessionsDir, (_, filename) => {
			if (!filename) return;
			const fullPath = join(sessionsDir, filename);
			if (!existsSync(fullPath) || !statSync(fullPath).isDirectory()) return;

			const sid = `${runName}/${filename}`;
			if (this.activeSessions_.has(sid)) return;

			// Watch for agent-0.jsonl to appear
			const agent0 = join(fullPath, "agent-0.jsonl");
			if (existsSync(agent0)) {
				this.watchSessionDir(sid, fullPath, runName);
				this.broadcastControl({ type: "session_active", sessionId: sid, runName });
			} else {
				// Wait for agent-0.jsonl to appear
				const innerWatcher = watch(fullPath, (_, innerFile) => {
					if (innerFile === "agent-0.jsonl") {
						if (!this.activeSessions_.has(sid)) {
							this.watchSessionDir(sid, fullPath, runName);
							this.broadcastControl({ type: "session_active", sessionId: sid, runName });
						}
					}
				});
				this.dirWatchers.push(innerWatcher);
			}
		});
		this.dirWatchers.push(watcher);
	}

	watchDirectories(): void {
		if (!existsSync(RUNS_ROOT)) {
			mkdirSync(RUNS_ROOT, { recursive: true });
		}

		// Watch each existing run's sessions dir
		for (const runEntry of readdirSync(RUNS_ROOT, { withFileTypes: true })) {
			if (!runEntry.isDirectory()) continue;
			const sessionsDir = join(RUNS_ROOT, runEntry.name, "sessions");
			if (existsSync(sessionsDir)) {
				this.watchRunSessionsDir(runEntry.name, sessionsDir);
			}
		}

		// Watch runs/ root for new run directories appearing
		const rootWatcher = watch(RUNS_ROOT, (_, filename) => {
			if (!filename) return;
			const runDir = join(RUNS_ROOT, filename);
			if (!existsSync(runDir) || !statSync(runDir).isDirectory()) return;

			const sessionsDir = join(runDir, "sessions");

			// If sessions dir already exists, watch it
			if (existsSync(sessionsDir)) {
				this.watchRunSessionsDir(filename, sessionsDir);
				this.broadcastControl({ type: "run_active", runName: filename });
				return;
			}

			// Watch for sessions dir to be created
			const runWatcher = watch(runDir, (_, innerName) => {
				if (innerName === "sessions" && existsSync(join(runDir, "sessions"))) {
					this.watchRunSessionsDir(filename, join(runDir, "sessions"));
					this.broadcastControl({ type: "run_active", runName: filename });
				}
			});
			this.dirWatchers.push(runWatcher);
		});
		this.dirWatchers.push(rootWatcher);
	}

	// ── Helpers ─────────────────────────────────────────────────────

	isActive(sessionId: string): boolean {
		return this.activeSessions_.has(sessionId);
	}

	activeSessions(): string[] {
		return [...this.activeSessions_.keys()];
	}

	/** Get active session IDs for a given run */
	activeSessionsInRun(runName: string): string[] {
		const result: string[] = [];
		for (const sid of this.activeSessions_.keys()) {
			if (sid.startsWith(runName + "/")) {
				result.push(sid);
			}
		}
		return result;
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
	console.log(`  Scanning: ${RUNS_ROOT}/*/`);

	fileWatcher = new FileWatcher(wss);
	fileWatcher.scanForActiveSessions();
	fileWatcher.watchDirectories();
	fileWatcher.startStaleCheck();
});
