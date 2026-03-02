#!/usr/bin/env bun
/**
 * ARC Trace Viewer — API server for session logs and results.
 *
 * Usage: bun pi-rlm/src/arc-viewer.ts [--port <n>]
 *
 * Serves JSON APIs consumed by the Vite frontend in arc-viewer/.
 */

import { readdirSync, readFileSync, existsSync, watch, openSync, readSync, closeSync, statSync, type FSWatcher } from "node:fs";
import { join, basename } from "node:path";
import { createServer } from "node:http";
import { WebSocketServer, type WebSocket } from "ws";

// ── CLI args ────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const portIdx = args.indexOf("--port");
const port = portIdx !== -1 && args[portIdx + 1]
	? parseInt(args[portIdx + 1], 10)
	: process.env.PORT ? parseInt(process.env.PORT, 10) : 3334;

// Resolve relative to pi-rlm/ (arc-runner writes to ./logs/arc and ./results/arc from pi-rlm cwd)
const piRlmRoot = join(import.meta.dirname, "..");
const LOGS_DIR = join(piRlmRoot, "logs/arc");
const RESULTS_DIR = join(piRlmRoot, "results/arc");

// ── Data helpers ────────────────────────────────────────────────────

interface SessionSummary {
	sessionId: string;
	filename: string;
	ts: number;
	taskId?: string;
	split?: string;
	model?: string;
	turns: number;
	usage?: { totalTokens: number; totalCost: number };
	active?: boolean;
}

function loadAnnotations(): Record<string, { model?: string; taskId?: string }> {
	const annoPath = join(LOGS_DIR, "annotations.json");
	if (!existsSync(annoPath)) return {};
	try {
		const data = JSON.parse(readFileSync(annoPath, "utf-8"));
		return data.sessions ?? {};
	} catch {
		return {};
	}
}

function listSessions(): SessionSummary[] {
	if (!existsSync(LOGS_DIR)) return [];
	const annotations = loadAnnotations();
	const files = readdirSync(LOGS_DIR).filter((f) => f.endsWith(".jsonl")).sort().reverse();
	const summaries: SessionSummary[] = [];

	for (const file of files) {
		try {
			const raw = readFileSync(join(LOGS_DIR, file), "utf-8");
			const lines = raw.split("\n").filter((l) => l.trim());
			if (lines.length === 0) continue;

			const first = JSON.parse(lines[0]);
			const last = JSON.parse(lines[lines.length - 1]);

			// Count turns via string scan (fast)
			let turns = 0;
			for (const line of lines) {
				if (line.includes('"turn_start"')) turns++;
			}

			const sid = first.sessionId ?? basename(file, ".jsonl");
			const anno = annotations[sid];
			const summary: SessionSummary = {
				sessionId: sid,
				filename: file,
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

	return summaries;
}

const RENDERABLE_TYPES = new Set([
	"session_start",
	"agent_start",
	"turn_start",
	"message_end",
	"tool_execution_start",
	"tool_execution_end",
	"agent_end",
	"session_end",
]);

function loadSession(id: string): object[] {
	const filePath = join(LOGS_DIR, `${id}.jsonl`);
	if (!existsSync(filePath)) return [];

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

interface ResultFile {
	filename: string;
	config: Record<string, unknown>;
	score: { correct: number; total: number; pct: number };
	totals: { cost: number; tokens: number; timeMs: number };
	results: Array<{
		taskId: string;
		correct: boolean;
		cost: number;
		tokens: number;
		timeMs: number;
	}>;
}

function listResults(): ResultFile[] {
	if (!existsSync(RESULTS_DIR)) return [];
	const files = readdirSync(RESULTS_DIR).filter((f) => f.endsWith(".json")).sort().reverse();
	const results: ResultFile[] = [];

	for (const file of files) {
		try {
			const raw = readFileSync(join(RESULTS_DIR, file), "utf-8");
			const data = JSON.parse(raw);
			results.push({
				filename: file,
				config: data.config ?? {},
				score: data.score ?? { correct: 0, total: 0, pct: 0 },
				totals: data.totals ?? { cost: 0, tokens: 0, timeMs: 0 },
				results: (data.results ?? []).map((r: Record<string, unknown>) => ({
					taskId: r.taskId,
					correct: r.correct,
					cost: r.cost,
					tokens: r.tokens,
					timeMs: r.timeMs,
				})),
			});
		} catch {
			// skip
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

	res.writeHead(404, { "Content-Type": "text/plain" });
	res.end("Not found");
});

// ── FileWatcher — detects active sessions and streams new events ────

const STALE_THRESHOLD_MS = 10 * 60 * 1000; // 10 minutes

class FileWatcher {
	private active = new Map<string, { offset: number; watcher: FSWatcher }>();
	private dirWatcher: FSWatcher | null = null;
	private wss: WebSocketServer;

	constructor(wss: WebSocketServer) {
		this.wss = wss;
	}

	/** Scan existing JSONL files for active (not yet ended) sessions. */
	scanForActiveSessions(): void {
		if (!existsSync(LOGS_DIR)) return;
		const files = readdirSync(LOGS_DIR).filter((f) => f.endsWith(".jsonl"));
		for (const file of files) {
			const filePath = join(LOGS_DIR, file);
			try {
				const raw = readFileSync(filePath, "utf-8");
				const lines = raw.split("\n").filter((l) => l.trim());
				if (lines.length === 0) continue;
				const last = JSON.parse(lines[lines.length - 1]);
				if (last.type === "session_end") continue;

				// Check staleness
				const stat = statSync(filePath);
				if (Date.now() - stat.mtimeMs > STALE_THRESHOLD_MS) continue;

				const sid = basename(file, ".jsonl");
				this.watchFile(sid, filePath, stat.size);
			} catch {
				// skip
			}
		}
	}

	/** Watch logs directory for new JSONL files. */
	watchDirectory(): void {
		if (!existsSync(LOGS_DIR)) return;
		this.dirWatcher = watch(LOGS_DIR, (eventType, filename) => {
			if (!filename || !filename.endsWith(".jsonl")) return;
			const sid = basename(filename, ".jsonl");
			if (this.active.has(sid)) return;

			const filePath = join(LOGS_DIR, filename);
			if (!existsSync(filePath)) return;

			this.watchFile(sid, filePath, 0);
			this.broadcast({ type: "session_active", sessionId: sid });
		});
	}

	isActive(sessionId: string): boolean {
		return this.active.has(sessionId);
	}

	activeSessions(): string[] {
		return [...this.active.keys()];
	}

	private watchFile(sessionId: string, filePath: string, offset: number): void {
		const watcher = watch(filePath, () => {
			this.readNewLines(sessionId, filePath);
		});
		this.active.set(sessionId, { offset, watcher });
		console.log(`Watching active session: ${sessionId}`);
	}

	private readNewLines(sessionId: string, filePath: string): void {
		const entry = this.active.get(sessionId);
		if (!entry) return;

		let stat;
		try {
			stat = statSync(filePath);
		} catch {
			return;
		}
		if (stat.size <= entry.offset) return;

		const bytesToRead = stat.size - entry.offset;
		const buf = Buffer.alloc(bytesToRead);

		let fd: number | undefined;
		try {
			fd = openSync(filePath, "r");
			readSync(fd, buf, 0, bytesToRead, entry.offset);
		} catch {
			return;
		} finally {
			if (fd !== undefined) closeSync(fd);
		}

		entry.offset = stat.size;

		const chunk = buf.toString("utf-8");
		const lines = chunk.split("\n").filter((l) => l.trim());

		for (const line of lines) {
			try {
				const evt = JSON.parse(line);
				if (RENDERABLE_TYPES.has(evt.type)) {
					this.broadcast({ type: "event", sessionId, event: evt });
				}
				if (evt.type === "session_end") {
					this.stopWatching(sessionId);
					this.broadcast({ type: "session_ended", sessionId });
				}
			} catch {
				// skip malformed lines
			}
		}
	}

	private stopWatching(sessionId: string): void {
		const entry = this.active.get(sessionId);
		if (!entry) return;
		entry.watcher.close();
		this.active.delete(sessionId);
		console.log(`Session ended: ${sessionId}`);
	}

	private broadcast(data: unknown): void {
		const msg = JSON.stringify(data);
		for (const client of this.wss.clients) {
			if (client.readyState === 1 /* WebSocket.OPEN */) {
				client.send(msg);
			}
		}
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

	fileWatcher = new FileWatcher(wss);
	fileWatcher.scanForActiveSessions();
	fileWatcher.watchDirectory();
});
