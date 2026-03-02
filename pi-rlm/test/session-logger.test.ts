import { describe, it, expect, afterEach } from "vitest";
import { readFileSync, rmSync, existsSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { SessionLogger } from "../src/session-logger.js";
import type { TaggedAgentEvent } from "../src/event-tagging.js";

function tempDir(): string {
	const dir = join(tmpdir(), `pi-rlm-test-${Date.now()}-${Math.random().toString(36).slice(2)}`);
	return dir;
}

const cleanupDirs: string[] = [];

afterEach(() => {
	for (const dir of cleanupDirs) {
		if (existsSync(dir)) {
			rmSync(dir, { recursive: true, force: true });
		}
	}
	cleanupDirs.length = 0;
});

function makeEvent(overrides: Partial<TaggedAgentEvent> = {}): TaggedAgentEvent {
	return {
		type: "agent_start",
		_depth: 0,
		_label: "Test",
		_agentId: 0,
		...overrides,
	} as TaggedAgentEvent;
}

describe("SessionLogger", () => {
	it("creates log file in specified directory", () => {
		const logDir = tempDir();
		cleanupDirs.push(logDir);

		const logger = new SessionLogger({ logDir, sessionId: "test-session" });
		logger.start();

		expect(existsSync(join(logDir, "test-session.jsonl"))).toBe(true);
	});

	it("writes valid JSONL", () => {
		const logDir = tempDir();
		cleanupDirs.push(logDir);

		const logger = new SessionLogger({ logDir, sessionId: "jsonl-test" });
		logger.start();
		logger.logEvent(makeEvent({ type: "agent_start" }));
		logger.logEvent(makeEvent({ type: "agent_end" }));
		const path = logger.close();

		const content = readFileSync(path, "utf-8").trim();
		const lines = content.split("\n");

		// Each line should parse as valid JSON
		for (const line of lines) {
			expect(() => JSON.parse(line)).not.toThrow();
		}
	});

	it("includes session_start and session_end lines", () => {
		const logDir = tempDir();
		cleanupDirs.push(logDir);

		const logger = new SessionLogger({ logDir, sessionId: "bookend-test" });
		logger.start();
		const path = logger.close();

		const content = readFileSync(path, "utf-8").trim();
		const lines = content.split("\n").map((l) => JSON.parse(l));

		expect(lines[0].type).toBe("session_start");
		expect(lines[0].sessionId).toBe("bookend-test");
		expect(lines[lines.length - 1].type).toBe("session_end");
	});

	it("logs events with agentId and depth", () => {
		const logDir = tempDir();
		cleanupDirs.push(logDir);

		const logger = new SessionLogger({ logDir, sessionId: "event-test" });
		logger.start();
		logger.logEvent(makeEvent({ type: "agent_start", _agentId: 3, _depth: 2 }));
		const path = logger.close();

		const content = readFileSync(path, "utf-8").trim();
		const lines = content.split("\n").map((l) => JSON.parse(l));

		const agentStart = lines.find((l: any) => l.type === "agent_start");
		expect(agentStart.agentId).toBe(3);
		expect(agentStart.depth).toBe(2);
	});

	it("logPath returns the file path", () => {
		const logDir = tempDir();
		cleanupDirs.push(logDir);

		const logger = new SessionLogger({ logDir, sessionId: "path-test" });
		expect(logger.logPath).toBe(join(logDir, "path-test.jsonl"));
	});

	it("generates a session ID when none provided", () => {
		const logDir = tempDir();
		cleanupDirs.push(logDir);

		const logger = new SessionLogger({ logDir });
		logger.start();

		// Session ID should be a timestamp-like string
		expect(logger.logPath).toMatch(/\d{8}_\d{6}\.jsonl$/);
	});
});
