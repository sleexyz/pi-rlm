#!/usr/bin/env bun
/**
 * ARC Trace Viewer — renders JSONL session logs to stdout in debug-TUI format.
 *
 * Usage: bun src/arc-trace.ts <logfile> [options]
 *   --no-thinking   hide thinking blocks
 *   --no-sysprompt  hide system prompt
 *   --compact       only show eval code + results + accuracy lines
 *   --turn <n>      show only turn N
 */

import { readFileSync } from "node:fs";
import { Chalk } from "chalk";

const chalk = new Chalk({ level: 3 });

// ── Types for JSONL log events ──────────────────────────────────────

interface LogEvent {
	type: string;
	ts: number;
	agentId?: number;
	depth?: number;
	label?: string;
	systemPrompt?: string;
	userMessage?: string;
	content?: ContentBlock[];
	usage?: { input: number; output: number; totalTokens: number; cost: number };
	toolName?: string;
	code?: string;
	result?: { content?: { type: string; text: string }[] };
	sessionId?: string;
}

interface ContentBlock {
	type: string;
	text?: string;
	thinking?: string;
	name?: string;
	arguments?: { code?: string };
}

// ── CLI arg parsing ─────────────────────────────────────────────────

const args = process.argv.slice(2);
const flags = {
	noThinking: args.includes("--no-thinking"),
	noSysprompt: args.includes("--no-sysprompt"),
	compact: args.includes("--compact"),
	turn: -1,
};

const turnIdx = args.indexOf("--turn");
if (turnIdx !== -1 && args[turnIdx + 1]) {
	flags.turn = parseInt(args[turnIdx + 1], 10);
}

const logFile = args.find((a) => !a.startsWith("--") && (turnIdx === -1 || a !== args[turnIdx + 1]));
if (!logFile) {
	console.error("Usage: bun src/arc-trace.ts <logfile> [--no-thinking] [--no-sysprompt] [--compact] [--turn <n>]");
	process.exit(1);
}

// ── Read JSONL ──────────────────────────────────────────────────────

const raw = readFileSync(logFile, "utf-8");
const events: LogEvent[] = raw
	.split("\n")
	.filter((l) => l.trim())
	.map((l) => JSON.parse(l));

// ── Helpers ─────────────────────────────────────────────────────────

function makeGutter(depth: number): string {
	return chalk.dim("│ ".repeat(depth));
}

function formatToolResultRaw(result: unknown): string {
	if (result == null) return "";
	const r = result as { content?: { type: string; text: string }[] };
	if (!Array.isArray(r.content)) return "";
	return r.content
		.filter((c) => c.type === "text" && c.text)
		.map((c) => c.text.trim())
		.join("\n")
		.trim();
}

function printLines(gutter: string, prefix: string, continuation: string, text: string, colorFn?: (s: string) => string): void {
	const lines = text.split("\n");
	for (let i = 0; i < lines.length; i++) {
		const pfx = i === 0 ? prefix : continuation;
		const line = colorFn ? colorFn(lines[i]) : lines[i];
		console.log(gutter + pfx + line);
	}
}

function isAccuracyLine(line: string): boolean {
	return /accuracy|correct|score/i.test(line);
}

// ── Render ──────────────────────────────────────────────────────────

let turnCount = 0;
let totalTokens = 0;
let totalCost = 0;
let totalTurns = 0;

for (const event of events) {
	const depth = event.depth ?? 0;
	const gutter = makeGutter(depth);
	const inner = makeGutter(depth + 1);

	switch (event.type) {
		case "session_start":
			if (!flags.compact) {
				console.log(chalk.dim(`Session: ${event.sessionId}`));
			}
			break;

		case "agent_start": {
			if (flags.compact) break;
			const label = event.label ?? "Agent";
			const idTag = event.agentId != null ? `#${event.agentId} ` : "";
			const fullLabel = `${idTag}${label}`;
			console.log(gutter + chalk.dim(`┌─ ${fullLabel} ${"─".repeat(Math.max(0, 50 - fullLabel.length))}`));

			// System prompt
			if (event.systemPrompt && !flags.noSysprompt) {
				printLines(inner, chalk.magenta("SYS  "), "     ", event.systemPrompt, (s) => chalk.dim(s));
			}

			// User message
			if (event.userMessage) {
				printLines(inner, chalk.blue("USR  "), "     ", event.userMessage, (s) => chalk.blue(s));
				console.log(inner);
			}
			break;
		}

		case "turn_start": {
			turnCount++;
			totalTurns++;
			if (flags.turn >= 0 && turnCount !== flags.turn) break;
			if (!flags.compact && turnCount > 1) {
				console.log(inner + chalk.dim(`── Turn ${turnCount} ${"─".repeat(Math.max(0, 45))}`));
			}
			break;
		}

		case "message_end": {
			if (flags.turn >= 0 && turnCount !== flags.turn) break;
			if (!event.content) break;

			// Accumulate usage
			if (event.usage) {
				totalTokens += event.usage.totalTokens;
				totalCost += event.usage.cost;
			}

			for (const block of event.content) {
				// Thinking blocks
				if (block.type === "thinking" && block.thinking) {
					if (flags.noThinking || flags.compact) continue;
					printLines(inner, chalk.cyan("THK  "), "     ", block.thinking, (s) => chalk.cyan(s));
				}

				// Text blocks (LLM output)
				if (block.type === "text" && block.text) {
					if (flags.compact) {
						// In compact mode, only show accuracy-related lines
						const accLines = block.text.split("\n").filter(isAccuracyLine);
						if (accLines.length > 0) {
							for (const line of accLines) {
								console.log(inner + chalk.green("  " + line));
							}
						}
					} else {
						printLines(inner, chalk.white("LLM  "), "     ", block.text);
					}
				}
			}
			break;
		}

		case "tool_execution_start": {
			if (flags.turn >= 0 && turnCount !== flags.turn) break;
			const code = event.code;
			if (code) {
				if (flags.compact) {
					// In compact mode, show shortened code
					const lines = code.trim().split("\n");
					const preview = lines.length > 5
						? [...lines.slice(0, 3), chalk.dim(`  ... (${lines.length - 3} more lines)`)].join("\n")
						: code.trim();
					printLines(inner, chalk.yellow("EVAL "), "     ", preview, (s) => chalk.yellow(s));
				} else {
					printLines(inner, chalk.yellow("EVAL "), "     ", code.trim(), (s) => chalk.yellow(s));
				}
			} else if (!flags.compact) {
				console.log(inner + chalk.dim(`${event.toolName}...`));
			}
			break;
		}

		case "tool_execution_end": {
			if (flags.turn >= 0 && turnCount !== flags.turn) break;
			const resultText = formatToolResultRaw(event.result);
			if (resultText) {
				const lines = resultText.split("\n");
				for (const line of lines) {
					if (line.startsWith("ERROR:")) {
						console.log(inner + "     " + chalk.red(line));
					} else if (line.startsWith("→") || isAccuracyLine(line)) {
						console.log(inner + "     " + chalk.green(line));
					} else if (flags.compact) {
						// In compact, skip non-notable output lines
						continue;
					} else {
						console.log(inner + "     " + chalk.dim(line));
					}
				}
			}
			break;
		}

		case "agent_end": {
			if (!flags.compact) {
				console.log(gutter + chalk.dim(`└${"─".repeat(55)}`));
			}
			break;
		}

		case "session_end": {
			// Usage summary
			const sessionUsage = event.usage as { totalTokens?: number; totalCost?: number } | undefined;
			const tokens = sessionUsage?.totalTokens ?? totalTokens;
			const cost = sessionUsage?.totalCost ?? totalCost;
			console.log("");
			const gtr = makeGutter(1);
			console.log(gtr + chalk.dim(`── Usage ${"─".repeat(45)}`));
			console.log(gtr + `Tokens: ${tokens.toLocaleString()} | Cost: $${cost.toFixed(4)} | Turns: ${totalTurns}`);
			break;
		}
	}
}

// If no session_end was found, still print accumulated usage
if (!events.some((e) => e.type === "session_end") && (totalTokens > 0 || totalTurns > 0)) {
	console.log("");
	const gtr = makeGutter(1);
	console.log(gtr + chalk.dim(`── Usage ${"─".repeat(45)}`));
	console.log(gtr + `Tokens: ${totalTokens.toLocaleString()} | Cost: $${totalCost.toFixed(4)} | Turns: ${totalTurns}`);
}
