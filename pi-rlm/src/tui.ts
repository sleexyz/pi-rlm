import { Chalk } from "chalk";
import type { AgentEvent } from "@mariozechner/pi-agent-core";
import type { AssistantMessage, TextContent } from "@mariozechner/pi-ai";
import type { TaggedAgentEvent } from "./event-tagging.js";
import type { UsageTracker } from "./usage-tracker.js";
import {
	TUI,
	ProcessTerminal,
	Container,
	Text,
	Loader,
	Markdown,
	type MarkdownTheme,
} from "@mariozechner/pi-tui";

const chalk = new Chalk({ level: 3 });

const markdownTheme: MarkdownTheme = {
	heading: (t) => chalk.bold.cyan(t),
	link: (t) => chalk.blue(t),
	linkUrl: (t) => chalk.dim(t),
	code: (t) => chalk.yellow(t),
	codeBlock: (t) => chalk.green(t),
	codeBlockBorder: (t) => chalk.dim(t),
	quote: (t) => chalk.italic(t),
	quoteBorder: (t) => chalk.dim(t),
	hr: (t) => chalk.dim(t),
	listBullet: (t) => chalk.cyan(t),
	bold: (t) => chalk.bold(t),
	italic: (t) => chalk.italic(t),
	strikethrough: (t) => chalk.strikethrough(t),
	underline: (t) => chalk.underline(t),
};

function getAssistantText(message: unknown): string {
	const msg = message as AssistantMessage;
	if (msg?.role !== "assistant" || !Array.isArray(msg.content)) return "";
	return msg.content
		.filter((c): c is TextContent => c.type === "text")
		.map((c) => c.text)
		.join("");
}

export interface OrchestratorTUIOptions {
	/** Show eval errors and code previews. Default: false */
	debug?: boolean;
	/** Optional usage tracker to display summary on stop. */
	usageTracker?: UsageTracker;
}

/**
 * Lightweight TUI for watching an Orchestrator run.
 * Maps AgentEvent → pi-tui components.
 */
export class OrchestratorTUI {
	private ui: TUI;
	private chatContainer: Container;
	private statusContainer: Container;
	private loader: Loader | null = null;
	private streamingComponent: Markdown | null = null;
	private debug: boolean;
	private usageTracker?: UsageTracker;

	// Debug mode state
	private debugStreamingText: Text | null = null;
	private debugStreamingContent = "";

	constructor(options: OrchestratorTUIOptions = {}) {
		this.debug = options.debug ?? false;
		this.usageTracker = options.usageTracker;
		const terminal = new ProcessTerminal();
		this.ui = new TUI(terminal);

		this.chatContainer = new Container();
		this.statusContainer = new Container();

		this.ui.addChild(this.chatContainer);
		this.ui.addChild(this.statusContainer);
	}

	start(): void {
		this.ui.start();

		// Clean exit on Ctrl+C
		this.ui.addInputListener((data) => {
			if (data === "\x03") {
				this.stop();
				process.exit(0);
			}
			return undefined;
		});
	}

	stop(): void {
		if (this.loader) {
			this.loader.stop();
			this.loader = null;
		}
		this.ui.stop();

		// Print usage summary after TUI is stopped (goes to raw stdout)
		if (this.usageTracker) {
			const total = this.usageTracker.totalUsage();
			if (total.totalTokens > 0) {
				console.log("\n" + this.usageTracker.summary());
			}
		}
	}

	handleEvent = (event: AgentEvent): void => {
		if (this.debug) {
			this.handleDebugEvent(event as TaggedAgentEvent);
		} else {
			this.handleNormalEvent(event);
		}
		this.ui.requestRender();
	};

	// ── Normal mode (unchanged) ──────────────────────────────────────

	private handleNormalEvent(event: AgentEvent): void {
		switch (event.type) {
			case "agent_start":
				this.showLoader("Thinking...");
				break;

			case "message_start":
				if ((event.message as AssistantMessage)?.role === "assistant") {
					this.hideLoader();
					this.streamingComponent = new Markdown("", 1, 0, markdownTheme);
					this.chatContainer.addChild(this.streamingComponent);
				}
				break;

			case "message_update": {
				if (!this.streamingComponent) break;
				const text = getAssistantText(event.message);
				if (text) {
					this.streamingComponent.setText(text);
				}
				break;
			}

			case "message_end":
				if (this.streamingComponent) {
					const text = getAssistantText(event.message);
					if (text) {
						this.streamingComponent.setText(text);
					}
					this.streamingComponent = null;
				}
				break;

			case "tool_execution_start":
				this.hideLoader();
				this.showLoader("Running...");
				break;

			case "tool_execution_end": {
				this.hideLoader();
				const resultText = formatToolResult(event.result, false);
				if (resultText) {
					this.chatContainer.addChild(new Text(resultText, 1, 0));
				}
				break;
			}

			case "agent_end":
				this.hideLoader();
				break;
		}
	}

	// ── Debug mode ───────────────────────────────────────────────────

	private handleDebugEvent(event: TaggedAgentEvent): void {
		const depth = event._depth ?? 0;
		const gutter = makeGutter(depth);

		switch (event.type) {
			case "agent_start": {
				this.hideLoader();

				// Box header with agent ID
				const label = event._label ?? "Agent";
				const idTag = event._agentId != null ? `#${event._agentId} ` : "";
				const fullLabel = `${idTag}${label}`;
				const header = gutter + chalk.dim(`┌─ ${fullLabel} ${"─".repeat(Math.max(0, 50 - fullLabel.length))}`);
				this.chatContainer.addChild(new Text(header, 0, 0));

				// System prompt (full)
				if (event._systemPrompt) {
					const sysLines = event._systemPrompt.split("\n");
					const inner = makeGutter(depth + 1);
					const sysText = sysLines
						.map((line, i) => {
							const prefix = i === 0 ? chalk.magenta("SYS  ") : "     ";
							return inner + prefix + chalk.dim(line);
						})
						.join("\n");
					this.chatContainer.addChild(new Text(sysText, 0, 0));
				}

				// User message (full)
				if (event._userMessage) {
					const inner = makeGutter(depth + 1);
					const usrLines = event._userMessage.split("\n");
					const usrText = usrLines
						.map((line, i) => {
							const prefix = i === 0 ? chalk.blue("USR  ") : "     ";
							return inner + prefix + chalk.blue(line);
						})
						.join("\n");
					this.chatContainer.addChild(new Text(usrText, 0, 0));

					// Blank separator
					this.chatContainer.addChild(new Text(inner, 0, 0));
				}

				break;
			}

			case "message_start": {
				if ((event.message as AssistantMessage)?.role === "assistant") {
					this.hideLoader();
					// Start a streaming text component with LLM prefix
					this.debugStreamingContent = "";
					const inner = makeGutter(depth + 1);
					this.debugStreamingText = new Text(inner + chalk.white("LLM  "), 0, 0);
					this.chatContainer.addChild(this.debugStreamingText);
				}
				break;
			}

			case "message_update": {
				if (!this.debugStreamingText) break;
				const text = getAssistantText(event.message);
				if (text) {
					this.debugStreamingContent = text;
					const inner = makeGutter(depth + 1);
					const formatted = text.split("\n")
						.map((line, i) => {
							const prefix = i === 0 ? chalk.white("LLM  ") : "     ";
							return inner + prefix + line;
						})
						.join("\n");
					this.debugStreamingText.setText(formatted);
				}
				break;
			}

			case "message_end": {
				if (this.debugStreamingText) {
					const text = getAssistantText(event.message);
					if (text) {
						const inner = makeGutter(depth + 1);
						const formatted = text.split("\n")
							.map((line, i) => {
								const prefix = i === 0 ? chalk.white("LLM  ") : "     ";
								return inner + prefix + line;
							})
							.join("\n");
						this.debugStreamingText.setText(formatted);
					}
					this.debugStreamingText = null;
					this.debugStreamingContent = "";
				}
				break;
			}

			case "tool_execution_start": {
				this.hideLoader();
				const inner = makeGutter(depth + 1);
				const code = event.args?.code;
				if (code) {
					const codeLines = code.split("\n") as string[];
					const formatted = codeLines
						.map((line: string, i: number) => {
							const prefix = i === 0 ? chalk.yellow("EVAL ") : "     ";
							return inner + prefix + chalk.yellow(line);
						})
						.join("\n");
					this.chatContainer.addChild(new Text(formatted, 0, 0));
				} else {
					this.chatContainer.addChild(
						new Text(inner + chalk.dim(`${event.toolName}...`), 0, 0),
					);
				}
				this.showLoader(gutter + chalk.dim("  running..."));
				break;
			}

			case "tool_execution_end": {
				this.hideLoader();
				const inner = makeGutter(depth + 1);
				const resultText = formatToolResultRaw(event.result);
				if (resultText) {
					const lines = resultText.split("\n");
					const formatted = lines
						.map((line) => {
							if (line.startsWith("ERROR:")) return inner + "   " + chalk.red(line);
							if (line.startsWith("→")) return inner + "   " + chalk.green(line);
							return inner + "   " + chalk.dim(line);
						})
						.join("\n");
					this.chatContainer.addChild(new Text(formatted, 0, 0));
				}
				break;
			}

			case "agent_end": {
				this.hideLoader();
				const footer = gutter + chalk.dim(`└${"─".repeat(55)}`);
				this.chatContainer.addChild(new Text(footer, 0, 0));
				break;
			}
		}
	}

	private showLoader(message: string): void {
		this.hideLoader();
		this.loader = new Loader(
			this.ui,
			(s) => chalk.cyan(s),
			(s) => chalk.dim(s),
			message,
		);
		this.statusContainer.addChild(this.loader);
	}

	private hideLoader(): void {
		if (this.loader) {
			this.loader.stop();
			this.statusContainer.removeChild(this.loader);
			this.loader = null;
		}
	}
}

/** Build the gutter prefix for a given depth: "│ " repeated. */
function makeGutter(depth: number): string {
	return chalk.dim("│ ".repeat(depth));
}

/** Extract raw text from a tool result (shared by both modes). */
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

function formatToolResult(result: unknown, debug: boolean): string {
	const raw = formatToolResultRaw(result);
	if (!raw) return "";

	const lines = raw.split("\n");

	// In normal mode, skip ERROR lines — the agent retries on its own
	const filtered = debug ? lines : lines.filter((l) => !l.startsWith("ERROR:"));
	if (filtered.length === 0) return "";

	return filtered
		.map((line) => {
			if (line.startsWith("ERROR:")) return chalk.red("  " + line);
			if (line.startsWith("→")) return chalk.green("  " + line);
			return chalk.dim("  " + line);
		})
		.join("\n");
}
