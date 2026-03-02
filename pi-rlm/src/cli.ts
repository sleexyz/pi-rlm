#!/usr/bin/env node
import { getModel } from "@mariozechner/pi-ai";
import { Orchestrator } from "./orchestrator.js";
import { OrchestratorTUI } from "./tui.js";
import { createDemoAdapter } from "./demo-adapter.js";
import { createOAuthResolver } from "./auth.js";
import { SessionLogger } from "./session-logger.js";
import type { TaggedAgentEvent } from "./event-tagging.js";

const args = process.argv.slice(2);
const debug = args.includes("--debug");
const logFlagIndex = args.indexOf("--log");
const logEnabled = logFlagIndex !== -1;

// --log takes an optional dir argument, but only if it looks like a path (not another flag, not bare words)
// To pass a custom dir, use: --log=./mydir or --log ./mydir/
// A value is treated as a dir only if it contains a path separator or starts with "."
const logNextArg = logEnabled ? args[logFlagIndex + 1] : undefined;
const logArgIsDir = logNextArg && !logNextArg.startsWith("--") && (logNextArg.includes("/") || logNextArg.startsWith("."));
const logDir = logEnabled ? (logArgIsDir ? logNextArg : "./logs") : undefined;

// Filter out flags to get the task
const flagArgs = new Set(["--debug", "--log"]);
const taskArgs = args.filter((a, i) => {
	if (flagArgs.has(a)) return false;
	// Skip the value after --log if it was consumed as the log dir
	if (i > 0 && args[i - 1] === "--log" && logArgIsDir && a === logNextArg) return false;
	return true;
});
const task = taskArgs.join(" ") || "What is 6 * 7? Use the eval tool to compute it and submit the answer.";

// Set up session logger if enabled
const sessionLogger = logDir ? new SessionLogger({ logDir }) : undefined;
if (sessionLogger) {
	sessionLogger.start();
}

const orchestrator = new Orchestrator({
	model: getModel("anthropic", "claude-sonnet-4-6"),
	adapter: createDemoAdapter(),
	thinkingLevel: "off",
	getApiKey: createOAuthResolver(),
	onEvent: (event) => {
		tui.handleEvent(event);
		if (sessionLogger) {
			sessionLogger.logEvent(event as TaggedAgentEvent);
		}
	},
});

const tui = new OrchestratorTUI({
	debug,
	usageTracker: orchestrator.getUsageTracker(),
});
tui.start();

try {
	const result = await orchestrator.run(task);
	// Give the TUI a moment to render the final state
	await new Promise((resolve) => setTimeout(resolve, 100));
	tui.stop();

	if (sessionLogger) {
		const logPath = sessionLogger.close(orchestrator.getUsageTracker());
		console.log(`\nSession log: ${logPath}`);
	}

	if (result !== undefined) {
		console.log(`\nResult: ${JSON.stringify(result)}`);
	}
} catch (err) {
	tui.stop();
	if (sessionLogger) {
		sessionLogger.close(orchestrator.getUsageTracker());
	}
	console.error(err);
	process.exit(1);
}
