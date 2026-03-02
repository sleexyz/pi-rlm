// Types
export type { EvalResult, Memory } from "./types.js";

// Eval runtime
export { EvalRuntime } from "./eval-runtime.js";

// Eval tool
export { createEvalTool, type EvalToolOptions } from "./eval-tool.js";

// Agent handle & spawn
export { AgentHandle, createSpawnAgent, type AgentHandleOptions, type SpawnAgentOptions } from "./agent-handle.js";

// Memories
export { Memories, MemoryQueryError, type MemoriesOptions } from "./memories.js";

// Domain adapter
export type { DomainAdapter } from "./domain-adapter.js";

// System prompt
export { generateSystemPrompt } from "./system-prompt.js";

// Orchestrator
export { Orchestrator, type OrchestratorOptions } from "./orchestrator.js";

// Auth
export { createOAuthResolver } from "./auth.js";

// Event tagging
export { createTaggedOnEvent, resetAgentIdCounter, type EventMeta, type TaggedAgentEvent } from "./event-tagging.js";

// TUI
export { OrchestratorTUI, type OrchestratorTUIOptions } from "./tui.js";

// Usage tracking
export { UsageTracker, type TokenUsage } from "./usage-tracker.js";

// Budgeted actions
export { createBudgetedAction, type BudgetedAction, type BudgetedActionOptions } from "./budgeted-action.js";

// Action history
export { ActionHistory, createTrackedAction, type HistoryEntry } from "./action-history.js";

// Session logging
export { SessionLogger, type SessionLoggerOptions } from "./session-logger.js";

// ARC-AGI-2
export type { ArcGrid, ArcExample, ArcTask, ArcAttempt, ArcResult } from "./arc/types.js";
export { createArcAdapter } from "./arc/adapter.js";
export { loadTask, loadTasksFromDir, selectDevSet } from "./arc/task-loader.js";
export {
	renderGrid,
	gridShape,
	makeGrid,
	copyGrid,
	gridsEqual,
	rotate90,
	rotate180,
	rotate270,
	flipH,
	flipV,
	transpose,
	crop,
	paste,
	tile,
	findColor,
	colorCounts,
	replaceColor,
	uniqueColors,
	connectedComponents,
	accuracy,
	softAccuracy,
	type Component,
	type ConnectedComponentsOptions,
} from "./arc/grid-helpers.js";
