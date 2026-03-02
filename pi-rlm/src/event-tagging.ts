import type { AgentEvent } from "@mariozechner/pi-agent-core";

/**
 * Extra metadata attached to AgentEvent objects for debug rendering.
 * These fields are added by wrapping the onEvent callback at each agent level.
 */
export interface EventMeta {
	_depth: number;
	_label: string;
	_agentId: number;
	_systemPrompt?: string;
	_userMessage?: string;
}

/** Auto-incrementing agent ID counter. */
let nextAgentId = 0;

/** Reset the agent ID counter (for testing). */
export function resetAgentIdCounter(): void {
	nextAgentId = 0;
}

export type TaggedAgentEvent = AgentEvent & EventMeta;

/**
 * Creates a wrapped onEvent callback that attaches depth/label metadata
 * to every event, and system prompt + user message to agent_start events.
 *
 * Returns a wrapper object so the userMessage can be set lazily
 * (it's only known at run/call time, not at construction time).
 */
export function createTaggedOnEvent(
	onEvent: ((event: AgentEvent) => void) | undefined,
	meta: { depth: number; label: string; systemPrompt?: string },
): { handler: ((event: AgentEvent) => void) | undefined; setUserMessage: (msg: string) => void } {
	if (!onEvent) return { handler: undefined, setUserMessage: () => {} };

	const agentId = nextAgentId++;
	let userMessage: string | undefined;

	const handler = (event: AgentEvent) => {
		const tagged: TaggedAgentEvent = {
			...event,
			_depth: meta.depth,
			_label: meta.label,
			_agentId: agentId,
		};
		if (event.type === "agent_start") {
			tagged._systemPrompt = meta.systemPrompt;
			tagged._userMessage = userMessage;
		}
		onEvent(tagged);
	};

	return {
		handler,
		setUserMessage: (msg: string) => {
			userMessage = msg;
		},
	};
}
