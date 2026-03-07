/**
 * A Adapter is the complete description of a problem space.
 * The framework assembles the system prompt, wires up eval + spawnAgent + submit.
 * Everything else is domain scope.
 */
export interface Adapter {
	/** Adapter name (used in system prompt heading). */
	name: string;

	/** Agent instructions — the core of the system prompt. */
	premise: string;

	/** SDK/API reference docs. Also exposed as `DOMAIN_REFERENCE` in eval scope. */
	reference: string;

	/** System prompt for sub-agents spawned via `spawnAgent()` with no args.
	 *  `spawnAgent("custom")` bypasses this and uses the custom prompt as-is. */
	defaultSubAgentPrompt?: string;

	/** Everything the agent can use — helpers, data, libraries, whatever the domain needs. */
	scope: Record<string, unknown>;

	/** Optional string appended to every eval result (e.g., game score, budget remaining). */
	onEvalResult?: () => string | undefined;
}
