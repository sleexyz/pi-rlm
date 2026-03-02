/**
 * Interface for domain plugins. Any domain (ARC-AGI, code generation, etc.)
 * plugs into the orchestrator by implementing this interface.
 */
export interface DomainAdapter {
	/** Domain name used in system prompt context. */
	name: string;

	/** Domain reference documentation (equivalent to GAME_REFERENCE). */
	reference: string;

	/** Orchestrator role/instructions (equivalent to premise()). */
	premise: string;

	/** Domain-specific objects to inject into the eval scope. */
	getScope(): Record<string, unknown>;

	/** Optional check for whether the domain task is complete. */
	isComplete?(): boolean;

	/** Optional status line appended to eval results (e.g., current score, level). */
	getStatus?(): string;

	/**
	 * Names of scope functions that are domain "actions" (trackable, budgetable).
	 * If omitted, no functions are auto-tracked in the action history.
	 */
	actionNames?: string[];

	/**
	 * Optional custom system prompt generator. When provided, bypasses the
	 * default orchestrator system prompt (generateSystemPrompt in system-prompt.ts).
	 * Used by domains like ARC-AGI-2 that act as direct solvers, not orchestrators.
	 */
	generateSystemPrompt?(): string;
}
