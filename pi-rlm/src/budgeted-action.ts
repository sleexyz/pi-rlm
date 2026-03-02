/**
 * Options for createBudgetedAction.
 */
export interface BudgetedActionOptions {
	/** Predicate: returns true if the call should NOT count toward the budget. */
	isFree?: (...args: unknown[]) => boolean;
}

/**
 * A function augmented with budget tracking properties.
 */
export type BudgetedAction<T extends (...args: any[]) => any> = T & {
	readonly remaining: number;
	readonly used: number;
	readonly limit: number;
};

/**
 * Wraps a function with an action budget.
 * Returns a callable with .remaining, .used, .limit properties.
 * Throws Error when budget is exhausted (on non-free calls).
 */
export function createBudgetedAction<T extends (...args: any[]) => any>(
	fn: T,
	limit: number,
	options?: BudgetedActionOptions,
): BudgetedAction<T> {
	let used = 0;
	const isFree = options?.isFree;

	const wrapped = function (this: unknown, ...args: unknown[]) {
		const free = isFree ? isFree(...args) : false;
		if (!free) {
			if (used >= limit) {
				throw new Error(
					`Action budget exhausted: ${used}/${limit} actions used. No more non-free calls allowed.`,
				);
			}
			used++;
		}
		return fn.apply(this, args);
	} as unknown as BudgetedAction<T>;

	Object.defineProperty(wrapped, "remaining", {
		get: () => limit - used,
		enumerable: true,
	});

	Object.defineProperty(wrapped, "used", {
		get: () => used,
		enumerable: true,
	});

	Object.defineProperty(wrapped, "limit", {
		get: () => limit,
		enumerable: true,
	});

	return wrapped;
}
