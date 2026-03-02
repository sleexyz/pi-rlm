/**
 * A recorded action entry.
 */
export interface HistoryEntry {
	action: string;
	args: unknown[];
	result: unknown;
	timestamp: number;
}

/**
 * Ring-buffer action history shared across all agents.
 */
export class ActionHistory {
	private buffer: HistoryEntry[] = [];
	private maxEntries: number;

	constructor(maxEntries = 50) {
		this.maxEntries = maxEntries;
	}

	/** Record an action. */
	record(action: string, args: unknown[], result: unknown): void {
		this.buffer.push({
			action,
			args,
			result,
			timestamp: Date.now(),
		});
		// Evict oldest if over capacity
		if (this.buffer.length > this.maxEntries) {
			this.buffer.splice(0, this.buffer.length - this.maxEntries);
		}
	}

	/** Get last N entries (oldest first). If N is omitted, returns all. */
	entries(n?: number): HistoryEntry[] {
		if (n == null || n >= this.buffer.length) return [...this.buffer];
		return this.buffer.slice(-n);
	}

	/** Number of recorded entries. */
	get length(): number {
		return this.buffer.length;
	}
}

/**
 * Wraps a function to automatically record calls in an ActionHistory.
 * Returns a new function with the same signature.
 */
export function createTrackedAction<T extends (...args: any[]) => any>(
	fn: T,
	name: string,
	history: ActionHistory,
): T {
	const tracked = function (this: unknown, ...args: unknown[]) {
		const result = fn.apply(this, args);
		history.record(name, args, result);
		return result;
	} as unknown as T;
	return tracked;
}
