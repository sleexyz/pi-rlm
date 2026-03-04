/** A 2D grid of integers 0-9 representing colored cells. */
export type ArcGrid = number[][];

/** An input-output pair for an ARC task. */
export interface ArcExample {
	input: ArcGrid;
	output: ArcGrid;
}

/** A complete ARC task with training and test examples. */
export interface ArcTask {
	train: ArcExample[];
	test: ArcExample[];
}

/** Result of a single attempt on an ARC task. */
export interface ArcAttempt {
	predicted: ArcGrid;
	correct: boolean;
	failed?: boolean;
	error?: string;
	cost: number;
	tokens: number;
	timeMs: number;
}

/** Result of running pi-rlm on a single ARC task (pass@K). */
export interface ArcResult {
	taskId: string;
	correct: boolean;
	failed?: boolean;
	attempts: ArcAttempt[];
	expected: ArcGrid;
	cost: number;
	tokens: number;
	timeMs: number;
}
