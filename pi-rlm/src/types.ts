export interface EvalResult {
	stdout: string;
	stderr: string;
	returnValue: unknown;
	error: string | null;
	durationMs: number;
}

export interface Memory {
	summary: string;
	details: string;
	timestamp: Date;
}
