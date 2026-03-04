import type { Adapter } from "./domain-adapter.js";

/**
 * Simple demo domain for testing the TUI.
 * Provides basic math/utility functions in scope.
 */
export function createDemoAdapter(): Adapter {
	return {
		name: "Demo",
		premise:
			"You are a helpful assistant with access to a code eval environment. " +
			"Use the eval tool to compute answers. Call resolve(result) with the final answer.",
		reference: [
			"This is a demo environment with basic utility functions.",
			"",
			"Available functions:",
			"  add(a, b) — Add two numbers",
			"  multiply(a, b) — Multiply two numbers",
			"  range(n) — Return array [0, 1, ..., n-1]",
			"  sleep(ms) — Wait for ms milliseconds",
			"",
			"Call resolve(value) when you have the final answer.",
		].join("\n"),
		scope: {
			add: (a: number, b: number) => a + b,
			multiply: (a: number, b: number) => a * b,
			range: (n: number) => Array.from({ length: n }, (_, i) => i),
			sleep: (ms: number) => new Promise((resolve) => setTimeout(resolve, ms)),
		},
	};
}
