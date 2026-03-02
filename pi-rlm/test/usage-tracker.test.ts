import { describe, it, expect } from "vitest";
import { UsageTracker } from "../src/usage-tracker.js";
import type { Usage } from "@mariozechner/pi-ai";

function mockUsage(overrides: Partial<Usage> = {}): Usage {
	return {
		input: 100,
		output: 50,
		cacheRead: 10,
		cacheWrite: 5,
		totalTokens: 165,
		cost: {
			input: 0.001,
			output: 0.002,
			cacheRead: 0.0001,
			cacheWrite: 0.0002,
			total: 0.0033,
		},
		...overrides,
	};
}

describe("UsageTracker", () => {
	it("starts with zero totals", () => {
		const tracker = new UsageTracker();
		const total = tracker.totalUsage();
		expect(total.inputTokens).toBe(0);
		expect(total.outputTokens).toBe(0);
		expect(total.totalTokens).toBe(0);
		expect(total.totalCost).toBe(0);
	});

	it("records and accumulates usage for a single agent", () => {
		const tracker = new UsageTracker();
		tracker.recordUsage(0, mockUsage());
		tracker.recordUsage(0, mockUsage());

		const total = tracker.totalUsage();
		expect(total.inputTokens).toBe(200);
		expect(total.outputTokens).toBe(100);
		expect(total.cacheReadTokens).toBe(20);
		expect(total.cacheWriteTokens).toBe(10);
		expect(total.totalTokens).toBe(330);
		expect(total.totalCost).toBeCloseTo(0.0066);
	});

	it("tracks per-agent usage separately", () => {
		const tracker = new UsageTracker();
		tracker.recordUsage(0, mockUsage({ input: 100, output: 50 }));
		tracker.recordUsage(1, mockUsage({ input: 200, output: 80 }));

		const perAgent = tracker.perAgentUsage();
		expect(perAgent.size).toBe(2);
		expect(perAgent.get(0)!.inputTokens).toBe(100);
		expect(perAgent.get(1)!.inputTokens).toBe(200);
	});

	it("total sums across all agents", () => {
		const tracker = new UsageTracker();
		tracker.recordUsage(0, mockUsage({ input: 100 }));
		tracker.recordUsage(1, mockUsage({ input: 200 }));

		const total = tracker.totalUsage();
		expect(total.inputTokens).toBe(300);
	});

	it("produces a formatted summary", () => {
		const tracker = new UsageTracker();
		tracker.recordUsage(0, mockUsage());
		const summary = tracker.summary();
		expect(summary).toContain("Token Usage Summary");
		expect(summary).toContain("#0");
		expect(summary).toContain("TOTAL");
	});

	it("produces a JSON-serializable snapshot", () => {
		const tracker = new UsageTracker();
		tracker.recordUsage(0, mockUsage());
		const json = tracker.toJSON();
		expect(json).toHaveProperty("total");
		expect(json).toHaveProperty("perAgent");
		// Ensure it round-trips through JSON
		const str = JSON.stringify(json);
		expect(() => JSON.parse(str)).not.toThrow();
	});

	it("perAgentUsage returns a copy", () => {
		const tracker = new UsageTracker();
		tracker.recordUsage(0, mockUsage());
		const map1 = tracker.perAgentUsage();
		const map2 = tracker.perAgentUsage();
		expect(map1).not.toBe(map2);
	});
});
