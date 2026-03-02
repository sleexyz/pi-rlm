import { describe, it, expect } from "vitest";
import { ActionHistory, createTrackedAction } from "../src/action-history.js";

describe("ActionHistory", () => {
	it("starts empty", () => {
		const h = new ActionHistory();
		expect(h.length).toBe(0);
		expect(h.entries()).toEqual([]);
	});

	it("records and retrieves entries", () => {
		const h = new ActionHistory();
		h.record("ACTION1", [], "result1");
		h.record("ACTION2", [1, 2], "result2");

		expect(h.length).toBe(2);
		const all = h.entries();
		expect(all).toHaveLength(2);
		expect(all[0].action).toBe("ACTION1");
		expect(all[1].action).toBe("ACTION2");
		expect(all[1].args).toEqual([1, 2]);
	});

	it("entries(n) returns last N oldest-first", () => {
		const h = new ActionHistory();
		h.record("A", [], null);
		h.record("B", [], null);
		h.record("C", [], null);

		const last2 = h.entries(2);
		expect(last2).toHaveLength(2);
		expect(last2[0].action).toBe("B");
		expect(last2[1].action).toBe("C");
	});

	it("entries(n) returns all if n >= length", () => {
		const h = new ActionHistory();
		h.record("A", [], null);
		expect(h.entries(100)).toHaveLength(1);
	});

	it("evicts oldest entries when exceeding maxEntries", () => {
		const h = new ActionHistory(3);
		h.record("A", [], null);
		h.record("B", [], null);
		h.record("C", [], null);
		h.record("D", [], null); // Should evict "A"

		expect(h.length).toBe(3);
		const all = h.entries();
		expect(all[0].action).toBe("B");
		expect(all[2].action).toBe("D");
	});

	it("entries include timestamps", () => {
		const h = new ActionHistory();
		const before = Date.now();
		h.record("X", [], null);
		const after = Date.now();

		const entry = h.entries()[0];
		expect(entry.timestamp).toBeGreaterThanOrEqual(before);
		expect(entry.timestamp).toBeLessThanOrEqual(after);
	});

	it("entries() returns a copy", () => {
		const h = new ActionHistory();
		h.record("A", [], null);
		const e1 = h.entries();
		const e2 = h.entries();
		expect(e1).not.toBe(e2);
	});
});

describe("createTrackedAction", () => {
	it("wraps a function to auto-record calls", () => {
		const h = new ActionHistory();
		const fn = (a: number, b: number) => a + b;
		const tracked = createTrackedAction(fn, "add", h);

		const result = tracked(3, 4);
		expect(result).toBe(7);
		expect(h.length).toBe(1);

		const entry = h.entries()[0];
		expect(entry.action).toBe("add");
		expect(entry.args).toEqual([3, 4]);
		expect(entry.result).toBe(7);
	});

	it("records multiple calls", () => {
		const h = new ActionHistory();
		const fn = (x: string) => x.toUpperCase();
		const tracked = createTrackedAction(fn, "upper", h);

		tracked("hello");
		tracked("world");
		expect(h.length).toBe(2);
	});
});
