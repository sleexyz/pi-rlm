import { describe, it, expect } from "vitest";
import { Memories, MemoryQueryError } from "../src/memories.js";
import { getModel } from "@mariozechner/pi-ai";

const model = getModel("anthropic", "claude-sonnet-4-6");

describe("Memories", () => {
	it("adds and retrieves memories", () => {
		const mem = new Memories(model);
		mem.add("First insight", "Details about first insight");
		mem.add("Second insight", "Details about second insight");

		expect(mem.stack).toHaveLength(2);
		expect(mem.get(0).summary).toBe("First insight");
		expect(mem.get(1).summary).toBe("Second insight");
	});

	it("supports negative indexing", () => {
		const mem = new Memories(model);
		mem.add("A", "a");
		mem.add("B", "b");
		mem.add("C", "c");

		expect(mem.get(-1).summary).toBe("C");
		expect(mem.get(-2).summary).toBe("B");
	});

	it("throws on out-of-range index", () => {
		const mem = new Memories(model);
		mem.add("A", "a");

		expect(() => mem.get(5)).toThrow(RangeError);
		expect(() => mem.get(-5)).toThrow(RangeError);
	});

	it("returns summaries", () => {
		const mem = new Memories(model);
		mem.add("Alpha", "a");
		mem.add("Beta", "b");

		const sums = mem.summaries();
		expect(sums).toEqual(["[0] Alpha", "[1] Beta"]);
	});

	it("evicts by index", () => {
		const mem = new Memories(model);
		mem.add("A", "a");
		mem.add("B", "b");
		mem.add("C", "c");

		const evicted = mem.evict(1);
		expect(evicted.summary).toBe("B");
		expect(mem.stack).toHaveLength(2);
		expect(mem.get(0).summary).toBe("A");
		expect(mem.get(1).summary).toBe("C");
	});

	it("evicts last by default", () => {
		const mem = new Memories(model);
		mem.add("A", "a");
		mem.add("B", "b");

		const evicted = mem.evict();
		expect(evicted.summary).toBe("B");
		expect(mem.stack).toHaveLength(1);
	});

	it("throws on evict out-of-range", () => {
		const mem = new Memories(model);
		expect(() => mem.evict(0)).toThrow(RangeError);
	});

	it("memories have timestamps", () => {
		const mem = new Memories(model);
		const before = new Date();
		mem.add("Timestamped", "details");
		const after = new Date();

		const ts = mem.get(0).timestamp;
		expect(ts.getTime()).toBeGreaterThanOrEqual(before.getTime());
		expect(ts.getTime()).toBeLessThanOrEqual(after.getTime());
	});
});

describe("MemoryQueryError", () => {
	it("is an Error subclass", () => {
		const err = new MemoryQueryError("test");
		expect(err).toBeInstanceOf(Error);
		expect(err).toBeInstanceOf(MemoryQueryError);
		expect(err.name).toBe("MemoryQueryError");
		expect(err.message).toBe("test");
	});
});
