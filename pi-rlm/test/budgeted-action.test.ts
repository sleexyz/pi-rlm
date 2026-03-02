import { describe, it, expect } from "vitest";
import { createBudgetedAction } from "../src/budgeted-action.js";

describe("createBudgetedAction", () => {
	it("preserves original function behavior", () => {
		const add = (a: number, b: number) => a + b;
		const bounded = createBudgetedAction(add, 10);
		expect(bounded(3, 4)).toBe(7);
	});

	it("tracks used and remaining counts", () => {
		const fn = () => "ok";
		const bounded = createBudgetedAction(fn, 3);

		expect(bounded.limit).toBe(3);
		expect(bounded.used).toBe(0);
		expect(bounded.remaining).toBe(3);

		bounded();
		expect(bounded.used).toBe(1);
		expect(bounded.remaining).toBe(2);

		bounded();
		bounded();
		expect(bounded.used).toBe(3);
		expect(bounded.remaining).toBe(0);
	});

	it("throws when budget is exhausted", () => {
		const fn = () => "ok";
		const bounded = createBudgetedAction(fn, 2);

		bounded();
		bounded();
		expect(() => bounded()).toThrow(/budget exhausted/i);
	});

	it("free calls do not count toward budget", () => {
		const fn = (action: string) => action;
		const bounded = createBudgetedAction(fn, 2, {
			isFree: (action: unknown) => action === "RESET",
		});

		// Free call
		expect(bounded("RESET")).toBe("RESET");
		expect(bounded.used).toBe(0);
		expect(bounded.remaining).toBe(2);

		// Non-free calls
		bounded("ACTION1");
		expect(bounded.used).toBe(1);

		bounded("ACTION2");
		expect(bounded.used).toBe(2);

		// Exhausted for non-free
		expect(() => bounded("ACTION3")).toThrow(/budget exhausted/i);

		// But free calls still work
		expect(bounded("RESET")).toBe("RESET");
		expect(bounded.used).toBe(2);
	});

	it("properties are read-only getters", () => {
		const bounded = createBudgetedAction(() => {}, 5);

		// Properties should be defined as getters
		const desc = Object.getOwnPropertyDescriptor(bounded, "remaining");
		expect(desc).toBeDefined();
		expect(desc!.get).toBeDefined();
		expect(desc!.set).toBeUndefined();
	});

	it("works with async functions", async () => {
		const asyncFn = async (x: number) => x * 2;
		const bounded = createBudgetedAction(asyncFn, 2);

		const result = await bounded(5);
		expect(result).toBe(10);
		expect(bounded.used).toBe(1);
	});
});
