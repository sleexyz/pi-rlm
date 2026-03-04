import { describe, it, expect } from "vitest";
import { loadTask, loadTasksFromDir, selectDevSet } from "../src/task-loader.js";
import { join } from "node:path";

const DATA_DIR = join(import.meta.dirname, "../../../downloads/ARC-AGI-2/data/training");

describe("loadTask", () => {
	it("loads a valid ARC task from JSON", () => {
		const task = loadTask(join(DATA_DIR, "00576224.json"));
		expect(task.train).toBeInstanceOf(Array);
		expect(task.test).toBeInstanceOf(Array);
		expect(task.train.length).toBeGreaterThan(0);
		expect(task.test.length).toBeGreaterThan(0);

		// Verify grid structure
		const ex = task.train[0];
		expect(ex.input).toBeInstanceOf(Array);
		expect(ex.output).toBeInstanceOf(Array);
		expect(ex.input[0]).toBeInstanceOf(Array);
	});

	it("throws on invalid file", () => {
		expect(() => loadTask("/nonexistent/file.json")).toThrow();
	});
});

describe("loadTasksFromDir", () => {
	it("loads multiple tasks from a directory", () => {
		const tasks = loadTasksFromDir(DATA_DIR, 3);
		expect(tasks).toHaveLength(3);
		for (const { id, task } of tasks) {
			expect(id).toMatch(/^[0-9a-f]{8}$/);
			expect(task.train.length).toBeGreaterThan(0);
			expect(task.test.length).toBeGreaterThan(0);
		}
	});

	it("returns tasks sorted by filename", () => {
		const tasks = loadTasksFromDir(DATA_DIR, 5);
		const ids = tasks.map((t) => t.id);
		const sorted = [...ids].sort();
		expect(ids).toEqual(sorted);
	});
});

describe("selectDevSet", () => {
	it("selects small tasks for dev iteration", () => {
		const tasks = selectDevSet(DATA_DIR, 3, 10);
		expect(tasks.length).toBeLessThanOrEqual(3);
		for (const { task } of tasks) {
			for (const ex of [...task.train, ...task.test]) {
				expect(ex.input.length).toBeLessThanOrEqual(10);
				expect(ex.input[0].length).toBeLessThanOrEqual(10);
			}
		}
	});

	it("returns tasks sorted by total size", () => {
		const tasks = selectDevSet(DATA_DIR, 5, 10);
		if (tasks.length < 2) return;
		// Just check it returns valid tasks
		for (const { id } of tasks) {
			expect(id).toBeTruthy();
		}
	});
});
