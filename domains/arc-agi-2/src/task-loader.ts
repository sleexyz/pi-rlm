import { readFileSync, readdirSync } from "node:fs";
import { join, basename } from "node:path";
import type { ArcTask } from "./types.js";

/** Load a single ARC task from a JSON file. */
export function loadTask(filePath: string): ArcTask {
	const raw = readFileSync(filePath, "utf-8");
	const data = JSON.parse(raw);
	if (!data.train || !data.test || !Array.isArray(data.train) || !Array.isArray(data.test)) {
		throw new Error(`Invalid ARC task format in ${filePath}`);
	}
	return data as ArcTask;
}

/** Load tasks from a directory. Optionally limit to `count`. */
export function loadTasksFromDir(
	dirPath: string,
	count?: number,
): { id: string; task: ArcTask }[] {
	const files = readdirSync(dirPath)
		.filter((f) => f.endsWith(".json"))
		.sort();
	const selected = count !== undefined ? files.slice(0, count) : files;
	return selected.map((f) => ({
		id: basename(f, ".json"),
		task: loadTask(join(dirPath, f)),
	}));
}

/**
 * Select N tasks with small grids (max dimension < maxDim) for faster dev iteration.
 * Returns tasks sorted by total grid size (smallest first).
 */
export function selectDevSet(
	dirPath: string,
	count: number,
	maxDim: number = 10,
): { id: string; task: ArcTask }[] {
	const all = loadTasksFromDir(dirPath);
	const small = all.filter(({ task }) => {
		for (const ex of [...task.train, ...task.test]) {
			if (ex.input.length > maxDim) return false;
			if (ex.input[0]?.length > maxDim) return false;
			if (ex.output.length > maxDim) return false;
			if (ex.output[0]?.length > maxDim) return false;
		}
		return true;
	});

	// Sort by total grid cells (smallest first)
	small.sort((a, b) => {
		const sizeA = totalCells(a.task);
		const sizeB = totalCells(b.task);
		return sizeA - sizeB;
	});

	return small.slice(0, count);
}

function totalCells(task: ArcTask): number {
	let total = 0;
	for (const ex of [...task.train, ...task.test]) {
		total += ex.input.length * (ex.input[0]?.length ?? 0);
		total += ex.output.length * (ex.output[0]?.length ?? 0);
	}
	return total;
}
