import { describe, it, expect } from "vitest";
import {
	renderGrid,
	gridShape,
	makeGrid,
	copyGrid,
	gridsEqual,
	rotate90,
	rotate180,
	rotate270,
	flipH,
	flipV,
	transpose,
	crop,
	paste,
	tile,
	findColor,
	colorCounts,
	replaceColor,
	uniqueColors,
	connectedComponents,
	accuracy,
	softAccuracy,
} from "../src/grid-helpers.js";

// ── Rendering ──

describe("renderGrid", () => {
	it("renders a simple grid with indices", () => {
		const grid = [[1, 2], [3, 4]];
		const rendered = renderGrid(grid);
		expect(rendered).toContain("1");
		expect(rendered).toContain("4");
		// Should have column header and 2 data rows
		expect(rendered.split("\n")).toHaveLength(3);
	});

	it("handles empty grid", () => {
		expect(renderGrid([])).toBe("(empty grid)");
	});
});

// ── Shape & Construction ──

describe("gridShape", () => {
	it("returns [rows, cols]", () => {
		expect(gridShape([[1, 2, 3], [4, 5, 6]])).toEqual([2, 3]);
	});

	it("handles empty grid", () => {
		expect(gridShape([])).toEqual([0, 0]);
	});
});

describe("makeGrid", () => {
	it("creates a grid filled with default value", () => {
		const g = makeGrid(2, 3);
		expect(g).toEqual([[0, 0, 0], [0, 0, 0]]);
	});

	it("creates a grid filled with custom value", () => {
		const g = makeGrid(2, 2, 5);
		expect(g).toEqual([[5, 5], [5, 5]]);
	});
});

describe("copyGrid", () => {
	it("creates a deep copy", () => {
		const original = [[1, 2], [3, 4]];
		const copied = copyGrid(original);
		expect(copied).toEqual(original);
		copied[0][0] = 99;
		expect(original[0][0]).toBe(1); // original unchanged
	});
});

describe("gridsEqual", () => {
	it("returns true for identical grids", () => {
		expect(gridsEqual([[1, 2], [3, 4]], [[1, 2], [3, 4]])).toBe(true);
	});

	it("returns false for different values", () => {
		expect(gridsEqual([[1, 2]], [[1, 3]])).toBe(false);
	});

	it("returns false for different shapes", () => {
		expect(gridsEqual([[1, 2]], [[1], [2]])).toBe(false);
	});
});

// ── Transforms ──

describe("rotate90", () => {
	it("rotates 90° clockwise", () => {
		const grid = [[1, 2], [3, 4]];
		expect(rotate90(grid)).toEqual([[3, 1], [4, 2]]);
	});

	it("rotating 4 times returns original", () => {
		const grid = [[1, 2, 3], [4, 5, 6]];
		expect(rotate90(rotate90(rotate90(rotate90(grid))))).toEqual(grid);
	});
});

describe("rotate180", () => {
	it("rotates 180°", () => {
		const grid = [[1, 2], [3, 4]];
		expect(rotate180(grid)).toEqual([[4, 3], [2, 1]]);
	});
});

describe("rotate270", () => {
	it("rotates 270° clockwise (= 90° counter-clockwise)", () => {
		const grid = [[1, 2], [3, 4]];
		expect(rotate270(grid)).toEqual([[2, 4], [1, 3]]);
	});
});

describe("flipH", () => {
	it("mirrors left-right", () => {
		expect(flipH([[1, 2, 3], [4, 5, 6]])).toEqual([[3, 2, 1], [6, 5, 4]]);
	});
});

describe("flipV", () => {
	it("mirrors top-bottom", () => {
		expect(flipV([[1, 2], [3, 4]])).toEqual([[3, 4], [1, 2]]);
	});
});

describe("transpose", () => {
	it("swaps rows and columns", () => {
		expect(transpose([[1, 2, 3], [4, 5, 6]])).toEqual([[1, 4], [2, 5], [3, 6]]);
	});
});

// ── Region Operations ──

describe("crop", () => {
	it("extracts a sub-grid", () => {
		const grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
		expect(crop(grid, 0, 1, 2, 3)).toEqual([[2, 3], [5, 6]]);
	});
});

describe("paste", () => {
	it("pastes source onto target", () => {
		const target = makeGrid(3, 3);
		const source = [[1, 2], [3, 4]];
		const result = paste(target, source, 1, 1);
		expect(result).toEqual([[0, 0, 0], [0, 1, 2], [0, 3, 4]]);
	});

	it("clips when source extends beyond target", () => {
		const target = makeGrid(2, 2);
		const source = [[1, 2], [3, 4]];
		const result = paste(target, source, 1, 1);
		expect(result).toEqual([[0, 0], [0, 1]]);
	});
});

describe("tile", () => {
	it("tiles a grid", () => {
		const grid = [[1, 2], [3, 4]];
		const result = tile(grid, 2, 2);
		expect(result).toEqual([
			[1, 2, 1, 2],
			[3, 4, 3, 4],
			[1, 2, 1, 2],
			[3, 4, 3, 4],
		]);
	});

	it("tiles 1x3", () => {
		const grid = [[1]];
		expect(tile(grid, 1, 3)).toEqual([[1, 1, 1]]);
	});
});

// ── Color Operations ──

describe("findColor", () => {
	it("finds all positions of a color", () => {
		const grid = [[1, 0, 1], [0, 1, 0]];
		expect(findColor(grid, 1)).toEqual([[0, 0], [0, 2], [1, 1]]);
	});

	it("returns empty array for missing color", () => {
		expect(findColor([[0, 0]], 5)).toEqual([]);
	});
});

describe("colorCounts", () => {
	it("counts each color", () => {
		const grid = [[1, 2, 1], [2, 2, 3]];
		expect(colorCounts(grid)).toEqual({ 1: 2, 2: 3, 3: 1 });
	});
});

describe("replaceColor", () => {
	it("replaces all occurrences", () => {
		expect(replaceColor([[1, 2, 1]], 1, 9)).toEqual([[9, 2, 9]]);
	});
});

describe("uniqueColors", () => {
	it("returns sorted distinct colors", () => {
		expect(uniqueColors([[3, 1, 2], [1, 3, 0]])).toEqual([0, 1, 2, 3]);
	});
});

// ── Connected Components ──

describe("connectedComponents", () => {
	it("finds separate components", () => {
		const grid = [
			[1, 0, 2],
			[1, 0, 2],
			[0, 0, 0],
		];
		const comps = connectedComponents(grid);
		expect(comps).toHaveLength(2);

		const c1 = comps.find((c) => c.color === 1)!;
		expect(c1.cells).toHaveLength(2);
		expect(c1.bbox).toEqual([0, 0, 1, 0]);

		const c2 = comps.find((c) => c.color === 2)!;
		expect(c2.cells).toHaveLength(2);
		expect(c2.bbox).toEqual([0, 2, 1, 2]);
	});

	it("skips background color 0 by default", () => {
		const grid = [[0, 1], [0, 0]];
		const comps = connectedComponents(grid);
		expect(comps).toHaveLength(1);
		expect(comps[0].color).toBe(1);
	});

	it("handles diagonal adjacency when enabled", () => {
		const grid = [
			[1, 0],
			[0, 1],
		];
		// Without diagonal: 2 separate components
		const nodiag = connectedComponents(grid);
		expect(nodiag).toHaveLength(2);

		// With diagonal: 1 component
		const diag = connectedComponents(grid, { diagonal: true });
		expect(diag).toHaveLength(1);
		expect(diag[0].cells).toHaveLength(2);
	});

	it("finds L-shaped component", () => {
		const grid = [
			[1, 0],
			[1, 1],
		];
		const comps = connectedComponents(grid);
		expect(comps).toHaveLength(1);
		expect(comps[0].cells).toHaveLength(3);
		expect(comps[0].bbox).toEqual([0, 0, 1, 1]);
	});

	it("handles single-color grid", () => {
		const grid = [[5, 5], [5, 5]];
		const comps = connectedComponents(grid);
		expect(comps).toHaveLength(1);
		expect(comps[0].color).toBe(5);
		expect(comps[0].cells).toHaveLength(4);
	});

	it("handles 1×1 grid", () => {
		const comps = connectedComponents([[3]]);
		expect(comps).toHaveLength(1);
		expect(comps[0].color).toBe(3);
	});

	it("returns empty for all-background grid", () => {
		const comps = connectedComponents([[0, 0], [0, 0]]);
		expect(comps).toHaveLength(0);
	});

	it("custom background color", () => {
		const grid = [[1, 1], [0, 0]];
		const comps = connectedComponents(grid, { background: 1 });
		expect(comps).toHaveLength(1);
		expect(comps[0].color).toBe(0);
	});
});

// ── Scoring ──

describe("accuracy", () => {
	it("returns 1.0 for exact match", () => {
		expect(accuracy([[1, 2], [3, 4]], [[1, 2], [3, 4]])).toBe(1.0);
	});

	it("returns 0.0 for any difference", () => {
		expect(accuracy([[1, 2]], [[1, 3]])).toBe(0.0);
	});

	it("returns 0.0 for shape mismatch", () => {
		expect(accuracy([[1]], [[1, 2]])).toBe(0.0);
	});
});

describe("softAccuracy", () => {
	it("returns 1.0 for exact match", () => {
		expect(softAccuracy([[1, 2], [3, 4]], [[1, 2], [3, 4]])).toBe(1.0);
	});

	it("returns 0.5 for half matching", () => {
		expect(softAccuracy([[1, 2, 3, 4]], [[1, 2, 9, 9]])).toBe(0.5);
	});

	it("returns 0.0 for shape mismatch", () => {
		expect(softAccuracy([[1]], [[1, 2]])).toBe(0.0);
	});

	it("returns 0.0 for no matches", () => {
		expect(softAccuracy([[1, 2]], [[3, 4]])).toBe(0.0);
	});
});
