import type { ArcGrid } from "./types.js";

// ── Rendering ──

/** ASCII render of a grid with row/col indices. */
export function renderGrid(grid: ArcGrid): string {
	if (grid.length === 0) return "(empty grid)";
	const rows = grid.length;
	const cols = grid[0].length;
	const rowLabelWidth = String(rows - 1).length;

	// Column header
	const colHeader =
		" ".repeat(rowLabelWidth + 1) +
		Array.from({ length: cols }, (_, c) => String(c).padStart(2)).join("");

	const lines = [colHeader];
	for (let r = 0; r < rows; r++) {
		const label = String(r).padStart(rowLabelWidth);
		const cells = grid[r].map((v) => String(v).padStart(2)).join("");
		lines.push(`${label} ${cells}`);
	}
	return lines.join("\n");
}

// ── Shape & Construction ──

/** Returns [rows, cols] of the grid. */
export function gridShape(grid: ArcGrid): [number, number] {
	return [grid.length, grid.length > 0 ? grid[0].length : 0];
}

/** Create a new grid filled with a value (default 0). */
export function makeGrid(rows: number, cols: number, fill: number = 0): ArcGrid {
	return Array.from({ length: rows }, () => Array(cols).fill(fill));
}

/** Deep copy a grid. */
export function copyGrid(grid: ArcGrid): ArcGrid {
	return grid.map((row) => [...row]);
}

/** Check if two grids are identical. */
export function gridsEqual(a: ArcGrid, b: ArcGrid): boolean {
	if (a.length !== b.length) return false;
	for (let r = 0; r < a.length; r++) {
		if (a[r].length !== b[r].length) return false;
		for (let c = 0; c < a[r].length; c++) {
			if (a[r][c] !== b[r][c]) return false;
		}
	}
	return true;
}

// ── Transforms ──

/** Rotate 90° clockwise. */
export function rotate90(grid: ArcGrid): ArcGrid {
	const rows = grid.length;
	const cols = grid[0]?.length ?? 0;
	const out: ArcGrid = makeGrid(cols, rows);
	for (let r = 0; r < rows; r++) {
		for (let c = 0; c < cols; c++) {
			out[c][rows - 1 - r] = grid[r][c];
		}
	}
	return out;
}

/** Rotate 180°. */
export function rotate180(grid: ArcGrid): ArcGrid {
	return rotate90(rotate90(grid));
}

/** Rotate 270° clockwise (= 90° counter-clockwise). */
export function rotate270(grid: ArcGrid): ArcGrid {
	return rotate90(rotate90(rotate90(grid)));
}

/** Flip horizontally (mirror left-right). */
export function flipH(grid: ArcGrid): ArcGrid {
	return grid.map((row) => [...row].reverse());
}

/** Flip vertically (mirror top-bottom). */
export function flipV(grid: ArcGrid): ArcGrid {
	return [...grid].reverse().map((row) => [...row]);
}

/** Transpose (swap rows and columns). */
export function transpose(grid: ArcGrid): ArcGrid {
	const rows = grid.length;
	const cols = grid[0]?.length ?? 0;
	const out: ArcGrid = makeGrid(cols, rows);
	for (let r = 0; r < rows; r++) {
		for (let c = 0; c < cols; c++) {
			out[c][r] = grid[r][c];
		}
	}
	return out;
}

// ── Region Operations ──

/** Extract sub-grid (r2/c2 exclusive). */
export function crop(grid: ArcGrid, r1: number, c1: number, r2: number, c2: number): ArcGrid {
	const out: ArcGrid = [];
	for (let r = r1; r < r2; r++) {
		out.push(grid[r].slice(c1, c2));
	}
	return out;
}

/** Paste source onto target at position (r, c). Returns new grid. */
export function paste(target: ArcGrid, source: ArcGrid, r: number, c: number): ArcGrid {
	const out = copyGrid(target);
	for (let sr = 0; sr < source.length; sr++) {
		for (let sc = 0; sc < source[sr].length; sc++) {
			const tr = r + sr;
			const tc = c + sc;
			if (tr >= 0 && tr < out.length && tc >= 0 && tc < out[0].length) {
				out[tr][tc] = source[sr][sc];
			}
		}
	}
	return out;
}

/** Tile the grid into a bigger grid of (rows × cols) tiles. */
export function tile(grid: ArcGrid, rows: number, cols: number): ArcGrid {
	const h = grid.length;
	const w = grid[0]?.length ?? 0;
	const out = makeGrid(h * rows, w * cols);
	for (let tr = 0; tr < rows; tr++) {
		for (let tc = 0; tc < cols; tc++) {
			for (let r = 0; r < h; r++) {
				for (let c = 0; c < w; c++) {
					out[tr * h + r][tc * w + c] = grid[r][c];
				}
			}
		}
	}
	return out;
}

// ── Color Operations ──

/** Find all [row, col] positions of a given color. */
export function findColor(grid: ArcGrid, color: number): [number, number][] {
	const positions: [number, number][] = [];
	for (let r = 0; r < grid.length; r++) {
		for (let c = 0; c < grid[r].length; c++) {
			if (grid[r][c] === color) positions.push([r, c]);
		}
	}
	return positions;
}

/** Count occurrences of each color. */
export function colorCounts(grid: ArcGrid): Record<number, number> {
	const counts: Record<number, number> = {};
	for (const row of grid) {
		for (const v of row) {
			counts[v] = (counts[v] ?? 0) + 1;
		}
	}
	return counts;
}

/** Replace all cells of one color with another. Returns new grid. */
export function replaceColor(grid: ArcGrid, from: number, to: number): ArcGrid {
	return grid.map((row) => row.map((v) => (v === from ? to : v)));
}

/** Array of distinct colors present in the grid. */
export function uniqueColors(grid: ArcGrid): number[] {
	const seen = new Set<number>();
	for (const row of grid) {
		for (const v of row) seen.add(v);
	}
	return [...seen].sort((a, b) => a - b);
}

// ── Connected Components ──

export interface Component {
	label: number;
	color: number;
	cells: [number, number][];
	bbox: [number, number, number, number]; // [minRow, minCol, maxRow, maxCol]
}

export interface ConnectedComponentsOptions {
	/** Color to skip (default: 0). Set to -1 to include all colors. */
	background?: number;
	/** Whether diagonal adjacency counts (default: false). */
	diagonal?: boolean;
}

/**
 * Flood-fill labeling of connected components.
 * Returns an array of components, each with label, color, cells, and bounding box.
 */
export function connectedComponents(
	grid: ArcGrid,
	options?: ConnectedComponentsOptions,
): Component[] {
	const bg = options?.background ?? 0;
	const diag = options?.diagonal ?? false;
	const rows = grid.length;
	const cols = grid[0]?.length ?? 0;
	const visited: boolean[][] = Array.from({ length: rows }, () => Array(cols).fill(false));

	const neighbors: [number, number][] = [
		[-1, 0], [1, 0], [0, -1], [0, 1],
	];
	if (diag) {
		neighbors.push([-1, -1], [-1, 1], [1, -1], [1, 1]);
	}

	const components: Component[] = [];
	let nextLabel = 1;

	for (let r = 0; r < rows; r++) {
		for (let c = 0; c < cols; c++) {
			if (visited[r][c]) continue;
			const color = grid[r][c];
			if (color === bg) {
				visited[r][c] = true;
				continue;
			}

			// BFS flood fill
			const cells: [number, number][] = [];
			const queue: [number, number][] = [[r, c]];
			visited[r][c] = true;

			while (queue.length > 0) {
				const [cr, cc] = queue.shift()!;
				cells.push([cr, cc]);

				for (const [dr, dc] of neighbors) {
					const nr = cr + dr;
					const nc = cc + dc;
					if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && !visited[nr][nc] && grid[nr][nc] === color) {
						visited[nr][nc] = true;
						queue.push([nr, nc]);
					}
				}
			}

			let minR = rows, minC = cols, maxR = 0, maxC = 0;
			for (const [cr, cc] of cells) {
				if (cr < minR) minR = cr;
				if (cr > maxR) maxR = cr;
				if (cc < minC) minC = cc;
				if (cc > maxC) maxC = cc;
			}

			components.push({
				label: nextLabel++,
				color,
				cells,
				bbox: [minR, minC, maxR, maxC],
			});
		}
	}

	return components;
}

// ── Scoring ──

/** 1.0 if shapes and all elements match, 0.0 otherwise. */
export function accuracy(predicted: ArcGrid, expected: ArcGrid): number {
	return gridsEqual(predicted, expected) ? 1.0 : 0.0;
}

/** Element-wise match ratio (0.0-1.0). 0.0 if shapes differ. */
export function softAccuracy(predicted: ArcGrid, expected: ArcGrid): number {
	if (predicted.length !== expected.length) return 0.0;
	let total = 0;
	let matches = 0;
	for (let r = 0; r < predicted.length; r++) {
		if (predicted[r].length !== expected[r].length) return 0.0;
		for (let c = 0; c < predicted[r].length; c++) {
			total++;
			if (predicted[r][c] === expected[r][c]) matches++;
		}
	}
	return total === 0 ? 1.0 : matches / total;
}
