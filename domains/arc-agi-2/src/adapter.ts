import type { Adapter } from "pi-rlm";
import type { ArcTask } from "./types.js";
import { ARC_PREMISE, ARC_REFERENCE, ARC_SUB_AGENT_PROMPT } from "./prompt.js";
import * as helpers from "./grid-helpers.js";

/**
 * Creates a Adapter for a single ARC task.
 * Injects training examples, test input, and grid helpers into scope.
 */
export function createArcAdapter(task: ArcTask): Adapter {
	return {
		name: "ARC-AGI-2",
		premise: ARC_PREMISE,
		reference: ARC_REFERENCE,
		defaultSubAgentPrompt: ARC_SUB_AGENT_PROMPT,

		scope: {
			// Data
			trainingExamples: task.train.map((ex) => ({
				input: ex.input,
				output: ex.output,
			})),
			testInputs: task.test.map((ex) => ex.input),

			// Grid helpers
			renderGrid: helpers.renderGrid,
			gridShape: helpers.gridShape,
			makeGrid: helpers.makeGrid,
			copyGrid: helpers.copyGrid,
			gridsEqual: helpers.gridsEqual,
			rotate90: helpers.rotate90,
			rotate180: helpers.rotate180,
			rotate270: helpers.rotate270,
			flipH: helpers.flipH,
			flipV: helpers.flipV,
			transpose: helpers.transpose,
			crop: helpers.crop,
			paste: helpers.paste,
			tile: helpers.tile,
			findColor: helpers.findColor,
			colorCounts: helpers.colorCounts,
			replaceColor: helpers.replaceColor,
			uniqueColors: helpers.uniqueColors,
			connectedComponents: helpers.connectedComponents,

			// Scoring
			accuracy: helpers.accuracy,
			softAccuracy: helpers.softAccuracy,
		},
	};
}
