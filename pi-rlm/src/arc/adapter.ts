import type { DomainAdapter } from "../domain-adapter.js";
import type { ArcTask } from "./types.js";
import { ARC_PREMISE, ARC_REFERENCE } from "./prompt.js";
import * as helpers from "./grid-helpers.js";

/**
 * Creates a DomainAdapter for a single ARC task.
 * The adapter injects training examples, test input, and grid helpers
 * into the eval scope. The agent calls resolve(grid) to submit an answer.
 */
export function createArcAdapter(task: ArcTask): DomainAdapter {
	return {
		name: "ARC-AGI-2",
		premise: ARC_PREMISE,
		reference: ARC_REFERENCE,

		generateSystemPrompt() {
			return `${ARC_PREMISE}

## Code Execution Environment

You have a single tool: \`eval\`. It executes JavaScript code in a persistent environment.

### Variable Persistence
All declarations (\`let\`, \`const\`, \`var\`) persist across eval calls. You can re-declare variables freely — the latest value wins.

### Top-level await
You can use \`await\` directly in your code.

### Console output
Use \`console.log()\` to print values. Output is captured and returned to you.

## Sub-agents (Optional)

\`spawnAgent(systemPrompt?)\` creates a sub-agent for parallel hypothesis exploration.
\`agent.call(task, objects?)\` sends a task to the agent. Returns the resolved value.

Include \`DOMAIN_REFERENCE\` in sub-agent system prompts:
\`\`\`js
const agent = await spawnAgent("You are an ARC solver.\\n\\n" + DOMAIN_REFERENCE);
const result = await agent.call("Test hypothesis X", { trainingExamples, testInput });
\`\`\`

## ARC-AGI-2 Reference

${ARC_REFERENCE}`;
		},

		getScope() {
			return {
				// Data
				trainingExamples: task.train.map((ex) => ({
					input: ex.input,
					output: ex.output,
				})),
				testInput: task.test[0].input,

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
			};
		},
	};
}
