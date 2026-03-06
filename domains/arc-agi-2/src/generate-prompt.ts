/**
 * One-shot prompt generator for ARC data preparation.
 *
 * Reads task JSON from stdin, prints the system prompt to stdout.
 * Used by arc_data.py to generate system prompts without porting TS to Python.
 *
 * Self-contained: inlines generateCodeBlockSystemPrompt to avoid importing
 * repl-stream.ts (which has heavy external deps not available in Modal).
 *
 * Usage: echo '{"train":[...],"test":[...]}' | bun run domains/arc-agi-2/src/generate-prompt.ts
 */

import { ARC_PREMISE, ARC_REFERENCE } from "./prompt.ts";
import type { ArcTask } from "./types.ts";

// Inlined from pi-rlm/src/repl-stream.ts to avoid external deps
function generateCodeBlockSystemPrompt(domain: { name: string; premise: string; reference: string }): string {
	return `${domain.premise}

## Code Execution Environment

You have a persistent JavaScript REPL. To execute code, wrap it in a \`\`\`js code block:

\`\`\`js
console.log("hello");
\`\`\`

After each code block, you'll see the execution result. You can then write more code blocks to continue.

### Variable Persistence
All declarations (\`let\`, \`const\`, \`var\`) persist across code blocks. You can re-declare variables freely — the latest value wins.

### Top-level await
You can use \`await\` directly in your code.

### Console output
Use \`console.log()\` to print values. Output is captured and returned to you.

### Important
- Write ONE code block at a time, then wait for the result before continuing.
- Do NOT put multiple \`\`\`js blocks in a single response — write one, see the result, then write the next.

## Sub-agents

\`spawnAgent(systemPrompt?)\` creates a sub-agent. Returns synchronously — the async work happens in \`.call()\`.
- No args: auto-configured with ${domain.name} context and reference.
- With custom prompt: uses exactly what you pass. Include \`DOMAIN_REFERENCE\` for domain context.

\`agent.call(task, objects?)\` sends a task to the agent. Returns the submitted value.

\`\`\`js
// One-step (preferred for one-off tasks)
const result = await spawnAgent().call("Do something", { data });

// Parallel exploration
const [r1, r2] = await Promise.all([
  spawnAgent().call("Hypothesis A", { data }),
  spawnAgent().call("Hypothesis B", { data }),
]);
\`\`\`

Sub-agents have access to \`spawnAgent\` for further delegation.

**submit(value)** — signals completion and returns a value.

## ${domain.name} Reference

${domain.reference}`;
}

const input = await Bun.stdin.text();
const task: ArcTask = JSON.parse(input);
const prompt = generateCodeBlockSystemPrompt({ name: "ARC-AGI-2", premise: ARC_PREMISE, reference: ARC_REFERENCE });
process.stdout.write(prompt);
