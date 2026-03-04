import type { Adapter } from "./domain-adapter.js";

/**
 * Assembles the system prompt from domain fields.
 * premise + eval environment docs + sub-agent docs + reference.
 */
export function generateSystemPrompt(domain: Adapter): string {
	return `${domain.premise}

## Code Execution Environment

You have a single tool: \`eval\`. It executes JavaScript code in a persistent environment.

### Variable Persistence
All declarations (\`let\`, \`const\`, \`var\`) persist across eval calls. You can re-declare variables freely — the latest value wins.

### Top-level await
You can use \`await\` directly in your code.

### Console output
Use \`console.log()\` to print values. Output is captured and returned to you.

## Sub-agents

\`spawnAgent(systemPrompt?)\` creates a sub-agent. Returns synchronously — the async work happens in \`.call()\`.
- No args: auto-configured with ${domain.name} context and reference.
- With custom prompt: uses exactly what you pass. Include \`DOMAIN_REFERENCE\` for domain context.

\`agent.call(task, objects?)\` sends a task to the agent. Returns the resolved value.

\`\`\`js
// One-step (preferred for one-off tasks)
const result = await spawnAgent().call("Do something", { data });

// Parallel exploration
const [r1, r2] = await Promise.all([
  spawnAgent().call("Hypothesis A", { data }),
  spawnAgent().call("Hypothesis B", { data }),
]);

// Multi-call stateful agent (retains context between calls)
const agent = spawnAgent("Custom prompt.\\n\\n" + DOMAIN_REFERENCE);
const a = await agent.call("First task", { data });
const b = await agent.call("Follow-up");
\`\`\`

Sub-agents have access to \`spawnAgent\` for further delegation.

**resolve(value)** — signals completion and returns a value.
**reject(reason)** — signals failure.

## ${domain.name} Reference

${domain.reference}`;
}
