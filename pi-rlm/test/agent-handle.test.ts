import { describe, it, expect } from "vitest";
import { AgentHandle, createSpawnAgent } from "../src/agent-handle.js";
import { getModel } from "@mariozechner/pi-ai";

// These tests verify the structural setup without calling real LLMs.
// Integration tests with actual LLM calls would go in a separate file.

describe("AgentHandle", () => {
	const model = getModel("anthropic", "claude-sonnet-4-6");

	it("creates an agent with eval tool", () => {
		const handle = new AgentHandle(model, "off", "test prompt", {});
		expect(handle.agent).toBeTruthy();
		expect(handle.runtime).toBeTruthy();
		expect(handle.agent.state.tools).toHaveLength(1);
		expect(handle.agent.state.tools[0].name).toBe("eval");
	});

	it("injects resolve into eval scope", async () => {
		const handle = new AgentHandle(model, "off", "test prompt", {});
		const result = await handle.runtime.eval("typeof resolve");
		expect(result.returnValue).toBe("function");
	});

	it("injects custom scope objects", async () => {
		const handle = new AgentHandle(model, "off", "test prompt", {
			myVar: 42,
			myFunc: () => "hello",
		});
		const r1 = await handle.runtime.eval("myVar");
		expect(r1.returnValue).toBe(42);
		const r2 = await handle.runtime.eval("myFunc()");
		expect(r2.returnValue).toBe("hello");
	});

	it("sets system prompt on agent", () => {
		const handle = new AgentHandle(model, "off", "my custom prompt", {});
		expect(handle.agent.state.systemPrompt).toBe("my custom prompt");
	});

	it("sets model and thinking level", () => {
		const handle = new AgentHandle(model, "high", "test", {});
		expect(handle.agent.state.model).toBe(model);
		expect(handle.agent.state.thinkingLevel).toBe("high");
	});
});

describe("createSpawnAgent", () => {
	const model = getModel("anthropic", "claude-sonnet-4-6");

	it("returns a function", () => {
		const spawn = createSpawnAgent(model, "off", {});
		expect(typeof spawn).toBe("function");
	});

	it("spawns an agent handle with spawnAgent in scope", async () => {
		const spawn = createSpawnAgent(model, "off", { baseVar: "test" });
		const handle = spawn("You are a test agent.");
		expect(handle).toBeInstanceOf(AgentHandle);

		// Check base scope is injected
		const r1 = await handle.runtime.eval("baseVar");
		expect(r1.returnValue).toBe("test");

		// Check spawnAgent is available
		const r2 = await handle.runtime.eval("typeof spawnAgent");
		expect(r2.returnValue).toBe("function");
	});

	it("uses custom system prompt", () => {
		const spawn = createSpawnAgent(model, "off", {});
		const handle = spawn("Custom agent instructions");
		expect(handle.agent.state.systemPrompt).toBe("Custom agent instructions");
	});

	it("uses default system prompt when none provided", () => {
		const spawn = createSpawnAgent(model, "off", {});
		const handle = spawn();
		expect(handle.agent.state.systemPrompt).toContain("sub-agent");
	});

	it("auto-wraps with defaultSubAgentPrompt when no args", () => {
		const spawn = createSpawnAgent(model, "off", {}, {
			defaultSubAgentPrompt: "You are a domain expert.",
			domainReference: "Some reference docs.",
		});
		const handle = spawn();
		expect(handle.agent.state.systemPrompt).toContain("domain expert");
		expect(handle.agent.state.systemPrompt).toContain("Some reference docs");
	});

	it("skips auto-wrap when custom prompt provided", () => {
		const spawn = createSpawnAgent(model, "off", {}, {
			defaultSubAgentPrompt: "You are a domain expert.",
		});
		const handle = spawn("My custom prompt");
		expect(handle.agent.state.systemPrompt).toBe("My custom prompt");
		expect(handle.agent.state.systemPrompt).not.toContain("domain expert");
	});

	it("enforces maxDepth", async () => {
		const spawn = createSpawnAgent(model, "off", {}, { maxDepth: 1 });
		const handle = spawn();

		// The child's spawnAgent should fail
		const result = await handle.runtime.eval("spawnAgent()");
		expect(result.error).toContain("Maximum agent spawn depth");
	});

	it("enforces maxAgents flat pool", () => {
		const counter = { count: 0 };
		const spawn = createSpawnAgent(model, "off", {}, {
			maxAgents: 2,
			agentCounter: counter,
			maxDepth: 10,
		});

		// First two spawns succeed
		const h1 = spawn();
		expect(h1).toBeInstanceOf(AgentHandle);
		expect(counter.count).toBe(1);

		const h2 = spawn();
		expect(h2).toBeInstanceOf(AgentHandle);
		expect(counter.count).toBe(2);

		// Third spawn exceeds pool
		expect(() => spawn()).toThrow("Maximum total number of agents");
	});

	it("shares flat pool counter across recursive spawn levels", async () => {
		const counter = { count: 0 };
		const spawn = createSpawnAgent(model, "off", {}, {
			maxAgents: 2,
			agentCounter: counter,
			maxDepth: 10,
		});

		// Spawn one agent at level 0
		const h1 = spawn();
		expect(counter.count).toBe(1);

		// Spawn a child from h1 (level 1) — shares the same counter
		const result = await h1.runtime.eval("spawnAgent()");
		expect(result.error).toBeFalsy();
		expect(counter.count).toBe(2);

		// Third spawn from either level should fail
		const result2 = await h1.runtime.eval("spawnAgent()");
		expect(result2.error).toContain("Maximum total number of agents");
	});
});
