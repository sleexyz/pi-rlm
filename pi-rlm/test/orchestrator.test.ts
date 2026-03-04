import { describe, it, expect } from "vitest";
import { Orchestrator } from "../src/orchestrator.js";
import { generateSystemPrompt } from "../src/system-prompt.js";
import type { Adapter } from "../src/domain-adapter.js";
import { getModel } from "@mariozechner/pi-ai";

const model = getModel("anthropic", "claude-sonnet-4-6");

function makeToyAdapter(): Adapter {
	return {
		name: "TestAdapter",
		premise: "You are an orchestrator for a test domain.",
		reference: "This is a test domain. The answer is always 42.",
		scope: {
			getAnswer: () => 42,
		},
		onEvalResult: () => "Status: running",
	};
}

describe("generateSystemPrompt", () => {
	it("includes domain premise", () => {
		const adapter = makeToyAdapter();
		const prompt = generateSystemPrompt(adapter);
		expect(prompt).toContain("orchestrator for a test domain");
	});

	it("includes code execution environment docs", () => {
		const adapter = makeToyAdapter();
		const prompt = generateSystemPrompt(adapter);
		expect(prompt).toContain("spawnAgent");
		expect(prompt).toContain("resolve");
		expect(prompt).toContain("reject");
		expect(prompt).toContain("eval");
	});

	it("includes variable persistence guidance", () => {
		const adapter = makeToyAdapter();
		const prompt = generateSystemPrompt(adapter);
		expect(prompt).toContain("let");
		expect(prompt).toContain("const");
		expect(prompt).toContain("persist");
	});

	it("includes domain reference", () => {
		const adapter = makeToyAdapter();
		const prompt = generateSystemPrompt(adapter);
		expect(prompt).toContain("TestAdapter Reference");
		expect(prompt).toContain("answer is always 42");
	});
});

describe("Orchestrator", () => {
	it("constructs with required options", () => {
		const adapter = makeToyAdapter();
		const orch = new Orchestrator({ model, adapter });
		expect(orch.getAgent()).toBeTruthy();
		expect(orch.getRuntime()).toBeTruthy();
	});

	it("has eval tool on the agent", () => {
		const adapter = makeToyAdapter();
		const orch = new Orchestrator({ model, adapter });
		const tools = orch.getAgent().state.tools;
		expect(tools).toHaveLength(1);
		expect(tools[0].name).toBe("eval");
	});

	it("injects domain scope into runtime", async () => {
		const adapter = makeToyAdapter();
		const orch = new Orchestrator({ model, adapter });
		const rt = orch.getRuntime();
		const result = await rt.eval("getAnswer()");
		expect(result.returnValue).toBe(42);
	});

	it("injects spawnAgent into runtime", async () => {
		const adapter = makeToyAdapter();
		const orch = new Orchestrator({ model, adapter });
		const rt = orch.getRuntime();
		const result = await rt.eval("typeof spawnAgent");
		expect(result.returnValue).toBe("function");
	});

	it("injects resolve into runtime", async () => {
		const adapter = makeToyAdapter();
		const orch = new Orchestrator({ model, adapter });
		const rt = orch.getRuntime();
		const result = await rt.eval("typeof resolve");
		expect(result.returnValue).toBe("function");
	});

	it("injects reject into runtime", async () => {
		const adapter = makeToyAdapter();
		const orch = new Orchestrator({ model, adapter });
		const rt = orch.getRuntime();
		const result = await rt.eval("typeof reject");
		expect(result.returnValue).toBe("function");
	});

	it("injects DOMAIN_REFERENCE into runtime", async () => {
		const adapter = makeToyAdapter();
		const orch = new Orchestrator({ model, adapter });
		const rt = orch.getRuntime();
		const result = await rt.eval("DOMAIN_REFERENCE");
		expect(result.returnValue).toContain("answer is always 42");
	});

	it("sets system prompt from domain fields", () => {
		const adapter = makeToyAdapter();
		const orch = new Orchestrator({ model, adapter });
		const sp = orch.getAgent().state.systemPrompt;
		expect(sp).toContain("orchestrator for a test domain");
		expect(sp).toContain("TestAdapter Reference");
	});

	it("sets thinking level", () => {
		const adapter = makeToyAdapter();
		const orch = new Orchestrator({ model, adapter, thinkingLevel: "high" });
		expect(orch.getAgent().state.thinkingLevel).toBe("high");
	});

	it("collects events via onEvent", () => {
		const events: string[] = [];
		const adapter = makeToyAdapter();
		new Orchestrator({
			model,
			adapter,
			onEvent: (e) => events.push(e.type),
		});
	});
});
