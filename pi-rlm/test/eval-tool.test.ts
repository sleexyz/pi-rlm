import { describe, it, expect } from "vitest";
import { EvalRuntime } from "../src/eval-runtime.js";
import { createEvalTool } from "../src/eval-tool.js";

describe("createEvalTool", () => {
	it("has correct tool metadata", () => {
		const rt = new EvalRuntime();
		const tool = createEvalTool(rt);
		expect(tool.name).toBe("eval");
		expect(tool.label).toBe("Eval");
		expect(tool.description).toBeTruthy();
	});

	it("executes code and returns text content", async () => {
		const rt = new EvalRuntime();
		const tool = createEvalTool(rt);
		const result = await tool.execute("tc-1", { code: 'console.log("hi"); 42' });
		expect(result.content).toHaveLength(1);
		expect(result.content[0].type).toBe("text");
		const text = (result.content[0] as { type: "text"; text: string }).text;
		expect(text).toContain("hi");
		expect(text).toContain("42");
	});

	it("returns error output", async () => {
		const rt = new EvalRuntime();
		const tool = createEvalTool(rt);
		const result = await tool.execute("tc-2", { code: 'throw new Error("fail")' });
		const text = (result.content[0] as { type: "text"; text: string }).text;
		expect(text).toContain("ERROR");
		expect(text).toContain("fail");
	});

	it("truncates long output", async () => {
		const rt = new EvalRuntime();
		const tool = createEvalTool(rt, { maxOutputChars: 50 });
		const result = await tool.execute("tc-3", { code: 'console.log("a".repeat(200))' });
		const text = (result.content[0] as { type: "text"; text: string }).text;
		expect(text).toContain("truncated");
	});

	it("returns (no output) for empty results", async () => {
		const rt = new EvalRuntime();
		const tool = createEvalTool(rt);
		const result = await tool.execute("tc-4", { code: "undefined" });
		const text = (result.content[0] as { type: "text"; text: string }).text;
		expect(text).toBe("(no output)");
	});

	it("calls onResult callback and appends status", async () => {
		const rt = new EvalRuntime();
		const tool = createEvalTool(rt, {
			onResult: () => "Status: OK",
		});
		const result = await tool.execute("tc-5", { code: "1 + 1" });
		const text = (result.content[0] as { type: "text"; text: string }).text;
		expect(text).toContain("Status: OK");
	});

	it("includes EvalResult in details", async () => {
		const rt = new EvalRuntime();
		const tool = createEvalTool(rt);
		const result = await tool.execute("tc-6", { code: "42" });
		expect(result.details.returnValue).toBe(42);
		expect(result.details.error).toBeNull();
		expect(result.details.durationMs).toBeGreaterThanOrEqual(0);
	});
});
