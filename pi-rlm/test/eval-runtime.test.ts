import { describe, it, expect } from "vitest";
import { EvalRuntime } from "../src/eval-runtime.js";

describe("EvalRuntime", () => {
	it("evaluates simple expressions", async () => {
		const rt = new EvalRuntime();
		const result = await rt.eval("1 + 2");
		expect(result.error).toBeNull();
		expect(result.returnValue).toBe(3);
	});

	it("captures console.log output", async () => {
		const rt = new EvalRuntime();
		const result = await rt.eval('console.log("hello"); console.log("world");');
		expect(result.stdout).toBe("hello\nworld");
		expect(result.error).toBeNull();
	});

	it("captures console.error output", async () => {
		const rt = new EvalRuntime();
		const result = await rt.eval('console.error("oops");');
		expect(result.stderr).toBe("oops");
	});

	it("persists var declarations across eval calls", async () => {
		const rt = new EvalRuntime();
		await rt.eval("var x = 42;");
		const result = await rt.eval("x + 8");
		expect(result.returnValue).toBe(50);
	});

	it("persists multiple var declarations", async () => {
		const rt = new EvalRuntime();
		await rt.eval("var a = 1, b = 2;");
		const result = await rt.eval("a + b");
		expect(result.returnValue).toBe(3);
	});

	it("persists let declarations across calls", async () => {
		const rt = new EvalRuntime();
		await rt.eval("let y = 99;");
		const result = await rt.eval("y + 1");
		expect(result.returnValue).toBe(100);
	});

	it("persists const declarations across calls", async () => {
		const rt = new EvalRuntime();
		await rt.eval("const z = 42;");
		const result = await rt.eval("z");
		expect(result.returnValue).toBe(42);
	});

	it("supports top-level await", async () => {
		const rt = new EvalRuntime();
		const result = await rt.eval("var val = await Promise.resolve(42); val");
		expect(result.returnValue).toBe(42);
		expect(result.error).toBeNull();
	});

	it("handles errors gracefully", async () => {
		const rt = new EvalRuntime();
		const result = await rt.eval("throw new Error('boom');");
		expect(result.error).toContain("boom");
		expect(result.returnValue).toBeUndefined();
	});

	it("handles syntax errors", async () => {
		const rt = new EvalRuntime();
		const result = await rt.eval("function {{{");
		expect(result.error).not.toBeNull();
	});

	it("reports durationMs", async () => {
		const rt = new EvalRuntime();
		const result = await rt.eval("1 + 1");
		expect(result.durationMs).toBeGreaterThanOrEqual(0);
	});

	it("times out long-running code", async () => {
		const rt = new EvalRuntime();
		const result = await rt.eval(
			"await new Promise(resolve => setTimeout(resolve, 5000));",
			100,
		);
		expect(result.error).toContain("timed out");
	});

	it("injects scope objects", async () => {
		const rt = new EvalRuntime({ greeting: "hi" });
		const result = await rt.eval("greeting");
		expect(result.returnValue).toBe("hi");
	});

	it("injectScope adds variables after construction", async () => {
		const rt = new EvalRuntime();
		rt.injectScope({ foo: "bar" });
		const result = await rt.eval("foo");
		expect(result.returnValue).toBe("bar");
	});

	it("can call injected functions", async () => {
		const calls: string[] = [];
		const rt = new EvalRuntime({
			myFunc: (arg: string) => {
				calls.push(arg);
				return `got ${arg}`;
			},
		});
		const result = await rt.eval('myFunc("test")');
		expect(result.returnValue).toBe("got test");
		expect(calls).toEqual(["test"]);
	});

	it("can call async injected functions", async () => {
		const rt = new EvalRuntime({
			asyncFunc: async () => 42,
		});
		const result = await rt.eval("await asyncFunc()");
		expect(result.returnValue).toBe(42);
	});

	it("captures stdout even on error", async () => {
		const rt = new EvalRuntime();
		const result = await rt.eval('console.log("before"); throw new Error("after");');
		expect(result.stdout).toBe("before");
		expect(result.error).toContain("after");
	});

	it("has access to standard globals", async () => {
		const rt = new EvalRuntime();
		const result = await rt.eval("JSON.stringify({ a: 1 })");
		expect(result.returnValue).toBe('{"a":1}');
	});

	it("getScope reads from context", async () => {
		const rt = new EvalRuntime({ x: 10 });
		expect(rt.getScope("x")).toBe(10);

		await rt.eval("var y = 20;");
		expect(rt.getScope("y")).toBe(20);
	});

	it("formats objects in console.log", async () => {
		const rt = new EvalRuntime();
		const result = await rt.eval('console.log({ a: 1, b: "two" });');
		expect(result.stdout).toContain('"a": 1');
		expect(result.stdout).toContain('"b": "two"');
	});

	it("persists var across multiple calls", async () => {
		const rt = new EvalRuntime();
		await rt.eval("var counter = 0;");
		await rt.eval("counter++;");
		await rt.eval("counter++;");
		const result = await rt.eval("counter");
		expect(result.returnValue).toBe(2);
	});

	it("allows var to be reassigned to different type", async () => {
		const rt = new EvalRuntime();
		await rt.eval("var x = 42;");
		await rt.eval('var x = "hello";');
		const result = await rt.eval("x");
		expect(result.returnValue).toBe("hello");
	});
});

describe("REPL persistence", () => {
	it("let re-declaration across evals: second version wins", async () => {
		const rt = new EvalRuntime();
		await rt.eval("let transform = 'v1';");
		await rt.eval("let transform = 'v2';");
		const result = await rt.eval("transform");
		expect(result.returnValue).toBe("v2");
	});

	it("const re-declaration across evals: second version wins", async () => {
		const rt = new EvalRuntime();
		await rt.eval("const N = 5;");
		await rt.eval("const N = 10;");
		const result = await rt.eval("N");
		expect(result.returnValue).toBe(10);
	});

	it("function re-definition across evals: second version wins", async () => {
		const rt = new EvalRuntime();
		await rt.eval("function transform(g) { return 'v1'; }");
		await rt.eval("function transform(g) { return 'v2'; }");
		const result = await rt.eval("transform()");
		expect(result.returnValue).toBe("v2");
	});

	it("cross-eval access: let declaration used in subsequent eval", async () => {
		const rt = new EvalRuntime();
		await rt.eval("let data = [1, 2, 3];");
		const result = await rt.eval("data.map(x => x * 2)");
		expect(result.returnValue).toEqual([2, 4, 6]);
	});

	it("scope injection coexists with let declarations", async () => {
		const rt = new EvalRuntime({
			trainingExamples: [{ input: 1 }, { input: 2 }],
		});
		await rt.eval("let count = trainingExamples.length;");
		const result = await rt.eval("count");
		expect(result.returnValue).toBe(2);
	});

	it("iterative refinement pattern: declare, test, re-declare, test", async () => {
		const rt = new EvalRuntime();
		// Eval 1: declare transform
		await rt.eval("let transform = (x) => x * 2;");
		// Eval 2: test it (no re-declaration)
		const r2 = await rt.eval("transform(5)");
		expect(r2.returnValue).toBe(10);
		// Eval 3: re-declare with improved version
		await rt.eval("let transform = (x) => x * 3;");
		// Eval 4: test the new version
		const r4 = await rt.eval("transform(5)");
		expect(r4.returnValue).toBe(15);
	});

	it("top-level await with let: result persists", async () => {
		const rt = new EvalRuntime();
		await rt.eval("let result = await Promise.resolve(42);");
		const r = await rt.eval("result");
		expect(r.returnValue).toBe(42);
	});

	it("console.log capture with let persistence", async () => {
		const rt = new EvalRuntime();
		const r = await rt.eval("let x = 10; console.log(x);");
		expect(r.stdout).toBe("10");
		const r2 = await rt.eval("x");
		expect(r2.returnValue).toBe(10);
	});

	it("multi-line with mixed declarations: all persist", async () => {
		const rt = new EvalRuntime({
			connectedComponents: (g: number[][]) => [{ color: 1 }],
			uniqueColors: (g: number[][]) => [0, 1, 2],
			makeGrid: (r: number, c: number) => Array.from({ length: r }, () => Array(c).fill(0)),
			testInput: [[1]],
		});
		await rt.eval(`
let components = connectedComponents([[1]]);
const colors = uniqueColors([[0, 1, 2]]);
let transform = function(g) {
  let out = makeGrid(3, 3);
  return out;
};
transform(testInput)
		`);
		// All variables persist
		const r1 = await rt.eval("components");
		expect(r1.returnValue).toEqual([{ color: 1 }]);
		const r2 = await rt.eval("colors");
		expect(r2.returnValue).toEqual([0, 1, 2]);
		const r3 = await rt.eval("typeof transform");
		expect(r3.returnValue).toBe("function");
	});

	it("implicit return: last expression is returned", async () => {
		const rt = new EvalRuntime();
		const result = await rt.eval("let x = 1; let y = 2; x + y");
		expect(result.returnValue).toBe(3);
	});

	it("error recovery: sandbox not corrupted after error", async () => {
		const rt = new EvalRuntime();
		// Eval 1 throws
		const r1 = await rt.eval("undefinedVar.foo");
		expect(r1.error).not.toBeNull();
		// Eval 2 works normally
		await rt.eval("let x = 42;");
		const r3 = await rt.eval("x");
		expect(r3.returnValue).toBe(42);
	});
});
