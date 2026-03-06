import { describe, it, expect } from "bun:test";
import { parseReplBlocks, createCodeBlockStreamFn, createVllmModel, transformMessagesForRepl } from "../src/repl-stream.js";
import type { AssistantMessage, Message, ToolResultMessage, UserMessage } from "@mariozechner/pi-ai";

describe("parseReplBlocks", () => {
	it("parses a single ```js code block", () => {
		const text = "Let me try this.\n```js\nprint('hello')\n```\nDone.";
		const segments = parseReplBlocks(text);
		expect(segments).toEqual([
			{ type: "text", text: "Let me try this." },
			{ type: "code", code: "print('hello')" },
			{ type: "text", text: "Done." },
		]);
	});

	it("parses ```repl blocks too (backwards compat)", () => {
		const text = "```repl\nx = 1\n```";
		const segments = parseReplBlocks(text);
		expect(segments).toEqual([{ type: "code", code: "x = 1" }]);
	});

	it("parses multiple code blocks", () => {
		const text = "First block:\n```js\nx = 1\n```\nSecond block:\n```js\ny = 2\nprint(y)\n```";
		const segments = parseReplBlocks(text);
		expect(segments).toEqual([
			{ type: "text", text: "First block:" },
			{ type: "code", code: "x = 1" },
			{ type: "text", text: "Second block:" },
			{ type: "code", code: "y = 2\nprint(y)" },
		]);
	});

	it("handles text with no code blocks", () => {
		const text = "Just some plain text without any code.";
		const segments = parseReplBlocks(text);
		expect(segments).toEqual([{ type: "text", text: "Just some plain text without any code." }]);
	});

	it("handles code block at start", () => {
		const text = "```js\nfoo()\n```\nAfter.";
		const segments = parseReplBlocks(text);
		expect(segments).toEqual([
			{ type: "code", code: "foo()" },
			{ type: "text", text: "After." },
		]);
	});

	it("handles multiline code blocks", () => {
		const text = "```js\nfor (let i = 0; i < 10; i++) {\n    console.log(i);\n    if (i > 5) break;\n}\n```";
		const segments = parseReplBlocks(text);
		expect(segments).toEqual([
			{
				type: "code",
				code: "for (let i = 0; i < 10; i++) {\n    console.log(i);\n    if (i > 5) break;\n}",
			},
		]);
	});

	it("ignores non-js/repl code blocks", () => {
		const text = '```python\nprint("not js")\n```\n```js\nconsole.log("yes js")\n```';
		const segments = parseReplBlocks(text);
		expect(segments).toEqual([
			{ type: "text", text: '```python\nprint("not js")\n```' },
			{ type: "code", code: 'console.log("yes js")' },
		]);
	});

	it("handles empty text", () => {
		const segments = parseReplBlocks("");
		expect(segments).toEqual([]);
	});

	it("strips whitespace-only segments", () => {
		const text = "\n\n```js\nx = 1\n```\n\n";
		const segments = parseReplBlocks(text);
		expect(segments).toEqual([{ type: "code", code: "x = 1" }]);
	});
});

describe("transformMessagesForRepl", () => {
	const ts = Date.now();
	const baseAssistant: Omit<AssistantMessage, "content"> = {
		role: "assistant",
		api: "openai-completions",
		provider: "openai",
		model: "test-model",
		usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
		stopReason: "toolUse",
		timestamp: ts,
	};

	it("converts assistant tool calls to ```js blocks", () => {
		const messages: Message[] = [
			{ role: "user", content: "do something", timestamp: ts } as UserMessage,
			{
				...baseAssistant,
				content: [
					{ type: "text", text: "Let me check." },
					{ type: "toolCall", id: "tc1", name: "eval", arguments: { code: 'console.log("hi")' } },
				],
			} as AssistantMessage,
		];

		const result = transformMessagesForRepl(messages);
		expect(result).toHaveLength(2);
		expect(result[0]).toEqual(messages[0]); // user passes through
		expect(result[1].role).toBe("assistant");
		const text = (result[1] as AssistantMessage).content[0];
		expect(text.type).toBe("text");
		expect((text as any).text).toContain("Let me check.");
		expect((text as any).text).toContain('```js\nconsole.log("hi")\n```');
	});

	it("converts tool results to user messages with REPL format", () => {
		const messages: Message[] = [
			{ role: "user", content: "go", timestamp: ts } as UserMessage,
			{
				...baseAssistant,
				content: [
					{ type: "toolCall", id: "tc1", name: "eval", arguments: { code: "1 + 1" } },
				],
			} as AssistantMessage,
			{
				role: "toolResult",
				toolCallId: "tc1",
				toolName: "eval",
				content: [{ type: "text", text: "→ 2" }],
				isError: false,
				timestamp: ts,
			} as ToolResultMessage,
		];

		const result = transformMessagesForRepl(messages);
		expect(result).toHaveLength(3);
		expect(result[2].role).toBe("user");
		const content = (result[2] as UserMessage).content as string;
		expect(content).toContain("Code executed:");
		expect(content).toContain("```js\n1 + 1\n```");
		expect(content).toContain("REPL output:\n→ 2");
	});

	it("handles multiple tool calls in sequence", () => {
		const messages: Message[] = [
			{ role: "user", content: "go", timestamp: ts } as UserMessage,
			{
				...baseAssistant,
				content: [
					{ type: "text", text: "First:" },
					{ type: "toolCall", id: "tc1", name: "eval", arguments: { code: "let x = 1" } },
				],
			} as AssistantMessage,
			{
				role: "toolResult",
				toolCallId: "tc1",
				toolName: "eval",
				content: [{ type: "text", text: "(no output)" }],
				isError: false,
				timestamp: ts,
			} as ToolResultMessage,
			{
				...baseAssistant,
				content: [
					{ type: "text", text: "Now:" },
					{ type: "toolCall", id: "tc2", name: "eval", arguments: { code: "console.log(x)" } },
				],
			} as AssistantMessage,
			{
				role: "toolResult",
				toolCallId: "tc2",
				toolName: "eval",
				content: [{ type: "text", text: "1" }],
				isError: false,
				timestamp: ts,
			} as ToolResultMessage,
		];

		const result = transformMessagesForRepl(messages);
		expect(result).toHaveLength(5);
		// user, assistant(text+js), user(result), assistant(text+js), user(result)
		expect(result.map((m) => m.role)).toEqual(["user", "assistant", "user", "assistant", "user"]);
	});

	it("passes through plain user messages unchanged", () => {
		const messages: Message[] = [
			{ role: "user", content: "hello", timestamp: ts } as UserMessage,
		];
		const result = transformMessagesForRepl(messages);
		expect(result).toEqual(messages);
	});
});

describe("createVllmModel", () => {
	it("creates a model with correct defaults", () => {
		const model = createVllmModel({
			id: "rlm-qwen3-8b-v0.1",
			baseUrl: "http://localhost:8000/v1",
		});
		expect(model.id).toBe("rlm-qwen3-8b-v0.1");
		expect(model.api).toBe("openai-completions");
		expect(model.baseUrl).toBe("http://localhost:8000/v1");
		expect(model.reasoning).toBe(false);
		expect(model.contextWindow).toBe(32768);
		expect(model.maxTokens).toBe(8192);
		expect(model.compat?.supportsStore).toBe(false);
		expect(model.compat?.maxTokensField).toBe("max_tokens");
	});

	it("accepts custom options", () => {
		const model = createVllmModel({
			id: "my-model",
			name: "My Model",
			baseUrl: "http://gpu-box:8000/v1",
			contextWindow: 65536,
			maxTokens: 16384,
		});
		expect(model.name).toBe("My Model");
		expect(model.contextWindow).toBe(65536);
		expect(model.maxTokens).toBe(16384);
	});
});

describe("createCodeBlockStreamFn", () => {
	it("returns a function", () => {
		const model = createVllmModel({ id: "test", baseUrl: "http://localhost:8000/v1" });
		const fn = createCodeBlockStreamFn(model);
		expect(typeof fn).toBe("function");
	});
});
