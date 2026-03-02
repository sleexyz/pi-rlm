import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { writeFileSync, mkdirSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { createOAuthResolver } from "../src/auth.js";

describe("createOAuthResolver", () => {
	let tempDir: string;
	let authPath: string;

	beforeEach(() => {
		tempDir = join(tmpdir(), `pi-auth-test-${Date.now()}-${Math.random().toString(36).slice(2)}`);
		mkdirSync(tempDir, { recursive: true });
		authPath = join(tempDir, "auth.json");
	});

	afterEach(() => {
		rmSync(tempDir, { recursive: true, force: true });
	});

	it("returns undefined when auth.json does not exist", async () => {
		const resolve = createOAuthResolver(join(tempDir, "nonexistent.json"));
		expect(await resolve("anthropic")).toBeUndefined();
	});

	it("returns undefined when auth.json is invalid JSON", async () => {
		writeFileSync(authPath, "not json");
		const resolve = createOAuthResolver(authPath);
		expect(await resolve("anthropic")).toBeUndefined();
	});

	it("returns undefined when provider is not in auth.json", async () => {
		writeFileSync(authPath, JSON.stringify({ openai: { type: "oauth", access: "tok", refresh: "ref", expires: 0 } }));
		const resolve = createOAuthResolver(authPath);
		expect(await resolve("anthropic")).toBeUndefined();
	});

	it("returns undefined when provider entry is not oauth type", async () => {
		writeFileSync(authPath, JSON.stringify({ anthropic: { type: "api-key", key: "sk-test" } }));
		const resolve = createOAuthResolver(authPath);
		expect(await resolve("anthropic")).toBeUndefined();
	});

	it("reads valid oauth credentials and returns the access token", async () => {
		const token = "sk-ant-oat-test-token-123";
		writeFileSync(
			authPath,
			JSON.stringify({
				anthropic: {
					type: "oauth",
					access: token,
					refresh: "refresh-token",
					// Set expires far in the future so no refresh is attempted
					expires: Date.now() + 3600000,
				},
			}),
		);
		const resolve = createOAuthResolver(authPath);
		const result = await resolve("anthropic");
		// getOAuthApiKey returns the access token as the apiKey
		expect(result).toBe(token);
	});
});
