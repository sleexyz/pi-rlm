import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { dirname } from "node:path";
import { join } from "node:path";
import { homedir } from "node:os";
import { getOAuthApiKey, type OAuthCredentials } from "@mariozechner/pi-ai";

const DEFAULT_AUTH_PATH = join(homedir(), ".pi", "agent", "auth.json");

type AuthData = Record<string, { type: string; [k: string]: unknown }>;

/**
 * Creates a getApiKey callback that reads OAuth tokens from auth.json,
 * auto-refreshes expired tokens, and persists refreshed tokens back.
 *
 * Usage:
 *   const orchestrator = new Orchestrator({
 *     model: getModel("anthropic", "claude-opus-4-6"),
 *     adapter: myAdapter,
 *     getApiKey: createOAuthResolver(),
 *   });
 */
export function createOAuthResolver(
	authPath: string = DEFAULT_AUTH_PATH,
): (provider: string) => Promise<string | undefined> {
	return async (provider: string): Promise<string | undefined> => {
		if (!existsSync(authPath)) return undefined;

		let data: AuthData;
		try {
			data = JSON.parse(readFileSync(authPath, "utf-8"));
		} catch {
			return undefined;
		}

		const cred = data[provider];
		if (!cred || cred.type !== "oauth") return undefined;

		const oauthCred = cred as unknown as OAuthCredentials;

		// Use pi-ai's getOAuthApiKey — handles refresh if expired
		const result = await getOAuthApiKey(provider, { [provider]: oauthCred });
		if (!result) return undefined;

		// Persist refreshed token if it changed
		if (result.newCredentials !== oauthCred) {
			data[provider] = { type: "oauth", ...result.newCredentials };
			const dir = dirname(authPath);
			if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
			writeFileSync(authPath, JSON.stringify(data, null, 2), { mode: 0o600 });
		}

		return result.apiKey;
	};
}
