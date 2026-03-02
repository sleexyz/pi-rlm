import type { Usage } from "@mariozechner/pi-ai";

/**
 * Aggregated token usage for display and serialization.
 */
export interface TokenUsage {
	inputTokens: number;
	outputTokens: number;
	cacheReadTokens: number;
	cacheWriteTokens: number;
	totalTokens: number;
	totalCost: number;
}

function emptyUsage(): TokenUsage {
	return {
		inputTokens: 0,
		outputTokens: 0,
		cacheReadTokens: 0,
		cacheWriteTokens: 0,
		totalTokens: 0,
		totalCost: 0,
	};
}

function addUsage(a: TokenUsage, b: TokenUsage): TokenUsage {
	return {
		inputTokens: a.inputTokens + b.inputTokens,
		outputTokens: a.outputTokens + b.outputTokens,
		cacheReadTokens: a.cacheReadTokens + b.cacheReadTokens,
		cacheWriteTokens: a.cacheWriteTokens + b.cacheWriteTokens,
		totalTokens: a.totalTokens + b.totalTokens,
		totalCost: a.totalCost + b.totalCost,
	};
}

/**
 * Accumulates per-agent token usage from message_end events.
 */
export class UsageTracker {
	private perAgent = new Map<number, TokenUsage>();

	/** Record usage from a message_end event. */
	recordUsage(agentId: number, usage: Usage): void {
		const existing = this.perAgent.get(agentId) ?? emptyUsage();
		const delta: TokenUsage = {
			inputTokens: usage.input,
			outputTokens: usage.output,
			cacheReadTokens: usage.cacheRead,
			cacheWriteTokens: usage.cacheWrite,
			totalTokens: usage.totalTokens,
			totalCost: usage.cost.total,
		};
		this.perAgent.set(agentId, addUsage(existing, delta));
	}

	/** Total usage across all agents. */
	totalUsage(): TokenUsage {
		let total = emptyUsage();
		for (const u of this.perAgent.values()) {
			total = addUsage(total, u);
		}
		return total;
	}

	/** Per-agent usage breakdown. */
	perAgentUsage(): Map<number, TokenUsage> {
		return new Map(this.perAgent);
	}

	/** Formatted summary table. */
	summary(): string {
		const perAgent = this.perAgentUsage();
		const total = this.totalUsage();

		const lines: string[] = ["Token Usage Summary", "=".repeat(70)];

		if (perAgent.size > 0) {
			lines.push(
				`${"Agent".padStart(8)}  ${"Input".padStart(10)}  ${"Output".padStart(10)}  ${"Cache Read".padStart(10)}  ${"Cache Write".padStart(11)}  ${"Cost".padStart(8)}`,
			);
			lines.push("-".repeat(70));

			const sortedIds = [...perAgent.keys()].sort((a, b) => a - b);
			for (const aid of sortedIds) {
				const u = perAgent.get(aid)!;
				lines.push(
					`${("#" + aid).padStart(8)}  ${fmt(u.inputTokens)}  ${fmt(u.outputTokens)}  ${fmt(u.cacheReadTokens)}  ${fmt(u.cacheWriteTokens).padStart(11)}  ${fmtCost(u.totalCost)}`,
				);
			}
			lines.push("-".repeat(70));
		}

		lines.push(
			`${"TOTAL".padStart(8)}  ${fmt(total.inputTokens)}  ${fmt(total.outputTokens)}  ${fmt(total.cacheReadTokens)}  ${fmt(total.cacheWriteTokens).padStart(11)}  ${fmtCost(total.totalCost)}`,
		);
		lines.push(`\nTotal tokens: ${total.totalTokens.toLocaleString()}`);

		return lines.join("\n");
	}

	/** JSON-serializable snapshot. */
	toJSON(): object {
		const perAgent = this.perAgentUsage();
		const total = this.totalUsage();

		const perAgentObj: Record<string, object> = {};
		const sortedIds = [...perAgent.keys()].sort((a, b) => a - b);
		for (const aid of sortedIds) {
			const u = perAgent.get(aid)!;
			perAgentObj[String(aid)] = { ...u };
		}

		return {
			total: { ...total },
			perAgent: perAgentObj,
		};
	}
}

function fmt(n: number): string {
	return n.toLocaleString().padStart(10);
}

function fmtCost(n: number): string {
	return `$${n.toFixed(4)}`.padStart(8);
}
