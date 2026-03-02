#!/usr/bin/env node
/**
 * Merge sharded ARC-AGI-2 result files into a single summary.
 *
 * Usage: bun src/merge-results.ts results/arc/*shard*.json
 */

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const files = process.argv.slice(2);
if (files.length === 0) {
	console.error("Usage: bun src/merge-results.ts <result-files...>");
	process.exit(1);
}

interface Attempt {
	predicted: number[][];
	correct: boolean;
	cost: number;
	tokens: number;
	timeMs: number;
}

interface Result {
	taskId: string;
	correct: boolean;
	attempts: Attempt[];
	expected: number[][];
	cost: number;
	tokens: number;
	timeMs: number;
}

interface ShardFile {
	config: Record<string, unknown>;
	score: { correct: number; total: number; pct: number };
	totals: { cost: number; tokens: number; timeMs: number };
	results: Result[];
}

const allResults: Result[] = [];
let config: Record<string, unknown> = {};

for (const file of files) {
	const data: ShardFile = JSON.parse(readFileSync(file, "utf-8"));
	allResults.push(...data.results);
	config = data.config; // use last shard's config (they should all match)
}

// Deduplicate by taskId: later files override earlier ones (for retry merging)
const byTaskId = new Map<string, Result>();
for (const r of allResults) byTaskId.set(r.taskId, r);
const deduped = Array.from(byTaskId.values());

// Sort by task ID for consistent ordering
deduped.sort((a, b) => a.taskId.localeCompare(b.taskId));

const correctCount = deduped.filter((r) => r.correct).length;
const totalCost = deduped.reduce((sum, r) => sum + r.cost, 0);
const totalTokens = deduped.reduce((sum, r) => sum + r.tokens, 0);
const totalTime = deduped.reduce((sum, r) => sum + r.timeMs, 0);

// Print summary
console.log("=".repeat(60));
console.log("Merged Results Summary");
console.log("=".repeat(60));
console.log();

for (const r of deduped) {
	const pass2Benefit = !r.attempts[0]?.correct && r.correct ? " (saved by attempt 2)" : "";
	console.log(`  ${r.taskId}: ${r.correct ? "CORRECT" : "WRONG   "} | $${r.cost.toFixed(2)} | ${(r.timeMs / 1000).toFixed(0)}s${pass2Benefit}`);
}

const attempt1Correct = deduped.filter((r) => r.attempts[0]?.correct).length;
const savedByAttempt2 = correctCount - attempt1Correct;

console.log();
console.log(`Score: ${correctCount}/${deduped.length} correct (${((correctCount / deduped.length) * 100).toFixed(1)}%)`);
console.log(`  pass@1: ${attempt1Correct}/${deduped.length} (${((attempt1Correct / deduped.length) * 100).toFixed(1)}%)`);
console.log(`  pass@2 benefit: +${savedByAttempt2} tasks saved by second attempt`);
console.log(`Total cost: $${totalCost.toFixed(2)}`);
console.log(`Avg cost/task: $${(totalCost / deduped.length).toFixed(2)}`);
console.log(`Total tokens: ${totalTokens.toLocaleString()}`);
console.log(`Total time: ${(totalTime / 1000 / 60).toFixed(0)} minutes (sum across shards)`);

// Save merged file
const resultsDir = join(process.cwd(), "results/arc");
mkdirSync(resultsDir, { recursive: true });
const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
const mergedPath = join(resultsDir, `${timestamp}-merged.json`);

const { shard: _s, numShards: _n, count: _c, ...cleanConfig } = config as any;

writeFileSync(
	mergedPath,
	JSON.stringify(
		{
			config: { ...cleanConfig, count: deduped.length, shardsMerged: files.length },
			score: { correct: correctCount, total: deduped.length, pct: correctCount / deduped.length },
			pass1: { correct: attempt1Correct, total: deduped.length, pct: attempt1Correct / deduped.length },
			pass2Benefit: savedByAttempt2,
			totals: { cost: totalCost, tokens: totalTokens, timeMs: totalTime },
			results: deduped,
		},
		null,
		2,
	),
);
console.log(`\nMerged results saved: ${mergedPath}`);
