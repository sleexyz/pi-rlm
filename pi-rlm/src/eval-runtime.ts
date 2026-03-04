import vm from "node:vm";
import type { EvalResult } from "./types.js";

/**
 * Persistent JavaScript evaluation environment using node:vm.
 *
 * The vm context IS the persistent namespace — variables written to it
 * survive across eval() calls. Top-level declarations (var, let, const,
 * function) in user code are automatically persisted via a __persist
 * callback injected after wrapping the code in an async IIFE.
 */
export class EvalRuntime {
	private context: vm.Context;

	constructor(scope: Record<string, unknown> = {}) {
		const sandbox: Record<string, unknown> = {
			// Standard globals
			console: { log: () => {}, error: () => {}, warn: () => {}, info: () => {} },
			setTimeout,
			setInterval,
			clearTimeout,
			clearInterval,
			Promise,
			JSON,
			Array,
			Object,
			Map,
			Set,
			WeakMap,
			WeakSet,
			Date,
			RegExp,
			Error,
			TypeError,
			RangeError,
			Math,
			Number,
			String,
			Boolean,
			Symbol,
			parseInt,
			parseFloat,
			isNaN,
			isFinite,
			encodeURIComponent,
			decodeURIComponent,
			encodeURI,
			decodeURI,
			// Node globals useful in eval
			Buffer: globalThis.Buffer,
			URL: globalThis.URL,
			URLSearchParams: globalThis.URLSearchParams,
			TextEncoder: globalThis.TextEncoder,
			TextDecoder: globalThis.TextDecoder,
			AbortController: globalThis.AbortController,
			AbortSignal: globalThis.AbortSignal,
			fetch: globalThis.fetch,
			structuredClone: globalThis.structuredClone,
			queueMicrotask: globalThis.queueMicrotask,
			atob: globalThis.atob,
			btoa: globalThis.btoa,
			// User-provided scope
			...scope,
		};

		this.context = vm.createContext(sandbox);
	}

	/**
	 * Add or update variables in the eval namespace.
	 */
	injectScope(entries: Record<string, unknown>): void {
		for (const [key, value] of Object.entries(entries)) {
			this.context[key] = value;
		}
	}

	/**
	 * Read a variable from the eval namespace.
	 */
	getScope(key: string): unknown {
		return this.context[key];
	}

	/**
	 * Evaluate JavaScript code in the persistent context.
	 *
	 * - Code is wrapped in an async IIFE for top-level await support
	 * - Top-level var declarations are scanned and persisted back to the context
	 * - console.log/error output is captured per-call
	 * - Execution is raced against a timeout (default 60s)
	 */
	async eval(code: string, timeoutMs: number = 60_000): Promise<EvalResult> {
		const stdoutLines: string[] = [];
		const stderrLines: string[] = [];

		// Override console for this eval call
		const prevConsole = this.context.console;
		this.context.console = {
			log: (...args: unknown[]) => {
				stdoutLines.push(args.map(formatArg).join(" "));
			},
			error: (...args: unknown[]) => {
				stderrLines.push(args.map(formatArg).join(" "));
			},
			warn: (...args: unknown[]) => {
				stderrLines.push(args.map(formatArg).join(" "));
			},
			info: (...args: unknown[]) => {
				stdoutLines.push(args.map(formatArg).join(" "));
			},
		};

		// Scan for top-level declarations (var/let/const/function) to persist after IIFE
		const varNames = scanDeclarations(code);

		// Build the persist callback
		const persistCode =
			varNames.length > 0
				? `\n__persist(${JSON.stringify(varNames)}, [${varNames.join(", ")}]);`
				: "";

		// Inject __persist into context
		this.context.__persist = (names: string[], values: unknown[]) => {
			for (let i = 0; i < names.length; i++) {
				this.context[names[i]] = values[i];
			}
		};

		// Wrap in async IIFE with implicit return for the last expression
		const wrapped = wrapInAsyncIIFE(code, persistCode);

		const start = performance.now();

		try {
			const script = new vm.Script(wrapped, { filename: "eval" });
			const promise = script.runInContext(this.context) as Promise<unknown>;

			// Race against timeout
			const returnValue = await raceTimeout(promise, timeoutMs);

			return {
				stdout: stdoutLines.join("\n"),
				stderr: stderrLines.join("\n"),
				returnValue,
				error: null,
				durationMs: performance.now() - start,
			};
		} catch (err: unknown) {
			// Bun's vm.Script doesn't always reject invalid syntax at compile
			// time, so expression-mode wrapping can fail with SyntaxError at
			// runtime. Since no code executes before a SyntaxError, it's safe
			// to retry with statement-mode wrapping.
			if ((err as Record<string, unknown>)?.name === "SyntaxError") {
				const stmtWrapped = wrapAsStatements(code, persistCode);
				if (stmtWrapped !== wrapped) {
					try {
						const script2 = new vm.Script(stmtWrapped, { filename: "eval" });
						const promise2 = script2.runInContext(this.context) as Promise<unknown>;
						const returnValue = await raceTimeout(promise2, timeoutMs);
						return {
							stdout: stdoutLines.join("\n"),
							stderr: stderrLines.join("\n"),
							returnValue,
							error: null,
							durationMs: performance.now() - start,
						};
					} catch (err2: unknown) {
						// Statement mode also failed — report this error instead
						const msg2 = err2 instanceof Error ? err2.message : String(err2);
						const stack2 = err2 instanceof Error ? err2.stack : undefined;
						return {
							stdout: stdoutLines.join("\n"),
							stderr: stderrLines.join("\n"),
							returnValue: undefined,
							error: stack2 || msg2,
							durationMs: performance.now() - start,
						};
					}
				}
			}

			const errorMessage = err instanceof Error ? err.message : String(err);
			const errorStack = err instanceof Error ? err.stack : undefined;
			return {
				stdout: stdoutLines.join("\n"),
				stderr: stderrLines.join("\n"),
				returnValue: undefined,
				error: errorStack || errorMessage,
				durationMs: performance.now() - start,
			};
		} finally {
			this.context.console = prevConsole;
			delete this.context.__persist;
		}
	}
}

/**
 * Scan for top-level declarations (var, let, const, function) in the code.
 *
 * Only scans at brace-depth 0 to avoid capturing declarations inside
 * nested function bodies or blocks. All declaration types are persisted
 * to the vm.Context via __persist so they survive across eval calls.
 *
 * Handles `var a = expr, b = expr` (and let/const equivalents) by tracking
 * nesting depth to skip over initializer expressions when looking for commas.
 */
function scanDeclarations(code: string): string[] {
	const names = new Set<string>();

	// Pre-compute brace depth at each position (to skip nested scopes)
	const braceDepth = computeBraceDepth(code);

	// Scan var/let/const declarations at top level only
	const declRegex = /\b(var|let|const)\s/g;
	let declMatch;

	while ((declMatch = declRegex.exec(code)) !== null) {
		// Skip declarations inside nested blocks (function bodies, etc.)
		if (braceDepth[declMatch.index] > 0) continue;

		// Skip let/const inside for-loop headers: `for (let i = ...)`
		// These are block-scoped to the loop and can't be persisted.
		const before = code.slice(0, declMatch.index).trimEnd();
		if ((declMatch[1] === "let" || declMatch[1] === "const") && before.endsWith("(")) {
			// Check if the open paren belongs to a `for` keyword
			const parenIdx = before.length - 1;
			const preSlice = code.slice(0, parenIdx).trimEnd();
			if (/\bfor$/.test(preSlice)) continue;
		}

		let pos = declMatch.index + declMatch[0].length;
		let expectIdent = true;
		let depth = 0;

		while (pos < code.length) {
			const ch = code[pos];

			if (depth > 0) {
				if (ch === "(" || ch === "[" || ch === "{") depth++;
				else if (ch === ")" || ch === "]" || ch === "}") depth--;
				pos++;
				continue;
			}

			if (ch === ";" || ch === "\n") break;

			if (expectIdent) {
				if (/\s/.test(ch)) { pos++; continue; }
				const idMatch = code.slice(pos).match(/^([a-zA-Z_$][\w$]*)/);
				if (idMatch) {
					names.add(idMatch[1]);
					pos += idMatch[1].length;
					expectIdent = false;
				} else {
					pos++;
				}
			} else {
				if (ch === "=") {
					pos++;
					// Skip initializer expression until comma or semicolon at depth 0
					while (pos < code.length) {
						const c2 = code[pos];
						if (c2 === "(" || c2 === "[" || c2 === "{") depth++;
						else if (c2 === ")" || c2 === "]" || c2 === "}") depth--;
						else if (depth === 0 && (c2 === "," || c2 === ";" || c2 === "\n")) break;
						pos++;
					}
				} else if (ch === ",") {
					pos++;
					expectIdent = true;
				} else {
					pos++;
				}
			}
		}
	}

	// Scan named function declarations at top level only
	const funcRegex = /^\s*function\s+([a-zA-Z_$][\w$]*)\s*\(/gm;
	let funcMatch;
	while ((funcMatch = funcRegex.exec(code)) !== null) {
		if (braceDepth[funcMatch.index] > 0) continue;
		names.add(funcMatch[1]);
	}

	return [...names];
}

/** Compute brace depth at each position in the code, respecting strings. */
function computeBraceDepth(code: string): number[] {
	const depth = new Array<number>(code.length);
	let d = 0;
	let inString: string | null = null;

	for (let i = 0; i < code.length; i++) {
		const ch = code[i];

		if (inString) {
			depth[i] = d;
			if (ch === inString && code[i - 1] !== "\\") inString = null;
			continue;
		}

		if (ch === '"' || ch === "'" || ch === "`") {
			inString = ch;
			depth[i] = d;
			continue;
		}

		if (ch === "{") {
			depth[i] = d;
			d++;
		} else if (ch === "}") {
			d = Math.max(0, d - 1);
			depth[i] = d;
		} else {
			depth[i] = d;
		}
	}

	return depth;
}

/**
 * Wrap user code in an async IIFE that returns the last expression's value.
 *
 * Strategy:
 * 1. Try to compile the entire code as a single expression (`return (code)`).
 *    If it works, we know it's an expression and can return its value directly.
 * 2. Otherwise treat as statements: find the last expression-like line and
 *    transform it into a return statement.
 */
function wrapInAsyncIIFE(code: string, persistCode: string): string {
	const trimmed = code.trim();
	if (!trimmed) return `(async () => {${persistCode}})()`;

	// Attempt 1: entire code is a single expression
	// Strip trailing semicolons — they're valid in statements but not
	// inside the parenthesised expression wrapper, and Bun's vm.Script
	// won't always reject the resulting invalid syntax at compile time.
	const exprCode = trimmed.replace(/;+\s*$/, "");

	// Skip expression attempt if any line starts with a statement keyword.
	// This is a fast-path optimisation; the runtime SyntaxError retry in
	// eval() is the real safety net for Bun's vm.Script which doesn't
	// always reject invalid syntax at compile time.
	const hasStatements = exprCode
		.split("\n")
		.some((line) => STMT_KEYWORD_RE.test(line.trim()));
	if (!hasStatements) {
		const exprWrapped = persistCode
			? `(async () => { var __r = (\n${exprCode}\n);${persistCode}\n return __r; })()`
			: `(async () => { return (\n${exprCode}\n); })()`;
		try {
			new vm.Script(exprWrapped);
			return exprWrapped;
		} catch {
			// Not a single expression — fall through to statement mode
		}
	}

	// Attempt 2: statements with implicit return on last expression.
	return wrapAsStatements(code, persistCode);
}

/** Wrap code as statements (no expression attempt) with implicit return on last expression. */
function wrapAsStatements(code: string, persistCode: string): string {
	const trimmed = code.trim();
	if (!trimmed) return `(async () => {${persistCode}})()`;

	if (persistCode) {
		// Use __retval assignment (not return) so persistCode runs before exit.
		const withImplicit = addImplicitReturn(trimmed, "__retval");
		return `(async () => { let __retval;\n${withImplicit}${persistCode}\nreturn __retval; })()`;
	}
	const withReturn = addImplicitReturn(trimmed);
	return `(async () => {\n${withReturn}\n})()`;
}

/**
 * Heuristic: find the last expression-like statement and capture its value.
 * Handles lines with multiple semicolon-separated statements by splitting
 * at top-level semicolons and only transforming the final segment.
 *
 * @param assignTo - Variable name to assign the last expression to.
 *   If provided, uses `assignTo = (expr)` (for use with __retval pattern).
 *   If not provided, uses `return (expr)`.
 */
function addImplicitReturn(code: string, assignTo?: string): string {
	const lines = code.split("\n");

	for (let i = lines.length - 1; i >= 0; i--) {
		const line = lines[i].trim();
		if (!line || line.startsWith("//") || line.startsWith("/*")) continue;

		if (line.endsWith("}")) return code;

		// Split line into statements at top-level semicolons
		const stmts = splitTopLevelStatements(line);
		const lastStmt = stmts[stmts.length - 1].trim();

		if (!lastStmt) return code;

		// Don't transform keyword-led statements
		if (STMT_KEYWORD_RE.test(lastStmt)) return code;

		// Transform last statement
		const transformed = assignTo
			? `${assignTo} = (${lastStmt});`
			: `return (${lastStmt});`;
		if (stmts.length > 1) {
			const prefix = stmts.slice(0, -1).join("; ") + "; ";
			lines[i] = `${prefix}${transformed}`;
		} else {
			lines[i] = transformed;
		}
		return lines.join("\n");
	}

	return code;
}

const STMT_KEYWORD_RE =
	/^(var|let|const|function|class|if|else|for|while|do|switch|try|catch|finally|return|throw|break|continue|import|export)\b/;

/** Split a line into statements at top-level semicolons (respecting nesting). */
function splitTopLevelStatements(line: string): string[] {
	const result: string[] = [];
	let depth = 0;
	let current = "";
	let inString: string | null = null;

	for (let i = 0; i < line.length; i++) {
		const ch = line[i];

		if (inString) {
			current += ch;
			if (ch === inString && line[i - 1] !== "\\") inString = null;
			continue;
		}

		if (ch === '"' || ch === "'" || ch === "`") {
			inString = ch;
			current += ch;
			continue;
		}

		if (ch === "(" || ch === "[" || ch === "{") depth++;
		else if (ch === ")" || ch === "]" || ch === "}") depth--;

		if (ch === ";" && depth === 0) {
			if (current.trim()) result.push(current.trim());
			current = "";
			continue;
		}

		current += ch;
	}

	if (current.trim()) result.push(current.trim());
	return result;
}

function formatArg(arg: unknown): string {
	if (typeof arg === "string") return arg;
	try {
		return JSON.stringify(arg, null, 2);
	} catch {
		return String(arg);
	}
}

function raceTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
	if (timeoutMs <= 0) return promise;
	return new Promise<T>((resolve, reject) => {
		const timer = setTimeout(() => {
			reject(new Error(`Eval timed out after ${timeoutMs}ms`));
		}, timeoutMs);
		promise.then(
			(val) => {
				clearTimeout(timer);
				resolve(val);
			},
			(err) => {
				clearTimeout(timer);
				reject(err);
			},
		);
	});
}
