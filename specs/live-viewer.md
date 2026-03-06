# Live Streaming Viewer on Modal

## Problem

When running ARC evaluations on Modal (multiple GPU shards evaluating tasks in parallel), we can't see what's happening until the run finishes. We want to watch traces live in the browser — see the model's code, sandbox output, and whether it's solving tasks — as the eval runs.

## Architecture

```
                        Modal
  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
  │  │ Shard 0  │  │ Shard 1  │  │ Shard 2  │  ...            │
  │  │ (A100)   │  │ (A100)   │  │ (A100)   │                 │
  │  │          │  │          │  │          │                 │
  │  │ Writes   │  │ Writes   │  │ Writes   │                 │
  │  │ JSONL    │  │ JSONL    │  │ JSONL    │                 │
  │  └────┬─────┘  └────┬─────┘  └────┬─────┘                 │
  │       │              │              │                       │
  │       ▼              ▼              ▼                       │
  │  ┌────────────────────────────────────────────┐            │
  │  │           Modal Volume (/data/)            │            │
  │  │  eval-runs/<run>/sessions/*/agent-0.jsonl  │            │
  │  │  eval-runs/<run>/shard-*.json              │            │
  │  └───────────────────┬────────────────────────┘            │
  │                      │ periodic rescan                     │
  │                      ▼                                     │
  │  ┌────────────────────────────────────────────┐            │
  │  │       Viewer Server (web_server)           │            │
  │  │       @modal.web_server(port=3334)         │            │
  │  │                                            │            │
  │  │  viewer.ts (bun process)                   │            │
  │  │  ├─ HTTP: /api/runs, /api/session/...      │            │
  │  │  ├─ WebSocket: /ws (live events)           │            │
  │  │  ├─ Static: /, /main.js, /style.css        │            │
  │  │  └─ Periodic rescan (no fs.watch)          │            │
  │  └────────────────────────────────────────────┘            │
  │                      │                                     │
  └──────────────────────┼─────────────────────────────────────┘
                         │ HTTPS (Modal auto-URL)
                         ▼
                    ┌──────────┐
                    │ Browser  │
                    │ (local)  │
                    └──────────┘
```

## Key Design Decisions

### 1. Reuse viewer.ts, don't rebuild

The viewer.ts already has everything we need:
- HTTP API (`/api/runs`, `/api/session/:id`)
- WebSocket server (`/ws`) with subscription model, event push, replay
- FileWatcher with polling-based file change detection (250ms intervals via `statSync`)
- JSONL parsing (`convertMessagesToEvents`)

We make two modifications:
1. **Add static file serving** — serve `index.html`, `main.js`, `style.css` from the viewer directory
2. **Replace `fs.watch()` with periodic rescan** — `fs.watch()` won't fire for cross-container writes on Modal volumes. The polling via `statSync` already works for file growth detection; we just need periodic directory scanning to discover *new* session directories.

### 2. Volume-based communication (not direct push)

Shards write JSONL traces to the Modal volume exactly as they do now. The viewer server reads from the same volume. No direct shard→viewer communication needed.

**Why not direct push?** Simpler. The shards don't need to know the viewer's URL. The viewer doesn't need a special ingest API. The existing TraceLogger writes to files — no changes needed. The volume is the communication channel.

**Tradeoff:** A few seconds of delay (volume sync + rescan interval). Acceptable for monitoring — you're watching a 30-60 minute run, not a real-time game.

### 3. @modal.web_server for Bun

Modal's `@modal.web_server(port=N)` works with any HTTP server that binds to `0.0.0.0:N`. We run `bun domains/arc-agi-2/src/viewer.ts --port 3334` inside the container. Modal handles TLS termination and URL assignment.

WebSocket support is confirmed: "WebSockets on Modal maintain a single function call per connection" with RFC 6455 support.

### 4. RUNS_ROOT on Modal

Locally, viewer.ts reads from `runs/` at repo root. On Modal, it should read from the volume mount: `/data/eval-runs/`. This is configurable via environment variable or CLI arg.

## What to Build

### 1. Modify viewer.ts for Modal compatibility (~30 lines changed)

**a. Add static file serving:**
```typescript
// Serve static files for paths not matching /api/ or /ws
const viewerDir = join(import.meta.dirname, "../../viewer");
const STATIC_FILES: Record<string, { path: string; mime: string }> = {
  "/": { path: "index.html", mime: "text/html" },
  "/main.js": { path: "main.js", mime: "application/javascript" },
  "/style.css": { path: "style.css", mime: "text/css" },
};

// In the request handler, before the 404:
if (STATIC_FILES[path]) {
  const { path: filePath, mime } = STATIC_FILES[path];
  const content = readFileSync(join(viewerDir, filePath));
  res.writeHead(200, { "Content-Type": mime });
  res.end(content);
  return;
}
```

**b. Replace `fs.watch()` with periodic rescan:**

The FileWatcher currently uses three levels of `fs.watch()`:
1. `watch(RUNS_ROOT)` — detect new run directories
2. `watch(sessionsDir)` — detect new session directories within a run
3. `watch(sessionDir)` — detect new agent-*.jsonl files within a session

On Modal volumes, `fs.watch()` won't fire for cross-container writes. Replace with a periodic `rescanDirectories()` that runs every 2-3 seconds:

```typescript
startPeriodicRescan(): void {
  setInterval(() => {
    // Scan for new runs
    if (!existsSync(RUNS_ROOT)) return;
    for (const runEntry of readdirSync(RUNS_ROOT, { withFileTypes: true })) {
      if (!runEntry.isDirectory()) continue;
      const sessionsDir = join(RUNS_ROOT, runEntry.name, "sessions");
      if (!existsSync(sessionsDir)) continue;

      // Scan for new session directories
      for (const sessionEntry of readdirSync(sessionsDir, { withFileTypes: true })) {
        if (!sessionEntry.isDirectory()) continue;
        const sid = `${runEntry.name}/${sessionEntry.name}`;
        if (this.activeSessions_.has(sid)) continue; // already watching

        const agent0 = join(sessionsDir, sessionEntry.name, "agent-0.jsonl");
        if (existsSync(agent0)) {
          this.watchSessionDir(sid, join(sessionsDir, sessionEntry.name), runEntry.name);
          this.broadcastControl({ type: "session_active", sessionId: sid, runName: runEntry.name });
        }
      }
    }
  }, 3000); // Every 3 seconds
}
```

The existing per-file polling (250ms `statSync` checks in `readNewAgentLines`) continues to work — it doesn't use `fs.watch()`, it uses `setInterval` + `statSync`. So file growth detection (new lines appended to agent-0.jsonl) already works cross-container.

**c. Configurable RUNS_ROOT:**
```typescript
const RUNS_ROOT = process.env.VIEWER_RUNS_ROOT
  ?? join(repoRoot, "runs");
```

### 2. Add viewer to Modal app (~20 lines in modal_app.py)

```python
@app.function(
    image=image,
    volumes={DATA_PATH: data_volume},
    keep_warm=1,  # Stay alive between requests
    allow_concurrent_inputs=100,  # Handle many WebSocket connections
    cpu=2,
    memory=2048,
)
@modal.web_server(port=3334, startup_timeout=30)
def viewer():
    """Live trace viewer — serves the viewer UI and streams events."""
    import subprocess

    viewer_script = "/app/domains/arc-agi-2/src/viewer.ts"
    env = {
        **os.environ,
        "VIEWER_RUNS_ROOT": f"{DATA_PATH}/eval-runs",
        "PORT": "3334",
    }

    # viewer.ts binds to 0.0.0.0:3334
    subprocess.Popen(
        ["/root/.bun/bin/bun", "run", viewer_script],
        env=env,
    )
```

Also copy the viewer static files to the Modal image:
```python
.add_local_dir("domains/arc-agi-2/viewer", "/app/domains/arc-agi-2/viewer", copy=True)
```

### 3. Volume reload strategy

The viewer server container needs to see writes from shard containers. Modal volumes require explicit reload to see new data. Options:

**Option A: Periodic reload via Python wrapper**
Instead of `subprocess.Popen` for bun, use a Python wrapper that:
1. Starts bun viewer.ts as a subprocess
2. Periodically calls `data_volume.reload()` every 5 seconds

```python
@modal.web_server(port=3334, startup_timeout=30)
def viewer():
    import subprocess, threading, time

    # Start viewer.ts
    proc = subprocess.Popen(
        ["/root/.bun/bin/bun", "run", "/app/domains/arc-agi-2/src/viewer.ts"],
        env={**os.environ, "VIEWER_RUNS_ROOT": f"{DATA_PATH}/eval-runs", "PORT": "3334"},
    )

    # Periodic volume reload in background thread
    def reload_loop():
        while proc.poll() is None:
            try:
                data_volume.reload()
            except Exception:
                pass
            time.sleep(5)

    threading.Thread(target=reload_loop, daemon=True).start()
```

**Option B: Let the filesystem handle it**
Some Modal volume implementations may not require explicit reload for reads (just for seeing newly committed data). If `statSync` returns updated sizes for files being written by other containers, the polling works without reload. Worth testing — if it works, no reload needed.

**Recommendation:** Implement Option A. If testing shows it's unnecessary, remove the reload loop.

## Usage Flow

### 1. Deploy the viewer (once, stays alive):
```bash
modal deploy rl/modal_app.py
# Prints: https://YOUR-ORG--sdpo-arc-viewer.modal.run
```

### 2. Launch an eval run:
```bash
modal run rl/modal_app.py::evaluate -- --model Qwen/Qwen3-8B --dataset ARC-AGI-1 --count 100
```

### 3. Open the viewer in your browser:
Navigate to `https://YOUR-ORG--sdpo-arc-viewer.modal.run`

The viewer shows:
- Sidebar: runs with tasks, live badges for active sessions
- Click a task: see the multi-turn conversation unfolding in real-time
- Dashboard mode: watch all active sessions simultaneously

### 4. After the run:
Traces persist on the volume. The viewer continues to serve them. You can browse traces from any past run.

## What Doesn't Change

- **TraceLogger** — writes JSONL files exactly as before. No push mechanism needed.
- **eval_loop.py** — evaluates tasks and writes traces. No viewer awareness.
- **Frontend (main.js)** — uses relative URLs (`/api/runs`, `/ws`). Works unchanged whether served locally or from Modal.
- **WebSocket protocol** — same subscribe/unsubscribe/event messages.
- **JSONL format** — same `agent-0.jsonl` format the viewer already reads.

## Complexity Assessment

| Component | Lines | Risk |
|-----------|-------|------|
| Static file serving in viewer.ts | ~15 | Low — trivial file reads |
| Periodic rescan (replace fs.watch) | ~30 | Low — similar to existing scanForActiveSessions |
| Configurable RUNS_ROOT | ~3 | None |
| Modal web_server function | ~20 | Medium — need to verify volume reload behavior |
| Copy viewer static files to image | ~1 | None |

**Total: ~70 lines of changes.** Most of the viewer infrastructure already exists.

## Risks

| Risk | Mitigation |
|------|-----------|
| Volume sync delay (new files not visible) | `volume.reload()` every 5s; accept small delay |
| `statSync` returns stale data cross-container | Test empirically; if broken, fall back to POST-based push |
| WebSocket disconnects on Modal | Frontend already has auto-reconnect (2s retry) |
| Viewer container cold-starts | `keep_warm=1` keeps it alive; first load may take ~10s |
| Bun process crash in web_server | Modal will restart the container; viewer.ts is stable |

## Future Enhancements (not for initial version)

- **Direct event push from shards** — HTTP POST to viewer's `/api/ingest` for sub-second latency. Only needed if volume delay is unacceptable.
- **Training rollout viewing** — add TraceLogger to ArcInteraction for training runs, same viewer works.
- **Tailscale integration** — access viewer over VPN instead of public URL. Modal supports Tailscale natively.
- **Multi-run comparison view** — viewer already has run.json data; add side-by-side comparison UI.
