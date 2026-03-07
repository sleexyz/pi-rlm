# Live Trace Viewer on Modal

## Overview

Real-time monitoring of ARC evaluation runs on Modal. Eval shards write JSONL traces to a shared volume; a viewer server reads from the same volume and streams events to a browser via WebSocket.

**URL:** `https://websim-ai--sdpo-arc-viewer.modal.run`

## Architecture

```
                        Modal
  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
  │  │ Shard 0  │  │ Shard 1  │  │ Shard 2  │  ...            │
  │  │ (A100)   │  │ (A100)   │  │ (A100)   │                 │
  │  │          │  │          │  │          │                 │
  │  │ TraceLog │  │ TraceLog │  │ TraceLog │                 │
  │  │ → JSONL  │  │ → JSONL  │  │ → JSONL  │                 │
  │  └────┬─────┘  └────┬─────┘  └────┬─────┘                 │
  │       │              │              │                       │
  │       ▼              ▼              ▼                       │
  │  ┌────────────────────────────────────────────┐            │
  │  │     Modal Volume (sdpo-arc-data, /data/)   │            │
  │  │                                            │            │
  │  │  eval-runs/<run>/run.json                  │            │
  │  │  eval-runs/<run>/sessions/*/agent-0.jsonl  │            │
  │  └───────────────────┬────────────────────────┘            │
  │                      │                                     │
  │       volume.reload() every 5s                             │
  │       directory rescan every 3s                            │
  │       file poll (statSync) every 250ms                     │
  │                      │                                     │
  │  ┌───────────────────▼────────────────────────┐            │
  │  │       Viewer Container (CPU-only)          │            │
  │  │       @modal.web_server(port=3334)         │            │
  │  │                                            │            │
  │  │  Python wrapper:                           │            │
  │  │  ├─ Starts bun viewer.ts subprocess        │            │
  │  │  └─ Background thread: volume.reload()     │            │
  │  │                                            │            │
  │  │  viewer.ts (bun process):                  │            │
  │  │  ├─ HTTP: /api/runs, /api/session/:id      │            │
  │  │  ├─ WebSocket: /ws (live event stream)     │            │
  │  │  ├─ Static: /, /main.js, /style.css        │            │
  │  │  ├─ FileWatcher: 250ms statSync polling    │            │
  │  │  └─ Periodic rescan: 3s directory scan     │            │
  │  └────────────────────────────────────────────┘            │
  │                      │                                     │
  └──────────────────────┼─────────────────────────────────────┘
                         │ HTTPS + WSS (Modal auto-TLS)
                         ▼
                    ┌──────────┐
                    │ Browser  │
                    └──────────┘
```

## How It Works

### Communication Channel: The Volume

Shards and viewer never communicate directly. The Modal volume is the channel:

1. **Eval shards** write `agent-0.jsonl` files via `TraceLogger` (one per task session)
2. **Viewer container** calls `volume.reload()` every 5s to see new files from other containers
3. **viewer.ts** rescans directories every 3s to discover new runs/sessions
4. **viewer.ts** polls known files via `statSync` every 250ms to detect new lines

This gives ~5-8s latency for new sessions to appear, and ~250ms latency for new lines within an already-discovered session.

### Three Layers of Polling

| Layer | Interval | What it catches | Mechanism |
|-------|----------|----------------|-----------|
| Volume reload | 5s | New files written by other containers | Python `data_volume.reload()` |
| Directory rescan | 3s | New run dirs, session dirs, agent files | `readdirSync` in `startPeriodicRescan()` |
| File polling | 250ms | New lines appended to known JSONL files | `statSync` size check in `readNewAgentLines()` |

### JSONL Trace Format

Each session produces `agent-0.jsonl` with these entry types:

```jsonl
{"type":"session","version":1,"agentIndex":0,"timestamp":"...","metadata":{"taskId":"...","model":"..."}}
{"type":"message","ts":1234,"message":{"role":"user","content":"task description..."}}
{"type":"message","ts":1234,"message":{"role":"assistant","content":[{"type":"text","text":"..."},{"type":"tool_use","name":"eval","arguments":{"code":"..."}}]}}
{"type":"message","ts":1234,"message":{"role":"tool","content":"output..."}}
{"type":"session_end","ts":1234,"usage":{"totalTokens":12345}}
```

The viewer's `convertMessagesToEvents()` transforms these into frontend events:
- `session` → `session_start` + `agent_start`
- `message` (assistant with `tool_use`) → `message_end` + `tool_execution_start`
- `message` (tool) → `tool_execution_end`
- `session_end` → `session_end`

### Frontend

The viewer frontend (`domains/arc-agi-2/viewer/`) connects via WebSocket to `/ws` and subscribes to sessions. It renders:

- **Sidebar**: runs auto-expanded, task rows with status badges (LIVE/checkmark/X), clickable to load traces
- **Trace view**: multi-turn conversation with code blocks, sandbox output, thinking sections
- **Dashboard mode**: all active sessions rendered simultaneously

The frontend uses relative URLs (`/api/runs`, `/ws`) so it works identically whether served locally or from Modal.

## Files

| File | Role |
|------|------|
| `domains/arc-agi-2/src/viewer.ts` | HTTP API, WebSocket, FileWatcher, static serving |
| `domains/arc-agi-2/viewer/` | Frontend (index.html, main.js, style.css) |
| `rl/modal_app.py` (`viewer()`) | Modal web_server wrapper with volume reload |
| `rl/trace_logger.py` | Writes JSONL traces in viewer format |
| `rl/eval_loop.py` | Eval loop that writes traces via TraceLogger |

## Viewer Modifications for Modal

Four changes to `viewer.ts` (additive, local usage still works):

### 1. Static file serving
Serves `index.html`, `main.js`, `style.css` from `../viewer/` directory. Simple path→file map, not a general file server.

### 2. Configurable RUNS_ROOT
```typescript
const RUNS_ROOT = process.env.VIEWER_RUNS_ROOT ?? join(repoRoot, "runs");
```
Modal sets `VIEWER_RUNS_ROOT=/data/eval-runs`.

### 3. Periodic directory rescan
`startPeriodicRescan()` runs every 3s alongside the existing `fs.watch()` watchers. `fs.watch()` works locally but doesn't fire for cross-container writes on Modal volumes. The rescan catches new runs/sessions that `fs.watch()` misses. Skips completed sessions (those with `session_end` in their JSONL).

### 4. Bind to 0.0.0.0
Required for Modal's `@modal.web_server` to proxy traffic to the container.

## Modal Deployment

### viewer() function in modal_app.py

```python
@app.function(
    image=image,
    volumes={DATA_PATH: data_volume},
    min_containers=1,       # Always warm, no cold starts
    cpu=2, memory=2048,     # CPU-only, no GPU
)
@modal.concurrent(max_inputs=100)  # Many WebSocket connections
@modal.web_server(port=3334, startup_timeout=30)
def viewer():
    proc = subprocess.Popen(
        ["/root/.bun/bin/bun", "run", "/app/domains/arc-agi-2/src/viewer.ts"],
        env={**os.environ, "VIEWER_RUNS_ROOT": f"{DATA_PATH}/eval-runs", "PORT": "3334"},
    )
    # Background volume reload
    def reload_loop():
        while proc.poll() is None:
            try: data_volume.reload()
            except Exception: pass
            time.sleep(5)
    threading.Thread(target=reload_loop, daemon=True).start()
```

### Image requirements
- Viewer static files: `.add_local_dir("domains/arc-agi-2/viewer", ...)`
- `ws` npm package: `.run_commands("cd /app/domains/arc-agi-2 && bun add ws")`
- Bun runtime (already in base image for sandbox-server.ts)

## Usage

### Deploy (persistent):
```bash
modal deploy rl/modal_app.py
# → https://websim-ai--sdpo-arc-viewer.modal.run
```

### Run eval (traces auto-appear in viewer):
```bash
modal run rl/modal_app.py::evaluate --model Qwen/Qwen3-8B --dataset ARC-AGI-1 --count 10
```

### Local usage (same viewer, local files):
```bash
VIEWER_RUNS_ROOT=runs bun domains/arc-agi-2/src/viewer.ts
# → http://localhost:3334
```

## Design Decisions

### Why volume-based, not direct push?
Simpler. Shards don't need the viewer's URL. TraceLogger just writes files. No ingest API. The volume is the only coordination mechanism. Tradeoff: ~5-8s latency for new sessions. Acceptable for monitoring 30-60 minute eval runs.

### Why bun subprocess, not a Python web server?
viewer.ts already exists (~960 lines) with full WebSocket support, JSONL parsing, file watching, and event streaming. Rewriting in Python would be a massive effort for zero benefit. The Python wrapper just handles volume.reload() which the TS process can't do.

### Why keep fs.watch() alongside periodic rescan?
Local development. `fs.watch()` gives instant detection on local filesystems. The periodic rescan is additive — it catches things `fs.watch()` misses on network filesystems without breaking local usage.

### Why min_containers=1?
The viewer must be instantly available when you open the URL. Cold-starting a container takes ~10-30s which makes the viewer feel broken. `min_containers=1` keeps one container warm at all times (minimal cost since it's CPU-only).

## Observed Behavior

- New sessions appear in sidebar within ~8-10s of the first JSONL write
- Once a session is discovered, new lines stream within ~250ms
- WebSocket reconnects automatically on disconnect (2s retry in frontend)
- Volume reload has no measurable CPU cost
- The viewer handles 6+ concurrent eval runs without issues
- `watchDirectories()` gracefully handles missing RUNS_ROOT (logs warning, relies on periodic rescan)

## Future Work

- **Direct event push** — shards POST to viewer's `/api/ingest` for sub-second new-session latency. Only worth it if volume delay is unacceptable.
- **Training rollout viewing** — add TraceLogger to verl's tool_agent_loop for training-time traces.
- **Run comparison** — side-by-side view of different model runs on same tasks.
