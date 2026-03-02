# pi-rlm development tasks

# Default: show available commands
default:
    @just --list

# ─────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────

# Run the TUI with a task
start *args:
    bun pi-rlm/src/cli.ts {{args}}

# ─────────────────────────────────────────────────────────────
# Development
# ─────────────────────────────────────────────────────────────

# Type check
typecheck:
    cd pi-rlm && bun tsc --noEmit

# Run tests
test:
    cd pi-rlm && bun vitest --run

# Build
build:
    cd pi-rlm && npm run build

# Clean build artifacts
clean:
    cd pi-rlm && npm run clean

# ARC trace viewer — API server + Vite HMR (localhome: http://arc/)
dev:
    NAME=arc bun pi-rlm/src/arc-viewer.ts & \
    cd pi-rlm && bunx vite arc-viewer --open & \
    wait

# ─────────────────────────────────────────────────────────────
# Ralph
# ─────────────────────────────────────────────────────────────

# Run ralph loop for a session
ralph session:
    polo ralph --session {{session}}
