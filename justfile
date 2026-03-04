# pi-rlm development tasks

# Default: show available commands
default:
    @just --list

# ─────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────

# Install dependencies
install:
    bun install

# Run the TUI with a task
start *args:
    ./bin/pi-rlm {{args}}

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

# Unified viewer for all domains — API server + Vite HMR (localhome: http://arc/)
dev:
    NAME=arc bun domains/arc-agi-2/src/viewer.ts & \
    bunx vite domains/arc-agi-2/viewer --open & \
    wait

# ─────────────────────────────────────────────────────────────
# ARC evaluation
# ─────────────────────────────────────────────────────────────

# Run ARC-AGI-1 evaluation
arc-1 *args:
    bun domains/arc-agi-2/src/runner.ts --data-dir downloads/ARC-AGI-1/data --log-dir logs/arc-1 --results-dir results/arc-1 {{args}}

# Run ARC-AGI-2 evaluation
arc-2 *args:
    bun domains/arc-agi-2/src/runner.ts --log-dir logs/arc-2 --results-dir results/arc-2 {{args}}

# Run ARC-AGI-2 tests
arc-test:
    cd domains/arc-agi-2 && bun vitest --run

# ─────────────────────────────────────────────────────────────
# Ralph
# ─────────────────────────────────────────────────────────────

# Run ralph loop for a session
ralph session:
    polo ralph --session {{session}}
