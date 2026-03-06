"""SDPO Training & Baseline Evaluation for ARC on Modal

Usage:
    # Training (4xA100-80GB):
    modal run rl/modal_app.py::prep_data
    modal run --detach rl/modal_app.py::train

    # Evaluation (1xA100-80GB per shard):
    modal run rl/modal_app.py::evaluate --model Qwen/Qwen3-8B --dataset ARC-AGI-1 --count 5
"""

import os
import subprocess

import modal

app = modal.App("sdpo-arc")

# Volumes
data_volume = modal.Volume.from_name("sdpo-arc-data", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("sdpo-arc-checkpoints", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("hf-cache", create_if_missing=True)

# Paths
SDPO_DIR = "/app/SDPO"
DATA_PATH = "/data"
CHECKPOINTS_PATH = "/checkpoints"
HF_CACHE_PATH = "/root/.cache/huggingface"
RL_DIR = "/app/rl"

# Training config
MODEL = "Qwen/Qwen3-8B"
N_GPUS = 4

# Image
image = (
    modal.Image.from_registry("verlai/verl:vllm011.latest")
    .apt_install("git", "curl", "unzip")
    .entrypoint([])
    # Python deps
    .pip_install(
        "codetiming",
        "pylatexenc",
        "wandb",
        "dill",
        "pybind11",
        "liger-kernel",
        "word2number",
        "latex2sympy2",
        "math-verify[antlr4_9_3]==%s" % "0.8.0",
        "latex2sympy2_extended",
        "sortedcontainers",
        "pandas",
        "pyarrow",
    )
    # Clone SDPO repo
    .run_commands(
        f"git clone https://github.com/lasgroup/SDPO.git {SDPO_DIR}",
        f"cd {SDPO_DIR} && pip install -e .",
    )
    # Install Bun
    .run_commands(
        "curl -fsSL https://bun.sh/install | bash",
        'echo \'export BUN_INSTALL="$HOME/.bun"\' >> /root/.bashrc',
        'echo \'export PATH="$HOME/.bun/bin:$PATH"\' >> /root/.bashrc',
    )
    # Copy TS source files + package.json for workspace resolution
    .add_local_file("package.json", "/app/package.json", copy=True)
    .add_local_file("pi-rlm/package.json", "/app/pi-rlm/package.json", copy=True)
    .add_local_dir("pi-rlm/src", "/app/pi-rlm/src", copy=True)
    .add_local_dir("pi-rlm/dist", "/app/pi-rlm/dist", copy=True)
    .add_local_file("domains/arc-agi-2/package.json", "/app/domains/arc-agi-2/package.json", copy=True)
    .add_local_dir("domains/arc-agi-2/src", "/app/domains/arc-agi-2/src", copy=True)
    .add_local_dir("domains/arc-agi-2/viewer", "/app/domains/arc-agi-2/viewer", copy=True)
    # Install TS dependencies (workspace-aware)
    .run_commands("cd /app && /root/.bun/bin/bun install")
    # Copy Python RL code
    .add_local_file("rl/__init__.py", f"{RL_DIR}/__init__.py", copy=True)
    .add_local_file("rl/arc_env.py", f"{RL_DIR}/arc_env.py", copy=True)
    .add_local_file("rl/arc_data.py", f"{RL_DIR}/arc_data.py", copy=True)
    .add_local_file("rl/arc_interaction.py", f"{RL_DIR}/arc_interaction.py", copy=True)
    .add_local_file("rl/arc_reward.py", f"{RL_DIR}/arc_reward.py", copy=True)
    .add_local_file("rl/eval_loop.py", f"{RL_DIR}/eval_loop.py", copy=True)
    .add_local_file("rl/trace_logger.py", f"{RL_DIR}/trace_logger.py", copy=True)
    # Copy ARC data
    .add_local_dir("downloads/ARC-AGI-2/data", "/app/downloads/ARC-AGI-2/data", copy=True)
    .add_local_dir("downloads/ARC-AGI-1/data", "/app/downloads/ARC-AGI-1/data", copy=True)
    # Copy configs
    .add_local_file("rl/config/sdpo_arc.yaml", f"{SDPO_DIR}/verl/trainer/config/sdpo_arc.yaml", copy=True)
    .add_local_file("rl/config/arc_interaction_config.yaml", f"{RL_DIR}/config/arc_interaction_config.yaml", copy=True)
    .run_commands(f"mkdir -p {SDPO_DIR}/datasets")
    .env(
        {
            "VLLM_USE_V1": "1",
            "PYTHONUNBUFFERED": "1",
            "HF_HOME": HF_CACHE_PATH,
            "PYTHONPATH": f"{SDPO_DIR}:{RL_DIR}/..",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "PATH": "/root/.bun/bin:/usr/local/bin:/usr/bin:/bin",
        }
    )
)


def _start_vllm_server(
    model: str,
    port: int = 8000,
    max_model_len: int = 32768,
    max_logprobs: int = 100,
    gpu_memory_utilization: float = 0.9,
) -> subprocess.Popen:
    """Start vLLM as an OpenAI-compatible HTTP server."""
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--max-logprobs", str(max_logprobs),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
    ]
    print(f"Starting vLLM server: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _wait_for_server(port: int = 8000, timeout: int = 300):
    """Poll vLLM health endpoint until ready."""
    import time
    import urllib.request
    import urllib.error

    url = f"http://localhost:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            print(f"vLLM server ready on port {port}")
            return
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            time.sleep(2)
    raise TimeoutError(f"vLLM server not ready after {timeout}s")


@app.function(
    image=image,
    volumes={DATA_PATH: data_volume, HF_CACHE_PATH: hf_cache_volume},
    timeout=3600,
    cpu=4,
    memory=16384,
)
def prep_data():
    """Load ARC tasks, generate system prompts via TS, create parquet files."""
    import sys
    sys.path.insert(0, f"{RL_DIR}/..")

    out_dir = f"{DATA_PATH}/arc"
    if os.path.exists(f"{out_dir}/train.parquet") and os.path.exists(f"{out_dir}/test.parquet"):
        print(f"Data already exists at {out_dir}, skipping.")
        return

    os.makedirs(out_dir, exist_ok=True)

    # Verify bun works
    subprocess.run(["/root/.bun/bin/bun", "--version"], check=True)

    from rl.arc_data import load_arc_tasks, prepare_verl_dataset, save_parquet

    print("=== Loading ARC training tasks ===")
    train_tasks = load_arc_tasks("training")
    print(f"Loaded {len(train_tasks)} training tasks")

    print("=== Loading ARC evaluation tasks ===")
    eval_tasks = load_arc_tasks("evaluation")
    print(f"Loaded {len(eval_tasks)} evaluation tasks")

    print("=== Generating system prompts and creating parquet ===")
    train_df = prepare_verl_dataset(train_tasks)
    save_parquet(train_df, f"{out_dir}/train.parquet")
    print(f"Saved {len(train_df)} train rows")

    test_df = prepare_verl_dataset(eval_tasks)
    save_parquet(test_df, f"{out_dir}/test.parquet")
    print(f"Saved {len(test_df)} test rows")

    data_volume.commit()
    print("=== Data prep complete ===")


@app.function(
    image=image,
    gpu="A100-80GB:4",
    volumes={
        DATA_PATH: data_volume,
        CHECKPOINTS_PATH: checkpoints_volume,
        HF_CACHE_PATH: hf_cache_volume,
    },
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=86400,
)
def train(*arglist):
    """Launch SDPO training with multi-turn ARC interaction."""
    data_volume.reload()
    hf_cache_volume.reload()

    # Symlink data
    data_dir = f"{SDPO_DIR}/datasets/arc"
    src = f"{DATA_PATH}/arc"
    if not os.path.exists(data_dir):
        os.symlink(src, data_dir)

    # Verify data
    for f in ["train.parquet", "test.parquet"]:
        if not os.path.exists(f"{data_dir}/{f}"):
            raise FileNotFoundError(f"{data_dir}/{f} not found. Run prep_data first.")

    os.environ.setdefault("USER", "modal")
    os.environ["EXPERIMENT"] = "SDPO-ARC-Qwen3-8B"
    os.environ["TASK"] = "datasets/arc"
    os.environ["ARC_DATA_DIR"] = data_dir

    interaction_config = f"{RL_DIR}/config/arc_interaction_config.yaml"

    cmd = [
        "python", "-m", "verl.trainer.main_ppo",
        "--config-name", "sdpo_arc",
        f"actor_rollout_ref.model.path={MODEL}",
        f"actor_rollout_ref.rollout.multi_turn.interaction_config_path={interaction_config}",
        "trainer.resume_mode=auto",
        f"trainer.default_local_dir={CHECKPOINTS_PATH}/sdpo-arc",
    ]

    if arglist:
        cmd.extend(arglist)

    print("=" * 64)
    print("SDPO Multi-Turn Training for ARC-AGI-2")
    print(f"Model: {MODEL}")
    print(f"GPUs: {N_GPUS}")
    print("=" * 64)

    subprocess.run(cmd, check=True)
    checkpoints_volume.commit()


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={
        DATA_PATH: data_volume,
        HF_CACHE_PATH: hf_cache_volume,
    },
    timeout=86400,
)
def evaluate(
    model: str = "Qwen/Qwen3-8B",
    dataset: str = "ARC-AGI-1",
    split: str = "training",
    count: int = 100,
    shard: int = 0,
    num_shards: int = 1,
    run_name: str = "",
    max_turns: int = 15,
    temperature: float = 0.6,
    top_p: float = 0.95,
    timeout: int = 300,
):
    """Evaluate ARC tasks on a single GPU.

    Starts vLLM as an HTTP server, then spawns agent-runner.ts per task.
    Bun owns all agent logic; Python just dispatches and collects results.
    """
    import json
    import sys
    from datetime import datetime, timezone

    sys.path.insert(0, f"{RL_DIR}/..")
    hf_cache_volume.reload()

    from rl.arc_data import load_arc_tasks
    from rl.eval_loop import evaluate_tasks

    data_dir = f"/app/downloads/{dataset}/data"

    # Load and shard tasks
    print(f"=== Loading {dataset} {split} tasks ===")
    all_tasks = load_arc_tasks(split, data_dir=data_dir)[:count]
    tasks = [t for i, t in enumerate(all_tasks) if i % num_shards == shard]
    print(f"Shard {shard}/{num_shards}: {len(tasks)} tasks (of {len(all_tasks)} total)")

    if not tasks:
        print("No tasks for this shard, exiting.")
        return

    # Generate run name
    if not run_name:
        model_short = model.split("/")[-1].lower()
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_name = f"{model_short}_{dataset}_{split}_{count}_{ts}"

    run_dir = f"{DATA_PATH}/eval-runs/{run_name}"
    os.makedirs(run_dir, exist_ok=True)

    # Start vLLM HTTP server
    print("=== Starting vLLM server ===")
    server = _start_vllm_server(model)
    try:
        _wait_for_server()

        vllm_base_url = "http://localhost:8000/v1"

        print("=" * 64)
        print("ARC Evaluation (unified harness)")
        print(f"Model: {model}")
        print(f"Dataset: {dataset} / {split}")
        print(f"Tasks: {len(tasks)} (shard {shard}/{num_shards})")
        print(f"Run: {run_name}")
        print(f"vLLM: {vllm_base_url}")
        print("=" * 64)

        evaluate_tasks(
            model_name=model,
            tasks=tasks,
            run_dir=run_dir,
            run_name=run_name,
            vllm_base_url=vllm_base_url,
            max_turns=max_turns,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
        )
    finally:
        server.terminate()
        server.wait(timeout=10)

    # Write per-shard results
    shard_path = os.path.join(run_dir, f"shard-{shard}.json")
    run_json_path = os.path.join(run_dir, "run.json")
    if os.path.exists(run_json_path):
        with open(run_json_path) as f:
            run_data = json.load(f)
        with open(shard_path, "w") as f:
            json.dump(run_data, f, indent=2)

    # Create tar archive for reliable download
    import tarfile
    sessions_dir = os.path.join(run_dir, "sessions")
    if os.path.exists(sessions_dir):
        tar_path = os.path.join(run_dir, f"shard-{shard}-traces.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(sessions_dir, arcname="sessions")

    data_volume.commit()
    print(f"\n=== Shard {shard} complete. Results at {shard_path} ===")


@app.function(
    image=image,
    volumes={DATA_PATH: data_volume},
    timeout=300,
)
def pack_run(run_name: str = ""):
    """Pack a run into a single tar.gz for reliable download."""
    import json
    import tarfile

    data_volume.reload()
    run_dir = f"{DATA_PATH}/eval-runs/{run_name}"

    if not os.path.exists(run_dir):
        print(f"Run dir not found: {run_dir}")
        return

    tar_path = os.path.join(run_dir, "run.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        for item in ["run.json", "sessions"]:
            path = os.path.join(run_dir, item)
            if os.path.exists(path):
                tar.add(path, arcname=item)
        for f in sorted(os.listdir(run_dir)):
            if f.startswith("shard-") and f.endswith(".json"):
                tar.add(os.path.join(run_dir, f), arcname=f)

    data_volume.commit()
    size_mb = os.path.getsize(tar_path) / (1024 * 1024)
    print(f"Packed {run_name} -> {tar_path} ({size_mb:.1f} MB)")
    print(f"  modal volume get sdpo-arc-data eval-runs/{run_name}/run.tar.gz .")


@app.function(
    image=image,
    volumes={DATA_PATH: data_volume},
    cpu=2,
    memory=2048,
)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=3334, startup_timeout=30)
def viewer():
    """Live trace viewer — streams eval traces via WebSocket."""
    import threading
    import time

    proc = subprocess.Popen(
        ["/root/.bun/bin/bun", "run", "/app/domains/arc-agi-2/src/viewer.ts"],
        env={
            **os.environ,
            "VIEWER_RUNS_ROOT": f"{DATA_PATH}/eval-runs",
            "PORT": "3334",
        },
    )

    def reload_loop():
        while proc.poll() is None:
            try:
                data_volume.reload()
            except Exception:
                pass
            time.sleep(5)

    threading.Thread(target=reload_loop, daemon=True).start()


@app.local_entrypoint()
def main():
    """Run full pipeline: prep data then train."""
    prep_data.remote()
    train.remote()
