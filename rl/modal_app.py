"""SDPO Training for ARC-AGI-2 on Modal

Multi-turn SDPO using the existing verl + SDPO infrastructure from kirby,
with ARC code execution via the TypeScript sandbox (EvalRuntime + createArcAdapter).

Usage:
    # One-time setup: create WandB secret
    modal secret create wandb-secret WANDB_API_KEY=<your-key>

    # Step 1: Prepare dataset (~5 min)
    modal run rl/modal_app.py::prep_data

    # Step 2: Launch training (~8-12 hours on 4xA100-80GB)
    modal run --detach rl/modal_app.py::train

    # Override hyperparams:
    modal run --detach rl/modal_app.py::train -- trainer.total_epochs=2 data.train_batch_size=2
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
        "math-verify[antlr4_9_3]==0.8.0",
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
    # Install Bun (for TS sandbox)
    .run_commands(
        "curl -fsSL https://bun.sh/install | bash",
        'echo \'export BUN_INSTALL="$HOME/.bun"\' >> /root/.bashrc',
        'echo \'export PATH="$HOME/.bun/bin:$PATH"\' >> /root/.bashrc',
    )
    # Copy TS source files — must match js_sandbox.py's PROJECT_ROOT layout (/app/)
    .add_local_dir("pi-rlm/src", "/app/pi-rlm/src", copy=True)
    .add_local_dir("domains/arc-agi-2/src", "/app/domains/arc-agi-2/src", copy=True)
    # Copy Python RL code (exclude venv/cache)
    .add_local_file("rl/__init__.py", f"{RL_DIR}/__init__.py", copy=True)
    .add_local_file("rl/parser.py", f"{RL_DIR}/parser.py", copy=True)
    .add_local_file("rl/js_sandbox.py", f"{RL_DIR}/js_sandbox.py", copy=True)
    .add_local_file("rl/arc_interaction.py", f"{RL_DIR}/arc_interaction.py", copy=True)
    .add_local_file("rl/arc_data.py", f"{RL_DIR}/arc_data.py", copy=True)
    .add_local_file("rl/arc_reward.py", f"{RL_DIR}/arc_reward.py", copy=True)
    # Copy ARC data
    .add_local_dir("downloads/ARC-AGI-2/data", "/app/downloads/ARC-AGI-2/data", copy=True)
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

    # No path overrides needed — TS files are at /app/ matching js_sandbox.py layout

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


@app.local_entrypoint()
def main():
    """Run full pipeline: prep data then train."""
    prep_data.remote()
    train.remote()
