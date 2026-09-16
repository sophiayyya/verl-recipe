# Dynamo rollout backend for verl

Run verl training with **vLLM or SGLang** behind Dynamo's **KV-aware router**.
The recipe supports shared or separate training/rollout GPUs, weight updates,
and engine sleep/wake. All training launchers use `verl.trainer.main_ppo`.

[Architecture](#architecture) · [Quick start](#quick-start) ·
[Configuration](#configuration) · [Results](#results) · [Key files](#key-files)

## Architecture

```mermaid
flowchart LR
    V("verl<br/>PPO trainer + agent loops")
    F("Shared Dynamo frontend<br/>KV-aware router")
    subgraph P["Rollout worker pool"]
        direction TB
        W1("Node 1<br/>vLLM or SGLang workers")
        WN("Node N<br/>vLLM or SGLang workers")
    end
    V -->|Generation requests| F
    F --> W1
    F --> WN

    classDef trainer fill:#eff6ff,stroke:#2563eb,color:#1e3a8a,stroke-width:1.5px
    classDef router fill:#ecfdf5,stroke:#059669,color:#064e3b,stroke-width:2px
    classDef worker fill:#f8fafc,stroke:#64748b,color:#0f172a,stroke-width:1.5px
    class V trainer
    class F router
    class W1,WN worker
    style P fill:transparent,stroke:#94a3b8,stroke-dasharray:5 5
```

**One frontend per rollout pool, shared across nodes and replicas.** Ray manages
one server actor per node; only the pool's first node starts the frontend,
etcd and NATS. Workers publish KV-cache events for routing. Weight updates use
node-local CUDA IPC; separate pools receive weights through verl's checkpoint
engine before applying them locally.

## Quick start

### Required versions

Use separate environments for the two inference engines. The tested versions are:

| Component | Version |
| --- | --- |
| Python | 3.12 |
| verl | [6cbca9ce](REQUIRED_VERL.txt) |
| Dynamo | [8c5a737](https://github.com/ai-dynamo/dynamo/commit/8c5a73723109058f96c15fff3fc912231d65ad6e), after [PR #13951](https://github.com/ai-dynamo/dynamo/pull/13951) |
| Inference engine | vLLM 0.28.0 **or** SGLang 0.5.19 |
| V1 trainer | TransferQueue 0.1.9; CuPy matching CUDA for the separate-pool NCCL backend |

### Install Dynamo

The DFW runs used existing verl `.sqsh` images, a separate `uv` environment per
engine, and **two local Dynamo wheels built from `8c5a737`**. Python components
were loaded from the same source checkout through `PYTHONPATH`.

Inside the prepared experiment container, the Dynamo installation was:

```bash
BENCH_ENGINE=sglang  # or vllm
DYNAMO_PYTHON="/experiment/engines/$BENCH_ENGINE/venv/bin/python"
uv pip install --python "$DYNAMO_PYTHON" \
    /experiment/runtime/wheels/ai_dynamo_runtime-1.5.0-cp310-abi3-manylinux_2_39_x86_64.whl \
    /experiment/runtime/wheels/ai_dynamo-1.5.0-py3-none-any.whl \
    'aisimulate==0.12.0.dev2' 'protobuf>=6.33.5,<7'
export PATH="/experiment/engines/$BENCH_ENGINE/venv/bin:/experiment/bin:$PATH"
export PYTHONPATH="/dynamo/components/src:/experiment/src/verl:/experiment/setup"
```

The runtime wheel supplies `dynamo._core`; the mounted checkout supplies the
frontend and engine Python code. See the [installation guide](INSTALL.md) for
the actual wheel-build commands, image/mount layout, environment preparation
and verification. `/experiment` and `/dynamo` above are paths inside the job's
container; the staged experiment files and wheels must already be present.

Run training from the verl checkout with this repository at `recipe/`. Keep
`etcd` and `nats-server` on `PATH`; for SGLang, unset `PYTORCH_CUDA_ALLOC_CONF`
and `PYTORCH_ALLOC_CONF`.

### Launch

Choose a preset and supply your model, datasets, and resource overrides:

```bash
export VERL_USE_EXTERNAL_MODULES=recipe.dynamo.register
python3 -m verl.trainer.main_ppo \
    --config-path ../../recipe/dynamo/config \
    --config-name dynamo_trainer_v1_colocate_sglang \
    actor_rollout_ref.model.path=/path/to/model \
    data.train_files=/path/to/train.parquet \
    data.val_files=/path/to/val.parquet \
    trainer.n_gpus_per_node=1 trainer.nnodes=1
```

| Preset | Trainer / placement |
| --- | --- |
| [dynamo_trainer_v1_colocate](config/dynamo_trainer_v1_colocate.yaml) | V1, vLLM, shared GPUs |
| [dynamo_trainer_v1_colocate_sglang](config/dynamo_trainer_v1_colocate_sglang.yaml) | V1, SGLang, shared GPUs |
| [dynamo_trainer_v1_separate](config/dynamo_trainer_v1_separate.yaml) | V1, separate rollout pool; vLLM by default |
| [dynamo_trainer](config/dynamo_trainer.yaml) | Legacy V0, shared GPUs |

For a small training smoke, set `MODEL_PATH`, `TRAIN_FILE` and `TEST_FILE`, then
choose one launcher. Colocate defaults to 1 GPU; separate needs at least 2:

| Placement | vLLM | SGLang |
| --- | --- | --- |
| Shared GPUs | [Colocate smoke](smoke_dynamo_v1_colocate.sh) | [Colocate smoke](smoke_dynamo_v1_colocate_sglang.sh) |
| Separate pools | [Separate smoke](smoke_dynamo_v1_separate.sh) | [Separate smoke](smoke_dynamo_v1_separate_sglang.sh) |

```bash
TOTAL_STEPS=2 bash recipe/dynamo/smoke_dynamo_v1_colocate_sglang.sh
```

The SGLang launchers include its required engine and sleep-mode settings.
The separate launchers also set the rollout pool size and batch constraints.
For 30B examples, see the [vLLM launcher](train_30b_rl_dynamo_kv_metrics.sh) and
[SGLang Slurm launcher](train_qwen3_30b_sglang.sh); adapt paths and cluster resources.

## Configuration

Dynamo options live under `actor_rollout_ref.rollout.engine_kwargs.dynamo`.
The shared defaults are in [dynamo_base.yaml](config/dynamo_base.yaml).

| Option | Purpose |
| --- | --- |
| `engine` | `vllm` or `sglang`; keep `rollout.name=dynamo` for both |
| `router_mode` | `kv`, `round-robin`, `random`, or `least-loaded` |
| `enable_kv_events` | Keep `true` for KV-aware routing |
| `thunderagent.enabled` | `false` for the built-in KV router; V1 presets set this explicitly |
| `request_completion_token_ids` | Return exact generated token IDs for RL |
| `enable_worker_system_metrics` | Required for SGLang's native control routes |
| `free_engine_on_train` | Must match `rollout.free_cache_engine` |

- **V1:** keep `agent.agent_loop_manager_class=null` to use TransferQueue.
  The legacy `DynamoAgentLoopManager` only supports V0.
- **Rewards:** use `reward.custom_reward_function.*` with either trainer.
- **Multiple nodes:** presets forward the registry through Ray's runtime environment;
  scripts also forward extra modules from the driver. Set paths and engine
  environment variables on every node before starting Ray.
- **Optional extensions:** [UniAgent variants](run_uniagent_variant.sh) cover
  ThunderAgent and native baselines; [NIXL smoke](run_nixl_smoke.sh) and
  [bandwidth probe](nixl_bench.py) cover checkpoint-engine weight transfer.

### Enable ThunderAgent

Set `thunderagent.enabled=true` on a Dynamo launcher. For a vLLM V1 smoke:

```bash
bash recipe/dynamo/smoke_dynamo_v1_colocate.sh \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.thunderagent.enabled=true
```

For the UniAgent workload used in the results below:

```bash
VARIANT=ta RAY_DATA_HOME=/path/to/verl-data bash recipe/dynamo/run_uniagent_variant.sh
```

Use `VARIANT=dynamo` for the built-in KV-router baseline or `VARIANT=global`
for native verl vLLM. V1 supports ThunderAgent in `colocate_async` and
`separate_async`. ThunderAgent GPU validation covers vLLM; SGLang routing
validation is still pending.

## Results

### Qwen3-30B-A3B-Base · ReTool · 8×H100

V0 GRPO on one 8×H100 80GB node per arm, with two TP4 replicas. Dynamo uses
KV-aware routing, KV events enabled, and ThunderAgent disabled. This comparison
uses **steps 1–30** of each independent 50-step run.

Lower is better. These are all 30 raw logged steps, including step 1.

| Arm | Mean timing_per_token_ms/gen | Mean vs. native |
| --- | ---: | ---: |
| Dynamo + vLLM | 0.23511 | −5.63% |
| Native vLLM | 0.24915 | baseline |
| Dynamo + SGLang | 0.26091 | −3.66% |
| Native SGLang | 0.27082 | baseline |

Means are in ms/token, with equal weight per step and no W&B smoothing.
This metric includes tool-return context tokens. These runs used additional
verl/data/tool preparations; evolving policies and sparse tool use limit
conclusions about KV routing alone. See the [benchmark report](benchmarks/retool_h100_20260916.md)
for the setup, W&B links, metric definitions and full 50-step results.

### ThunderAgent · UniAgent · 8×H20-3e

Qwen3-Coder-30B-A3B-Instruct on 8×H20-3e (140.4 GiB/GPU), with two TP4 replicas
and colocated, synchronous GRPO. The comparison is Dynamo ThunderAgent versus
verl's native Global LB on vLLM.

| Concurrency | Rollout-phase speedup | Full-step speedup |
| ---: | ---: | ---: |
| 384 | 1.94× | 1.39× |
| 512 | 2.40× | 1.60× |

At concurrency 64–256, performance was near parity. Rollout throughput counts
model-generated tokens per rollout second; full-step measurements also include
reward/advantage computation, log-probability evaluation, actor updates and
weight synchronization.

### Entry-point validation

The subsequent `main_ppo` entry-point cleanup passed **141 CPU tests per engine**
and **6 native CLI configuration checks**. GPU training was not repeated for that cleanup.

## Key files

| File | Role |
| --- | --- |
| [register.py](register.py) | Register the Dynamo backend with verl |
| [config/](config/) | V0/V1 presets and shared defaults |
| [dynamo_async_server.py](dynamo_async_server.py) | Shared frontend, worker pool and lifecycle |
| [dynamo_rollout.py](dynamo_rollout.py) | Select the vLLM or SGLang adapter |
| [dynamo_agent_loop.py](dynamo_agent_loop.py) | Agent-loop clients and async integration |
