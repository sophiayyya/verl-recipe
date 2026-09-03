# Dynamo rollout backend for verl

This recipe plugs [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) into verl
as a first-class **async rollout backend**, alongside the built-in `vllm`,
`sglang` backends. Turning it on is a one-line config change
(`actor_rollout_ref.rollout.name=dynamo`); everything Dynamo-specific
is driven from `rollout.engine_kwargs.dynamo.*`. The backend can front **either inference engine**: `dynamo.vllm` (the default)
or `dynamo.sglang` (`engine_kwargs.dynamo.engine=sglang`). 

Dynamo owns request routing behind a single logical frontend, so its
**KV-cache-aware router** can raise the prefix-cache hit rate across a rollout
step. Weight updates still flow through verl's colocated CUDA-IPC path, so the
trainer and the engine workers share GPUs the same way the native backends do.

**Contents** —
[How it works](#how-it-works) ·
[Configuration](#configuration) ·
[Quick start](#quick-start) ·
[NIXL weight sync](#nixl-weight-sync-checkpoint-engine) ·
[KV-aware routing result](#kv-aware-routing-result) ·
[Agent-loop RL](#running-a-full-agent-loop-rl-run) ·
[ThunderAgent](#thunderagent-extension) ·
[SGLang engine](#sglang-engine-engine_kwargsdynamoenginesglang) ·
[Sibling-repo changes](#changes-required-in-sibling-repos)

## How it works

The Dynamo backend keeps verl's AgentLoop execution model but replaces the
rollout server with a Dynamo deployment. A single Ray actor per node
(`DynamoHttpServer`) supervises the whole Dynamo stack as subprocesses; it
reserves **no** GPUs of its own — the colocated trainer workers already own
them, and the actor only forwards `CUDA_VISIBLE_DEVICES` into the engine
shards.

```
 verl trainer (colocated, owns GPUs)
        │  HTTP chat/completions            control RPC (sleep / wake /
        │  (per-rank ServerAdapter)         update_weights) via Ray
        ▼                                             │
 ┌─────────────────── DynamoHttpServer (Ray actor, 1 / node) ────────────────────┐
 │  supervises + watchdogs subprocesses, forwards CUDA_VISIBLE_DEVICES           │
 │                                                                               │
 │   dynamo.frontend ──► KV-aware router ──► engine workers × N (one / DP        │
 │        ▲                                        │     shard; dynamo.vllm      │
 │        │                       CUDA-IPC weight sync    or dynamo.sglang)      │
 │        │                       (ZMQ receiver for vLLM;                        │
 │        │                        native control route for sglang)              │
 │   etcd + nats-server (service discovery / messaging)                          │
 │                                                                               │
 │   optional: per-worker metrics sidecar                                        │
 └───────────────────────────────────────────────────────────────────────────────┘
```

Request routing happens inside Dynamo's KV router, **not** in verl's
`GlobalRequestLoadBalancer` — verl only ever talks to the one shared frontend.

### Key files

| File | Role |
| --- | --- |
| [`register.py`](register.py) | Registers `dynamo` in verl's rollout registries; loaded via `VERL_USE_EXTERNAL_MODULES=recipe.dynamo.register`. |
| [`main_dynamo.py`](main_dynamo.py) | Compatibility shim only — the launchers now call `verl.trainer.main_ppo` directly; see the entry-point note under [Quick start](#quick-start). |
| [`config/dynamo_trainer.yaml`](config/dynamo_trainer.yaml) | Hydra config: inherits `ppo_trainer`, sets `rollout.name=dynamo`, `rollout.mode=async`. |
| [`dynamo_async_server.py`](dynamo_async_server.py) | `DynamoReplica` / `DynamoHttpServer` — spawns and watchdogs etcd, nats-server, engine workers, and `dynamo.frontend`, for both engines. |
| [`dynamo_rollout.py`](dynamo_rollout.py) | `ServerAdapter` — engine-agnostic facade; dispatches on `engine_kwargs.dynamo.engine` and lazily imports the chosen adapter (no module-scope engine imports), so it loads on an image that ships only one engine. |
| [`dynamo_vllm_rollout.py`](dynamo_vllm_rollout.py) | `VllmDynamoServerAdapter` — per-rank client for the vLLM engine; HTTP generation via the frontend, control RPCs (sleep/wake/`update_weights`) to the shared per-node actor. |
| [`dynamo_sglang_rollout.py`](dynamo_sglang_rollout.py) | `SGLangServerAdapter` — per-rank client for the sglang engine (shard-local TP group, CUDA-IPC weight sync via `update_weights_from_tensor`). |
| [`dynamo_sglang_engine.py`](dynamo_sglang_engine.py) | HTTP client for `dynamo.sglang`'s native `/engine/control/*` RL routes. |
| [`dynamo_naming.py`](dynamo_naming.py) | `control_actor_name()` — the one place the `dynamo_server_{replica}_{node}` actor-name contract is spelled out. |
| [`dynamo_agent_loop.py`](dynamo_agent_loop.py) | `DynamoServerManager` / `DynamoLLMServerManager` — talk to the single shared frontend instead of load-balancing across replicas. |
| [`dynamo_worker_extension.py`](dynamo_worker_extension.py) | vLLM `worker_extension_cls` that maps each DP shard to a node-global rank so trainer and engine agree on the CUDA-IPC socket path. |
| [`_dynamo_vllm_with_control.py`](_dynamo_vllm_with_control.py) | Private ZMQ control sidecar that bridges verl's `collective_rpc` into the `dynamo.vllm` subprocess (vLLM only; sglang has native control routes). |
| [`metrics_sidecar.py`](metrics_sidecar.py) | Optional per-worker system-status / metrics scraper. |



Enable the backend by pointing verl at the recipe's registration module:

```bash
export VERL_USE_EXTERNAL_MODULES=recipe.dynamo.register
```

## Configuration

Everything Dynamo-specific lives under
`actor_rollout_ref.rollout.engine_kwargs.dynamo`. All keys are optional; sane
defaults are applied in `DynamoHttpServer`.

| Key | Values / example | Purpose |
| --- | --- | --- |
| `engine` | `vllm` (default), `sglang` | Which inference engine the workers run. |
| `router_mode` | `kv` (default), `round-robin`, `random`, `least-loaded` | Dynamo request-routing policy; `kv` enables KV-cache-aware routing. |
| `frontend_http_port` / `etcd_port` / `nats_port` | `0` = auto-assign | Fixed ports if you need them. |
| `served_model_name` | falls back to `model_config.local_path` | Model name the frontend advertises. |
| `request_engine_data` / `request_completion_token_ids` | `true` / `false` | Ask the frontend to return `nvext.engine_data` (vLLM only) / raw `completion_token_ids` (token-in/token-out for RL). **Required for RL.** The sglang engine refuses to start when `request_completion_token_ids` is left unset (an explicit `false` is honored). If it is `true` and the frontend still returns no token ids, generation raises rather than silently re-tokenizing the text; when it is off, the text re-encode fallback is logged at ERROR (first 3 hits + every 100th). |
| `return_tokens_as_token_ids` | `true` / `false` | Emit token ids instead of detokenized text. |
| `request_timeout_s` | `600` (default; scripts use `1800`) | Per-request timeout. |
| `free_engine_on_train` | `true` (set by `config/dynamo_trainer.yaml`; code default `false`) | Free the engine (sleep) during the training phase. Keep it aligned with verl's `rollout.free_cache_engine` (default `true`); the sglang engine additionally requires `rollout.enable_sleep_mode=true` when this is on (that flag adds `--enable-memory-saver`) and refuses to start otherwise. This is what makes CUDA graph affordable on tight-memory footprints. |
| `enable_worker_system_metrics` | `true` / `false` | Expose the per-worker system-status / metrics port (paired with `metrics_sidecar.py`). Must stay `true` for sglang — that port carries its control plane. |
| `extra_args` | `["--generation-config","vllm","--stream-interval=100"]` | Extra CLI args forwarded verbatim to the engine worker. |

## Quick start

All examples use verl's standard entry point. Three things make it work with this
recipe on the pinned checkout (`6cbca9ce`):

- `--config-path ../../recipe/dynamo/config --config-name dynamo_trainer` loads the
  recipe's Hydra config. Hydra resolves the relative path against `verl/trainer/`
  (the module declaring `@hydra.main`), so it is CWD-independent as long as this
  repository sits at `recipe/` inside the verl checkout.
- **`trainer.use_v1=False` is required.** The V1 trainer on this checkout hands a
  `TensorDict` to the async agent loop and dies with an `AttributeError`; the flag
  pins the v0 `TaskRunner`.
- Custom rewards must use the canonical `reward.custom_reward_function.*` namespace,
  not the legacy top-level `custom_reward_function.*`: the v0 runner reads
  `config.reward.*` and `main_ppo` never runs the legacy-key migration, so legacy
  keys are **silently ignored** (the run proceeds with the default reward).

[`main_dynamo.py`](main_dynamo.py) is the convenience entry point: it pins the v0
`TaskRunner` and runs verl's legacy-reward-key migration itself, so neither
`trainer.use_v1=False` nor the `reward.custom_reward_function.*` namespace is
required when you launch through it. The launchers call `verl.trainer.main_ppo`
directly and therefore set both explicitly.

### 1. Generation-only smoke

Verifies the full Dynamo stack (etcd + nats + workers + frontend) can serve a
completion, no training loop. Passes when the log prints `PASS:`.

```bash
bash recipe/dynamo/smoke_vllm_generate.sh          # Qwen2.5-0.5B-Instruct, 1 GPU
```

### 2. One-node training smoke

```bash
export VERL_USE_EXTERNAL_MODULES=recipe.dynamo.register
python3 -m verl.trainer.main_ppo \
    --config-path ../../recipe/dynamo/config --config-name dynamo_trainer \
    trainer.use_v1=False \
    algorithm.adv_estimator=grpo \
    data.train_files=.../gsm8k/train.parquet \
    data.val_files=.../gsm8k/test.parquet \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.rollout.name=dynamo \
    actor_rollout_ref.rollout.mode=async \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.router_mode=kv \
    trainer.n_gpus_per_node=2 trainer.nnodes=1 \
    trainer.total_training_steps=2
```

### 3. Multi-node 30B RL

Proven `Qwen3-30B-A3B-Base` launchers (SLURM, 4 × 8 H100 unless noted). vLLM and
sglang **cannot share one job**: installing the `ai_dynamo[sglang]` extra
downgrades vLLM's guided-decoding stack (`llguidance`, `outlines_core`, …), and
the breakage only surfaces at runtime — so each engine gets its own job with a
conditional install, never a baked image.

| Script | Engine | What it runs |
| --- | --- | --- |
| [`train_30b_rl_dynamo_kv_metrics.sh`](train_30b_rl_dynamo_kv_metrics.sh) | vLLM | KV router + metrics sidecar RL run (inner command, `NNODES` default 2; driven by the sbatch below). |
| [`train_qwen3_30b_sglang.sh`](train_qwen3_30b_sglang.sh) | sglang | The verified 100-step retool GRPO run. Defaults reproduce it (`ENFORCE_EAGER=False`, `DISABLE_PIECEWISE=0`, deferred optimizer load / fused kernels / eager experts all off); every env knob is listed in the script header, e.g. `sbatch --export=ALL,TOTAL_STEPS=3 …`. |
| [`baseline_qwen3_30b_sglang_native.sh`](baseline_qwen3_30b_sglang_native.sh) | — | **Baseline**: verl's native `rollout.name=sglang`, no Dynamo, for A/B comparison. |

## NIXL weight sync (checkpoint engine)

By default the trainer pushes weights to the Dynamo workers through verl's
naive CUDA-IPC path. Setting

```bash
actor_rollout_ref.rollout.checkpoint_engine.backend=nixl \
actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024
```

routes refit through verl's `CheckpointEngineManager` instead: the recipe
spawns one `CheckpointEngineWorker` Ray actor per rollout rank (colocated on
the paired GPU — CUDA IPC requires same-GPU pairing) and NIXL moves the
buckets down a trainer → CE₁ → … → CEₙ chain, which crosses nodes at most
twice regardless of world size.

### Support matrix

| backend | single-node | multi-node |
| --- | --- | --- |
| naive | ✅ | ✅ |
| NIXL | ✅ | ✅ (2×8 GPU validated) |

### Transport selection (read this before multi-node)

Cross-node measured on 2 nodes × 8×H100 (RDMA fabric), 3-step GRPO,
step-3 `update_weights`:

| model | naive | NIXL UCX (tcp / one-sided read emulated over send/recv) | NIXL LIBFABRIC |
| --- | --- | --- | --- |
| Qwen2.5-0.5B | 3.08 s | 12.2 s / 11.6 s | 3.09 s |
| Qwen3-8B | 29.7 s | — | **27.1 s** |

At 0.5B the LIBFABRIC path is at parity with naive (the shared per-rank
engine-consume dominates); at 8B it is ~9% faster. The UCX column shows
the send/recv emulation ceiling — protocol-level, not tunable.

Two helpers ship with the recipe: [`run_nixl_smoke.sh`](run_nixl_smoke.sh) (a
3-step GRPO training smoke parameterised over `NNODES` / `CE_BACKEND`) and
[`nixl_bench.py`](nixl_bench.py) (a standalone cross-node bandwidth probe for
checking what a fabric actually delivers before debugging the training path).
When running in containers/Kubernetes, give worker pods the fabric's
RDMA device resource (e.g. `rdma/ib`) and the `IPC_LOCK` capability —
NIXL needs it to pin memory for RDMA registration, and transfers hang
without it.

> `engine_kwargs.nixl.backends` requires a small verl-side change (a
> `backends` kwarg on `NIXLCheckpointEngine`, pending as a separate verl
> PR); on IB/RoCE fabrics the stock UCX backend needs no verl change.

## KV-aware routing result

The matched comparison below keeps only Dynamo KV routing with
`stream-interval=100` and the native vLLM baseline. Lower `ms/token` is better;
the similar response lengths are a sanity check that generation behavior stayed
comparable.

| Backend | ms/token | Mean response length | KV-cache hits / queries | KV-cache hit rate |
| --- | --- | --- | --- | --- |
| Dynamo KV (`stream-interval=100`) | 1.5956 | 876.1 | 2,248,368 / 2,520,362 | 89.21% |
| vLLM baseline | 1.7220 | 872.3 | 1,860,816 / 2,432,064 | 76.51% |

Dynamo KV shows approximately **7.3% lower per-token latency** (≈7.9% faster)
in this comparison and improves the KV-cache hit rate by **12.70 percentage points**.

## Running a full agent-loop RL run

The quick-start examples above are single-turn smoke tests. A real RL run drives
a **multi-turn agent loop** (tool calls, an external environment, a custom
reward) through the Dynamo frontend. The pieces below are what that adds on top
of the smoke test; everything is parameterised, so drop in your own model, data,
and agent loop.

### Trainer: currently no fully_async

Dynamo has no `fully_async` trainer. Run it through `verl.trainer.main_ppo` with
`trainer.use_v1=False` (see [Quick start](#quick-start)) and
`actor_rollout_ref.hybrid_engine=True`.
`DynamoLLMServerManager` is colocated — it forwards the trainer's
`CUDA_VISIBLE_DEVICES` into the engine shards — so `hybrid_engine=True` is
required, not optional.

### Wire in your agent loop

These overrides turn a plain GRPO run into a multi-turn agent-loop run served by
Dynamo. The config path and loop name are **yours** — this recipe does not ship
an agent loop:

```bash
actor_rollout_ref.rollout.mode=async \
actor_rollout_ref.rollout.multi_turn.enable=True \
actor_rollout_ref.rollout.agent.num_workers=64 \
actor_rollout_ref.rollout.agent.agent_loop_config_path=/path/to/agent_config.yaml \
actor_rollout_ref.rollout.agent.default_agent_loop=<your_loop_name> \
++actor_rollout_ref.rollout.agent.agent_loop_manager_class=recipe.dynamo.dynamo_agent_loop.DynamoAgentLoopManager
```

The `agent_loop_manager_class` override is the key one (`++`, not `+`: `config/dynamo_trainer.yaml` already sets it, and Hydra rejects a bare `+` on an existing key): it swaps verl's default
manager for `DynamoAgentLoopManager`, which talks to the single shared Dynamo
frontend instead of load-balancing across replicas.

### Recommended `engine_kwargs.dynamo` for RL

Token-in/token-out generation (so the trainer scores the exact tokens the engine
produced), KV-aware routing, and freeing engine memory during the training
phase. See the [Configuration](#configuration) table for every key. The
`extra_args` shown are **vLLM-only** — drop them for the sglang engine (the
verified sglang run used none).

```bash
++actor_rollout_ref.rollout.engine_kwargs.dynamo.router_mode=kv \
++actor_rollout_ref.rollout.engine_kwargs.dynamo.request_engine_data=true \
++actor_rollout_ref.rollout.engine_kwargs.dynamo.request_completion_token_ids=true \
++actor_rollout_ref.rollout.engine_kwargs.dynamo.return_tokens_as_token_ids=false \
++actor_rollout_ref.rollout.engine_kwargs.dynamo.request_timeout_s=1800 \
++actor_rollout_ref.rollout.engine_kwargs.dynamo.free_engine_on_train=true \
++actor_rollout_ref.rollout.engine_kwargs.dynamo.enable_worker_system_metrics=true \
'++actor_rollout_ref.rollout.engine_kwargs.dynamo.extra_args=["--generation-config","vllm","--stream-interval=100"]'
```

### Optional: KV-metrics sidecar

With `enable_worker_system_metrics=true`, each worker writes an
`.endpoints` file under `$VERL_DYNAMO_WORKER_METRICS_DIR`. Run the sidecar
alongside training to scrape those `/metrics` endpoints into JSONL (KV-cache hit
rate, queue depth, …):

```bash
python3 recipe/dynamo/metrics_sidecar.py \
    --endpoints-glob "$VERL_DYNAMO_WORKER_METRICS_DIR/*.endpoints" \
    --output /path/to/logs/kv_metrics.jsonl \
    --label dynamo_kv --interval 30 &
```

## ThunderAgent extension

This recipe extends the Dynamo rollout backend with **ThunderAgent**
(from [verl-recipe PR #110](https://github.com/verl-project/verl-recipe/pull/110)):
program-aware scheduling for multi-turn agent trajectories. It does not
modify core `verl` or Dynamo.

### Required versions

- Dynamo: source commit `59d614641837e593f0567b79d75394aae5f864e0`, including
  [PR #11185](https://github.com/ai-dynamo/dynamo/pull/11185). This recipe is
  currently validated against `94accc7389` (the #11185 merge commit) — see
  [Sibling-repo patches](#patches-for-repos-outside-this-recipe). When in doubt,
  check out `94accc7389`: it satisfies this section and is the base commit the
  patches apply to.

### Topology

With ThunderAgent enabled, verl launches processes in this order:

```text
etcd -> NATS -> Dynamo vLLM workers -> ThunderAgent router -> frontend
```

The frontend uses round-robin only to reach the registered ThunderAgent
handler. ThunderAgent owns the internal KV router and forwards to the worker
endpoint `<namespace>.backend.generate`. Shutdown reverses the consumer side:
frontend, ThunderAgent, workers, NATS, then etcd.

### Configuration

The recipe's default config ([`config/dynamo_trainer.yaml`](config/dynamo_trainer.yaml))
enables ThunderAgent — vLLM path only; it is not enabled for the sglang engine
(see [Not wired yet](#not-wired-yet)):

```yaml
actor_rollout_ref:
  rollout:
    agent:
      agent_loop_manager_class: recipe.dynamo.dynamo_agent_loop.DynamoAgentLoopManager
    engine_kwargs:
      dynamo:
        thunderagent:
          enabled: true
          router_block_size: 16
```

`router_block_size` is applied to both vLLM and ThunderAgent. Pass scheduler
CLI options without hard-coding them in the recipe:

```yaml
thunderagent:
  enabled: true
  router_block_size: 16
  extra_args:
    - --pause-threshold
    - "0.95"
```

Optional finalization controls are `finalize_max_attempts` (default `3`) and
`finalize_retry_delay_s` (default `0.1`). Set `thunderagent.enabled=false` for
the PR #110 KV-router baseline.

### Run

From a verl checkout containing this repository at `recipe/`:

```bash
VERL_USE_EXTERNAL_MODULES=recipe.dynamo.register \
python -m verl.trainer.main_ppo \
  --config-path ../../recipe/dynamo/config --config-name dynamo_trainer \
  trainer.use_v1=False \
  actor_rollout_ref.model.path=/path/to/model \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
```

Supply the remaining dataset, trainer, and resource overrides required by the
standard verl PPO configuration.

### UniAgent variants

[`run_uniagent_variant.sh`](run_uniagent_variant.sh) is a concise UniAgent
training example. Select one rollout path with `VARIANT`:

- `ta` (default): Dynamo with ThunderAgent enabled.
- `dynamo`: the native Dynamo KV-router baseline with ThunderAgent disabled.
- `global`: the native verl vLLM rollout baseline, bypassing Dynamo. The
  historical name does not mean that this script explicitly configures a
  GlobalLoadBalancer.

Run it from the verl root:

```bash
VARIANT=ta     RAY_DATA_HOME=/path/to/verl-data bash recipe/dynamo/run_uniagent_variant.sh
VARIANT=dynamo RAY_DATA_HOME=/path/to/verl-data bash recipe/dynamo/run_uniagent_variant.sh
VARIANT=global RAY_DATA_HOME=/path/to/verl-data bash recipe/dynamo/run_uniagent_variant.sh
```

### End-to-end ThunderAgent result

The ThunderAgent comparison used the following matched setup:

- Uni-Agent × verl synchronous end-to-end GRPO, including rollout,
  reward/advantage, old log-probability, actor update, and weight
  synchronization.
- Training and inference colocated and time-multiplexed on 8 × NVIDIA H20-3e
  GPUs (140.4 GiB/GPU), with two TP4 replicas.
- Qwen3-Coder-30B-A3B-Instruct.
- The only backend change was verl Global LB versus Dynamo ThunderAgent.

Here, `rollout.mode=async` only enables concurrent agent requests within the
rollout phase; the RL algorithm remains synchronous and does not use a
`staleness_threshold`. Rollout throughput is generated response tokens divided
by rollout wall time. Full-step throughput also includes reward/advantage,
old-log-probability, actor-update, and weight-synchronization time.

<img src="assets/thunderagent_full_step_throughput.png" alt="ThunderAgent and Global LB complete synchronous RL-step throughput" width="800">

<img src="assets/thunderagent_rollout_throughput.png" alt="ThunderAgent and Global LB rollout-phase generated-token throughput" width="800">

At concurrency 64–256, ThunderAgent and Global LB are near parity. At
concurrency 384, ThunderAgent reaches **1.94× rollout-phase speedup** and
**1.39× observed full-step speedup**; at concurrency 512, the speedups reach
**2.40×** and **1.60×**, respectively.

<img src="assets/thunderagent_speedup.png" alt="ThunderAgent speedup over Global LB" width="800">

## SGLang engine (`engine_kwargs.dynamo.engine=sglang`)

The Dynamo backend can front **either** `dynamo.vllm` (default, everything
above) or `dynamo.sglang`. Engine selection is a single switch —
`rollout.name=dynamo` stays fixed for both engines, and
`recipe.dynamo.dynamo_rollout.ServerAdapter` dispatches on
`engine_kwargs.dynamo.engine` (an earlier revision used a separate
`rollout.name=dynamo_sglang`; that name is no longer registered):

```bash
actor_rollout_ref.rollout.name=dynamo \
++actor_rollout_ref.rollout.engine_kwargs.dynamo.engine=sglang
```

### What is different from the vLLM path

| | vLLM | SGLang |
| --- | --- | --- |
| Worker process | `python -m recipe.dynamo._dynamo_vllm_with_control` (verl wrapper) | stock `python -m dynamo.sglang` |
| Control plane | verl-private ZMQ REP sidecar → `engine.collective_rpc` | **native** `/engine/control/*` on `DYN_SYSTEM_PORT` |
| Weight sync | `BucketedWeightSender` → ZMQ-IPC socket → `update_weights_from_ipc` | `MultiprocessingSerializer` CUDA-IPC handles → `control/update_weights_from_tensor` |
| Sleep / wake | `engine.sleep(level=…)` | `release_memory_occupation(tags=…)` / `resume_memory_occupation` |
| KV events | `--kv-events-config <json>` | none needed — `DynamoSglangPublisher` forwards them |
| Cache flush | `reset_prefix_cache` | `call_tokenizer_manager("flush_cache")` (**needs `--enable-rl`**) |

`dynamo.sglang` registers its RL control routes itself
(`request_handlers/handler_base.py::register_engine_routes`), which is why this
path ships no sidecar. The trade is that `DYN_SYSTEM_PORT` stops being an
optional metrics extra and becomes the whole control plane —
`enable_worker_system_metrics=false` is rejected outright for this engine.

### Prerequisites

1. **A container with `dynamo.sglang`.** Two validated routes:
   - the smoke scripts install the `[sglang]` extra from the **local** 1.3.0
     wheelhouse into the vLLM image (`ai_dynamo==1.3.0` is not on PyPI, only
     `.post1`/`1.3.1`; the extra pulls `sglang==0.5.14`). Per-job only, never baked in:
     it downgrades vLLM's guided-decoding deps (see [Quick start §3](#3-multi-node-30b-rl));
   - the 30B launchers use an image that already ships sglang (`verl_sgl0512.dev4`:
     sglang 0.5.12 / transformers 5.3.0) and install the dynamo wheels `--no-deps`.
2. **The log-prob fidelity fix** —
   [required for trustworthy rollout log-probs](#engine-log-prob-fidelity-incremental-vs-cumulative-fix-required-fix):
   a dynamo build that includes upstream #11640, or its log-prob portion
   backported onto `94accc7389` (three files, described in that section).
3. **No patching needed for the weight-sync payload.** The CUDA-IPC blobs go on
   the wire base64-encoded, which is SGLang's own contract:
   `serialized_named_tensors` is typed `List[Union[str, bytes]]` and
   `MultiprocessingSerializer.deserialize` b64-decodes any `str`. Verified
   end-to-end against an unmodified dynamo 1.3.0 + sglang 0.5.14 on an H100.
   Note a malformed payload does **not** return an error — it
   kills the worker process.
4. **`--enable-memory-saver`** is added automatically when
   `rollout.enable_sleep_mode=true`; without it SGLang's torch_memory_saver
   never arms and memory release silently frees nothing (the server refuses
   `free_cache_engine=true` + `enable_sleep_mode=false` for this reason). Also
   **unset `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`** in the job
   environment: torch_memory_saver refuses to initialise under it, and the
   engine then sits on ~31 GB/GPU of weights through every training step.
5. **transformers 5.x memory.** sglang 0.5.x pins transformers 5.x, which stores
   MoE experts as one fused tensor and roughly **doubles** the actor-update peak
   versus 4.x (35.6 → 71.6 GB on Qwen3-30B-A3B). On 4×8 H100 it fits as is; on
   2×8 use `VERL_DEFER_OPTIMIZER_LOAD=1` (patch below). Do not reach for
   `_experts_implementation=eager` — it costs a 39× slower update.

### Run

```bash
# generation-only smoke (M1)
bash recipe/dynamo/smoke_dynamo_sglang.sh

# 2-step GRPO incl. weight sync + sleep/wake (M2)
STAGE=train bash recipe/dynamo/smoke_dynamo_sglang.sh

# the verified 4-node 30B run (100 steps; defaults reproduce it)
sbatch recipe/dynamo/train_qwen3_30b_sglang.sh
# shorter: sbatch --export=ALL,TOTAL_STEPS=3 recipe/dynamo/train_qwen3_30b_sglang.sh
```

### `engine_kwargs.dynamo.sglang.*`

| Key | Default | Purpose |
| --- | --- | --- |
| `enable_rl` | `true` | Adds `--enable-rl`; registers `call_tokenizer_manager`, the only route that can flush the radix cache after a weight update. |
| `verify_weight_sync` | `false` | Best-effort probe: reads one synced parameter back via `get_weights_by_name` after each sync (snapshot refreshed every sync) and raises on mismatch. That API is model-specific and **unimplemented for Qwen2/Qwen3**, so on those models the probe logs `INCONCLUSIVE` at ERROR and verifies nothing — a passing run is not a verified one. |
| `page_size` | falls back to `thunderagent.router_block_size` | KV-router block size (`--page-size`). |
| `skip_tokenizer_init` | `false` | Token-in/token-out. |
| `extra_args` | `[]` | Forwarded verbatim; `dynamo.sglang` exposes the whole `ServerArgs` CLI. |

### Not wired yet

- **ThunderAgent** — only validated against vLLM; not enabled for the sglang engine.
- **`checkpoint_engine.backend=delta_sharded`** — verl gates it on
  `rollout.name == "sglang"` (`verl/checkpoint_engine/base.py`), which the
  `dynamo` rollout name does not satisfy.
- **LoRA adapter sync** — needs `load_lora_adapter_from_tensors` as an engine route.
- **SGLang-native streaming `/generate`** (token-in/token-out) — needs a dynamo
  build with [#11640](https://github.com/ai-dynamo/dynamo/pull/11640);
  generation currently goes through the same frontend `/v1/completions` path as
  vLLM. (Only the log-prob portion of #11640 is needed by this recipe, not the
  `/generate` endpoint itself.)

### Engine log-prob fidelity: incremental-vs-cumulative fix (required fix)

**Status: fixed and verified end-to-end (2026-09-01).** Requires the dynamo-side fix below.

**Symptom.** With the sglang engine, the outlier-sensitive rollout-vs-actor
diagnostics are destroyed, while k1 KL is silently biased but still lands in a
plausible range — which is exactly why the bug hides:

| metric | dynamo+sglang (broken) | native sglang (21-step mean) |
| --- | --- | --- |
| `rollout_actor_probs_pearson_corr` | 0.05 – 0.62 | 0.9993 |
| `rollout_corr/k3_kl` | up to 1635 | 0.0020 |
| `rollout_corr/kl` (k1) | 0.0032 – 0.0037 (looks plausible!) | 0.0020 |

A dynamo+vLLM reference run (3 steps) shows the same healthy
profile as native sglang: pearson 0.9993, k3_kl 0.0019.

In the default recipe config, training itself learns normally despite this —
gradients use trainer-recomputed log-probs, so only diagnostics were corrupted.
**Any mode that consumes engine log-probs directly
(`actor.use_rollout_log_probs=true`, fully-async training) would have trained
on the corrupted values** and must not run without the patch below.

**Root cause.** sglang streams `meta_info["output_token_logprobs"]`
**incrementally** (each chunk carries only that chunk's tokens), but dynamo's
shared extractor `common/backend/logprobs.py::extract_from_sglang_meta` sliced
it as if it were **cumulative** (`arr[num_output_logprobs_so_far:]`). From
chunk 2 onward the slice was always empty, so the chunk carried no log-probs
while token ids kept flowing through `nvext.completion_token_ids`. Measured on
one request: 12 of 6944 positions (0.17%) had real log-probs; the rest were
padded. A pad of `0.0` means "probability 1.0", which the exponential in k3
amplifies to astronomical values while the linear k1 mean barely moves — hence
the signature above.

**Fix.** Backport of the log-prob portion of upstream
[ai-dynamo/dynamo#11640](https://github.com/ai-dynamo/dynamo/pull/11640)
(the PR title does not mention log-probs — the fix ships inside the
engine-native generate endpoint work; search by file, not by subject). Either
build dynamo from a commit that already contains #11640, or backport its
log-prob portion (~20 lines) onto `94accc7389`. Three files: `common/backend/logprobs.py` (slice the chunk head instead of a
running cursor), `sglang/request_handlers/llm/decode_handler.py` and
`sglang/llm_engine.py` (pass `num_output_tokens_in_chunk=len(output_ids)`, drop
the cursor state, fix the comments that still described the old cumulative
semantics).

**Recipe-side hardening** (already on this branch, in `dynamo_async_server.py`):

- Launch-time fail-fast: starting an sglang engine without
  `engine_kwargs.dynamo.request_completion_token_ids=true` now raises at startup
  instead of silently degrading (an explicit `false` is still honored).
- `_normalize_log_probs` pads missing positions with the sequence mean instead
  of `0.0`, and reports loudly (first 3 occurrences + every 100th). Padding is a
  symptom of an engine-side bug: with the patch above the counter stayed at 0
  for the full verification run, and any nonzero count should be treated as a
  regression.
- Each sglang worker gets an explicitly allocated `--nccl-port`
  (fixes `EADDRINUSE` when several workers share a node).

**Verification** — Qwen3-30B-A3B-Base retool GRPO, 4 nodes × 8 H100,
CUDA graph on, **no** `--stream-interval` workaround, 100 steps in 2h07
(65.5 s/step), 0 padding events, 0 OOM. Against a native-sglang run
(`rollout.name=sglang`, no dynamo) over the native run's full 21 steps:

| metric (steps 1–21 mean) | dynamo+sglang, patched | native sglang |
| --- | --- | --- |
| `rollout_actor_probs_pearson_corr` | 0.99920 | 0.99928 |
| `rollout_corr/k3_kl` | 0.00197 | 0.00204 |
| `rollout_corr/kl` (k1) | 0.00201 | 0.00204 |
| `rollout_probs_diff_mean` | 0.00526 | 0.00516 |
| `critic/score/mean` | -0.760 | -0.750 |
| `actor/grad_norm` | 0.173 | 0.173 |

Over the full 100 steps (20-step segment means): the k3/k1 ratio stays at ~1
throughout (0.98 / 1.00 / 1.04 / 1.01 / 1.00) — no re-divergence of the bug;
pearson declines slowly and smoothly (0.99921 → 0.99744) as the policy moves
off its initialization, with no discontinuity; score improves −0.77 → −0.18.

Throughput context (not a like-for-like engine comparison): the patched dynamo
arm ran 100 steps in 2h07 while the native arm hit the 4h wall at step 21.
dynamo's `free_engine_on_train` releases engine memory during training; with
it, CUDA graph fit on this footprint. In this recipe's native configuration
engine memory stays resident through training — the native CUDA-graph attempt
OOM'd and the arm had to run eager.

**Diagnostic rule of thumb.** Watch the `k3_kl / k1` **ratio**, not k3's
absolute value. Both estimate the same KL, so when the two log-prob streams
agree the ratio is ~1 (here 0.98–1.04 throughout). Extreme per-token outliers —
exactly what bad `0.0` padding produces — explode the exponential k3 but not
the linear k1 (broken run: ratio ~4.7e5), and unlike k3 itself the ratio does
not grow with KL magnitude. Two caveats: k1 is a signed mean, so very early in
training (KL ≈ 0) the ratio is ill-conditioned and can spike without any bug;
and mean-padded values (the hardening above) do **not** produce outliers, so
they will not trip this alarm — the padding counter is the alarm for that
failure mode.

**Known open item.** `actor/entropy` 0.88 vs 1.05 and `response_length/mean`
941 vs 983 (dynamo vs native, steps 1–21) reproduce across independent dynamo
runs — though all of those runs share the same confound, so reproduction does
not disentangle it. The clean fidelity metrics above are consistent with a
generation-side difference (what the engine samples), which they cannot see by
construction. Candidates: CUDA graph (on for dynamo, eager for native in all
data so far) and sampling-parameter passthrough at the OpenAI frontend. Not yet
root-caused.

## Changes required in sibling repos

This branch only carries `recipe/` files; the changes below live in the
corresponding checkouts. No wheel rebuild is needed: the launch scripts prepend
`$DYNAMO_SRC/components/src` to `PYTHONPATH`, which shadows the installed
`ai_dynamo` wheel, and run verl from the checkout mounted as
`$VERL_SRC_IN_CONTAINER`.

| repo | change | status |
| --- | --- | --- |
| ai-dynamo/dynamo | log-prob portion of [#11640](https://github.com/ai-dynamo/dynamo/pull/11640) (see the fix section above) | **required** for the sglang engine; upstream from that PR onward |
| verl | `bucket_size_mb` 512 → 4096 for the vLLM CUDA-IPC weight sender | applied at runtime by `train_qwen3_30b_sglang.sh` with `sed`; unconditional, and known to break DeepSeek-V4 fp8 requantize (`CUDA driver error: invalid argument`) — drop it there |
| verl | opt-in `VERL_DEFER_OPTIMIZER_LOAD=1` (Adam state on CPU through fwd/bwd, loaded only around `optimizer.step()`; ~7 GB/GPU at the actor-update peak on 32-GPU FSDP) | needed to fit 30B on 2×8 H100 only; the verified 4×8 run did not use it. Not on this branch; available as a separate verl change |
| verl | NIXL `backends` kwarg on `NIXLCheckpointEngine` (LIBFABRIC transport) | pending as a separate verl PR — see the NIXL section |

Known-good base versions:

| component | version |
| --- | --- |
| dynamo | `94accc7389` + the #11640 log-prob fix |
| verl | `6cbca9ce` (= `REQUIRED_VERL.txt`), stock |
| sglang | 0.5.14 (via `ai_dynamo[sglang]`) |
| `ai_dynamo` wheels | 1.3.0, local build (`$DYNAMO_WHEELHOUSE` in the launchers) |
