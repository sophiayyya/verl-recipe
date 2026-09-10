# Dynamo rollout backend for verl

This recipe plugs [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) into verl
as a first-class **async rollout backend**, alongside the built-in `vllm`,
`sglang` backends. Turning it on is a one-line config change
(`actor_rollout_ref.rollout.name=dynamo`); everything Dynamo-specific
is driven from `rollout.engine_kwargs.dynamo.*`. The backend can front **either inference engine**: `dynamo.vllm` (the default)
or `dynamo.sglang` (`engine_kwargs.dynamo.engine=sglang`). 

Dynamo owns request routing behind a single logical frontend, so its
**KV-cache-aware router** can raise the prefix-cache hit rate across a rollout
step. Weight updates flow through verl's colocated CUDA-IPC path when trainer
and engine workers share GPUs (`colocate_async` / legacy V0), and through a
two-hop checkpoint-engine path (nccl across pools → node-local CUDA-IPC) for
the standalone rollout pool in `separate_async`.

**Contents** —
[How it works](#how-it-works) ·
[Configuration](#configuration) ·
[Quick start](#quick-start) ·
[V1 async trainers](#running-a-full-rl-run-v1-trainer--the-main-path) ·
[NIXL weight sync](#nixl-weight-sync-checkpoint-engine) ·
[KV-aware routing result](#kv-aware-routing-result) ·
[Legacy V0 path](#legacy-v0-path-compatibility-only) ·
[ThunderAgent](#thunderagent-extension) ·
[SGLang engine](#sglang-engine-engine_kwargsdynamoenginesglang)

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


| File                                                           | Role                                                                                                                                                                                                              |
| -------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `[register.py](register.py)`                                   | Registers `dynamo` in verl's rollout registries; loaded via `VERL_USE_EXTERNAL_MODULES=recipe.dynamo.register`.                                                                                                   |
| `[main_dynamo.py](main_dynamo.py)`                             | Entry point used by the V1 smokes: dispatches on `trainer.use_v1` (`TaskRunnerV1` / legacy V0 `TaskRunner`), refuses V1 + `DynamoAgentLoopManager`, runs the legacy-reward-key migration; see [Quick start](#quick-start). |
| `[config/dynamo_trainer.yaml](config/dynamo_trainer.yaml)`     | Hydra config: inherits `ppo_trainer`, sets `rollout.name=dynamo`, `rollout.mode=async`.                                                                                                                           |
| `[dynamo_async_server.py](dynamo_async_server.py)`             | `DynamoReplica` / `DynamoHttpServer` — spawns and watchdogs etcd, nats-server, engine workers, and `dynamo.frontend`, for both engines.                                                                           |
| `[dynamo_rollout.py](dynamo_rollout.py)`                       | `ServerAdapter` — engine-agnostic facade; dispatches on `engine_kwargs.dynamo.engine` and lazily imports the chosen adapter (no module-scope engine imports), so it loads on an image that ships only one engine. |
| `[dynamo_vllm_rollout.py](dynamo_vllm_rollout.py)`             | `VllmDynamoServerAdapter` — per-rank client for the vLLM engine; HTTP generation via the frontend, control RPCs (sleep/wake/`update_weights`) to the shared per-node actor.                                       |
| `[dynamo_sglang_rollout.py](dynamo_sglang_rollout.py)`         | `SGLangServerAdapter` — per-rank client for the sglang engine (shard-local TP group, CUDA-IPC weight sync via `update_weights_from_tensor`).                                                                      |
| `[dynamo_sglang_engine.py](dynamo_sglang_engine.py)`           | HTTP client for `dynamo.sglang`'s native `/engine/control/*` RL routes.                                                                                                                                           |
| `[dynamo_naming.py](dynamo_naming.py)`                         | `control_actor_name()` — the one place the `dynamo_server_{replica}_{node}` actor-name contract is spelled out.                                                                                                   |
| `[dynamo_agent_loop.py](dynamo_agent_loop.py)`                 | `DynamoServerManager` / `DynamoLLMServerManager` — talk to the single shared frontend instead of load-balancing across replicas.                                                                                  |
| `[dynamo_worker_extension.py](dynamo_worker_extension.py)`     | vLLM `worker_extension_cls` that maps each DP shard to a node-global rank so trainer and engine agree on the CUDA-IPC socket path.                                                                                |
| `[_dynamo_vllm_with_control.py](_dynamo_vllm_with_control.py)` | Private ZMQ control sidecar that bridges verl's `collective_rpc` into the `dynamo.vllm` subprocess (vLLM only; sglang has native control routes).                                                                 |
| `[metrics_sidecar.py](metrics_sidecar.py)`                     | Optional per-worker system-status / metrics scraper.                                                                                                                                                              |


Enable the backend by pointing verl at the recipe's registration module:

```bash
export VERL_USE_EXTERNAL_MODULES=recipe.dynamo.register
```



## Configuration

Everything Dynamo-specific lives under
`actor_rollout_ref.rollout.engine_kwargs.dynamo`. All keys are optional; sane
defaults are applied in `DynamoHttpServer`.


| Key                                                    | Values / example                                                   | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ------------------------------------------------------ | ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `engine`                                               | `vllm` (default), `sglang`                                         | Which inference engine the workers run.                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `router_mode`                                          | `kv` (default), `round-robin`, `random`, `least-loaded`            | Dynamo request-routing policy; `kv` enables KV-cache-aware routing.                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `frontend_http_port` / `etcd_port` / `nats_port`       | `0` = auto-assign                                                  | Fixed ports if you need them.                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `served_model_name`                                    | falls back to `model_config.local_path`                            | Model name the frontend advertises.                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `request_engine_data` / `request_completion_token_ids` | `true` / `false`                                                   | Ask the frontend to return `nvext.engine_data` (vLLM only) / raw `completion_token_ids` (token-in/token-out for RL). **Required for RL.** The sglang engine refuses to start when `request_completion_token_ids` is left unset (an explicit `false` is honored). If it is `true` and the frontend still returns no token ids, generation raises rather than silently re-tokenizing the text; when it is off, the text re-encode fallback is logged at ERROR (first 3 hits + every 100th). |
| `return_tokens_as_token_ids`                           | `true` / `false`                                                   | Emit token ids instead of detokenized text.                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `request_timeout_s`                                    | `600` (default; scripts use `1800`)                                | Per-request timeout.                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `free_engine_on_train` | mirrors `rollout.free_cache_engine` (default `true`; set in `config/dynamo_base.yaml`) | Sleep the engine during the training phase. Validated against `rollout.free_cache_engine` at startup: an explicit value that contradicts it fails fast instead of silently no-op'ing (and OOM'ing) in colocate training. |
| `enable_kv_events`                                     | `true` (default) / `false`                                         | Pass `--kv-events-config` to every engine shard (vLLM and SGLang) so the KV router indexes real block residency. Without it the SGLang worker registers `use_kv_events=False` and the router only has its predict-on-route guesses (measured: router-estimated hit 0.87 vs engine 0.63). Set `false` for `round-robin`, where the events are pure overhead.                                                                                                                               |
| `router_session_affinity_ttl_secs`                     | unset (default, off) / `1`..`31536000`                             | Frontend `--router-session-affinity-ttl-secs`: pin every request carrying the same `x-dynamo-session-id` to one worker. The recipe sends verl's per-trajectory request_id as that header, so the later turns of a multi-turn rollout land on the worker that already holds their prefix. Works in every `router_mode`; with ThunderAgent on, the program id it sets is the key.                                                                                                                                                                    |
| `enable_worker_system_metrics`                         | `true` / `false`                                                   | Expose the per-worker system-status / metrics port (paired with `metrics_sidecar.py`). Must stay `true` for sglang — that port carries its control plane.                                                                                                                                                                                                                                                                                                                                 |
| `extra_args`                                           | `["--generation-config","vllm","--stream-interval=100"]`           | Extra CLI args forwarded verbatim to the engine worker.                                                                                                                                                                                                                                                                                                                                                                                                                                   |




## Quick start

All examples use verl's standard entry point. Three things make it work with this
recipe on the pinned checkout (`6cbca9ce`):

- `--config-path ../../recipe/dynamo/config --config-name dynamo_trainer` loads the
recipe's Hydra config. Hydra resolves the relative path against `verl/trainer/`
(the module declaring `@hydra.main`), so it is CWD-independent as long as this
repository sits at `recipe/` inside the verl checkout.
- `trainer.use_v1=False` selects the legacy V0 `TaskRunner` that `dynamo_trainer`
(agent-loop path, `DynamoAgentLoopManager`) is written for. The V1 unified trainer is
the main path now; it uses the `dynamo_trainer_v1_*` presets described below, which
set `trainer.use_v1: true` explicitly.
- Custom rewards must use the canonical `reward.custom_reward_function.*` namespace,
not the legacy top-level `custom_reward_function.*`: the v0 runner reads
`config.reward.*` and `main_ppo` never runs the legacy-key migration, so legacy
keys are **silently ignored** (the run proceeds with the default reward).

`[main_dynamo.py](main_dynamo.py)` is the convenience entry point: it dispatches on
`trainer.use_v1` exactly like upstream (`TaskRunnerV1` when it is `true`, the legacy V0
`TaskRunner` when it is `false`), refuses the V1 + `DynamoAgentLoopManager` combination
with a clear error, and runs verl's legacy-reward-key migration itself, so the
`reward.custom_reward_function.*` namespace is not required when you launch through it.
Its default primary config is `dynamo_trainer` (V0), so pass `--config-name=dynamo_trainer_v1_*`
for V1, as the V1 smokes do. The V0 launchers call `verl.trainer.main_ppo` directly and
therefore set both explicitly.

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


| Script                                                                       | Engine | What it runs                                                                                                                                                                                                                                                             |
| ---------------------------------------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `[train_30b_rl_dynamo_kv_metrics.sh](train_30b_rl_dynamo_kv_metrics.sh)`     | vLLM   | KV router + metrics sidecar RL run (inner command, `NNODES` default 2; driven by the sbatch below).                                                                                                                                                                      |
| `[train_qwen3_30b_sglang.sh](train_qwen3_30b_sglang.sh)`                     | sglang | The verified 100-step retool GRPO run. Defaults reproduce it (`ENFORCE_EAGER=False`, `DISABLE_PIECEWISE=0`, deferred optimizer load / fused kernels / eager experts all off); every env knob is listed in the script header, e.g. `sbatch --export=ALL,TOTAL_STEPS=3 …`. |




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


| backend | single-node | multi-node            |
| ------- | ----------- | --------------------- |
| naive   | ✅           | ✅                     |
| NIXL    | ✅           | ✅ (2×8 GPU validated) |




### Transport selection (read this before multi-node)

NIXL's default backend is UCX, and UCX picks its transport from `UCX_TLS`.
The right setting depends on the RDMA fabric:


| fabric                                                   | recommendation                                                                                                                                                                                                                                                                                                                |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| RDMA with native RDMA-read (e.g. InfiniBand / RoCE)      | `UCX_TLS=cuda_ipc,cuda_copy,rc,tcp` — `rc` gives native RDMA read at line rate.                                                                                                                                                                                                                                               |
| RDMA without native RDMA-read (send/recv-only protocols) | UCX can only emulate one-sided reads over send/recv — we measured 0.23 GB/s. Use NIXL's **LIBFABRIC** backend instead: `+actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.nixl.backends=[LIBFABRIC]`, with your fabric's libfabric provider (≥1.18) on `LD_LIBRARY_PATH`. Same 1 GiB cross-node read: **48.5 GB/s**. |


Cross-node measured on 2 nodes × 8×H100 (RDMA fabric), 3-step GRPO,
step-3 `update_weights`:


| model        | naive  | NIXL UCX (tcp / one-sided read emulated over send/recv) | NIXL LIBFABRIC |
| ------------ | ------ | ------------------------------------------------------- | -------------- |
| Qwen2.5-0.5B | 3.08 s | 12.2 s / 11.6 s                                         | 3.09 s         |
| Qwen3-8B     | 29.7 s | —                                                       | **27.1 s**     |


At 0.5B the LIBFABRIC path is at parity with naive (the shared per-rank
engine-consume dominates); at 8B it is ~9% faster. The UCX column shows
the send/recv emulation ceiling — protocol-level, not tunable.

Two helpers ship with the recipe: `[run_nixl_smoke.sh](run_nixl_smoke.sh)` (a
3-step GRPO training smoke parameterised over `NNODES` / `CE_BACKEND`) and
`[nixl_bench.py](nixl_bench.py)` (a standalone cross-node bandwidth probe for
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


| Backend                           | ms/token | Mean response length | KV-cache hits / queries | KV-cache hit rate |
| --------------------------------- | -------- | -------------------- | ----------------------- | ----------------- |
| Dynamo KV (`stream-interval=100`) | 1.5956   | 876.1                | 2,248,368 / 2,520,362   | 89.21%            |
| vLLM baseline                     | 1.7220   | 872.3                | 1,860,816 / 2,432,064   | 76.51%            |


Dynamo KV shows approximately **7.3% lower per-token latency** (≈7.9% faster)
in this comparison and improves the KV-cache hit rate by **12.70 percentage points**.

## Running a full RL run (V1 trainer — the main path)

The recipe now targets verl's **V1 unified trainer** (see `REQUIRED_VERL.txt`
for the tested pin). Three entry configs ship ready to run:

| Entry config | Mode | Engine | Placement |
| --- | --- | --- | --- |
| `--config-name=dynamo_trainer_v1_colocate` | `colocate_async` | vLLM | trainer + rollout share GPUs; replicas abort + sleep every train step |
| `--config-name=dynamo_trainer_v1_colocate_sglang` | `colocate_async` | SGLang | same shape; adds the switches sglang hard-requires (`enable_sleep_mode`, `request_completion_token_ids`, `enable_worker_system_metrics`) |
| `--config-name=dynamo_trainer_v1_separate` | `separate_async` | vLLM; SGLang with the four overrides `smoke_dynamo_v1_separate_sglang.sh` adds (`engine=sglang`, `request_completion_token_ids=true`, `enable_worker_system_metrics=true`, `rollout.enable_sleep_mode=true`) | standalone rollout pool (`rollout.nnodes × n_gpus_per_node`); weights flow nccl → node-local (CUDA-IPC for vLLM, the HTTP control route for sglang) |

Smoke them end-to-end (real training steps, tiny model), one script per engine:

```bash
bash recipe/dynamo/smoke_dynamo_v1_colocate.sh
bash recipe/dynamo/smoke_dynamo_v1_colocate_sglang.sh
bash recipe/dynamo/smoke_dynamo_v1_separate.sh          # needs >= 2 GPUs and cupy (nccl backend)
bash recipe/dynamo/smoke_dynamo_v1_separate_sglang.sh   # same requirements
```

Under V1, leave `agent.agent_loop_manager_class` at `null` (the presets do):
verl's `AgentLoopManagerTQ` drives the loop and writes TransferQueue, and
`DynamoLLMServerManager` upgrades the client to a partial-rollout-aware
`FullyAsyncLLMServerClient` (with ThunderAgent affinity when enabled).
Multi-turn agent frameworks (e.g. uni-agent) plug in unchanged via their own
`agent_loop_manager_class` — validated with the uni-agent mem-agent recipe by
only switching `rollout.name=vllm` → `dynamo`.

ThunderAgent under V1: programs are keyed by the caller's stable `request_id`
and auto-finalized per generate call (`thunderagent.auto_finalize`, default
true). Multi-turn callers that want cross-turn affinity set
`auto_finalize: false` and call the client's `finalize_program(session_id)`
from their trajectory-end hook. Program tables are frontend-local and
separate_async re-routes aborted retries across pools, so the client records
**every server that serves a generation attempt** and finalizes each of them
(each finalize RPC bounded by `thunderagent.finalize_timeout_s`, default
60 s); cleanup counts as confirmed only when all ack, and unconfirmed
cleanups count toward `thunderagent.finalize_leak_threshold`.

### Multi-node runs (pre-started Ray clusters)

On a pre-started Ray cluster (`ray start` on each node, driver connects with
`RAY_ADDRESS`), workers inherit their environment from the **raylet's** shell,
not from the driver — exports in your launch script never reach them. Two
rules:

1. **`VERL_USE_EXTERNAL_MODULES` is shipped automatically** when you launch
   through `recipe.dynamo.main_dynamo` (injected into the job's Ray
   `runtime_env`). If you launch through `verl.trainer.main_ppo` directly, add:

   ```
   "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES=recipe.dynamo.register"
   ```

   (`++` is required: the schema's `runtime_env` node is a struct without an
   `env_vars` field.) Without this, workers fail with
   `Rollout dynamo with mode async not found`.

2. **Everything else must be exported before `ray start` on every node** —
   deployment-specific paths cannot live in the repo config. Checklist (each
   row is a failure observed in practice when missing):

   | env before `ray start` | symptom when missing |
   | --- | --- |
   | `HF_HOME=<writable path>` | `OSError: Read-only file system` (datasets cache lands on the read-only model mount) |
   | `PYTHONPATH=<ws>:<ws>/verl`, `PATH+=<etcd/nats dir>` | recipe import / infra binary failures |
   | `unset DD_*`, `unset LD_PRELOAD` | raylet crashes parsing injected quoted-JSON env |
   | `unset PYTORCH_CUDA_ALLOC_CONF` | sglang torch_memory_saver incompatibility |

Small-GPU multi-node smokes only: size pods so no single node can hold every
per-node bundle (verl places each node's bundle as an independent placement
group with no cross-node constraint — full 8-GPU-per-node deployments are
unaffected), and assert the pool spans the expected node count
(`[DynamoReplica pool] ready: ... nodes=N`).

### Legacy V0 path (compatibility only)

The original PR #110/#126 flow — `--config-name=dynamo_trainer` (which pins
`trainer.use_v1=false`), colocated `hybrid_engine=True`, and the legacy
`DynamoAgentLoopManager` — still works but is a **compatibility path**:
upstream has deprecated the V0 trainer (removal planned in v0.9.0), and the
legacy manager must NOT be combined with `trainer.use_v1=true` (the entry
point fails fast on that combination because it does not write TransferQueue).

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



### Required versions

- Dynamo, vLLM engine: PyPI `ai-dynamo>=1.3.0.post1` (ships `dynamo.vllm`,
`dynamo.frontend`, and `dynamo.thunderagent_router`, including
[PR #11185](https://github.com/ai-dynamo/dynamo/pull/11185)). Supersedes the
previous source build at commit `59d614641837e593f0567b79d75394aae5f864e0`.
- Dynamo, sglang engine: additionally requires the incremental-logprobs fix
  (dynamo#11640 area) which no stable PyPI release contains yet — use the
  source build at `94accc7389` (the #11185 merge commit) with the
  incremental-logprobs backport (dynamo#11640) applied, or
  an image carrying that backport. Without it ~99.8% of sglang logprobs pad
  and rollout-correction metrics explode.
- verl: pinned commit in [REQUIRED_VERL.txt](REQUIRED_VERL.txt).
- `separate_async` additionally needs `cupy-cuda12x` (verl's nccl
checkpoint-engine backend registers only when cupy imports); the V1 trainer
itself needs `TransferQueue` (tested with `0.1.9`).


### Topology

With ThunderAgent enabled, verl launches processes in this order:

```text
etcd -> NATS -> Dynamo engine workers (vLLM or SGLang) -> ThunderAgent router -> frontend
```

ThunderAgent owns the internal KV router and forwards to the worker
endpoint `<namespace>.backend.generate`. Shutdown reverses the consumer side:
frontend, ThunderAgent, workers, NATS, then etcd.

### Configuration

The recipe's default config (`[config/dynamo_base.yaml](config/dynamo_base.yaml)`,
inherited by `dynamo_trainer.yaml`) enables ThunderAgent. It is validated end to end
on the vLLM path. The sglang engine has the worker-side glue (its workers register
under the internal `--verl-thunderagent-backend` name so only the router serves the
public model name), but its routing assertions still await a GPU rerun, so every
sglang launcher here passes `thunderagent.enabled=false` explicitly:

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

`router_block_size` is applied to the engine (vLLM `--block-size`, SGLang
`--page-size`) and to ThunderAgent alike. Pass scheduler
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

`[run_uniagent_variant.sh](run_uniagent_variant.sh)` is a concise UniAgent
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

At concurrency 64–256, ThunderAgent and Global LB are near parity. At
concurrency 384, ThunderAgent reaches **1.94× rollout-phase speedup** and
**1.39× observed full-step speedup**; at concurrency 512, the speedups reach
**2.40×** and **1.60×**, respectively.

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


|                | vLLM                                                                | SGLang                                                                              |
| -------------- | ------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Worker process | `python -m recipe.dynamo._dynamo_vllm_with_control` (verl wrapper)  | stock `python -m dynamo.sglang`                                                     |
| Control plane  | verl-private ZMQ REP sidecar → `engine.collective_rpc`              | **native** `/engine/control/`* on `DYN_SYSTEM_PORT`                                 |
| Weight sync    | `BucketedWeightSender` → ZMQ-IPC socket → `update_weights_from_ipc` | `MultiprocessingSerializer` CUDA-IPC handles → `control/update_weights_from_tensor` |
| Sleep / wake   | `engine.sleep(level=…)`                                             | `release_memory_occupation(tags=…)` / `resume_memory_occupation`                    |
| KV events      | `--kv-events-config <json>`                                         | same JSON, passed by default (`enable_kv_events`); sglang's ZMQ publisher plus `DynamoSglangPublisher` re-publish on the event plane |
| Cache flush    | `reset_prefix_cache`                                                | `call_tokenizer_manager("flush_cache")` (**needs** `--enable-rl`)                   |


`dynamo.sglang` registers its RL control routes itself
(`request_handlers/handler_base.py::register_engine_routes`), which is why this
path ships no sidecar. The trade is that `DYN_SYSTEM_PORT` stops being an
optional metrics extra and becomes the whole control plane —
`enable_worker_system_metrics=false` is rejected outright for this engine.

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


| Key                   | Default                                        | Purpose                                                                                                                                                                                                                                                                                                                                      |
| --------------------- | ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `enable_rl`           | `true`                                         | Adds `--enable-rl`; registers `call_tokenizer_manager`, the only route that can flush the radix cache after a weight update.                                                                                                                                                                                                                 |
| `verify_weight_sync`  | `false`                                        | Best-effort probe: reads one synced parameter back via `get_weights_by_name` after each sync (snapshot refreshed every sync) and raises on mismatch. That API is model-specific and **unimplemented for Qwen2/Qwen3**, so on those models the probe logs `INCONCLUSIVE` at ERROR and verifies nothing — a passing run is not a verified one. |
| `page_size`           | falls back to `thunderagent.router_block_size` | KV-router block size (`--page-size`).                                                                                                                                                                                                                                                                                                        |
| `skip_tokenizer_init` | `false`                                        | Token-in/token-out.                                                                                                                                                                                                                                                                                                                          |
| `attention_backend`   | `flashinfer`                                   | `--attention-backend`. sglang's Hopper default (fa3) decodes ~8% slower with the router page size; an explicit value or an `--attention-backend` in `extra_args` wins. |
| `extra_args`          | `[]`                                           | Forwarded verbatim; `dynamo.sglang` exposes the whole `ServerArgs` CLI.                                                                                                                                                                                                                                                                      |




