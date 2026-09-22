# ThunderAgent + async verl + vLLM 0.24

This guide uses the **`sopy/dynamo_sglang`** branch to run Dynamo's Python
ThunderAgent router with **vLLM 0.24.0** and verl V1 async training.
Both references use **Qwen3-30B-A3B-Base**, H100 80GB nodes with approximately
**2 TiB host RAM each**, two independent TP4 engines and internal DP=1:

| Mode | Training / rollout GPUs |
| --- | --- |
| `colocate_async` | One node, 8 GPUs shared between training and rollout |
| Fully async: `separate_async`, `hybrid_engine=false` | Two nodes, 8 training + 8 rollout GPUs |

Uni-Agent is pinned to upstream main as inspected on **2026-09-21**. Its
Gateway integration lives in the recipe; do not apply the old Uni-Agent
lifecycle patch to this version. Uni-Agent and verl need no source changes.
Keep the Dynamo/vLLM exception compatibility patch. The recipe also handles
the pinned fully async trainer's logging API mismatch at runtime.

The request path is:

```text
Uni-Agent → recipe Gateway adapter → verl V1 async client
          → shared Dynamo frontend → ThunderAgent → vLLM workers
```

The frontend uses `round-robin` to reach ThunderAgent. ThunderAgent selects the
backend worker and DP rank. The recipe starts the frontend, router, workers,
etcd and NATS; separate manual serving processes are not required.

## 1. Prerequisites and versions

Start with a working Linux **Python 3.12** verl/vLLM training environment, CUDA
drivers, and your workload's model, parquet data and reward implementation.
For Uni-Agent, also prepare its agent configuration, tools and sandbox runtime.
This guide adds Dynamo to that environment; it does not provision a training
container or a SWE sandbox.

| Component | Reference version |
| --- | --- |
| Recipe | `sopy/dynamo_sglang`, including `uniagent_gateway.py` |
| Uni-Agent | **`0825e1ea00640896fec04535e6f8bbf734210df4`**, unmodified |
| vLLM | **0.24.0** |
| PyTorch / CUDA | **2.11.0+cu130 / 13.0** |
| Transformers / FlashAttention | **5.5.3 / 2.8.3** |
| Accelerate | **1.15.0** |
| Ray / NIXL | **2.56.1 / 1.3.2** |
| Dynamo | **`8c5a73723109058f96c15fff3fc912231d65ad6e` + bundled compatibility patch** |
| verl | Uni-Agent submodule **`3efe38c759c14622fd1b2c9e3679f2d02f86bdac`** |
| TransferQueue | **`434f8c476b4be24bc087e6e95070e64efcc739f9`**, version 0.1.9.dev0 |
| Protobuf | **>=6.33.5,<7**; the reference environment used 6.33.6 |

The general [installation guide](INSTALL.md) describes a separate vLLM 0.28
environment. Use the constraints and compatibility patch in this guide for 0.24.
The pinned Dynamo `[vllm]` extra requests **vLLM 0.28.0**; install the base Dynamo
wheels, without that extra.

## 2. Check out Uni-Agent, verl and the recipe

Use new checkout directories and a working vLLM 0.24 training environment.

```bash
export UNIAGENT_ROOT=/path/to/uni-agent
export DYNAMO_SRC=/path/to/dynamo

git clone https://github.com/verl-project/uni-agent.git "$UNIAGENT_ROOT"
git -C "$UNIAGENT_ROOT" checkout 0825e1ea00640896fec04535e6f8bbf734210df4
git -C "$UNIAGENT_ROOT" submodule update --init --recursive
export VERL_SRC="$UNIAGENT_ROOT/verl"
git -C "$VERL_SRC" rev-parse HEAD  # expected: 3efe38c759c14622fd1b2c9e3679f2d02f86bdac

git clone --branch sopy/dynamo_sglang \
    https://github.com/sophiayyya/verl-recipe.git "$VERL_SRC/recipe"
export RECIPE_DIR="$VERL_SRC/recipe"
export PATCH_DIR="$RECIPE_DIR/dynamo/patches/vllm024"
export CONSTRAINTS="$PATCH_DIR/constraints.txt"
python3 -m pip install uv 'maturin[patchelf]'
python3 -m uv pip install --python "$(command -v python3)" --no-deps \
    -e "$VERL_SRC" -e "$UNIAGENT_ROOT"
```

Install the official Accelerate release used by this reference:

```bash
python3 -m uv pip install --python "$(command -v python3)" -c "$CONSTRAINTS" \
    'accelerate==1.15.0'
```

Accelerate 1.12.0 fails on Transformers 5.5.3 meta initialization with
`Parameter.__new__() got an unexpected keyword argument '_is_hf_initialized'`.
The [official 1.15.0 implementation](https://github.com/huggingface/accelerate/blob/v1.15.0/src/accelerate/big_modeling.py)
fixes that failure. Current verl already provides
`fsdp_config.use_no_sync_for_gradient_accumulation=false`, which the launcher
sets to keep accumulated gradients sharded. Do not apply the old
`verl-qwen3-fsdp2.patch` or `uniagent-program-lifecycle.patch` to these checkouts.

`--no-deps` preserves the working training stack; it does not bootstrap an empty
environment. Add the task/sandbox dependencies from the
[Uni-Agent installation guide](https://github.com/verl-project/uni-agent/blob/0825e1ea00640896fec04535e6f8bbf734210df4/docs/source/quickstart/installation.md).

## 3. Build and install Dynamo for vLLM 0.24

Install Rust and the compiler/system libraries from the
[official source-build guide](https://docs.dynamo.nvidia.com/dynamo/advanced-customizations/building-from-source).
The pinned checkout selects Rust 1.96.1. Build on Linux compatible with the
training environment.

```bash
git clone https://github.com/ai-dynamo/dynamo.git "$DYNAMO_SRC"
git -C "$DYNAMO_SRC" checkout 8c5a73723109058f96c15fff3fc912231d65ad6e
git -C "$DYNAMO_SRC" apply --check "$PATCH_DIR/dynamo-client-errors.patch"
git -C "$DYNAMO_SRC" apply "$PATCH_DIR/dynamo-client-errors.patch"

cd "$DYNAMO_SRC"
python3 -m uv build --wheel
cd "$DYNAMO_SRC/lib/bindings/python"
python3 -m maturin build --release --locked \
    --features 'kv-indexer,slot-tracker,select-service,mm-routing,aic-forward-pass,request-trace-s3' \
    --out "$DYNAMO_SRC/dist"

# Keep only this build's two wheels in dist/.
python3 -m uv pip install --python "$(command -v python3)" -c "$CONSTRAINTS" \
    --reinstall-package ai-dynamo --reinstall-package ai-dynamo-runtime \
    "$DYNAMO_SRC"/dist/ai_dynamo*.whl \
    'nixl[cu13]==1.3.2' 'protobuf>=6.33.5,<7' 'grpcio-tools==1.81.1' blake3 \
    'TransferQueue @ git+https://github.com/Ascend/TransferQueue.git@434f8c476b4be24bc087e6e95070e64efcc739f9'

export PYTHONPATH="$DYNAMO_SRC/components/src:$VERL_SRC:$UNIAGENT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
command -v etcd nats-server
python3 -m dynamo.frontend --help
```

Use the same patched `components/src` on the driver and Ray workers. The launcher
forwards `PYTHONPATH`, `PATH` and the recipe registry to workers. Install `etcd`
and `nats-server` on `PATH` before launching.

The patch is required because [vLLM 0.24's exception API](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/exceptions.py)
does not provide `VLLMClientError`, which this newer Dynamo backend otherwise
imports on the first generation. The patch preserves request-error status codes
and propagates engine errors and cancellation. This is a tested text-completion
path, not a claim that every Dynamo 1.5 feature supports vLLM 0.24.

## 4. Check the environment

```bash
cd "$VERL_SRC"
python3 - <<'PY'
import importlib.metadata as m
import torch
assert m.version("vllm") == "0.24.0"
assert torch.__version__.split("+")[0] == "2.11.0"
assert torch.version.cuda == "13.0"
assert m.version("transformers") == "5.5.3"
assert m.version("accelerate") == "1.15.0"
import dynamo._core, dynamo.vllm.main, recipe.dynamo.register
import recipe.dynamo.uniagent_gateway
print({name: m.version(name) for name in
       ["vllm", "torch", "transformers", "accelerate", "ai-dynamo", "ai-dynamo-runtime", "TransferQueue"]})
print("Dynamo runtime:", dynamo._core.__file__)
PY

python3 -m uv pip install --python "$(command -v python3)" -c "$CONSTRAINTS" pytest pytest-asyncio
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q -p pytest_asyncio.plugin \
    -c /dev/null --noconftest --import-mode=importlib \
    "$DYNAMO_SRC/components/src/dynamo/vllm/tests/test_client_error_compat.py"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q -p pytest_asyncio.plugin \
    --import-mode=importlib "$RECIPE_DIR/dynamo/tests"
```

The generator compatibility test matters: an import-only check did not expose
the original first-request failure.

## 5. Run two training steps

### Colocated async

The [launcher](run_uniagent_thunderagent_vllm024.sh) uses
`verl.trainer.main_ppo`, V1 `colocate_async`, 8 GPUs and two TP4 engines. Run it
inside an allocated 8-GPU job/container. The default is two prompts × two
responses per batch and two training steps.

Use a working **current Uni-Agent** workload: training parquet with serialized
Task Configs in `extra_info.tools_kwargs.task`, and a task YAML such as the
[upstream ReAct example](https://github.com/verl-project/uni-agent/blob/0825e1ea00640896fec04535e6f8bbf734210df4/examples/quickstart/training/task_config_react.yaml).
Configure its sandbox for your infrastructure and check a real tool/reward call.
The old `swe_agent` YAML and old SWE parquet format are not interchangeable with
this Gateway/TaskRunner interface.

```bash
export MODEL_PATH=/models/Qwen3-30B-A3B-Base
export TRAIN_FILE=/data/train.parquet
export TEST_FILE="$TRAIN_FILE"
export TASK_CONFIG=/data/task_config.yaml
export RUN_NAME=ta-vllm024-uniagent-smoke
export OUTPUT_DIR=/outputs/$RUN_NAME
export TOTAL_STEPS=2
export RAY_NUM_CPUS=96
mkdir -p "$OUTPUT_DIR"

bash "$RECIPE_DIR/dynamo/run_uniagent_thunderagent_vllm024.sh" \
    actor_rollout_ref.rollout.enforce_eager=true --cfg job --resolve \
    > "$OUTPUT_DIR/resolved-config.yaml"
bash "$RECIPE_DIR/dynamo/run_uniagent_thunderagent_vllm024.sh" \
    actor_rollout_ref.rollout.enforce_eager=true
```

The launcher selects
`recipe.dynamo.uniagent_gateway.DynamoAgentFrameworkRolloutAdapter` and
`thunderagent.auto_finalize=false`. It uses Uni-Agent's standard
`uni_agent.framework.task_runner.run_task`; an existing custom runner can be
selected with `AGENT_RUNNER_FQN`. Append workload-specific Hydra overrides to
the command. The script does not create datasets, sandbox images or reward code.

For W&B, run `wandb login` with your own account and append
`'trainer.logger=[console,wandb]'`. To run 40 steps, set `TOTAL_STEPS=40` and a new
`RUN_NAME`/`OUTPUT_DIR`; checkpoint resume is disabled, so this starts a new run.

### Fully async

Use **two 8×H100 nodes**, one training pool and one rollout pool. The tested
4-training + 4-rollout configuration exhausted GPU memory while allocating
Adam states. Retaining the same model and training precision worked with 8+8.

Install the NCCL checkpoint engine's dependency on every node:

```bash
python3 -m uv pip install --python "$(command -v python3)" -c "$CONSTRAINTS" \
    'cupy-cuda13x==14.0.1'
```

The launcher loads [verl_logging_compat.py](verl_logging_compat.py) through
`VERL_USE_EXTERNAL_MODULES` in both the driver and Ray workers. It removes the
unsupported `flush` argument from this trainer's logger only, preserving INFO
messages, normal handler flushing and the original training control flow.
No verl source patch or global log suppression is needed.

Make the same environment, model, dataset, task YAML and recipe available at
the same paths on both nodes. Start Ray on the allocated nodes:

```bash
# On both CUDA nodes, before starting Ray:
unset ROCR_VISIBLE_DEVICES PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF RAY_ADDRESS
# Training driver/head node: set its routable IP first.
ray start --head --node-ip-address="$HEAD_IP" --port=6379 \
    --num-gpus=8 --num-cpus=96 --include-dashboard=false
# Second node:
ray start --address="$HEAD_IP:6379" --num-gpus=8 --num-cpus=96
```

On the head node, retain the workload variables from the colocated example:

```bash
export RAY_CLUSTER_ADDRESS="$HEAD_IP:6379"
export RUN_NAME=ta-vllm024-uniagent-fully-async-smoke
export OUTPUT_DIR=/outputs/$RUN_NAME
mkdir -p "$OUTPUT_DIR"
bash "$RECIPE_DIR/dynamo/run_uniagent_thunderagent_vllm024_fully_async.sh" \
    --cfg job --resolve > "$OUTPUT_DIR/resolved-config.yaml"
bash "$RECIPE_DIR/dynamo/run_uniagent_thunderagent_vllm024_fully_async.sh"
```

This [launcher](run_uniagent_thunderagent_vllm024_fully_async.sh) still enters
`verl.trainer.main_ppo`. It sets `separate_async`, disables hybrid replicas,
uses two TP4 rollout engines, NCCL weight transfer and one update per sync.
Training and generation overlap; generation pauses briefly for weight updates.
The batch constraint is `train_batch_size = parameter_sync_step × ppo_mini_batch_size`.
The script connects to an existing Ray cluster; it does not allocate nodes.