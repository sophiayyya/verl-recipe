#!/bin/bash
# NOTE: retool_ab.sbatch (the original A/B driver) was removed, but this
# bootstrap stays: validate_vllm_kv_metrics.sbatch cats it into its per-node scripts.
# Per-node, in-container bootstrap for the retool A/B (runs on BOTH nodes).
#
# Techniques lifted from the proven vLLM run
# ($B/verl/recipe/dynamo/train_30b_rl_dynamo_kv_i100_metrics.sh) — a job-unique
# sandbox-fusion port, the detached-actor patch, and cd'ing into the verl repo
# before `ray start` so raylet-spawned actors inherit a cwd where relative config
# paths like recipe/retool/retool.py resolve.
set -x
B=/lustre/fsw/portfolios/coreai/users/sopyang
V=$B/verl_dynamo/verl
W=$B/dynamo_wheels_1.3.0_94accc7389d4
export PATH=$B/dynamo_bin:$PATH
export HF_HOME=$B/.cache/huggingface
export PYTHONPATH=$V:$PYTHONPATH
export PIP_CACHE_DIR=$B/pip_cache
# dfw GPU nodes export both; torch refuses to init and it only surfaces as an
# opaque Ray ActorDiedError.
unset ROCR_VISIBLE_DEVICES 2>/dev/null
# MoE backward allocates in wildly varying block sizes (per-expert token gathers),
# which fragments the caching allocator badly: the OOM that killed job 16446247 was
# a 384 MiB request failing with 73 GiB allocated and only 172 MiB reserved-but-free.
# Set on every node so Ray-spawned workers inherit it, not just the driver.
# expandable_segments cuts cross-step fragmentation on the vLLM path, but it is
# fundamentally incompatible with sglang's torch_memory_saver:
#   RuntimeError: TorchMemorySaver is disabled for the current process because
#                 expandable_segments is not supported yet.
# and torch_memory_saver is what release/resume_memory_occupation uses to hand GPU
# memory back during the actor update -- losing it is not survivable. Both sglang arms
# died on this in jobs 16565476 (as engine rc=-9) and 16565477 (as a raw RuntimeError).
# Unset rather than skip: the surrounding environment may already carry it.
if [[ "${ARM:-}" == *sglang* ]]; then
  unset PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF 2>/dev/null || true
  echo "ARM=$ARM is sglang-based -> PYTORCH_CUDA_ALLOC_CONF unset (torch_memory_saver needs it off)"
else
  export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
fi
# vLLM's KV-event publisher binds its ZMQ port inside a child process minutes after
# the recipe probed it (the probe happens at launch, the bind after model load), so
# an OS-assigned ephemeral port can be taken by someone else in between:
#   zmq.error.ZMQError: Address already in use (addr='tcp://*:11199')   [job 16488580]
# A job-derived base makes the ports deterministic (no probe/bind race) AND unique
# per job (no collision with a previous run's stale subprocesses, which is why the
# recipe defaults to ephemeral in the first place).
export VERL_DYNAMO_KV_EVENT_PORT_BASE=${VERL_DYNAMO_KV_EVENT_PORT_BASE:-$(( 20000 + ${SLURM_JOB_ID:-0} % 8000 ))}
export VERL_DYNAMO_LOG_DIR=$B/logs/retool_ab_${SLURM_JOB_ID}_$(hostname)
mkdir -p "$VERL_DYNAMO_LOG_DIR"

# Ray actors are created on BOTH nodes, and get_rollout_class resolves the
# registry inside each actor's own process. Exporting this only in the driver
# (head node) left node 2's workers without recipe.dynamo.register, which surfaced
# as "AssertionError: Rollout dynamo with mode async not found" from
# actor_rollout_init_model on the second node only.
if [[ "$ARM" == dynamo_* ]]; then
  export VERL_USE_EXTERNAL_MODULES=recipe.dynamo.register
else
  unset VERL_USE_EXTERNAL_MODULES
fi

echo "=== bootstrap on $(hostname) (ARM=$ARM, VERL_USE_EXTERNAL_MODULES=${VERL_USE_EXTERNAL_MODULES:-<unset>}) ==="

# The sglang engine is installed per job, never baked into the image: it
# downgrades vLLM's guided-decoding stack and two kernel packages, and
# `import vllm` still succeeds so the damage would only appear at runtime.
if [[ "$ARM" == "dynamo_sglang" || "$ARM" == "native_sglang" ]]; then
  python -m pip install -q "$W/ai_dynamo-1.3.0-py3-none-any.whl[sglang]" \
                           "$W/ai_dynamo_runtime-1.3.0-cp310-abi3-manylinux_2_39_x86_64.whl" 2>&1 | tail -3
  python -c "import zmq" 2>/dev/null || python -m pip install -q pyzmq 2>&1 | tail -1
  python -c "import aiohttp" 2>/dev/null || python -m pip install -q aiohttp 2>&1 | tail -1
fi

# transformers 5.x passes _is_hf_initialized into nn.Parameter(), which an older
# accelerate forwards verbatim from register_empty_parameter:
#   TypeError: Parameter.__new__() got an unexpected keyword argument '_is_hf_initialized'
# Only reachable via accelerate's meta-device init, i.e. the large-model path —
# invisible at 0.5B, fatal at Qwen3-30B.
#
# This lives OUTSIDE the sglang branch on purpose. It was first hit on the sglang
# arm and blamed on the [sglang] extra bumping transformers 5.5.3 -> 5.8.1, so the
# fix was scoped to that branch. Wrong diagnosis: the container's OWN transformers
# is already 5.5.3, so accelerate is behind for every arm. dynamo_vllm — which
# installs no extra at all — died with the identical TypeError (job 16353038).
python -m pip install -q -U accelerate 2>&1 | tail -2

# --- vLLM arm: base wheels only, plus the LOCAL Dynamo source overlay -------- #
# Mirrors the install in the proven run
# ($B/verl/recipe/dynamo/train_30b_rl_dynamo_kv_i100_metrics.sh:256): the two
# wheels go in with --force-reinstall --no-deps and NO extra, so vLLM's own
# dependency stack is left exactly as the container built it.
if [[ "$ARM" == "dynamo_vllm" ]]; then
  python -m pip install -q --force-reinstall --no-deps \
    "$W/ai_dynamo_runtime-1.3.0-cp310-abi3-manylinux_2_39_x86_64.whl" \
    "$W/ai_dynamo-1.3.0-py3-none-any.whl" 2>&1 | tail -3
  python -c "import zmq"     2>/dev/null || python -m pip install -q pyzmq 2>&1 | tail -1
  python -c "import aiohttp" 2>/dev/null || python -m pip install -q aiohttp 2>&1 | tail -1
  python -c "import cupy"    2>/dev/null || python -m pip install -q "${CUPY_PKG:-cupy-cuda12x==13.6.0}" 2>&1 | tail -2

  # The shipped wheel's dynamo.vllm.handlers cannot return completion_token_ids;
  # only the local checkout can. Without the overlay the frontend returns text
  # with no token ids, _fallback_token_ids() substitutes a single EOS, and the
  # run trains on 1-token responses while every metric looks healthy. The proven
  # script asserts this for the same reason — fail here, loudly, not 40 minutes
  # in with a plausible-looking reward curve.
  export DYNAMO_SRC=${DYNAMO_SRC:-$B/dynamo}
  export PYTHONPATH=$DYNAMO_SRC/components/src:$PYTHONPATH
  python - <<'PY' || exit 1
import importlib.util, os, sys
spec = importlib.util.find_spec("dynamo.vllm.handlers")
path = spec.origin if spec else ""
want = os.path.join(os.environ["DYNAMO_SRC"], "components", "src", "dynamo")
if not path.startswith(want + "/"):
    sys.exit(f"FATAL: dynamo.vllm.handlers resolved to {path}, expected overlay under {want}")
if "completion_token_ids" not in open(path, encoding="utf-8").read():
    sys.exit(f"FATAL: {path} has no completion_token_ids support -> would train on 1-token responses")
print(f"dynamo.vllm.handlers OK (overlay, completion_token_ids present): {path}")
PY
  python -c "import vllm; print('vllm', vllm.__version__)" || exit 1
fi

python -c "import accelerate, transformers; print('accelerate', accelerate.__version__, '| transformers', transformers.__version__)"
if [[ "$ARM" == "dynamo_vllm" ]]; then
  python -c "import verl; print('verl ok (vllm arm, sglang deliberately not installed)')" || exit 1
else
  python -c "import sglang, verl; print('sglang', sglang.__version__, '| verl ok')" || exit 1
fi

# --- sandbox-fusion (retool's code tool) ---------------------------------- #
# The PyPI package `sandbox-fusion` ships only the CLIENT (`sandbox_fusion`);
# the server lives in bytedance/SandboxFusion as the `sandbox` package, which is
# why `uvicorn sandbox.server.server:app` needs the repo on PYTHONPATH. Cloned
# once to lustre rather than per node per job: compute nodes are not guaranteed
# outbound network, and a fresh clone on every node is pure waste.
pkill -u "$(id -u)" -f "uvicorn sandbox.server.server" 2>/dev/null || true
python -m pip install -q setuptools sandbox-fusion 2>&1 | tail -1
export PYTHONPATH=$B/SandboxFusion:$B/symeval:$PYTHONPATH
# Install the server's missing deps BY NAME and UNPINNED. Do NOT
# `pip install $B/SandboxFusion`: its pyproject pins pydantic <2.7 and
# transformers ^4.44, and this container runs pydantic 2.13 / transformers 5.8 —
# honouring those pins would downgrade both and break vllm, verl and sglang at
# once. Unpinned names leave already-satisfied packages alone.
python -c "import sandbox.server.server" 2>/dev/null || {
  echo "installing SandboxFusion server deps (unpinned)"
  python -m pip install -q databases aiosqlite aiomysql sqlalchemy \
      tenacity psutil structlog aiofiles 2>&1 | tail -3
}
python -c "import sandbox.server.server; print('sandbox server importable')" 2>&1 | tail -3

# Sandbox actors must be detached+named or the second node's rollout workers
# cannot find the rate limiter created by the first.
VERL_PKG=$(python -c "import verl,os; print(os.path.dirname(verl.__file__))")
# Detached+named sandbox actors, or node 2's rollout workers cannot find the
# rate limiter node 1 created. Upstream moved this tool out of verl/tools into
# the recipe (verl 5a506cc5 deleted verl/tools/sandbox_fusion_tools.py); the
# maintained file is recipe/retool/sandbox_fusion_tool.py, singular.
SF_PY=$V/recipe/retool/sandbox_fusion_tool.py
if [ -f "$SF_PY" ]; then
  sed -i 's|TokenBucketWorker\.options(name="rate-limiter", get_if_exists=True)|TokenBucketWorker.options(name="rate-limiter", get_if_exists=True, lifetime="detached", namespace="verl_sandbox")|g' "$SF_PY"
  sed -i 's|\.options(name="sandbox-execution-pool", get_if_exists=True, max_concurrency=num_workers)|.options(name="sandbox-execution-pool", get_if_exists=True, lifetime="detached", namespace="verl_sandbox", max_concurrency=num_workers)|g' "$SF_PY"
  echo "patched detached sandbox actors in $SF_PY"
else
  echo "WARN: $SF_PY missing - cross-node rate limiter will not be shared"
fi

# SandboxFusion's own runner needs the recipe's patch to work outside its docker image.
if [ -f "$V/recipe/retool/patch_sf_runner.py" ]; then
  python "$V/recipe/retool/patch_sf_runner.py" "$B/SandboxFusion" 2>&1 | tail -3
fi

# Job-unique port: a fixed :8080 gets held by a previous job's leftover server.
SF_PORT=$(( 8100 + ${SLURM_JOB_ID:-0} % 700 ))
fuser -k "${SF_PORT}/tcp" 2>/dev/null || true
SANDBOX_LOG=$B/logs/sandbox_${SLURM_JOB_ID}_$(hostname).log
nohup python -m uvicorn sandbox.server.server:app --host 0.0.0.0 --port "$SF_PORT" > "$SANDBOX_LOG" 2>&1 &
for i in $(seq 1 30); do
  curl -sf "http://localhost:${SF_PORT}/v1/ping" >/dev/null 2>&1 && break
  sleep 5
done
RESULT=$(curl -s -m 30 -X POST "http://localhost:${SF_PORT}/run_code" \
  -H "Content-Type: application/json" \
  -d '{"code":"print(1+1)","language":"python","run_timeout":10}')
echo "sandbox selftest: ${RESULT:0:200}"
case "$RESULT" in *'"2'*) echo "SANDBOX OK on $(hostname):${SF_PORT}";; *) echo "SANDBOX FAILED on $(hostname):${SF_PORT}"; tail -20 "$SANDBOX_LOG";; esac

# Shared tool config so BOTH nodes' rollout workers hit their local sandbox.
export SF_TOOL_CFG=$B/logs/sf_tool_${SLURM_JOB_ID}.yaml
sed "s|localhost:8080|localhost:${SF_PORT}|g" "$V/recipe/retool/sandbox_fusion_tool_config.yaml" > "$SF_TOOL_CFG"
echo "SF_TOOL_CFG=$SF_TOOL_CFG"
