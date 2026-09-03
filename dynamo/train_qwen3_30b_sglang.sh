#!/bin/bash
# Qwen3-30B retool RL through Dynamo + SGLang on 4 nodes -- the A/B partner of
# baseline_qwen3_30b_sglang_native.sh.
#
# Generated FROM that script by reverting only the rollout path (rollout.name, the
# engine_kwargs.dynamo.* block, the dynamo agent-loop manager, and
# VERL_USE_EXTERNAL_MODULES=recipe.dynamo.register). Everything else -- image, node
# orchestration, memory fixes, checkpoint/resume, wandb -- is byte-identical, so a
# comparison between the two isolates the rollout backend and nothing else.
#SBATCH --job-name=verl-dyn-sglang-i100
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=batch
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --cpus-per-task=112
#SBATCH --mem=1800G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/sopyang/verl/slurm/logs/train_30b_rl_dynamo_sglang_%j.log
#
# Qwen3-30B retool RL through Dynamo + SGLang — 4x8 H100.
#
# Env knobs (defaults = the verified 100-step run, job 17562481, pearson 0.9994):
#   TOTAL_STEPS=100            training steps
#   ENFORCE_EAGER=False        False = CUDA graph on (sglang: ~8x faster generation than eager)
#   DISABLE_PIECEWISE=0        1 adds --disable-piecewise-cuda-graph
#   STREAM_INTERVAL=           empty = engine default (only a workaround before the logprob patch)
#   VERL_DEFER_OPTIMIZER_LOAD=0  1 = Adam on CPU through fwd/bwd (needed on 2x8, not on 4x8)
#   USE_FUSED_KERNELS=0 FUSED_KERNELS_BACKEND=torch EAGER_EXPERTS=0 PPO_MAX_TOKEN_LEN=18432
#   ULYSSES_SP / ROLLOUT_TP / RESUME_MODE / RESET_CHECKPOINT / CONTAINER / WANDB_KEY_FILE
#   e.g. sbatch --export=ALL,TOTAL_STEPS=3 train_qwen3_30b_sglang.sh
#
# Derived from the proven Dynamo+vLLM run (now train_30b_rl_dynamo_kv_metrics.sh). Everything that is not engine-specific is kept byte-identical to that
# script: SBATCH block, Ray bring-up, sandbox-fusion, datasets, and every
# trainer/actor hyper-parameter. The deltas below are the ones the engine swap
# actually forces, each with the reason it cannot be avoided.
#
#   1. CONTAINER -> official verlai/verl:sgl0512.dev4.
#      The proven image (verl_vllm020.sqsh) has no SGLang. Do NOT try to add it
#      with the ai_dynamo[sglang] extra either: that extra hard-pins
#      transformers==5.8.1 and drags vLLM's guided-decoding stack down with it.
#      An image that already ships a self-consistent sglang+transformers set is
#      the only clean way in.
#
#   2. NO `pip install "transformers>=4.56,<5"`.
#      The proven script pins it because vLLM 0.20.2 rejects transformers 5.x.
#      SGLang 0.5.x requires transformers==5.8.1, so the pin is not merely
#      unnecessary here, it is unsatisfiable. Consequence to keep in mind: the
#      transformers 5.x Qwen3-MoE path uses grouped_mm experts, which needs more
#      activation memory than 4.x's looped path — hence _experts_implementation
#      below.
#
#   3. VERL_SRC_IN_CONTAINER -> /workspace/verl_dynamo/verl.
#      recipe/dynamo/dynamo_sglang_{engine,rollout}.py only exist in that
#      checkout. Entry point is verl.trainer.main_ppo with trainer.use_v1=False:
#      the V1 TaskRunner on this checkout (6cbca9ce) hands a TensorDict to
#      agent_loop.generate_sequences and dies with AttributeError, so we pin the
#      v0 TaskRunner. --config-path loads the recipe's dynamo_trainer config
#      (resolved relative to verl/trainer/, i.e. CWD-independent).
#
#   4. Dynamo wheels installed --no-deps from the prebuilt wheelhouse instead of
#      being compiled in-container. The wheels are portable (py3-none-any and
#      cp310-abi3) and --no-deps keeps the image's sglang/transformers exactly as
#      shipped. The proven script's in-container Rust build exists to produce
#      those same wheels; reusing them skips ~20 min of rustup/maturin per job.
#
#   5. Engine-specific hydra args: engine=sglang, sglang.enable_rl=true, and the
#      vLLM-only extra_args (--generation-config vllm, --stream-interval) dropped.
#      enable_rl is what registers call_tokenizer_manager, the only way to flush
#      the radix cache on this path (control/flush_cache returns 404).
#
#   6. transformers 5.x MoE memory. Consequence of (2): 5.x stores the experts as
#      one fused 3D tensor and picks grouped_mm, which roughly DOUBLES the actor
#      update's peak memory versus 4.x (35.6 -> 71.6 GB on this footprint). The
#      EAGER_EXPERTS=1 knob below trades that for a 39x slower update and is off;
#      on 4x8 H100 the fused path fits as is.
set -ex
WORKSPACE=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/sopyang
CONTAINER="${CONTAINER:-${WORKSPACE}/images/verl_sgl0512.dev4.sqsh}"
# WANDB_API_KEY is intentionally NOT hardcoded here (the proven script has a live
# key in plaintext). Export it before sbatch to enable the wandb logger; the
# driver falls back to console-only when it is unset.

echo "=== Qwen3-30B Dynamo+SGLang retool — multi-node (${SLURM_NNODES:-?}x8 H100) ==="
echo "Node: $(hostname), $(date -Iseconds)"
test -f "$CONTAINER" || { echo "[fatal] container not found: $CONTAINER" >&2; exit 2; }

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=( $nodes )
node_1=${nodes_array[0]}
port=6379
ip_head=$node_1:$port
export ip_head
echo "Ray head at $ip_head; workers on ${nodes_array[@]:1} (${#nodes_array[@]} nodes total)"

read -r -d '' BOOTSTRAP <<'BOOT' || true
set -ex
export PIP_CACHE_DIR=/workspace/.cache/pip
export HF_HOME=/workspace/.cache/huggingface
# The official verl images ship a PEP 668 "externally managed" Python, so every
# plain `pip install` aborts with externally-managed-environment. The proven
# script never hit this because verl_vllm020.sqsh predates the marker. We are
# installing into a throwaway container, so overriding is the intended escape.
export PIP_BREAK_SYSTEM_PACKAGES=1
# The sglang recipe (dynamo_sglang_rollout.py etc.) lives in the verl_dynamo
# checkout, not the one the proven vLLM script uses.
export VERL_SRC_IN_CONTAINER="${VERL_SRC_IN_CONTAINER:-/workspace/verl_dynamo/verl}"
export PYTHONPATH="${VERL_SRC_IN_CONTAINER}:${PYTHONPATH:-}"
export VERL_USE_EXTERNAL_MODULES="${VERL_USE_EXTERNAL_MODULES:-recipe.dynamo.register}"
echo "Container verl source: ${VERL_SRC_IN_CONTAINER}"
echo "VERL_USE_EXTERNAL_MODULES=${VERL_USE_EXTERNAL_MODULES}"

# Per-job cache root. The job id is load-bearing (added 2026-08-31): without it
# every sglang job on a node writes into the same /tmp tree and nothing ever
# reclaims it. Three runs on 2026-08-31 (17014038/17015852/17016074) filled the
# node-local /tmp, and the fourth died with a failure that names neither disk nor
# cache -- sglang's piecewise-cudagraph compiler raised
#   OSError: [Errno 28] No space left on device
# from backend.py, which the engine turned into SIGQUIT and the watchdog reported
# as "dynamo vllm_workers[1] exited rc=-9". Looks like an OOM kill, is a full disk.
JOB_CACHE_BASE=${VERL_NODE_CACHE_BASE:-/tmp/verl_${USER:-user}_sgl0512_$(hostname)_${SLURM_JOB_ID:-manual}}
# Reap caches from finished jobs on this node. Guarded by -mmin so a concurrently
# starting sibling job is never touched.
find /tmp -maxdepth 1 -name "verl_${USER:-user}_sgl0512_*" -mmin +180 \
  -not -path "$JOB_CACHE_BASE" -exec rm -rf {} + 2>/dev/null || true
df -h /tmp | tail -1
# HOME must move too (2026-08-31). The container mounts only ${WORKSPACE}:/workspace,
# so the default HOME=/root lives in the container's writable layer -- a few GB, shared
# with everything else the image writes. Dynamo's model-discovery cache does NOT honour
# XDG_CACHE_HOME: it writes $HOME/.cache/dynamo/mdc/ directly, so setting XDG alone left
# it on the small layer. Job 17017698 filled it and the frontend logged
#   Error adding model from discovery ... symlinking /root/.cache/dynamo/mdc/... :
#   No space left on device (os error 28)
# 260 times, never once "added model" -- so /v1/completions returned 404 for every
# rollout request even though the engines were up and weight sync had completed
# (32 ranks, buckets=57). The control plane (/engine/control/*, its own port) was fine;
# only the data plane never got a route. df -h /tmp reported 37T free the whole time,
# which is why the earlier "disk is full" reading looked wrong -- wrong filesystem.
export HOME=${JOB_CACHE_BASE}/home
mkdir -p "$HOME"
# Carry the image's credentials into the relocated HOME. wandb authenticates from
# ~/.netrc, NOT from WANDB_API_KEY here: the export below happens in the outer
# script process and never reaches the Ray-actor trainer, so every prior run was
# in fact authenticating via the image's /root/.netrc. Moving HOME without this
# copy broke that -- job 17018473 got all the way through engine start, frontend
# registration (16 models, 0x 404) and into the trainer, then died on
#   wandb.errors.errors.UsageError: No API key configured.
# Copy only credentials; /root/.cache is exactly what we are moving away from.
for _cred in .netrc .netrc.gpg; do
  [ -f "/root/$_cred" ] && cp -p "/root/$_cred" "$HOME/$_cred" 2>/dev/null || true
done
[ -d /root/.config/wandb ] && { mkdir -p "$HOME/.config"; cp -rp /root/.config/wandb "$HOME/.config/" 2>/dev/null || true; }
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${JOB_CACHE_BASE}/xdg}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-${JOB_CACHE_BASE}/torch_extensions}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${JOB_CACHE_BASE}/triton}
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-${JOB_CACHE_BASE}/cuda}
export FLASHINFER_CACHE_DIR=${FLASHINFER_CACHE_DIR:-${JOB_CACHE_BASE}/flashinfer}
export FLASHINFER_JIT_CACHE_DIR=${FLASHINFER_JIT_CACHE_DIR:-${FLASHINFER_CACHE_DIR}}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-${JOB_CACHE_BASE}/flashinfer_workspace}
export FLASHINFER_CUBIN_DIR=${FLASHINFER_CUBIN_DIR:-${JOB_CACHE_BASE}/flashinfer_cubins}
export VERL_DYNAMO_FE_READY_TIMEOUT=${VERL_DYNAMO_FE_READY_TIMEOUT:-1800}
# The proven vLLM script sets expandable_segments:True here to cut cross-step
# fragmentation. It MUST NOT be set on the sglang path: sglang's memory saver
# refuses to initialise under it —
#   RuntimeError: TorchMemorySaver is disabled for the current process because
#                 expandable_segments is not supported yet.   [job 16512212]
# and torch_memory_saver is exactly what release_memory_occupation/​resume use to
# hand GPU memory back to the trainer during the update. Losing the allocator
# tweak is survivable; losing memory release is not — the engine would sit on
# ~31 GB of weights per GPU through every training step.
# Unset rather than merely skip: the surrounding environment may already carry it.
unset PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF 2>/dev/null || true
export HYDRA_FULL_ERROR=1
# Adam state is ~15-30 GB/GPU and verl loads it on train-mode entry, so it sits on the
# card through all of forward_backward_batch even though only optimizer_step() needs
# it. Step 1 already peaks at ~72 GB of 79 with no optimizer state allocated (Adam is
# lazy until the first .step()), which is why every run so far cleared step 1 and OOMed
# in step 2's backward. Exported here, not in the driver, so both raylets carry it and
# the Ray workers inherit it.
# Off by default: the verified 4x8 run did not need it. Set 1 on 2x8.
export VERL_DEFER_OPTIMIZER_LOAD=${VERL_DEFER_OPTIMIZER_LOAD:-0}
echo "VERL_DEFER_OPTIMIZER_LOAD=${VERL_DEFER_OPTIMIZER_LOAD}"
unset ROCR_VISIBLE_DEVICES 2>/dev/null
mkdir -p "$HOME" "$HOME/.cache" "$PIP_CACHE_DIR" "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$FLASHINFER_CACHE_DIR" "$FLASHINFER_JIT_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE" "$FLASHINFER_CUBIN_DIR"

# Stale subprocesses from a failed run keep ZMQ/etcd/nats ports bound and make
# the next engine startup fail.
pkill -u "$(id -u)" -f "recipe.dynamo._dynamo_vllm_with_control" 2>/dev/null || true
pkill -u "$(id -u)" -f "python3 -m dynamo.frontend" 2>/dev/null || true
# Anchor on the module invocation, NOT the bare string "dynamo.sglang": pkill -f
# matches the whole argv and "." is a regex wildcard, so the loose pattern also
# matches this very script when its path contains "dynamo_sglang" — the bootstrap
# SIGTERMs itself and the job dies at ~70s with exit code 15 (job 16510322).
# The proven script documents the same hazard for its own patterns.
pkill -u "$(id -u)" -f "python3 -m dynamo\.sglang" 2>/dev/null || true
pkill -u "$(id -u)" -f "nats-server -p" 2>/dev/null || true
pkill -u "$(id -u)" -f "etcd --listen-client-urls" 2>/dev/null || true
pkill -u "$(id -u)" -f "uvicorn sandbox.server.server" 2>/dev/null || true
sleep 2

# Report, do not change, the image's own stack. Pinning transformers here would
# break sglang (it requires ==5.8.1).
python3 - <<'PY'
from importlib.metadata import version
for pkg in ("torch", "transformers", "sglang", "verl"):
    try:
        print(f"{pkg} {version(pkg)}")
    except Exception as exc:
        print(f"{pkg} MISSING ({exc})")
PY
python3 -c "import sglang; print('sglang import OK')" || { echo "[fatal] image has no sglang" >&2; exit 2; }
pip install setuptools sandbox-fusion 2>&1 | tail -1
# verl's wandb logger imports wandb at construction time; the sgl0512 image is not
# guaranteed to ship it, and a missing import would kill the run only after the ~10 min
# engine bring-up. Install it here rather than discover it late.
python3 -c "import wandb" 2>/dev/null || pip install wandb 2>&1 | tail -1
python3 -c "import wandb; print('wandb', wandb.__version__)" 2>&1 | tail -1

# --- Dynamo SDK: prebuilt portable wheels, --no-deps ------------------------ #
DYNAMO_SRC=${DYNAMO_SRC:-/workspace/dynamo}
export DYNAMO_SRC
test -d "$DYNAMO_SRC" || { echo "[fatal] local Dynamo source not found: $DYNAMO_SRC" >&2; exit 2; }
export PYTHONPATH="${DYNAMO_SRC}/components/src:${PYTHONPATH:-}"

DYNAMO_WHEELHOUSE=${DYNAMO_WHEELHOUSE:-/workspace/dynamo_wheels_1.3.0_94accc7389d4}
DYNAMO_RUNTIME_WHEEL=$(ls "$DYNAMO_WHEELHOUSE"/ai_dynamo_runtime-*.whl | sort | tail -1)
DYNAMO_API_WHEEL=$(ls "$DYNAMO_WHEELHOUSE"/ai_dynamo-*.whl | sort | tail -1)
python3 -m pip install --force-reinstall --no-deps "$DYNAMO_RUNTIME_WHEEL" "$DYNAMO_API_WHEEL" 2>&1 | tail -3

# Same overlay assertion as the proven script, against the sglang handler. The
# published wheel's handlers do not necessarily return completion_token_ids; when
# they do not, dynamo hands back text with no token ids, verl's
# _fallback_token_ids() substitutes a single EOS, and training silently runs on
# 1-token responses with every metric still looking healthy. Fail here instead.
# Installing the wheels --no-deps keeps the image's sglang/transformers intact but
# also skips dynamo's own runtime deps. blake3 is the one the sglang handler pulls
# in (via request_handlers/multimodal/encode_worker_handler.py); without it the
# engine worker dies with ModuleNotFoundError ~10 min into the job (16511576).
python3 -c "import blake3" 2>/dev/null || pip install blake3 2>&1 | tail -1

python3 - <<'PY'
import importlib, os
from importlib.metadata import version

print(f"ai-dynamo {version('ai-dynamo')}")
print(f"ai-dynamo-runtime {version('ai-dynamo-runtime')}")
import dynamo, dynamo.runtime, dynamo.llm

# Actually IMPORT the handler package rather than find_spec it. find_spec only
# locates the file, so a missing transitive dep (blake3) sails through here and
# resurfaces much later as a dead engine worker. Importing makes the bootstrap
# the place that fails, which is ~10 minutes earlier and names the real cause.
mod = importlib.import_module("dynamo.sglang.request_handlers.handler_base")
path = mod.__file__
print(f"dynamo.sglang handler_base: {path}")
want = os.path.join(os.environ["DYNAMO_SRC"], "components", "src", "dynamo")
if not path.startswith(want + "/"):
    raise SystemExit(f"expected local Dynamo overlay under {want}, got {path}")
src = open(path, encoding="utf-8").read()
for route in ("update_weights_from_tensor", "release_memory_occupation", "resume_memory_occupation"):
    if route not in src:
        raise SystemExit(f"sglang handler missing RL control route: {route}")
print("dynamo.sglang RL control routes present")

# Import the worker entry point too: that is the module the engine subprocess
# actually runs, so anything missing from its import graph fails here instead.
importlib.import_module("dynamo.sglang.main")
print("dynamo.sglang.main importable")
PY
python3 -c "import zmq" 2>/dev/null || pip install pyzmq 2>&1 | tail -1
python3 -c "import aiohttp" 2>/dev/null || pip install aiohttp 2>&1 | tail -1

# Static etcd + nats-server, cached under the shared workspace.
DYN_BIN_DIR=/workspace/dynamo_bin
mkdir -p "$DYN_BIN_DIR"
ETCD_VER=v3.5.21
NATS_VER=v2.10.22
if [[ ! -x "$DYN_BIN_DIR/etcd" ]]; then
    cd /tmp
    curl -sSL "https://github.com/etcd-io/etcd/releases/download/${ETCD_VER}/etcd-${ETCD_VER}-linux-amd64.tar.gz" | tar xz
    install -m 755 "etcd-${ETCD_VER}-linux-amd64/etcd"    "$DYN_BIN_DIR/etcd"
    install -m 755 "etcd-${ETCD_VER}-linux-amd64/etcdctl" "$DYN_BIN_DIR/etcdctl"
    rm -rf "etcd-${ETCD_VER}-linux-amd64"*
    cd -
fi
if [[ ! -x "$DYN_BIN_DIR/nats-server" ]]; then
    cd /tmp
    curl -sSL "https://github.com/nats-io/nats-server/releases/download/${NATS_VER}/nats-server-${NATS_VER}-linux-amd64.tar.gz" | tar xz
    install -m 755 "nats-server-${NATS_VER}-linux-amd64/nats-server" "$DYN_BIN_DIR/nats-server"
    rm -rf "nats-server-${NATS_VER}-linux-amd64"*
    cd -
fi
export PATH="$DYN_BIN_DIR:$PATH"
etcd --version | head -1
nats-server --version

# Weight-sync bucket bump, same as the proven run.
VERL_PKG_DIR=$(python3 -c "import verl,os; print(os.path.dirname(verl.__file__))")
grep -rl "bucket_size_mb" "$VERL_PKG_DIR" --include="*.py" 2>/dev/null | xargs -r sed -i "s/bucket_size_mb=2048/bucket_size_mb=4096/g; s/bucket_size_mb: int = 512/bucket_size_mb: int = 4096/g"

# Detached+named sandbox actors: without lifetime=detached the rate limiter dies
# with whichever ExecutionWorker created it, and node 2's rollout workers then
# fail with ActorDiedError. Upstream moved this file out of verl/tools into the
# recipe, so patch whichever copy exists.
for SF_PY in "$VERL_PKG_DIR/tools/sandbox_fusion_tools.py" \
             "${VERL_SRC_IN_CONTAINER}/verl/tools/sandbox_fusion_tools.py" \
             "${VERL_SRC_IN_CONTAINER}/recipe/retool/sandbox_fusion_tool.py"; do
  if [ -f "$SF_PY" ]; then
    sed -i "s|TokenBucketWorker\.options(name=\"rate-limiter\", get_if_exists=True)|TokenBucketWorker.options(name=\"rate-limiter\", get_if_exists=True, lifetime=\"detached\", namespace=\"verl_sandbox\")|g" "$SF_PY"
    sed -i "s|\\.options(name=\"sandbox-execution-pool\", get_if_exists=True, max_concurrency=num_workers)|.options(name=\"sandbox-execution-pool\", get_if_exists=True, lifetime=\"detached\", namespace=\"verl_sandbox\", max_concurrency=num_workers)|g" "$SF_PY"
    echo "Patched detached SandboxFusion actors in $SF_PY"
  fi
done

cd /tmp && rm -rf symeval && git clone -q https://github.com/tongyx361/symeval.git && cd symeval
sed -i "s/from pkg_resources import parse_version/from packaging.version import parse as parse_version/" setup.py
pip install . 2>&1 | tail -1
pip install "antlr4-python3-runtime==4.9.3" 2>&1 | tail -1

export VERL_DYNAMO_LOG_DIR=/workspace/dynamorl_workspace/logs/dynamo_logs/${SLURM_JOB_ID:-manual}
mkdir -p "$VERL_DYNAMO_LOG_DIR"
python3 -c "import verl; print(f'verl {verl.__version__}')"

# === LOCAL sandbox-fusion server on this node =============================== #
echo "=== Setting up sandbox-fusion server on $(hostname) ==="
cd /tmp && rm -rf SandboxFusion
# Prefer the lustre clone (compute nodes are not guaranteed outbound network);
# fall back to a fresh clone, as the proven script does.
if [ -d /workspace/SandboxFusion ]; then
  cp -r /workspace/SandboxFusion /tmp/SandboxFusion
else
  git clone --depth 1 https://github.com/bytedance/SandboxFusion.git /tmp/SandboxFusion 2>&1 | tail -2
fi
cd /tmp/SandboxFusion
# By NAME and UNPINNED: SandboxFusion's pyproject pins pydantic<2.7 and
# transformers^4.44, which would downgrade the image's stack and break sglang.
pip install tenacity structlog psutil aiofiles aiohttp "databases[aiomysql,aiosqlite]" 2>&1 | tail -3
if [ -f "${VERL_SRC_IN_CONTAINER}/recipe/retool/patch_sf_runner.py" ]; then
  python3 "${VERL_SRC_IN_CONTAINER}/recipe/retool/patch_sf_runner.py" /tmp/SandboxFusion
fi
export PYTHONPATH=/tmp/SandboxFusion:${PYTHONPATH:-}

mkdir -p /workspace/dynamorl_workspace/logs /workspace/dynamorl_workspace/slurm
SF_PORT=$(( 8100 + ${SLURM_JOB_ID:-0} % 700 ))
fuser -k "${SF_PORT}/tcp" 2>/dev/null || true
for _p in $(ss -ltnHp "sport = :${SF_PORT}" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u); do kill -9 "$_p" 2>/dev/null || true; done
SANDBOX_LOG=/workspace/dynamorl_workspace/logs/sandbox_fusion_${SLURM_JOB_ID:-manual}_$(hostname).log
nohup python3 -m uvicorn sandbox.server.server:app --host 0.0.0.0 --port "$SF_PORT" > "$SANDBOX_LOG" 2>&1 &
disown
sleep 8
curl -sf "http://localhost:${SF_PORT}/v1/ping" && echo " Sandbox-Fusion OK on $(hostname):${SF_PORT}" || echo "Sandbox-Fusion FAILED on $(hostname):${SF_PORT}"

SF_TOOL_CFG=/workspace/dynamorl_workspace/slurm/sf_tool_sglang_${SLURM_JOB_ID:-manual}.yaml
sed "s|localhost:8080|localhost:${SF_PORT}|g" "${VERL_SRC_IN_CONTAINER}/recipe/retool/sandbox_fusion_tool_config.yaml" > "$SF_TOOL_CFG"
RESULT=$(curl -s -X POST "http://localhost:${SF_PORT}/run_code" -H "Content-Type: application/json" -d "{\"code\":\"print(1+1)\",\"language\":\"python\",\"run_timeout\":10}")
echo "Quick test: $RESULT"
BOOT

# Driver — runs ONLY on node_1, after the Ray cluster is up.
read -r -d '' DRIVER <<'DRV' || true
set -ex
rm -rf /tmp/verl_ckpt

cd "$VERL_SRC_IN_CONTAINER"
export DYNAMO_SRC="${DYNAMO_SRC:-/workspace/dynamo}"
export PYTHONPATH="$DYNAMO_SRC/components/src:$VERL_SRC_IN_CONTAINER":/tmp/SandboxFusion:${PYTHONPATH:-}
export VERL_USE_EXTERNAL_MODULES="${VERL_USE_EXTERNAL_MODULES:-recipe.dynamo.register}"
echo "VERL_USE_EXTERNAL_MODULES=${VERL_USE_EXTERNAL_MODULES}"

project_name=retool_30b_dynamo_sglang
experiment_name=${EXPERIMENT_NAME:-e2_30b_dynamo_sglang_4n_${SLURM_JOB_ID:-manual}}
# No ${SLURM_JOB_ID} suffix: a 50-step campaign spans several 4h jobs, and resume_mode=auto
# looks for global_step_* under this exact path. experiment_name already carries the job id
# when EXPERIMENT_NAME is unset, so one-off runs still get their own directory.
default_local_dir=/workspace/dynamorl_workspace/checkpoint/${experiment_name}
if [[ "${RESET_CHECKPOINT:-0}" == "1" ]]; then
  rm -rf "$default_local_dir"
fi
mkdir -p "$default_local_dir"

unset VERL_ROLLOUT_PROMPT_LOG_PATH

PROM_PORT=$(( 9200 + ${SLURM_JOB_ID:-0} % 500 ))
PROM_FILE=/workspace/dynamorl_workspace/slurm/prometheus_dynamo_sglang_${SLURM_JOB_ID:-manual}.yml
export VERL_DYNAMO_WORKER_METRICS_DIR=/workspace/dynamorl_workspace/slurm/worker_metrics_sglang_${SLURM_JOB_ID:-manual}
rm -rf "$VERL_DYNAMO_WORKER_METRICS_DIR"
mkdir -p "$VERL_DYNAMO_WORKER_METRICS_DIR"

SIDECAR_JSONL=/workspace/dynamorl_workspace/slurm/metrics_snapshots_sglang_${SLURM_JOB_ID:-manual}.jsonl
nohup python3 recipe/dynamo/metrics_sidecar.py --targets-glob "$PROM_FILE" \
  --endpoints-glob "$VERL_DYNAMO_WORKER_METRICS_DIR/*.endpoints" \
  --output "$SIDECAR_JSONL" --label dynamo_sglang --interval 30 \
  > /workspace/dynamorl_workspace/slurm/sidecar_sglang_${SLURM_JOB_ID:-manual}.log 2>&1 &
disown
echo "metrics sidecar started -> ${SIDECAR_JSONL}"

# `set -x` is active here, so anything that touches WANDB_API_KEY on a traced line
# prints the key into a group-readable log on lustre. Read it with tracing off, from a
# 0600 file, and never echo the value -- only whether one was found.
set +x
WANDB_KEY_FILE="${WANDB_KEY_FILE:-/workspace/.wandb_key}"
if [ -z "${WANDB_API_KEY:-}" ] && [ -r "$WANDB_KEY_FILE" ]; then
  WANDB_API_KEY="$(tr -d '\r\n' < "$WANDB_KEY_FILE")"
  export WANDB_API_KEY
fi
if [ -n "${WANDB_API_KEY:-}" ]; then
  TRAINER_LOGGER="[\"console\",\"wandb\"]"
  # Compute nodes are not guaranteed egress. wandb's online mode retries on a blocking
  # path, so an unreachable endpoint would stall the run rather than just lose metrics.
  # Offline still records everything locally for a later `wandb sync`.
  if ! curl -sf -o /dev/null --max-time 10 https://api.wandb.ai/ 2>/dev/null; then
    export WANDB_MODE="${WANDB_MODE:-offline}"
    echo "wandb: api.wandb.ai unreachable -> WANDB_MODE=$WANDB_MODE (sync later with 'wandb sync')"
  else
    export WANDB_MODE="${WANDB_MODE:-online}"
    echo "wandb: reachable, WANDB_MODE=$WANDB_MODE"
  fi
  export WANDB_DIR="${WANDB_DIR:-/workspace/dynamorl_workspace/slurm/wandb_${SLURM_JOB_ID:-manual}}"
  mkdir -p "$WANDB_DIR"
  # Segments of one campaign must append to a single wandb run, not create three.
  export WANDB_RUN_ID="${WANDB_RUN_ID:-$experiment_name}"
  export WANDB_RESUME="${WANDB_RESUME:-allow}"
  echo "wandb: key loaded (${#WANDB_API_KEY} chars), project=$project_name run=$experiment_name dir=$WANDB_DIR"
else
  TRAINER_LOGGER="[\"console\"]"
  echo "wandb: no key at $WANDB_KEY_FILE and WANDB_API_KEY unset -> console logger only"
fi
set -x

dapo_math_17k=/workspace/dynamorl_workspace/datasets/BytedTsinghua-SIA/DAPO-Math-17k
aime_2025=/workspace/dynamorl_workspace/datasets/yentinglin/aime_2025

# Rollout dump: chasing a degenerate response_length through aggregate metrics
# alone produced three wrong hypotheses in a row on this stack; one raw sample
# settles it immediately.
ROLLOUT_DUMP=/workspace/dynamorl_workspace/slurm/rollouts_sglang_${SLURM_JOB_ID:-manual}
mkdir -p "$ROLLOUT_DUMP"

# NB ppo_max_token_len_per_gpu is multiplied by ulysses_sequence_parallel_size
# inside verl (workers/engine/utils.py: max_token_len = per_gpu * sp_size), so
# 18432 * 4 = 73728 tokens per micro-batch — matching log_prob's 73728, which is
# NOT multiplied because the ref path runs at sp=1. Both numbers are copied from
# the proven run; 18432 itself is the floor (max_prompt + max_response), since a
# micro-batch must hold at least one whole sequence.
# use_fused_kernels routes the LM head through FusedLinearForPPO, which walks the
# sequence in 512-token chunks and never materialises the full [T, 151936] logits
# (nor its fp32 upcast). That tensor is what blew up step 1 here: job 16513027 died
# in prepare_model_outputs trying to allocate 2.79 GiB with 69.98 GiB already held.
# Qwen3-MoE has no dedicated fused forward, so it takes the generic dense_common one
# via monkey_patch's else-branch. Off by default: it changes how log_probs/entropy
# are computed, so a run that uses it is not numerically identical to one that does not.
FUSED_ARGS=()
# _experts_implementation=eager forces the 128-expert Python loop (48 layers x 128,
# paid again on every gradient-checkpoint recompute) instead of transformers 5.x's
# batched grouped_mm. It was added here to cut activation memory, but job 16442420
# ran grouped_mm on transformers 5.8.1 with update_actor=52s and a 72.13 GB peak,
# while this eager path measured 2810s in job 16517408 -- a 54x tax for memory that
# use_fused_kernels now saves more cheaply. Default off; set EAGER_EXPERTS=1 to restore.
if [[ "${EAGER_EXPERTS:-0}" == "1" ]]; then
  FUSED_ARGS+=( "+actor_rollout_ref.model.override_config._experts_implementation=eager" )
fi
if [[ "${USE_FUSED_KERNELS:-0}" == "1" ]]; then
  FUSED_ARGS+=( actor_rollout_ref.model.use_fused_kernels=True )
  FUSED_ARGS+=( "++actor_rollout_ref.model.fused_kernel_options.impl_backend=${FUSED_KERNELS_BACKEND:-torch}" )
fi
echo "FUSED_ARGS: ${FUSED_ARGS[*]:-<none>}"

# --disable-piecewise-cuda-graph（2026-08-31）：sglang 的 piecewise-cudagraph 编译器
# 把图写盘，两次跑（17016074/17016916）都在这里 OSError: [Errno 28] No space left。
# 注意 `df -h /tmp` 当时报 37T 可用，所以写入目标并不是 /tmp，而是编译器自己的路径；
# 与其继续找那个路径，不如按 sglang 自己的报错建议关掉它 —— 本脚本已经
# enforce_eager=True / disable_cuda_graph=True，本来就不打算用 cuda graph，
# piecewise 是一个独立的开关，之前漏掉了。
# stream_interval 实测无效，已撤除（2026-08-27）：dynamo 侧尾部饱和层
# interval 1 -> 100 的 per_token 均值变化 2.6981 -> 2.6982。注入方式本身是对的
# (dynamo 走 extra_args 进 CLI，native 走 engine_kwargs.sglang 进 ServerArgs)，
# 两侧都验证过真的到达引擎，所以这不是"设了没生效"。详见 HANDOFF.md 8.4。

# sglang engine extra CLI args, assembled as a JSON array for hydra.
# STREAM_INTERVAL (2026-08-31): the Dynamo frontend aggregates streamed chunks back
# into one non-streaming response, but keeps only the FIRST chunk's logprobs while
# token ids accumulate correctly -- measured live on job 4803: 7 usable logprobs out
# of 1647 tokens (0.425%), the rest padded to 0.0 which verl reads as probability
# 1.0. That is the whole of the pearson=0.05 / probs_diff=0.27 anomaly (HANDOFF 20).
# Setting the interval to the full response length makes the response a single chunk,
# so nothing is left to drop. Empty by default = engine default (1).
# DISABLE_PIECEWISE=1 adds --disable-piecewise-cuda-graph. It was added to dodge the
# compiler's ENOSPC (HANDOFF 12), which turned out to be the container writable layer
# filling up and is fixed by relocating HOME (HANDOFF 14). The verified 100-step run
# (17562481) ran with it OFF, so 0 is the default.
SGLANG_EXTRA=()
if [[ "${DISABLE_PIECEWISE:-0}" == "1" ]]; then
  SGLANG_EXTRA+=( "--disable-piecewise-cuda-graph" )
fi
if [[ -n "${STREAM_INTERVAL:-}" ]]; then
  SGLANG_EXTRA+=( "--stream-interval=${STREAM_INTERVAL}" )
fi
if (( ${#SGLANG_EXTRA[@]} )); then
  SGLANG_EXTRA_JSON=$(printf '"%s",' "${SGLANG_EXTRA[@]}"); SGLANG_EXTRA_JSON=${SGLANG_EXTRA_JSON%,}
else
  SGLANG_EXTRA_JSON=""
fi
echo "SGLANG_EXTRA_JSON: ${SGLANG_EXTRA_JSON}"

python3 -m verl.trainer.main_ppo \
    --config-path ../../recipe/dynamo/config --config-name dynamo_trainer \
    trainer.use_v1=False \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files="[\"$dapo_math_17k\"]" \
    data.val_files="[\"$aime_2025\"]" \
    data.return_raw_chat=True \
    data.train_batch_size=16 \
    data.max_prompt_length=2048 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.custom_cls.path=recipe/retool/retool.py \
    data.custom_cls.name=CustomRLHFDataset \
    reward.custom_reward_function.path=recipe/retool/retool.py \
    reward.custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=/workspace/hf_models/Qwen3-30B-A3B-Base \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN:-18432} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ULYSSES_SP:-4} \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=73728 \
    actor_rollout_ref.rollout.name=dynamo \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.engine=sglang \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.sglang.enable_rl=true \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.return_tokens_as_token_ids=false \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.request_engine_data=true \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.request_completion_token_ids=true \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.request_timeout_s=1800 \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.router_mode=kv \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.enable_nemo_router_tuning=true \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.enable_worker_system_metrics=true \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.stable_kv_event_ports=true \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.free_engine_on_train=true \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.thunderagent.enabled=false \
    ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=recipe.dynamo.dynamo_agent_loop.DynamoAgentLoopManager \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP:-2} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=8 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=8 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$SF_TOOL_CFG" \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=30 \
    actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER:-False} \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.prometheus.enable=True \
    actor_rollout_ref.rollout.prometheus.port=$PROM_PORT \
    actor_rollout_ref.rollout.prometheus.file="$PROM_FILE" \
    "++actor_rollout_ref.rollout.engine_kwargs.dynamo.extra_args=[${SGLANG_EXTRA_JSON}]" \
    trainer.logger="$TRAINER_LOGGER" \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=False \
    trainer.nnodes=4 \
    trainer.save_freq=${SAVE_FREQ:--1} \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.max_critic_ckpt_to_keep=2 \
    trainer.default_local_dir="$default_local_dir" \
    trainer.rollout_data_dir="$ROLLOUT_DUMP" \
    trainer.resume_mode=${RESUME_MODE:-disable} \
    trainer.test_freq=${TEST_FREQ:-999} \
    trainer.total_training_steps=${TOTAL_STEPS:-100} \
    "${FUSED_ARGS[@]}"

echo ">>> Sandbox-Fusion log (last 20 lines):"
tail -20 "$SANDBOX_LOG" 2>/dev/null
DRV

# Materialise per-role scripts and run them as FILES, not `bash -c "<text>"`:
# the bootstrap contains pkill patterns, and with -c that text lands in the bash
# process's own argv, so pkill -f matches and SIGTERMs the script itself.
RUN_DIR_HOST="${WORKSPACE}/verl/slurm/run_dynsgl_${SLURM_JOB_ID:-manual}"
RUN_DIR_CTR="/workspace/verl/slurm/run_dynsgl_${SLURM_JOB_ID:-manual}"
mkdir -p "$RUN_DIR_HOST"

{
  printf '%s\n' "$BOOTSTRAP"
  echo 'cd "$VERL_SRC_IN_CONTAINER"'
  echo "ray start --address=$ip_head --block"
} > "$RUN_DIR_HOST/worker.sh"

{
  printf '%s\n' "$BOOTSTRAP"
  echo 'cd "$VERL_SRC_IN_CONTAINER"'
  echo "ray start --head --node-ip-address=$node_1 --port=$port"
  echo "sleep 5"
  echo "ray status"
  echo "export RAY_ADDRESS=$ip_head"
  printf '%s\n' "$DRIVER"
} > "$RUN_DIR_HOST/head.sh"

# 1. Ray workers on every node except the head (background; each joins after its own
#    bootstrap). Looping matters: the original hardcoded node_2, so on a >2-node
#    allocation the extra nodes sat idle while trainer.nnodes demanded their GPUs.
for ((i = 1; i < ${#nodes_array[@]}; i++)); do
  srun --overlap --nodes=1 --ntasks=1 -w "${nodes_array[$i]}" \
    --container-image="$CONTAINER" --container-mounts="${WORKSPACE}:/workspace" \
    bash "$RUN_DIR_CTR/worker.sh" &
done
sleep 30

# 2. Ray head + driver on node_1.
srun --overlap --nodes=1 --ntasks=1 -w "$node_1" \
  --container-image="$CONTAINER" --container-mounts="${WORKSPACE}:/workspace" \
  bash "$RUN_DIR_CTR/head.sh"

echo "=== Done $(date -Iseconds) ==="
