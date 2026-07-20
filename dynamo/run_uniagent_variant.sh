#!/bin/bash
# Launch one arm of the uni-agent SWE x ThunderAgent benchmark.
#
# Usage:
#   VARIANT=global|dynamo|ta RUN_NAME=<name> [STEPS=3] [BATCH=32] [N=8] \
#   [MAX_TURNS=50] [CONCURRENCY=256] [MAX_NUM_SEQS=128] [GATE=0] \
#   [ROLLOUT_ONLY=0] [DYNAMO_REQUEST_TIMEOUT_S=1800] \
#   [MODEL_DIR=models/Qwen3-Coder-30B-A3B-Instruct] bash run_uniagent_variant.sh
#
# GATE=1 shrinks to a 1-step correctness gate (BATCH=8 N=2 unless overridden).
set -euo pipefail

BENCH=${BENCH:?set BENCH to this reproduction package directory}
VERL_WT=${VERL_WT:?set VERL_WT to the Verl worktree}
RECIPE_WT=${RECIPE_WT:?set RECIPE_WT to the verl-recipe worktree}
UNIAGENT=${UNIAGENT:?set UNIAGENT to the uni-agent source checkout}
IMAGE=${IMAGE:-verl-recipe-thunderagent:pr11185-fa2.8.3-uni-agent-docker}
DYNAMO_WT=${DYNAMO_WT:-}
DOCKER_GID=$(stat -c '%g' /var/run/docker.sock)
NETWORK=${NETWORK:-uniagent-bench}

VARIANT=${VARIANT:?global|dynamo|ta}
RUN_NAME=${RUN_NAME:?}

# Temporary benchmark queue hold: the current priority is the matched
# GlobalLoadBalancer vs ThunderAgent comparison.  Do not create a run directory
# for Dynamo KV Router arms, so they can be resumed cleanly later.
if [ "$VARIANT" = "dynamo" ] && [ "${ALLOW_DYNAMO_KV:-0}" != "1" ]; then
  echo "Dynamo KV Router arm is on hold: $RUN_NAME"
  exit 0
fi

GATE=${GATE:-0}
ROLLOUT_ONLY=${ROLLOUT_ONLY:-0}
DYNAMO_REQUEST_TIMEOUT_S=${DYNAMO_REQUEST_TIMEOUT_S:-1800}
STEPS=${STEPS:-3}
BATCH=${BATCH:-32}
N=${N:-8}
if [ "$GATE" = "1" ]; then STEPS=1; BATCH=${BATCH_OVERRIDE:-8}; N=${N_OVERRIDE:-2}; fi
if [ "$ROLLOUT_ONLY" = "1" ]; then STEPS=1; fi
MAX_TURNS=${MAX_TURNS:-50}
CONCURRENCY=${CONCURRENCY:-256}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR to Qwen3-Coder-30B-A3B-Instruct}
DATA=${DATA:-/bench/uni-agent/data/swe_bench_verified_local32.parquet}
MAX_PROMPT=${MAX_PROMPT:-4096}
MAX_RESP=${MAX_RESP:-28672}
MAX_MODEL_LEN=$((MAX_PROMPT + MAX_RESP))
TEMP=${TEMP:-0.8}
TOP_P=${TOP_P:-0.9}
DYNAMO_SESSION_AFFINITY=${DYNAMO_SESSION_AFFINITY:-$( [ "$VARIANT" = dynamo ] && echo 1 || echo 0 )}
DYNAMO_SESSION_AFFINITY_TTL_S=${DYNAMO_SESSION_AFFINITY_TTL_S:-7200}
SESSION_AFFINITY_OVERLAY=${SESSION_AFFINITY_OVERLAY:-$BENCH/uniagent-concurrency-sweep-20260713/dynamo_async_server_session_affinity.py}

RUN_DIR=$BENCH/runs/$RUN_NAME
if [ -e "$RUN_DIR/command.txt" ]; then
  echo "Run directory already contains command.txt: $RUN_DIR" >&2
  exit 4
fi
mkdir -p "$RUN_DIR" "$RUN_DIR/agent-logs" "$RUN_DIR/home"
chmod -R 777 "$RUN_DIR"
docker network inspect "$NETWORK" >/dev/null 2>&1 || docker network create "$NETWORK"

# Refuse to start while other benchmark trainers hold the GPUs.
if docker ps --format '{{.Names}}' | grep -qE 'retool32-|uniagent-swe'; then
  echo "GPUs busy (retool32/uniagent container running); aborting launch." >&2
  exit 3
fi

# Render an agent config with this run's turn/concurrency/parser knobs.
TOOL_PARSER=${TOOL_PARSER:-qwen3_coder}
sed -e "s/max_turns: 50/max_turns: $MAX_TURNS/" \
    -e "s/concurrency: 256/concurrency: $CONCURRENCY/" \
    -e "s/tool_parser: qwen3_coder/tool_parser: $TOOL_PARSER/" \
    "$BENCH/uni-agent/agent_config_local_swe.yaml" > "$RUN_DIR/agent_config.yaml"

CONTAINER_PYTHONPATH=/workspace/verl:/workspace:/workspace/uni-agent:/bench
if [ -n "$DYNAMO_WT" ]; then
  CONTAINER_PYTHONPATH=/workspace/dynamo/components/src:$CONTAINER_PYTHONPATH
fi
if [ "$ROLLOUT_ONLY" = "1" ]; then
  CONTAINER_PYTHONPATH=/run:$CONTAINER_PYTHONPATH
fi

COMMON_ENV=(
  -e HOME=/run/home
  -e PYTHONPATH="$CONTAINER_PYTHONPATH"
  -e PYTHONDONTWRITEBYTECODE=1
  -e TOKENIZERS_PARALLELISM=true
  -e RAY_DEDUP_LOGS=0
  -e VERL_FILE_LOGGER_PATH=/run/metrics.jsonl
)
MOUNTS=(
  -v /var/run/docker.sock:/var/run/docker.sock
  -v "$VERL_WT":/workspace/verl:ro
  -v "$RECIPE_WT":/workspace/recipe:ro
  -v "$UNIAGENT":/workspace/uni-agent:ro
  -v "$BENCH":/bench:ro
  -v "$RUN_DIR":/run
  -v "$MODEL_DIR":/model:ro
)
if [ -n "$DYNAMO_WT" ]; then
  MOUNTS+=(-v "$DYNAMO_WT":/workspace/dynamo:ro)
fi
if [ "$VARIANT" = dynamo ] && [ "$DYNAMO_SESSION_AFFINITY" = 1 ]; then
  if [ ! -f "$SESSION_AFFINITY_OVERLAY" ]; then
    echo "Missing Dynamo session-affinity overlay: $SESSION_AFFINITY_OVERLAY" >&2
    exit 5
  fi
  MOUNTS+=(-v "$SESSION_AFFINITY_OVERLAY":/workspace/recipe/dynamo/dynamo_async_server.py:ro)
fi
if [ "$ROLLOUT_ONLY" = "1" ]; then
  COMMON_ENV+=(
    -e ROLLOUT_BENCHMARK_AUTO_INSTALL=1
    -e ROLLOUT_BENCHMARK_PATH=/run/rollout-benchmark.json
  )
  cp "$BENCH/uni-agent/rollout_benchmark_entry.py" "$RUN_DIR/rollout_benchmark_entry.py"
fi

COMMON_ARGS=(
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=false
  algorithm.kl_ctrl.kl_coef=0.0
  data.train_files="$DATA"
  data.val_files="$DATA"
  data.return_raw_chat=true
  data.train_batch_size="$BATCH"
  data.max_prompt_length="$MAX_PROMPT"
  data.max_response_length="$MAX_RESP"
  data.filter_overlong_prompts=true
  data.filter_overlong_prompts_workers=1
  data.truncation=error
  data.shuffle=false
  actor_rollout_ref.model.path=/model
  actor_rollout_ref.model.use_remove_padding=true
  actor_rollout_ref.model.enable_gradient_checkpointing=true
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.use_kl_loss=false
  actor_rollout_ref.actor.kl_loss_coef=0.0
  actor_rollout_ref.actor.use_dynamic_bsz=true
  actor_rollout_ref.actor.ppo_mini_batch_size="$BATCH"
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$MAX_MODEL_LEN"
  actor_rollout_ref.actor.ulysses_sequence_parallel_size="${SP:-8}"
  actor_rollout_ref.actor.fsdp_config.param_offload=true
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.tensor_model_parallel_size=4
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_UTIL:-0.75}"
  actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN"
  actor_rollout_ref.rollout.max_num_batched_tokens=32768
  actor_rollout_ref.rollout.max_num_seqs="$MAX_NUM_SEQS"
  actor_rollout_ref.rollout.disable_log_stats=false
  actor_rollout_ref.rollout.temperature="$TEMP"
  actor_rollout_ref.rollout.top_p="$TOP_P"
  actor_rollout_ref.rollout.n="$N"
  actor_rollout_ref.rollout.agent.num_workers=8
  actor_rollout_ref.rollout.agent.agent_loop_config_path=/run/agent_config.yaml
  actor_rollout_ref.rollout.multi_turn.enable=true
  trainer.logger='[console,file]'
  trainer.project_name=verl-ta-uniagent
  trainer.experiment_name="$RUN_NAME"
  trainer.nnodes=1
  trainer.n_gpus_per_node=8
  trainer.val_before_train=$( [ "$ROLLOUT_ONLY" = "1" ] && echo true || echo false )
  trainer.test_freq=-1
  trainer.save_freq=-1
  trainer.resume_mode=disable
  trainer.default_local_dir=/run/checkpoints
  trainer.rollout_data_dir=/run/rollout_data
  trainer.total_epochs="$STEPS"
  trainer.total_training_steps="$STEPS"
  hydra.run.dir=/run/hydra
)

if [ "$ROLLOUT_ONLY" = "1" ]; then
  COMMON_ARGS+=(
    data.val_batch_size="$BATCH"
    data.val_max_samples="$BATCH"
    actor_rollout_ref.rollout.val_kwargs.n="$N"
    actor_rollout_ref.rollout.val_kwargs.temperature="$TEMP"
    actor_rollout_ref.rollout.val_kwargs.top_p="$TOP_P"
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.do_sample=true
    trainer.val_only=true
  )
fi

# Explicit per-GPU KV bytes sidestep vLLM's memory-profiling race with actor
# memory release (known phase-1 issue). Empty = use gpu_memory_utilization.
KV_BYTES=${KV_BYTES:-}

case "$VARIANT" in
  global)
    TARGET_MODULE=verl.trainer.main_ppo
    EXTRA_ENV=()
    if [ "$ROLLOUT_ONLY" = "1" ]; then
      EXTRA_ENV+=(-e VERL_USE_EXTERNAL_MODULES=rollout_benchmark_entry)
    fi
    EXTRA_ARGS=(actor_rollout_ref.rollout.name=vllm)
    if [ -n "$KV_BYTES" ]; then
      EXTRA_ARGS+=(+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes="$KV_BYTES")
    fi
    ;;
  dynamo|ta)
    TARGET_MODULE=recipe.dynamo.main_dynamo
    NS=$(echo "$RUN_NAME" | tr '-' '_')
    EXTERNAL_MODULES=recipe.dynamo.register
    if [ "$ROLLOUT_ONLY" = "1" ]; then
      EXTERNAL_MODULES=$EXTERNAL_MODULES,rollout_benchmark_entry
    fi
    EXTRA_ENV=(
      -e VERL_USE_EXTERNAL_MODULES="$EXTERNAL_MODULES"
      -e VERL_DYNAMO_LOG_DIR=/run/dynamo
      -e VERL_DYNAMO_WORKER_METRICS_DIR=/run/worker_metrics
      -e VERL_DYNAMO_KV_EVENT_PORT_BASE=24000
      -e VERL_DYNAMO_REFIT_STRICT=1
      -e VLLM_LOGGING_LEVEL=INFO
    )
    TA_ENABLED=false
    [ "$VARIANT" = "ta" ] && TA_ENABLED=true
    EXTRA_ARGS=(
      actor_rollout_ref.rollout.engine_kwargs.dynamo.thunderagent.enabled=$TA_ENABLED
      actor_rollout_ref.rollout.engine_kwargs.dynamo.thunderagent.router_block_size=16
      ++actor_rollout_ref.rollout.engine_kwargs.dynamo.request_engine_data=true
      ++actor_rollout_ref.rollout.engine_kwargs.dynamo.request_completion_token_ids=true
      ++actor_rollout_ref.rollout.engine_kwargs.dynamo.request_timeout_s="$DYNAMO_REQUEST_TIMEOUT_S"
      +actor_rollout_ref.rollout.engine_kwargs.dynamo.enable_worker_system_metrics=true
      +actor_rollout_ref.rollout.engine_kwargs.dynamo.stable_kv_event_ports=true
      +actor_rollout_ref.rollout.engine_kwargs.dynamo.free_engine_on_train=true
      +actor_rollout_ref.rollout.engine_kwargs.dynamo.namespace="$NS"
    )
    if [ "$VARIANT" = dynamo ] && [ "$DYNAMO_SESSION_AFFINITY" = 1 ]; then
      EXTRA_ARGS+=(
        +actor_rollout_ref.rollout.engine_kwargs.dynamo.session_affinity_enabled=true
        "+actor_rollout_ref.rollout.engine_kwargs.dynamo.frontend_extra_args=[--router-session-affinity-ttl-secs,$DYNAMO_SESSION_AFFINITY_TTL_S]"
      )
    fi
    if [ -n "$KV_BYTES" ]; then
      EXTRA_ARGS+=("+actor_rollout_ref.rollout.engine_kwargs.dynamo.extra_args=[--kv-cache-memory-bytes,$KV_BYTES]")
    fi
    ;;
  *) echo "unknown VARIANT=$VARIANT" >&2; exit 2 ;;
esac

ENTRY=(python -m "$TARGET_MODULE")

# GPU telemetry sampler (host side). Epoch timestamps align exactly with
# ThunderAgent scheduler snapshots and worker Prometheus samples.
printf 'timestamp,index,utilization_gpu_pct,memory_used_mib,power_draw_w\n' > "$RUN_DIR/gpu.csv"
(
  while true; do
    timestamp=$(date +%s.%N)
    nvidia-smi \
      --query-gpu=index,utilization.gpu,memory.used,power.draw \
      --format=csv,noheader,nounits | while IFS= read -r row; do
        printf '%s,%s\n' "$timestamp" "${row//, /,}"
      done
    sleep 1
  done
) >> "$RUN_DIR/gpu.csv" 2> "$RUN_DIR/gpu-sampler.stderr" &
SAMPLER=$!

SIDECAR=
if [ "$VARIANT" != "global" ]; then
  mkdir -p "$RUN_DIR/worker_metrics"
  chmod 777 "$RUN_DIR/worker_metrics"
  python "$RECIPE_WT/dynamo/metrics_sidecar.py" \
    --endpoints-glob "$RUN_DIR/worker_metrics/*.endpoints" \
    --output "$RUN_DIR/worker-metrics.jsonl" \
    --label "$VARIANT" --interval 1 --timeout 1 \
    > "$RUN_DIR/metrics-sidecar.log" 2>&1 &
  SIDECAR=$!
fi

cleanup() {
  kill "$SAMPLER" 2>/dev/null || true
  if [ -n "$SIDECAR" ]; then
    kill "$SIDECAR" 2>/dev/null || true
  fi
}
trap cleanup EXIT

set -o pipefail
git_diff_sha() {
  git -C "$1" diff --binary | sha256sum | awk '{print $1}'
}
IMAGE_ID=$(docker image inspect "$IMAGE" --format '{{.Id}}')
VERL_SHA=$(git -C "$VERL_WT" rev-parse HEAD)
VERL_DIFF_SHA=$(git_diff_sha "$VERL_WT")
RECIPE_SHA=$(git -C "$RECIPE_WT" rev-parse HEAD)
RECIPE_DIFF_SHA=$(git_diff_sha "$RECIPE_WT")
UNIAGENT_SHA=$(git -C "$UNIAGENT" rev-parse HEAD)
UNIAGENT_DIFF_SHA=$(git_diff_sha "$UNIAGENT")
DYNAMO_SHA=${DYNAMO_WT:+$(git -C "$DYNAMO_WT" rev-parse HEAD)}
DYNAMO_DIFF_SHA=${DYNAMO_WT:+$(git_diff_sha "$DYNAMO_WT")}
HARNESS_SHA=$(sha256sum "$0" | awk '{print $1}')
ROLLOUT_PROBE_SHA=$(sha256sum "$BENCH/uni-agent/rollout_benchmark_entry.py" | awk '{print $1}')
DATA_SHA=$(sha256sum "$BENCH/uni-agent/data/swe_bench_verified_local32.parquet" | awk '{print $1}')
MODEL_INDEX_SHA=$(sha256sum "$MODEL_DIR/model.safetensors.index.json" | awk '{print $1}')
SANDBOX_MANIFEST_SHA=$(sha256sum "$BENCH/uni-agent/data/sandbox-images-manifest.txt" | awk '{print $1}')
SANDBOX_IMAGE_COUNT=$(wc -l < "$BENCH/uni-agent/data/sandbox-images-manifest.txt" | tr -d ' ')
{
  echo "VARIANT=$VARIANT GATE=$GATE ROLLOUT_ONLY=$ROLLOUT_ONLY STEPS=$STEPS BATCH=$BATCH N=$N MAX_TURNS=$MAX_TURNS CONCURRENCY=$CONCURRENCY MAX_NUM_SEQS=$MAX_NUM_SEQS MODEL_DIR=$MODEL_DIR DYNAMO_WT=$DYNAMO_WT DYNAMO_REQUEST_TIMEOUT_S=$DYNAMO_REQUEST_TIMEOUT_S DYNAMO_SESSION_AFFINITY=$DYNAMO_SESSION_AFFINITY DYNAMO_SESSION_AFFINITY_TTL_S=$DYNAMO_SESSION_AFFINITY_TTL_S NGPUS_PER_NODE=8 ROLLOUT_TP=4 ACTOR_SP=${SP:-8} AGENT_WORKERS=8 GPU_DEVICES=0,1,2,3,4,5,6,7"
  echo "HOSTNAME=$(hostname) IMAGE_ID=$IMAGE_ID VERL_SHA=$VERL_SHA VERL_DIFF_SHA=$VERL_DIFF_SHA RECIPE_SHA=$RECIPE_SHA RECIPE_DIFF_SHA=$RECIPE_DIFF_SHA UNIAGENT_SHA=$UNIAGENT_SHA UNIAGENT_DIFF_SHA=$UNIAGENT_DIFF_SHA DYNAMO_SHA=${DYNAMO_SHA:-none} DYNAMO_DIFF_SHA=${DYNAMO_DIFF_SHA:-none} HARNESS_SHA=$HARNESS_SHA ROLLOUT_PROBE_SHA=$ROLLOUT_PROBE_SHA DATA_SHA=$DATA_SHA MODEL_INDEX_SHA=$MODEL_INDEX_SHA SANDBOX_MANIFEST_SHA=$SANDBOX_MANIFEST_SHA SANDBOX_IMAGE_COUNT=$SANDBOX_IMAGE_COUNT"
  if [ "$VARIANT" = dynamo ] && [ "$DYNAMO_SESSION_AFFINITY" = 1 ]; then
    echo "SESSION_AFFINITY_OVERLAY_SHA=$(sha256sum "$SESSION_AFFINITY_OVERLAY" | awk '{print $1}')"
  fi
  echo GPU_LAYOUT_BEGIN
  nvidia-smi -L
  echo GPU_LAYOUT_END
  echo "docker run ... ${ENTRY[*]} ${COMMON_ARGS[*]} ${EXTRA_ARGS[*]}"
} > "$RUN_DIR/command.txt"

set +e
docker run $( [ "${KEEP:-0}" = "1" ] || echo --rm ) --name "$RUN_NAME" -e HYDRA_FULL_ERROR=1 \
  --network "$NETWORK" \
  --gpus all \
  --ipc=host \
  --group-add "$DOCKER_GID" \
  "${COMMON_ENV[@]}" "${EXTRA_ENV[@]}" \
  "${MOUNTS[@]}" \
  -w /workspace/verl \
  "$IMAGE" \
  "${ENTRY[@]}" "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}" ${EXTRA:-} \
  2>&1 | tee "$RUN_DIR/console.log"
DOCKER_STATUS=${PIPESTATUS[0]}
set -e
printf '%s\n' "$DOCKER_STATUS" > "$RUN_DIR/exit_code"
python "$BENCH/uni-agent/validity.py" "$RUN_DIR" || true
exit "$DOCKER_STATUS"
