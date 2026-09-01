#!/bin/bash
#SBATCH --job-name=retool_ab
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --cpus-per-task=112
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/sopyang/logs/retool_%x_%j.out
#
# retool A/B on Qwen3-30B-A3B-Base, 2 nodes x 8 H100.
#
#   ARM=dynamo_sglang   rollout.name=dynamo  + engine_kwargs.dynamo.engine=sglang
#   ARM=dynamo_vllm     rollout.name=dynamo  + engine_kwargs.dynamo.engine=vllm
#   ARM=native_sglang   rollout.name=sglang  (verl's own sglang server path)
#
# All arms run the SAME entry point and config otherwise, so the only difference
# is which engine generation is routed through.
#
# dynamo_vllm is the control for "is this bug mine or the recipe's": it exercises
# the pre-existing vLLM path on the same container, dataset, batch shape and
# entry point as dynamo_sglang. It differs in two ways that are NOT incidental:
#   - it needs the local Dynamo source overlay on PYTHONPATH, because the shipped
#     wheel's dynamo.vllm.handlers has no completion_token_ids support (the
#     bootstrap asserts this rather than silently degrading to 1-token responses)
#   - it must NOT install the [sglang] extra, which downgrades vLLM's
#     guided-decoding stack while leaving `import vllm` working
#
# Usage:  sbatch --export=ALL,ARM=dynamo_sglang,STEPS=100 retool_ab.sbatch
set -x
B=/lustre/fsw/portfolios/coreai/users/sopyang
V=$B/verl_dynamo/verl
CONTAINER=$B/images/verl_vllm024.dev2.sqsh
ARM=${ARM:-dynamo_sglang}

# Verified-good settings from job 16547388 (3/3 steps, update_actor ~76s, peak 71.3GB).
# eager costs 39x in update_actor (2810s vs 71.9s) for memory that fused kernels save
# more cheaply -- see HANDOFF.md section 6. Defaults keep the OLD behaviour so an
# unqualified `sbatch retool_ab.sbatch` is unchanged; the campaign passes these in.
EAGER_EXPERTS=${EAGER_EXPERTS:-1}
USE_FUSED_KERNELS=${USE_FUSED_KERNELS:-0}
FUSED_KERNELS_BACKEND=${FUSED_KERNELS_BACKEND:-torch}
PPO_MAX_TOKEN_LEN=${PPO_MAX_TOKEN_LEN:-18432}
SAVE_FREQ=${SAVE_FREQ:--1}
RESUME_MODE=${RESUME_MODE:-disable}
export VERL_DEFER_OPTIMIZER_LOAD=${VERL_DEFER_OPTIMIZER_LOAD:-0}

OPT_ARGS=""
if [[ "$EAGER_EXPERTS" == "1" ]]; then
  OPT_ARGS="$OPT_ARGS +actor_rollout_ref.model.override_config._experts_implementation=eager"
fi
if [[ "$USE_FUSED_KERNELS" == "1" ]]; then
  OPT_ARGS="$OPT_ARGS actor_rollout_ref.model.use_fused_kernels=True"
  OPT_ARGS="$OPT_ARGS ++actor_rollout_ref.model.fused_kernel_options.impl_backend=$FUSED_KERNELS_BACKEND"
fi

# `set -x` is on, so never let the key land on a traced line: the job log is
# group-readable on lustre. Read it with tracing off and echo only the length.
set +x
WANDB_KEY_FILE=${WANDB_KEY_FILE:-$B/.wandb_key}
if [ -z "${WANDB_API_KEY:-}" ] && [ -r "$WANDB_KEY_FILE" ]; then
  WANDB_API_KEY="$(tr -d '\r\n' < "$WANDB_KEY_FILE")"
  export WANDB_API_KEY
fi
if [ -n "${WANDB_API_KEY:-}" ]; then
  TRAINER_LOGGER='["console","wandb"]'
  echo "wandb: key loaded (${#WANDB_API_KEY} chars)"
else
  TRAINER_LOGGER='["console"]'
  echo "wandb: no key at $WANDB_KEY_FILE -> console only"
fi
set -x
STEPS=${STEPS:-100}
MODEL=$B/hf_models/Qwen3-30B-A3B-Base
# retool's CustomRLHFDataset calls datasets.load_dataset(<path>)["train"], so these
# must be HF-dataset DIRECTORIES (containing train.parquet), not .parquet files.
# It also derives data_source from the last two path segments and switches map_fn
# on it: names in its AIME list take the raw-column branch (Problem/Answer), which
# our verl-preprocessed parquets do not have. "local/..." falls through to map_fn2,
# the branch that reads prompt[0].content — the right one here.
TRAIN_FILE=${TRAIN_FILE:-$B/datasets/local/dapo-math-17k}
VAL_FILE=${VAL_FILE:-$B/datasets/local/aime-2024}

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
node_1=${nodes[0]}; node_2=${nodes[1]}
port=$(( 6500 + SLURM_JOB_ID % 1000 ))
ip_head=$node_1:$port
echo "ARM=$ARM STEPS=$STEPS  ray head=$ip_head worker=$node_2"

# Metrics plumbing, copied from the proven run. Without prometheus.* and
# disable_log_stats=False there are no engine-side stats at all, and the KV-cache
# hit rate — the headline number this A/B exists to produce — is simply not
# collected. Ports are job-derived so concurrent arms do not collide.
PROM_PORT=$(( 9200 + SLURM_JOB_ID % 500 ))
PROM_FILE=$B/logs/prom_${ARM}_${SLURM_JOB_ID}.yml
# ROLLOUT_ARGS is single-quoted, so its $PROM_FILE survives into DRIVER unexpanded and
# is resolved by the container-side shell -- which only sees these if they are exported.
# Unexported, prometheus.file= arrives empty and verl's write_config_file does
# makedirs('') -> FileNotFoundError (job 16566566).
export PROM_FILE PROM_PORT
export VERL_DYNAMO_WORKER_METRICS_DIR=$B/logs/worker_metrics_${ARM}_${SLURM_JOB_ID}
mkdir -p "$VERL_DYNAMO_WORKER_METRICS_DIR"
CAMPAIGN=${CAMPAIGN:-$SLURM_JOB_ID}
CKPT_DIR=$B/logs/ckpt_${ARM}_${CAMPAIGN}
# Dump prompts/responses to disk. Chasing "response_length == 1" through metrics
# alone produced three wrong hypotheses in a row; the actual text is the only
# thing that settles it.
ROLLOUT_DUMP=${ROLLOUT_DUMP:-$B/logs/rollouts_${ARM}_${SLURM_JOB_ID}}
mkdir -p "$ROLLOUT_DUMP"

RUN_DIR=$B/logs/retool_run_${SLURM_JOB_ID}
mkdir -p "$RUN_DIR"

# _experts_implementation=eager is a MEMORY fix, not a speed knob, and it is the
# reason this config fits on 2 nodes at all.
#
# transformers 5.x picks an experts kernel at load time and lands on "grouped_mm",
# which materialises the projection for every (token x hit expert) pair at once:
#
#   transformers/integrations/moe.py:458 grouped_mm_experts_forward
#     weighted_out = proj_out * sample_weights_g.unsqueeze(-1)
#
# "eager" is special-cased in ExpertsInterface.get_interface() to fall through to
# the model's own forward, and Qwen3MoeExperts.forward loops over hit experts,
# gathering only the tokens routed to each. Same math, a fraction of the peak.
#
# Without it, update_actor OOMs on BOTH engines at 76.5 GiB with the engine down to
# 2.4 GiB (jobs 16442420 sglang / 16353597 vLLM) — i.e. it is not an engine problem
# and not a batch-size problem: lowering ppo_max_token_len_per_gpu 18432 -> 8192 did
# not help either arm. The proven 100-step run never hit this because it pins
# transformers>=4.56,<5, which only ever had the looped path.
#
# entropy_* : the second half of the same memory story as _experts_implementation.
# GRPO derives entropy from logits, and Qwen3's vocab is 151936, so one fp32 logits
# tensor for an 8192-token micro-batch is ~5.0 GB before the softmax temporaries.
# Unchunked, that is what backward OOMs on (75.3 GiB in use, 768 MiB request failing,
# jobs 16481584 vLLM / 16446247 sglang). Chunking computes the same value in pieces
# and checkpointing recomputes it in backward instead of storing it — both are
# numerically equivalent, neither changes the objective.
#
# The proven run leaves both off and still fits, because transformers 4.x's looped
# MoE left enough headroom. The budget did not change; 5.x just spends more of it on
# the experts, so it has to be clawed back here.
#
# Sampling params are copied verbatim from the proven vLLM run
# ($B/verl/recipe/dynamo/train_30b_rl_dynamo_kv_i100_metrics.sh). Omitting them
# let verl's defaults apply, and the first retool attempt produced
# response_length == 1 for every sample (greedy -> immediate EOS on the base
# model), i.e. a green run with no generation and zero gradient. Do not drop
# temperature/top_k/top_p here.
#
# Driver: identical for both arms except the rollout backend selection.
# NB entry point is recipe.dynamo.main_dynamo for BOTH arms — not because the
# baseline needs anything from dynamo, but because core verl at this checkout
# (6cbca9ce) routes verl.trainer.main_ppo to TaskRunnerV1, which hands a
# TensorDict to agent_loop.generate_sequences (still DataProto-only, verl PR
# #6572) and dies with AttributeError. main_dynamo forces the v0 TaskRunner;
# main_ppo_v0 itself exposes no hydra entry. The baseline additionally leaves
# VERL_USE_EXTERNAL_MODULES unset so recipe.dynamo.register never runs, and
# deletes the dynamo agent-loop manager override.
if [[ "$ARM" == "dynamo_sglang" ]]; then
  ARM_ARGS=(
    "export VERL_USE_EXTERNAL_MODULES=recipe.dynamo.register"
  )
  ROLLOUT_ARGS='actor_rollout_ref.rollout.name=dynamo \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.engine=sglang \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.router_mode=kv \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.enable_nemo_router_tuning=true \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.request_engine_data=true \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.request_completion_token_ids=true \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.return_tokens_as_token_ids=false \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.enable_worker_system_metrics=true \
    actor_rollout_ref.rollout.prometheus.enable=True \
    actor_rollout_ref.rollout.prometheus.port=$PROM_PORT \
    actor_rollout_ref.rollout.prometheus.file="$PROM_FILE" \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.request_timeout_s=1800 \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.free_engine_on_train=true \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.thunderagent.enabled=false \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.sglang.enable_rl=true \
    ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=recipe.dynamo.dynamo_agent_loop.DynamoAgentLoopManager'
elif [[ "$ARM" == "dynamo_vllm" ]]; then
  # Same knobs as the sglang arm minus the sglang-only ones. engine=vllm is
  # written out explicitly even though _engine_kind() already defaults to it:
  # this arm exists to be compared against, so the config should say what it is
  # rather than rely on a default that could move.
  ARM_ARGS=(
    "export VERL_USE_EXTERNAL_MODULES=recipe.dynamo.register"
  )
  ROLLOUT_ARGS='actor_rollout_ref.rollout.name=dynamo \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.engine=vllm \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.router_mode=kv \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.enable_nemo_router_tuning=true \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.request_engine_data=true \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.request_completion_token_ids=true \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.return_tokens_as_token_ids=false \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.enable_worker_system_metrics=true \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.stable_kv_event_ports=true \
    actor_rollout_ref.rollout.prometheus.enable=True \
    actor_rollout_ref.rollout.prometheus.port=$PROM_PORT \
    actor_rollout_ref.rollout.prometheus.file="$PROM_FILE" \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.request_timeout_s=1800 \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.free_engine_on_train=true \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.thunderagent.enabled=false \
    ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=recipe.dynamo.dynamo_agent_loop.DynamoAgentLoopManager'
else
  ARM_ARGS=( "unset VERL_USE_EXTERNAL_MODULES" )
  ROLLOUT_ARGS='actor_rollout_ref.rollout.name=sglang \
    ~actor_rollout_ref.rollout.agent.agent_loop_manager_class'
fi

read -r -d '' DRIVER <<DRV || true
${ARM_ARGS[0]}
export ARM=$ARM
python3 -m recipe.dynamo.main_dynamo \\
    algorithm.adv_estimator=grpo \\
    algorithm.use_kl_in_reward=False \\
    algorithm.kl_ctrl.kl_coef=0.0 \\
    data.train_files="[\\"$TRAIN_FILE\\"]" \\
    data.val_files="[\\"$VAL_FILE\\"]" \\
    data.return_raw_chat=True \\
    data.train_batch_size=16 \\
    data.max_prompt_length=2048 \\
    data.max_response_length=8192 \\
    data.filter_overlong_prompts=True \\
    data.truncation=error \\
    data.custom_cls.path=recipe/retool/retool.py \\
    data.custom_cls.name=CustomRLHFDataset \\
    custom_reward_function.path=recipe/retool/retool.py \\
    custom_reward_function.name=compute_score \\
    actor_rollout_ref.model.path=$MODEL \\
    actor_rollout_ref.model.use_remove_padding=True \\
    actor_rollout_ref.model.enable_gradient_checkpointing=True \\
    $OPT_ARGS \\
    actor_rollout_ref.actor.use_kl_loss=False \\
    actor_rollout_ref.actor.kl_loss_coef=0.0 \\
    actor_rollout_ref.actor.clip_ratio_low=0.2 \\
    actor_rollout_ref.actor.clip_ratio_high=0.28 \\
    actor_rollout_ref.actor.clip_ratio_c=10.0 \\
    actor_rollout_ref.actor.optim.lr=1e-6 \\
    actor_rollout_ref.actor.use_dynamic_bsz=True \\
    ++actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \\
    ++actor_rollout_ref.actor.entropy_checkpointing=True \\
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \\
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN \\
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \\
    actor_rollout_ref.actor.fsdp_config.param_offload=True \\
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \\
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=73728 \\
    actor_rollout_ref.rollout.mode=async \\
    actor_rollout_ref.rollout.calculate_log_probs=True \\
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \\
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \\
    actor_rollout_ref.rollout.n=16 \\
    actor_rollout_ref.rollout.temperature=1.0 \\
    actor_rollout_ref.rollout.top_k=-1 \\
    actor_rollout_ref.rollout.top_p=1.0 \\
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \\
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \\
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \\
    actor_rollout_ref.rollout.val_kwargs.n=30 \\
    actor_rollout_ref.rollout.enforce_eager=True \\
    actor_rollout_ref.rollout.disable_log_stats=False \\
    actor_rollout_ref.rollout.multi_turn.enable=True \\
    actor_rollout_ref.rollout.multi_turn.max_user_turns=8 \\
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=8 \\
    actor_rollout_ref.rollout.multi_turn.format=hermes \\
    actor_rollout_ref.rollout.multi_turn.tool_config_path=\$SF_TOOL_CFG \\
    $ROLLOUT_ARGS \\
    trainer.logger='$TRAINER_LOGGER' \\
    trainer.project_name=retool-ab \\
    trainer.experiment_name=retool_${ARM}_${CAMPAIGN} \\
    trainer.n_gpus_per_node=8 \\
    trainer.nnodes=2 \\
    trainer.val_before_train=False \\
    trainer.save_freq=$SAVE_FREQ \\
    trainer.test_freq=999 \\
    trainer.resume_mode=$RESUME_MODE \\
    trainer.default_local_dir="$CKPT_DIR" \\
    trainer.max_actor_ckpt_to_keep=2 \\
    trainer.max_critic_ckpt_to_keep=2 \\
    trainer.rollout_data_dir="$ROLLOUT_DUMP" \\
    trainer.total_training_steps=$STEPS ${EXTRA_HYDRA:-}
echo "RETOOL_AB_DONE arm=$ARM rc=\$?"
DRV

# Materialise per-role scripts and run them as FILES: the bootstrap contains
# pkill patterns, and with `bash -c "<text>"` that text lands in the bash
# process's own argv, so pkill -f matches and SIGTERMs the script itself.
{ echo "export ARM=$ARM"; cat "$V/recipe/dynamo/retool_ab_bootstrap.sh"
  echo "cd $V"; echo "ray start --address=$ip_head --block"; } > "$RUN_DIR/worker.sh"
{ echo "export ARM=$ARM"; cat "$V/recipe/dynamo/retool_ab_bootstrap.sh"
  echo "cd $V"
  echo "ray start --head --node-ip-address=$node_1 --port=$port"
  echo "sleep 10"; echo "ray status"; echo "export RAY_ADDRESS=$ip_head"
  # Scrape per-worker /metrics into JSONL for the whole run; this is where the
  # KV-cache hit rate for the A/B comes from. Dynamo arms only — the native sglang
  # path exposes no dynamo worker endpoints.
  if [[ "$ARM" == dynamo_* ]]; then
    echo "nohup python3 $V/recipe/dynamo/metrics_sidecar.py \
      --endpoints-glob \"$VERL_DYNAMO_WORKER_METRICS_DIR/*.endpoints\" \
      --output $B/logs/kv_metrics_${ARM}_${SLURM_JOB_ID}.jsonl \
      --label $ARM --interval 30 > $B/logs/sidecar_${ARM}_${SLURM_JOB_ID}.log 2>&1 &"
  fi
  printf '%s\n' "$DRIVER"; } > "$RUN_DIR/head.sh"

srun --overlap --nodes=1 --ntasks=1 -w "$node_2" \
  --container-image="$CONTAINER" --container-mounts="$B:$B" --no-container-mount-home \
  bash "$RUN_DIR/worker.sh" &
sleep 60

srun --overlap --nodes=1 --ntasks=1 -w "$node_1" \
  --container-image="$CONTAINER" --container-mounts="$B:$B" --no-container-mount-home \
  bash "$RUN_DIR/head.sh"
echo "=== retool A/B done: ARM=$ARM ==="
