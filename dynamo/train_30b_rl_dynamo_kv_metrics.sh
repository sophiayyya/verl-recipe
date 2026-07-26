#!/usr/bin/env bash
set -xeuo pipefail

project_name=${PROJECT_NAME:-retool-30b-dynamo}
exp_name=${EXP_NAME:-dynamo-kv-metrics}

max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-16384}
train_prompt_bsz=${TRAIN_BATCH_SIZE:-16}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-16}
train_prompt_mini_bsz=${MINI_BATCH_SIZE:-16}

NNODES=${NNODES:-2}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3-30B-A3B-Base"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}
TOOL_CONFIG=${TOOL_CONFIG:-recipe/retool/sandbox_fusion_tool_config.yaml}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
METRICS_DIR=${METRICS_DIR:-"${RAY_DATA_HOME}/logs/${exp_name}"}
METRICS_PORT=${METRICS_PORT:-9200}

export VERL_USE_EXTERNAL_MODULES=recipe.dynamo.register
export VERL_DYNAMO_WORKER_METRICS_DIR="${METRICS_DIR}/workers"
mkdir -p "${VERL_DYNAMO_WORKER_METRICS_DIR}"

python3 recipe/dynamo/metrics_sidecar.py \
    --targets-glob "${METRICS_DIR}/prometheus.yml" \
    --endpoints-glob "${VERL_DYNAMO_WORKER_METRICS_DIR}/*.endpoints" \
    --output "${METRICS_DIR}/kv_metrics.jsonl" \
    --label dynamo_kv --interval 30 &
sidecar_pid=$!
trap 'kill "${sidecar_pid}" 2>/dev/null || true' EXIT

python3 -m recipe.dynamo.main_dynamo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.return_raw_chat=True \
    data.train_batch_size="${train_prompt_bsz}" \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.custom_cls.path=recipe/retool/retool.py \
    data.custom_cls.name=CustomRLHFDataset \
    custom_reward_function.path=recipe/retool/retool.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="${train_prompt_mini_bsz}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 4)) \
    actor_rollout_ref.rollout.name=dynamo \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=8 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=8 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n="${n_resp_per_prompt}" \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.router_mode=kv \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.thunderagent.enabled=false \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.request_engine_data=true \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.request_completion_token_ids=true \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.request_timeout_s=1800 \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.enable_worker_system_metrics=true \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.free_engine_on_train=true \
    '++actor_rollout_ref.rollout.engine_kwargs.dynamo.extra_args=["--generation-config","vllm","--stream-interval=100"]' \
    actor_rollout_ref.rollout.prometheus.enable=True \
    actor_rollout_ref.rollout.prometheus.port="${METRICS_PORT}" \
    actor_rollout_ref.rollout.prometheus.file="${METRICS_DIR}/prometheus.yml" \
    trainer.logger='["console"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=999 \
    trainer.save_freq=-1 \
    trainer.total_training_steps=100 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    "$@"
