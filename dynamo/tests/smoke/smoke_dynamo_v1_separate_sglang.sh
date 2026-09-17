#!/usr/bin/env bash
set -xeuo pipefail

# V1 separate_async smoke on the SGLANG engine: standalone rollout pool
# receives weights over the nccl checkpoint engine; CE workers reach the
# engine through the sglang adapter's HTTP control route (no ZMQ hop).
#
# First GPU validation target for the CE-worker rank/TP-group contract
# (see the experiment log) — run AFTER the colocate smoke is green.
python3 -c "import dynamo.sglang" || {
  echo "dynamo.sglang not importable — install ai-dynamo[sglang] first" >&2
  exit 2
}
unset PYTORCH_CUDA_ALLOC_CONF || true

project_name=${PROJECT_NAME:-verl-dynamo}
exp_name=${EXP_NAME:-dynamo-v1-separate-sglang-smoke}

max_prompt_length=${MAX_PROMPT_LENGTH:-512}
max_response_length=${MAX_RESPONSE_LENGTH:-512}
TOTAL_STEPS=${TOTAL_STEPS:-2}
PARAMETER_SYNC_STEP=${PARAMETER_SYNC_STEP:-2}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-1}
# PPOTrainerSeparateAsync asserts train_batch_size == parameter_sync_step * ppo_mini_batch_size.
TRAIN_BATCH_SIZE=$((PARAMETER_SYNC_STEP * PPO_MINI_BATCH_SIZE))

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
ROLLOUT_NNODES=${ROLLOUT_NNODES:-1}
ROLLOUT_NGPUS_PER_NODE=${ROLLOUT_NGPUS_PER_NODE:-1}
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen2.5-0.5B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}

export VERL_USE_EXTERNAL_MODULES="${VERL_USE_EXTERNAL_MODULES:-recipe.dynamo.register}"
DYNAMO_CONFIG_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../config" && pwd)

python3 -m verl.trainer.main_ppo \
    --config-path "${DYNAMO_CONFIG_DIR}" \
    --config-name=dynamo_trainer_v1_separate \
    "ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES='${VERL_USE_EXTERNAL_MODULES}'" \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.engine=sglang \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.request_completion_token_ids=true \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.enable_worker_system_metrics=true \
    "+actor_rollout_ref.rollout.enable_sleep_mode=true" \
    algorithm.adv_estimator=grpo \
    algorithm.rollout_correction.bypass_mode=true \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_batch_size=1 \
    data.train_max_samples="${TRAIN_BATCH_SIZE}" \
    data.val_max_samples=1 \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.multi_turn.enable=False \
    actor_rollout_ref.rollout.nnodes="${ROLLOUT_NNODES}" \
    actor_rollout_ref.rollout.n_gpus_per_node="${ROLLOUT_NGPUS_PER_NODE}" \
    trainer.v1.separate_async.parameter_sync_step="${PARAMETER_SYNC_STEP}" \
    trainer.logger=console \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    trainer.total_epochs=100 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    "$@"
