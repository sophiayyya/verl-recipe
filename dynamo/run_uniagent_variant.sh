#!/usr/bin/env bash
set -xeuo pipefail

# VARIANT=ta runs ThunderAgent; use dynamo or global for the two baselines.
VARIANT=${VARIANT:-ta}
project_name=${PROJECT_NAME:-verl-ta-uniagent}
exp_name=${EXP_NAME:-uniagent-${VARIANT}}

max_prompt_length=${MAX_PROMPT_LENGTH:-4096}
max_response_length=${MAX_RESPONSE_LENGTH:-28672}
train_prompt_bsz=${TRAIN_BATCH_SIZE:-32}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-8}
train_prompt_mini_bsz=${MINI_BATCH_SIZE:-32}
total_training_steps=${TOTAL_TRAINING_STEPS:-3}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3-Coder-30B-A3B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/swe_bench_verified_local32.parquet"}
TEST_FILE=${TEST_FILE:-"${TRAIN_FILE}"}
AGENT_CONFIG=${AGENT_CONFIG:-"${RAY_DATA_HOME}/uni-agent/agent_config_local_swe.yaml"}
UNIAGENT_ROOT=${UNIAGENT_ROOT:-"${RAY_DATA_HOME}/uni-agent"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}

export PYTHONPATH="${UNIAGENT_ROOT}:${PYTHONPATH:-}"

case "${VARIANT}" in
    global)
        target_module=verl.trainer.main_ppo
        backend_args=(
            actor_rollout_ref.rollout.name=vllm
        )
        ;;
    dynamo | ta)
        target_module=recipe.dynamo.main_dynamo
        export VERL_USE_EXTERNAL_MODULES=recipe.dynamo.register
        thunderagent_enabled=false
        [[ "${VARIANT}" == "ta" ]] && thunderagent_enabled=true
        backend_args=(
            actor_rollout_ref.rollout.name=dynamo
            "++actor_rollout_ref.rollout.engine_kwargs.dynamo.thunderagent.enabled=${thunderagent_enabled}"
            ++actor_rollout_ref.rollout.engine_kwargs.dynamo.thunderagent.router_block_size=16
            ++actor_rollout_ref.rollout.engine_kwargs.dynamo.request_engine_data=true
            ++actor_rollout_ref.rollout.engine_kwargs.dynamo.request_completion_token_ids=true
            ++actor_rollout_ref.rollout.engine_kwargs.dynamo.request_timeout_s=1800
            ++actor_rollout_ref.rollout.engine_kwargs.dynamo.free_engine_on_train=true
        )
        ;;
    *)
        echo "VARIANT must be global, dynamo, or ta" >&2
        exit 2
        ;;
esac

python3 -m "${target_module}" \
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
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="${train_prompt_mini_bsz}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.n="${n_resp_per_prompt}" \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_CONFIG}" \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    "${backend_args[@]}" \
    trainer.logger='["console"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.total_training_steps="${total_training_steps}" \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    "$@"
