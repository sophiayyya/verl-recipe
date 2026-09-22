#!/usr/bin/env bash
# Single-node Qwen3-30B-A3B-Base reference: V1 colocate_async, 8 GPUs, two TP4 engines.
set -euo pipefail

: "${VERL_SRC:?Set VERL_SRC to your prepared verl checkout}"
: "${DYNAMO_SRC:?Set DYNAMO_SRC to the patched Dynamo checkout}"
: "${MODEL_PATH:?Set MODEL_PATH to the local model directory}"
: "${TRAIN_FILE:?Set TRAIN_FILE to the workload training parquet}"

CONFIG_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/config" && pwd)
TEST_FILE=${TEST_FILE:-$TRAIN_FILE}
RUN_NAME=${RUN_NAME:-thunderagent-vllm024-smoke}
OUTPUT_DIR=${OUTPUT_DIR:-$VERL_SRC/outputs/$RUN_NAME}
TOTAL_STEPS=${TOTAL_STEPS:-2}
AUTO_FINALIZE=${AUTO_FINALIZE:-true}
case "$AUTO_FINALIZE" in
    true|false) ;;
    *) echo "AUTO_FINALIZE must be true or false" >&2; exit 2 ;;
esac

export PYTHONPATH="$DYNAMO_SRC/components/src:$VERL_SRC${UNIAGENT_ROOT:+:$UNIAGENT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
case ",${VERL_USE_EXTERNAL_MODULES:-}," in
    *,recipe.dynamo.register,*) ;;
    *) export VERL_USE_EXTERNAL_MODULES="recipe.dynamo.register${VERL_USE_EXTERNAL_MODULES:+,$VERL_USE_EXTERNAL_MODULES}" ;;
esac
export VERL_DYNAMO_LOG_DIR="$OUTPUT_DIR/dynamo"
export VERL_DYNAMO_FE_READY_TIMEOUT=1800
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF RAY_ADDRESS ROCR_VISIBLE_DEVICES

agent_args=(actor_rollout_ref.rollout.multi_turn.enable=false)
if [[ -n ${AGENT_CONFIG:-} ]]; then
    agent_args=(
        "actor_rollout_ref.rollout.agent.agent_loop_config_path='$AGENT_CONFIG'"
        "actor_rollout_ref.rollout.agent.default_agent_loop=${AGENT_LOOP:-swe_agent}"
        actor_rollout_ref.rollout.multi_turn.enable=true
        actor_rollout_ref.rollout.multi_turn.tool_config_path=null
    )
fi

cd "$VERL_SRC"
exec python3 -m verl.trainer.main_ppo \
    --config-path "$CONFIG_DIR" --config-name dynamo_trainer_v1_colocate \
    trainer.use_v1=true trainer.v1.trainer_mode=colocate_async \
    trainer.v1.colocate_async.num_warmup_batches=1 \
    trainer.v1.sampler.max_off_policy_threshold=8 \
    trainer.v1.sampler.max_off_policy_strategy=drop \
    transfer_queue.enable=true \
    algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=false \
    algorithm.rollout_correction.bypass_mode=true \
    algorithm.rollout_correction.loss_type=ppo_clip \
    "data.train_files='$TRAIN_FILE'" "data.val_files='$TEST_FILE'" \
    data.return_raw_chat=true data.train_batch_size=2 \
    data.max_prompt_length=4096 data.max_response_length=4096 \
    data.filter_overlong_prompts=true data.filter_overlong_prompts_workers=1 \
    data.truncation=error data.shuffle=false data.seed=42 \
    "actor_rollout_ref.model.path='$MODEL_PATH'" \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    ++actor_rollout_ref.actor.optim.override_optimizer_config.foreach=false \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=true \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=false \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    actor_rollout_ref.actor.fsdp_config.use_no_sync_for_gradient_accumulation=false \
    actor_rollout_ref.rollout.name=dynamo actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.nnodes=0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_num_seqs=2048 \
    actor_rollout_ref.rollout.disable_log_stats=false \
    actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.enable_prefix_caching=true \
    actor_rollout_ref.rollout.enforce_eager=false \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    actor_rollout_ref.rollout.agent.agent_loop_manager_class=null \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.engine=vllm \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.router_mode=round-robin \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.enable_kv_events=true \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.router_session_affinity_ttl_secs=0 \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.direct_generate=false \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.request_engine_data=true \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.request_completion_token_ids=true \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.request_timeout_s=7200 \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.enable_worker_system_metrics=true \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.stable_kv_event_ports=true \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.free_engine_on_train=true \
    "++actor_rollout_ref.rollout.engine_kwargs.dynamo.namespace='$RUN_NAME'" \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.thunderagent.enabled=true \
    "++actor_rollout_ref.rollout.engine_kwargs.dynamo.thunderagent.auto_finalize=$AUTO_FINALIZE" \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.thunderagent.router_block_size=16 \
    '++actor_rollout_ref.rollout.engine_kwargs.dynamo.extra_args=["--seed","42","--block-size","16","--max-num-seqs","1024"]' \
    '++actor_rollout_ref.rollout.engine_kwargs.dynamo.frontend_extra_args=["--kv-cache-block-size","16"]' \
    'trainer.logger=[console]' trainer.project_name=thunderagent-vllm024 \
    "trainer.experiment_name='$RUN_NAME'" \
    trainer.nnodes=1 trainer.n_gpus_per_node=8 \
    trainer.val_before_train=false trainer.test_freq=-1 trainer.save_freq=-1 \
    trainer.total_epochs=1000 "trainer.total_training_steps=$TOTAL_STEPS" \
    trainer.resume_mode=disable \
    "trainer.default_local_dir='$OUTPUT_DIR/checkpoints'" \
    "trainer.rollout_data_dir='$OUTPUT_DIR/rollouts'" \
    ++ray_kwargs.ray_init.address=null \
    "ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS:-32}" \
    ++ray_kwargs.ray_init.runtime_env.py_executable=null \
    "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES='$VERL_USE_EXTERNAL_MODULES'" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH='$PYTHONPATH'" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.PATH='$PATH'" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_DYNAMO_LOG_DIR='$VERL_DYNAMO_LOG_DIR'" \
    "hydra.run.dir='$OUTPUT_DIR/hydra'" \
    "${agent_args[@]}" "$@"
