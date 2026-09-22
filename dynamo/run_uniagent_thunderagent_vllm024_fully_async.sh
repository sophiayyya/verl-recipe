#!/usr/bin/env bash
# Qwen3-30B-A3B-Base: 8 training GPUs + 8 rollout GPUs (two TP4 engines).
# Start a Ray cluster across both nodes before running this on its head node.
set -euo pipefail
: "${RAY_CLUSTER_ADDRESS:?Set RAY_CLUSTER_ADDRESS to the prepared 16-GPU Ray cluster}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
case ",${VERL_USE_EXTERNAL_MODULES:-}," in
    *,recipe.dynamo.verl_logging_compat,*) ;;
    *) export VERL_USE_EXTERNAL_MODULES="${VERL_USE_EXTERNAL_MODULES:+$VERL_USE_EXTERNAL_MODULES,}recipe.dynamo.verl_logging_compat" ;;
esac
exec bash "$SCRIPT_DIR/run_uniagent_thunderagent_vllm024.sh" \
    trainer.v1.trainer_mode=separate_async \
    actor_rollout_ref.hybrid_engine=false \
    trainer.v1.separate_async.hybrid_rollout.enable_switch=false \
    trainer.v1.separate_async.num_warmup_batches=1 \
    trainer.v1.separate_async.parameter_sync_step=1 \
    trainer.nnodes=1 trainer.n_gpus_per_node=8 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    actor_rollout_ref.rollout.nnodes=1 \
    actor_rollout_ref.rollout.n_gpus_per_node=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=512 \
    "++ray_kwargs.ray_init.address='$RAY_CLUSTER_ADDRESS'" \
    ray_kwargs.ray_init.num_cpus=null \
    "$@"
