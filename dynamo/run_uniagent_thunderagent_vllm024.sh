#!/usr/bin/env bash
# Uni-Agent 0825e1ea + pristine verl 3efe38c7; V1 colocate_async.
set -euo pipefail
: "${UNIAGENT_ROOT:?Set UNIAGENT_ROOT to the upstream Uni-Agent checkout}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUNNER=${AGENT_RUNNER_FQN:-uni_agent.framework.task_runner.run_task}
runner_args=("++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn='$RUNNER'")
if [[ $RUNNER == uni_agent.framework.task_runner.run_task ]]; then
    : "${TASK_CONFIG:?Set TASK_CONFIG to your current Uni-Agent task YAML}"
    runner_args+=("++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path='$TASK_CONFIG'")
fi
export AUTO_FINALIZE=false
unset AGENT_CONFIG
exec bash "$SCRIPT_DIR/run_thunderagent_vllm024.sh" \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.agent.default_agent_loop=task \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=null \
    actor_rollout_ref.rollout.agent.agent_loop_manager_class=recipe.dynamo.uniagent_gateway.DynamoAgentFrameworkRolloutAdapter \
    "++actor_rollout_ref.rollout.custom.agent_framework.gateway_count=${GATEWAY_COUNT:-2}" \
    "++actor_rollout_ref.rollout.custom.agent_framework.log_dir='${OUTPUT_DIR:-$VERL_SRC/outputs/uniagent}/agent_logs'" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=inline_async \
    "++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions=${AGENT_CONCURRENCY:-4}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.trajectory_selection=longest \
    '++actor_rollout_ref.rollout.custom.agent_framework.allowed_request_sampling_param_keys=[temperature,top_p,top_k]' \
    "${runner_args[@]}" "$@"
