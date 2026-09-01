"""Recipe-side Dynamo rollout registration for VERL_USE_EXTERNAL_MODULES."""

from __future__ import annotations

import sys

from verl.workers.rollout.base import _ROLLOUT_REGISTRY
from verl.workers.rollout.replica import RolloutReplicaRegistry


def _load_dynamo():
    from recipe.dynamo.dynamo_thunderagent import DynamoThunderAgentReplica

    return DynamoThunderAgentReplica


RolloutReplicaRegistry.register("dynamo", _load_dynamo)
_ROLLOUT_REGISTRY[("dynamo", "async")] = "recipe.dynamo.dynamo_rollout.ServerAdapter"

# One rollout name for both engines. recipe.dynamo.dynamo_rollout.ServerAdapter
# dispatches on engine_kwargs.dynamo.engine, so selecting SGLang is
#   actor_rollout_ref.rollout.name=dynamo
#   ++actor_rollout_ref.rollout.engine_kwargs.dynamo.engine=sglang
# and nothing else moves. (An earlier revision registered a separate
# "dynamo_sglang" name with its own trainer yaml and entry point; that spread the
# engine choice across three files that all had to agree.)


def _patch_dynamo_llm_server_manager():
    partial = sys.modules.get("recipe.dynamo.dynamo_agent_loop")
    if partial is not None and not hasattr(partial, "DynamoLLMServerManager"):
        return

    from recipe.dynamo.dynamo_agent_loop import DynamoLLMServerManager

    from verl.workers.rollout import llm_server

    llm_server.LLMServerManager = DynamoLLMServerManager
    try:
        from verl.trainer.ppo import ray_trainer

        ray_trainer.LLMServerManager = DynamoLLMServerManager
    except Exception:
        pass


_patch_dynamo_llm_server_manager()
