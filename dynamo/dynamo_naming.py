"""Single source of truth for the Dynamo control-actor name.

The name is produced in one place and looked up in two others, and a silent
divergence between them surfaces only as an opaque ``ray.get_actor`` failure at
weight-sync time. Keeping the format here makes a change to the convention break
in one spot rather than quietly at runtime.

Producer : ``dynamo_async_server.DynamoReplica.launch_servers``
Consumers: ``dynamo_vllm_rollout.VllmDynamoServerAdapter``
           ``dynamo_sglang_rollout.SGLangServerAdapter``

NB the producer also emits ``server_reward_`` / ``server_teacher_`` variants that
have no consumer here -- both adapters only ever look up the plain rollout actor.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

__all__ = ["control_actor_name", "DEFAULT_SERVER_NAME_PREFIX"]

DEFAULT_SERVER_NAME_PREFIX = "dynamo_"


def control_actor_name(
    engine_kwargs: Optional[Mapping[str, Any]],
    node_rank: int,
    *,
    prefix: str = DEFAULT_SERVER_NAME_PREFIX,
    name_suffix: str = "",
) -> str:
    """Name of the per-node Dynamo server actor that owns the control plane.

    Args:
        engine_kwargs: ``rollout.engine_kwargs``. The replica index is read from
            its ``dynamo.shared_pool_replica_rank`` entry -- every replica on a
            node shares one control actor, so this is 0 unless explicitly pooled.
        node_rank: Rank of the node *within the Dynamo replica*, not the trainer
            rank. The two differ whenever a replica spans fewer nodes than the job.
        prefix: Server-name prefix. The vLLM adapter overrides verl's
            ``_get_server_name_prefix``, so pass it through rather than hardcoding
            and let that override stay authoritative.
        name_suffix: Already-formatted suffix from ``RolloutReplica.name_suffix``
            (arrives as ``"_<x>"`` or ``""``). Upstream defaults it to empty; both
            adapters rely on that, and a non-empty value would otherwise make them
            miss the actor entirely.
    """
    dynamo_cfg = (engine_kwargs or {}).get("dynamo", {}) or {}
    shared_replica_rank = int(dynamo_cfg.get("shared_pool_replica_rank", 0))
    return f"{prefix}server_{shared_replica_rank}_{node_rank}{name_suffix}"
