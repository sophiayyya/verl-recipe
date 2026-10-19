"""Claude Code configuration for the shared agent-aware Dynamo Gateway."""

import ray

from .uniagent_gateway import DynamoAgentFrameworkRolloutAdapter, DynamoGatewayManager, _DynamoGatewayActor


class NativeClaudeGatewayActor(_DynamoGatewayActor):
    """Require Claude identity; Uni-Agent owns parsing and trajectory isolation."""

    _require_agent_context = True


RemoteNativeClaudeGatewayActor = ray.remote(NativeClaudeGatewayActor)


class NativeClaudeGatewayManager(DynamoGatewayManager):
    actor_class = RemoteNativeClaudeGatewayActor


class NativeClaudeRolloutAdapter(DynamoAgentFrameworkRolloutAdapter):
    gateway_manager_class = NativeClaudeGatewayManager

    @classmethod
    def create(cls, *, config, **kwargs):
        ta = config.actor_rollout_ref.rollout.engine_kwargs.dynamo.thunderagent
        if not ta.get("enabled") or ta.get("implementation") != "native" or ta.get("auto_finalize", True):
            raise ValueError("Native Claude integration requires native ThunderAgent and auto_finalize=false")
        return super().create(config=config, **kwargs)
