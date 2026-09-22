"""ThunderAgent session cleanup for the unmodified Uni-Agent Gateway API.

Integration baseline: Uni-Agent 0825e1ea and its verl submodule 3efe38c7.
Select DynamoAgentFrameworkRolloutAdapter through agent_loop_manager_class.
Only serial model calls within a session are supported; separate sessions run
concurrently. Backend state stays in the gateway actor that owns the client.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from dataclasses import replace

import ray
from fastapi import HTTPException
from omegaconf import OmegaConf
from uni_agent.framework.entry import AgentFrameworkRolloutAdapter, AgentFrameworkWorker
from uni_agent.gateway.config import GatewayActorConfig
from uni_agent.gateway.gateway import _GatewayActor
from uni_agent.gateway.manager import GatewayManager

from verl.utils.config import omega_conf_to_dataclass

logger = logging.getLogger(__name__)


async def _wait_for_cleanup(task):
    """Finish cleanup even if the waiter is cancelled repeatedly; then propagate."""
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    result = task.result()
    if cancelled:
        raise asyncio.CancelledError
    return result


class _DynamoGatewayActor(_GatewayActor):
    """Close the gateway session and the matching ThunderAgent program together."""

    _finalize_attempts = 3
    _finalize_timeout = 60.0
    _retry_delay = 0.2
    _drain_timeout = 60.0

    def __init__(self, config, backend):
        if not callable(getattr(backend, "finalize_program", None)):
            raise ValueError("Dynamo Gateway requires a client with finalize_program")
        if getattr(backend, "_auto_finalize", True):
            raise ValueError("Dynamo Gateway requires thunderagent.auto_finalize=false")
        # Dynamo uses vLLM tool parsers behind its own rollout registry name.
        super().__init__(replace(config, rollout_backend="vllm"), backend)
        self._requests = {}
        self._closing = {}
        self._closed = OrderedDict()

    async def create_session(self, session_id, **kwargs):
        if session_id in self._closing or session_id in self._closed:
            raise ValueError("Use a fresh session ID for every trajectory")
        return await super().create_session(session_id, **kwargs)

    async def _handle_request(self, handler, session_id, payload):
        if session_id in self._closing or session_id in self._closed:
            raise HTTPException(status_code=409, detail="Session is closing")
        if session_id in self._requests:
            raise HTTPException(status_code=409, detail="Dynamo Gateway requires serial calls within one session")
        self._requests[session_id] = asyncio.current_task()
        try:
            # Cancelling an HTTP waiter does not necessarily cancel its Ray
            # backend RPC. Keep the complete handler alive until that RPC and
            # token collection finish, so cleanup cannot release a live program.
            request = asyncio.create_task(handler(session_id=session_id, payload=payload))
            return await _wait_for_cleanup(request)
        finally:
            self._requests.pop(session_id, None)

    async def _handle_openai_chat_completions(self, *, session_id, payload):
        return await self._handle_request(super()._handle_openai_chat_completions, session_id, payload)

    async def _handle_anthropic_messages(self, *, session_id, payload):
        return await self._handle_request(super()._handle_anthropic_messages, session_id, payload)

    async def _release_program(self, session_id):
        for attempt in range(self._finalize_attempts):
            try:
                await asyncio.wait_for(self._backend.finalize_program(session_id), self._finalize_timeout)
                logger.info("ThunderAgent program finalized: session=%s", session_id)
                return
            except Exception:
                if attempt + 1 == self._finalize_attempts:
                    logger.exception("ThunderAgent program cleanup failed: session=%s", session_id)
                    raise
                await asyncio.sleep(self._retry_delay * 2**attempt)

    async def _close_session(self, session_id, abort):
        request = self._requests.get(session_id)
        if request is not None:
            if abort:
                request.cancel()
            # Await the complete HTTP handler, including token collection.
            # On timeout retain the closing state and surface failure.
            _, pending = await asyncio.wait({request}, timeout=self._drain_timeout)
            if pending:
                raise TimeoutError(f"Session {session_id} still has an active generation; program was not released")
        # The same actor owns the client's served-frontend records. A driver
        # copy of the client would not know every frontend visited on retries.
        await self._release_program(session_id)
        if abort:
            await super().abort_session(session_id)
            result = None
        else:
            result = await super().finalize_session(session_id)
        self._closed[session_id] = result
        self._closed.move_to_end(session_id)
        while len(self._closed) > 64:
            self._closed.popitem(last=False)
        self._closing.pop(session_id, None)
        return result

    async def _finish(self, session_id, *, abort):
        if session_id in self._closed:
            return None if abort else self._closed[session_id]
        if session_id not in self._sessions and session_id not in self._closing:
            if abort:
                return None
            raise KeyError(session_id)
        task = self._closing.get(session_id)
        if task is None:
            task = asyncio.create_task(self._close_session(session_id, abort))
            self._closing[session_id] = task
        # Reuse the task and its bounded retry budget, including failures.
        return await _wait_for_cleanup(task)

    async def finalize_session(self, session_id):
        return await self._finish(session_id, abort=False)

    async def abort_session(self, session_id):
        await self._finish(session_id, abort=True)

    async def shutdown(self):
        try:
            results = await asyncio.gather(
                *(self.abort_session(session_id) for session_id in tuple(self._sessions)), return_exceptions=True
            )
            for result in results:
                if isinstance(result, BaseException):
                    raise result
        finally:
            await super().shutdown()


DynamoGatewayActor = ray.remote(_DynamoGatewayActor)


class DynamoGatewayManager(GatewayManager):
    """Use the recipe actor while retaining upstream session routing methods."""

    def __init__(self, llm_client, *, gateway_count, gateway_actor_config):
        if gateway_count <= 0:
            raise ValueError("gateway_count must be positive")
        nodes = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        if not nodes:
            raise RuntimeError("No alive CPU nodes available for DynamoGatewayActor")
        self.gateways = [
            DynamoGatewayActor.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=nodes[i % len(nodes)], soft=True
                )
            ).remote(gateway_actor_config, backend=llm_client)
            for i in range(gateway_count)
        ]
        ray.get([gateway.start.remote() for gateway in self.gateways])
        self.gateway_count = len(self.gateways)
        self.active_sessions_per_gateway = [0 for _ in self.gateways]
        self._session_to_gateway_index = {}


def _build_gateway_manager(config, llm_client):
    """Mirror Uni-Agent's GatewayActorConfig wiring without replacing globals."""
    af = config.actor_rollout_ref.rollout.custom.agent_framework
    model = omega_conf_to_dataclass(config.actor_rollout_ref.model)
    rollout = omega_conf_to_dataclass(config.actor_rollout_ref.rollout)
    allowed = af.get("allowed_request_sampling_param_keys")
    if allowed is not None:
        if not (isinstance(allowed, list | tuple) or OmegaConf.is_list(allowed)):
            raise ValueError("allowed_request_sampling_param_keys must be a list of strings or null")
        if any(not isinstance(key, str) for key in allowed):
            raise ValueError("allowed_request_sampling_param_keys must be a list of strings or null")
    actor_config = GatewayActorConfig(
        tokenizer=model.tokenizer,
        processor=model.processor,
        tool_parser_name=rollout.multi_turn.format,
        rollout_backend="vllm",
        enable_tool_parser_cache=af.get("enable_tool_parser_cache", True),
        hf_model_type=getattr(model.hf_config, "model_type", None),
        apply_chat_template_kwargs=dict(config.data.get("apply_chat_template_kwargs", {})),
        mm_processor_kwargs=dict(config.data.get("mm_processor_kwargs", {})),
        prompt_length=rollout.prompt_length,
        response_length=rollout.response_length,
        enable_last_assistant_rollback=af.get("enable_last_assistant_rollback", True),
        allowed_request_sampling_param_keys=None if allowed is None else set(allowed),
        coalesce_reserved_exact_requests=af.get("coalesce_reserved_exact_requests", True),
    )
    return DynamoGatewayManager(llm_client, gateway_count=int(af["gateway_count"]), gateway_actor_config=actor_config)


class DynamoAgentFrameworkRolloutAdapter(AgentFrameworkRolloutAdapter):
    """V1 adapter for current Uni-Agent, without modifying its source tree."""

    @classmethod
    def create(cls, *, config, llm_client, teacher_client=None, reward_loop_worker_handles=None, **_):
        if teacher_client is not None:
            raise ValueError("Dynamo Uni-Agent Gateway does not support teacher_client")
        dynamo = config.actor_rollout_ref.rollout.engine_kwargs.dynamo
        if dynamo.get("engine", "vllm") != "vllm":
            raise ValueError("Dynamo Uni-Agent Gateway currently supports only vLLM")
        if not dynamo.thunderagent.enabled or dynamo.thunderagent.get("auto_finalize", True):
            raise ValueError("Enable ThunderAgent with auto_finalize=false for the Uni-Agent Gateway")
        gateway_manager = _build_gateway_manager(config, llm_client)
        instance = cls()
        instance.gateway_manager = gateway_manager
        instance.framework_worker = AgentFrameworkWorker.remote(
            config=config, gateway_manager=gateway_manager, reward_loop_worker_handles=reward_loop_worker_handles
        )
        return instance
