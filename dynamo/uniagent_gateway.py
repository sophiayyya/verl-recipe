"""ThunderAgent program lifecycle for the unmodified Uni-Agent Gateway API.

Integration baseline: Uni-Agent 0825e1ea and its verl submodule 3efe38c7.
Select DynamoAgentFrameworkRolloutAdapter through agent_loop_manager_class.

A gateway session is one trajectory, but not necessarily one program: a Claude
Code run that dispatches subagents is a tree of them. The agent announces that
tree in its request headers, which the Gateway terminates, so this module reads
them here (``agent_identity``), names every turn's program for the client, and
releases the tree -- subagents first, then the root -- when the session closes.
Model calls are serial within a program; programs run concurrently, whether they
are the subagents of one session or separate sessions. Backend state stays in
the gateway actor that owns the client.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import replace

import ray
from fastapi import HTTPException
from omegaconf import OmegaConf
from starlette.datastructures import Headers
from uni_agent.framework.entry import AgentFrameworkRolloutAdapter, AgentFrameworkWorker
from uni_agent.gateway.config import GatewayActorConfig
from uni_agent.gateway.gateway import _GatewayActor
from uni_agent.gateway.manager import GatewayManager

from verl.utils.config import omega_conf_to_dataclass

from .agent_identity import AgentIdentity, agent_identity_from_headers

logger = logging.getLogger(__name__)

# The identity the current HTTP request announced, bound for the request's task
# and every task it spawns, down to the backend call.
_REQUEST_IDENTITY: ContextVar[AgentIdentity | None] = ContextVar("dynamo_gateway_agent_identity", default=None)


class _AgentIdentityMiddleware:
    """Bind each request's coding-agent identity headers to its task.

    A pure ASGI middleware runs the route in the same task, so a ``ContextVar``
    set here is visible to the handler, to the task it spawns, and to the client
    call at the bottom.
    """

    def __init__(self, app):
        self._app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self._app(scope, receive, send)
        token = _REQUEST_IDENTITY.set(agent_identity_from_headers(Headers(scope=scope)))
        try:
            return await self._app(scope, receive, send)
        finally:
            _REQUEST_IDENTITY.reset(token)


def _program_of(session_id: str, identity: AgentIdentity | None) -> tuple[str, str | None]:
    """The program a turn belongs to and its parent; a turn without identity headers is the session."""
    if identity is None:
        return session_id, None
    return identity.session_id, identity.parent_session_id


class _ProgramClient:
    """The session's view of the ThunderAgent client: a program per agent identity.

    Uni-Agent's session calls ``generate`` with its session id as the request id
    and knows nothing about programs. The request id stays the routing key, so a
    whole tree reaches one frontend; the program named here is what the router
    schedules and what the actor later releases.
    """

    def __init__(self, client):
        self._client = client

    async def generate(self, request_id, **kwargs):
        program_id, parent_program_id = _program_of(request_id, _REQUEST_IDENTITY.get())
        return await self._client.generate(
            request_id, program_id=program_id, parent_program_id=parent_program_id, **kwargs
        )

    async def finalize_program(self, program_id):
        await self._client.finalize_program(program_id)


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
    """Close the gateway session and every ThunderAgent program it created together."""

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
        super().__init__(replace(config, rollout_backend="vllm"), _ProgramClient(backend))
        self._app.add_middleware(_AgentIdentityMiddleware)
        self._requests = {}
        # session -> {program: parent}, in creation order; a child is created after its parent.
        self._programs = {}
        self._closing = {}
        self._closed = OrderedDict()

    async def create_session(self, session_id, **kwargs):
        if session_id in self._closing or session_id in self._closed:
            raise ValueError("Use a fresh session ID for every trajectory")
        return await super().create_session(session_id, **kwargs)

    async def _handle_request(self, handler, session_id, payload):
        if session_id in self._closing or session_id in self._closed:
            raise HTTPException(status_code=409, detail="Session is closing")
        program_id, parent_program_id = _program_of(session_id, _REQUEST_IDENTITY.get())
        inflight = self._requests.setdefault(session_id, {})
        if program_id in inflight:
            raise HTTPException(status_code=409, detail="Dynamo Gateway requires serial calls within one program")
        inflight[program_id] = asyncio.current_task()
        try:
            self._programs.setdefault(session_id, {}).setdefault(program_id, parent_program_id)
            await self._release_returned_children(session_id, program_id)
            # Cancelling an HTTP waiter does not necessarily cancel its Ray
            # backend RPC. Keep the complete handler alive until that RPC and
            # token collection finish, so cleanup cannot release a live program.
            request = asyncio.create_task(handler(session_id=session_id, payload=payload))
            return await _wait_for_cleanup(request)
        finally:
            inflight.pop(program_id, None)
            if not inflight:
                self._requests.pop(session_id, None)

    async def _handle_openai_chat_completions(self, *, session_id, payload):
        return await self._handle_request(super()._handle_openai_chat_completions, session_id, payload)

    async def _handle_anthropic_messages(self, *, session_id, payload):
        return await self._handle_request(super()._handle_anthropic_messages, session_id, payload)

    async def _release_program(self, program_id):
        for attempt in range(self._finalize_attempts):
            try:
                await asyncio.wait_for(self._backend.finalize_program(program_id), self._finalize_timeout)
                logger.info("ThunderAgent program finalized: program=%s", program_id)
                return
            except Exception:
                if attempt + 1 == self._finalize_attempts:
                    logger.exception("ThunderAgent program cleanup failed: program=%s", program_id)
                    raise
                await asyncio.sleep(self._retry_delay * 2**attempt)

    async def _release_returned_children(self, session_id, program_id):
        """Release the idle subagents of a program that is taking a new turn.

        Nothing marks a subagent final, and the agent that dispatched it blocks
        until it returns, so the dispatcher's next turn is the signal that its
        idle children are done. Left resident until the session closes they keep
        charging the router's capacity, and under pressure can pin the whole
        tree: the root is paused for lack of room that only its own release
        could make. A child still mid-request (a background subagent) is kept.
        """
        tree = self._programs.get(session_id, {})
        inflight = self._requests.get(session_id, {})
        for child in [child for child, parent in tree.items() if parent == program_id and child not in inflight]:
            await self._release_returned_children(session_id, child)
            await self._release_program(child)
            tree.pop(child, None)

    def _session_programs(self, session_id):
        """Every program the session created, children before the root that dispatched them."""
        programs = list(self._programs.get(session_id, {}))
        return list(reversed(programs)) if programs else [session_id]

    async def _close_session(self, session_id, abort):
        requests = list(self._requests.get(session_id, {}).values())
        if requests:
            if abort:
                for request in requests:
                    request.cancel()
            # Await the complete HTTP handlers, including token collection.
            # On timeout retain the closing state and surface failure.
            _, pending = await asyncio.wait(requests, timeout=self._drain_timeout)
            if pending:
                raise TimeoutError(f"Session {session_id} still has an active generation; program was not released")
        # The same actor owns the client's served-frontend records. A driver
        # copy of the client would not know every frontend visited on retries.
        for program_id in self._session_programs(session_id):
            await self._release_program(program_id)
        self._programs.pop(session_id, None)
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
