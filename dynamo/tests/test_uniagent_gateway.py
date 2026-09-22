"""Exercise the unmodified Uni-Agent Gateway with the recipe lifecycle adapter.

Requires the optional Uni-Agent 0825e1ea checkout and its verl submodule.
"""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("uni_agent.gateway.gateway")

from fastapi import HTTPException
from recipe.dynamo.uniagent_gateway import _DynamoGatewayActor
from uni_agent.gateway.config import GatewayActorConfig
from uni_agent.gateway.gateway import _GatewayActor

from verl.workers.rollout.replica import TokenOutput


class Tokenizer:
    eos_token_id = ord("\n")

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **kwargs):
        text = "".join(f"{m['role']}:{m.get('content') or ''}\n" for m in messages)
        if add_generation_prompt:
            text += "assistant:"
        return [ord(c) for c in text] if tokenize else text

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, token_ids, **kwargs):
        return "".join(chr(int(token)) for token in token_ids)


class Backend:
    _auto_finalize = False

    def __init__(self):
        self.ids = []
        self.finalize_program = AsyncMock()

    async def generate(self, request_id, **kwargs):
        self.ids.append(request_id)
        return TokenOutput(token_ids=[65], log_probs=[-0.1], stop_reason="completed")


@pytest.fixture
def make_actor(monkeypatch):
    monkeypatch.setattr("ray.util.get_node_ip_address", lambda: "127.0.0.1")

    def make(cls=_DynamoGatewayActor):
        backend = Backend()
        actor = cls(GatewayActorConfig(tokenizer=Tokenizer()), backend)
        actor._server_base_url = "http://test"
        actor._retry_delay = 0
        return actor, backend

    return make


async def create(actor, session="session"):
    return await actor.create_session(session, sampling_params={"logprobs": True})


async def generate(actor, session="session", messages=None):
    return await actor._handle_openai_chat_completions(
        session_id=session,
        payload={"model": "test", "messages": messages or [{"role": "user", "content": "question"}]},
    )


@pytest.mark.asyncio
async def test_upstream_session_close_does_not_release_thunderagent(make_actor):
    # Regression evidence: a plain upstream upgrade alone cannot remove cleanup.
    actor, backend = make_actor(_GatewayActor)
    await create(actor)
    await generate(actor)
    await actor.finalize_session("session")
    backend.finalize_program.assert_not_awaited()


@pytest.mark.asyncio
async def test_two_turns_share_program_and_cleanup_preserves_tokens(make_actor):
    actor, backend = make_actor()
    await create(actor)
    first = await generate(actor)
    assistant = json.loads(first.body)["choices"][0]["message"]
    await generate(
        actor, messages=[{"role": "user", "content": "question"}, assistant, {"role": "user", "content": "continue"}]
    )
    backend.finalize_program.assert_not_awaited()
    trajectories = await actor.finalize_session("session")
    assert backend.ids == ["session", "session"]
    backend.finalize_program.assert_awaited_once_with("session")
    assert len(trajectories) == 1
    trajectory = trajectories[0]
    # Uni-Agent counts conversation messages in num_turns; backend.ids above
    # establishes the actual number of model calls.
    assert sum(trajectory.response_mask) == 2
    assert len(trajectory.response_ids) == len(trajectory.response_mask) == len(trajectory.response_logprobs)
    assert not actor._sessions
    assert await actor.finalize_session("session") is trajectories
    await actor.abort_session("session")
    assert backend.finalize_program.await_count == 1


@pytest.mark.asyncio
async def test_transient_cleanup_failure_retries(make_actor):
    actor, backend = make_actor()
    await create(actor)
    await generate(actor)
    backend.finalize_program.side_effect = [RuntimeError("busy"), TimeoutError(), None]
    await actor.finalize_session("session")
    assert backend.finalize_program.await_count == 3
    assert not actor._sessions


@pytest.mark.asyncio
async def test_exhausted_cleanup_is_visible_and_does_not_restart_retry_budget(make_actor):
    actor, backend = make_actor()
    await create(actor)
    await generate(actor)
    backend.finalize_program.side_effect = RuntimeError("unreachable")
    for method in [actor.finalize_session, actor.abort_session]:
        with pytest.raises(RuntimeError, match="unreachable"):
            await method("session")
    assert backend.finalize_program.await_count == 3
    assert "session" in actor._sessions
    with pytest.raises(HTTPException) as error:
        await generate(actor)
    assert error.value.status_code == 409


@pytest.mark.asyncio
async def test_cleanup_timeout_is_bounded(make_actor):
    actor, backend = make_actor()
    actor._finalize_timeout = 0.01
    await create(actor)

    async def hang(_):
        await asyncio.Future()

    backend.finalize_program.side_effect = hang
    with pytest.raises(TimeoutError):
        await actor.finalize_session("session")
    assert backend.finalize_program.await_count == 3
    assert "session" in actor._sessions


@pytest.mark.asyncio
async def test_repeated_waiter_cancellation_waits_for_cleanup(make_actor):
    actor, backend = make_actor()
    await create(actor)
    started, release = asyncio.Event(), asyncio.Event()

    async def finalize(_):
        started.set()
        await release.wait()

    backend.finalize_program.side_effect = finalize
    task = asyncio.create_task(actor.finalize_session("session"))
    await started.wait()
    for _ in range(3):
        task.cancel()
        await asyncio.sleep(0)
    assert not task.done()
    assert "session" in actor._sessions
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert "session" not in actor._sessions
    assert backend.finalize_program.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("abort", [False, True])
async def test_close_drains_or_cancels_the_complete_request(make_actor, abort):
    actor, backend = make_actor()
    await create(actor)
    started, release, stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = backend.generate

    async def pending(*args, **kwargs):
        started.set()
        try:
            await release.wait()
            return await original(*args, **kwargs)
        finally:
            stopped.set()

    async def finalized(_):
        assert stopped.is_set(), "program released while generation is still running"

    backend.generate = pending
    backend.finalize_program.side_effect = finalized
    request = asyncio.create_task(generate(actor))
    await started.wait()
    task = asyncio.create_task((actor.abort_session if abort else actor.finalize_session)("session"))
    await asyncio.sleep(0)
    backend.finalize_program.assert_not_awaited()
    assert not stopped.is_set()
    release.set()
    await task
    result = await asyncio.gather(request, return_exceptions=True)
    assert isinstance(result[0], asyncio.CancelledError) == abort
    assert not actor._sessions
    backend.finalize_program.assert_awaited_once_with("session")


@pytest.mark.asyncio
async def test_drain_timeout_keeps_live_program_and_reports_failure(make_actor):
    actor, backend = make_actor()
    actor._drain_timeout = 0.01
    await create(actor)
    started, release = asyncio.Event(), asyncio.Event()
    original = backend.generate

    async def pending(*args, **kwargs):
        started.set()
        await release.wait()
        return await original(*args, **kwargs)

    backend.generate = pending
    request = asyncio.create_task(generate(actor))
    await started.wait()
    with pytest.raises(TimeoutError, match="still has an active generation"):
        await actor.abort_session("session")
    backend.finalize_program.assert_not_awaited()
    assert not request.done()
    assert "session" in actor._sessions
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await request


@pytest.mark.asyncio
async def test_independent_sessions_run_concurrently_but_same_session_is_rejected(make_actor):
    actor, backend = make_actor()
    await create(actor, "one")
    await create(actor, "two")
    started, release = asyncio.Event(), asyncio.Event()
    original = backend.generate

    async def pending(*args, **kwargs):
        started.set()
        await release.wait()
        return await original(*args, **kwargs)

    backend.generate = pending
    request = asyncio.create_task(generate(actor, "one"))
    await started.wait()
    with pytest.raises(HTTPException) as error:
        await generate(actor, "one")
    assert error.value.status_code == 409
    second = asyncio.create_task(generate(actor, "two"))
    await asyncio.sleep(0)
    assert set(actor._requests) == {"one", "two"}
    release.set()
    await asyncio.gather(request, second)
    await asyncio.gather(actor.finalize_session("one"), actor.finalize_session("two"))
    assert backend.finalize_program.await_count == 2


@pytest.mark.asyncio
async def test_shutdown_attempts_all_sessions_even_if_one_cleanup_fails(make_actor):
    actor, backend = make_actor()
    for session in ["bad", "healthy"]:
        await create(actor, session)

    async def finalize(session):
        if session == "bad":
            raise RuntimeError("failed")

    backend.finalize_program.side_effect = finalize
    with pytest.raises(RuntimeError, match="failed"):
        await actor.shutdown()
    assert set(actor._sessions) == {"bad"}
    assert [call.args[0] for call in backend.finalize_program.await_args_list].count("healthy") == 1


@pytest.mark.asyncio
async def test_anthropic_requests_share_the_lifecycle(make_actor):
    actor, backend = make_actor()
    await create(actor)
    await actor._handle_anthropic_messages(
        session_id="session",
        payload={
            "model": "test",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "question"}],
        },
    )
    await actor.finalize_session("session")
    backend.finalize_program.assert_awaited_once_with("session")


@pytest.mark.asyncio
async def test_new_verl_routing_fields_reach_parent_client(monkeypatch):
    from collections import OrderedDict

    from recipe.dynamo.dynamo_agent_loop import DynamoFullyAsyncLLMServerClient

    from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient

    handle = object()
    acquire = AsyncMock(return_value=("worker", handle))
    monkeypatch.setattr(FullyAsyncLLMServerClient, "_acquire_server", acquire)
    client = object.__new__(DynamoFullyAsyncLLMServerClient)
    client._session_servers = OrderedDict()
    client._inflight_routing_sessions = {"request": "session"}
    assert await client._acquire_server("request", prompt_ids=[1, 2]) == ("worker", handle)
    acquire.assert_awaited_once_with("request", prompt_ids=[1, 2])
    assert client._session_servers["session"]["worker"] is handle
