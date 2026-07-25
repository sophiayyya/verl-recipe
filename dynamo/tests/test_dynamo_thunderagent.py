# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import inspect
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from recipe.dynamo import dynamo_async_server, dynamo_thunderagent, register
from recipe.dynamo.dynamo_agent_loop import (
    DynamoAgentLoopWorker,
    DynamoLLMServerManager,
    DynamoServerManager,
)
from recipe.dynamo.dynamo_async_server import DynamoHttpServer
from recipe.dynamo.dynamo_thunderagent import (
    DynamoThunderAgentHttpServer,
    DynamoThunderAgentReplica,
)

from verl.experimental.agent_loop.agent_loop import AgentLoopWorker

RECIPE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = RECIPE_ROOT.parent


def _make_http_server(thunderagent: dict | None = None) -> DynamoThunderAgentHttpServer:
    server = object.__new__(DynamoThunderAgentHttpServer)
    server.config = SimpleNamespace(engine_kwargs={"dynamo": {"thunderagent": thunderagent or {"enabled": True}}})
    server.model_config = SimpleNamespace(local_path="/models/test-model")
    server._namespace = "verl_dynamo"
    server._served_model_name = "test-model"
    server._router_mode = "round-robin"
    server.replica_rank = 0
    server._frontend_process = None
    server._thunderagent_process = None
    server._thunderagent_log_fp = None
    return server


class _RemoteMethod:
    def __init__(self, function):
        self.function = function

    async def remote(self, *args, **kwargs):
        result = self.function(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result


class _FakeServer:
    def __init__(self):
        self.generate_calls = []
        self.finalize_calls = []
        self.generate = _RemoteMethod(self._generate)
        self.finalize_program = _RemoteMethod(self._finalize_program)

    def _generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return kwargs

    def _finalize_program(self, session_id: str) -> None:
        self.finalize_calls.append(session_id)


def _generate_kwargs(prompt_id: int) -> dict:
    return {
        "request_id": None,
        "prompt_ids": [prompt_id],
        "sampling_params": {"max_tokens": 1},
    }


@pytest.mark.asyncio
async def test_client_reuses_program_id_for_all_turns_and_finalizes_once() -> None:
    server = _FakeServer()
    manager = DynamoServerManager([("frontend:8000", server)], thunderagent_enabled=True)

    async with manager.program_scope():
        first = await manager.generate(**_generate_kwargs(1))
        second = await manager.generate(**_generate_kwargs(2))

    session_id = first["thunderagent_session_id"]
    assert session_id
    assert second["thunderagent_session_id"] == session_id
    assert server.finalize_calls == [session_id]


@pytest.mark.asyncio
async def test_client_isolates_concurrent_programs() -> None:
    server = _FakeServer()
    manager = DynamoServerManager([("frontend:8000", server)], thunderagent_enabled=True)

    async def run(prompt_id: int) -> str:
        async with manager.program_scope():
            output = await manager.generate(**_generate_kwargs(prompt_id))
            return output["thunderagent_session_id"]

    first_id, second_id = await asyncio.gather(run(1), run(2))
    assert first_id != second_id
    assert sorted(server.finalize_calls) == sorted([first_id, second_id])


@pytest.mark.asyncio
async def test_enabled_client_fails_closed_without_program_scope() -> None:
    manager = DynamoServerManager([("frontend:8000", _FakeServer())], thunderagent_enabled=True)

    with pytest.raises(RuntimeError, match="active ThunderAgent program"):
        await manager.generate(**_generate_kwargs(1))


@pytest.mark.asyncio
async def test_disabled_client_preserves_pr110_request_path() -> None:
    server = _FakeServer()
    manager = DynamoServerManager([("frontend:8000", server)], thunderagent_enabled=False)

    async with manager.program_scope():
        await manager.generate(**_generate_kwargs(1))

    assert "thunderagent_session_id" not in server.generate_calls[0]
    assert server.finalize_calls == []


@pytest.mark.asyncio
async def test_client_omits_empty_audio_arguments_unsupported_by_pr110() -> None:
    server = _FakeServer()
    manager = DynamoServerManager([("frontend:8000", server)], thunderagent_enabled=False)

    await manager.generate(
        **_generate_kwargs(1),
        audio_data=None,
        mm_processor_kwargs=None,
    )

    assert "audio_data" not in server.generate_calls[0]
    assert "mm_processor_kwargs" not in server.generate_calls[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("unsupported_argument", "value"),
    [("audio_data", [b"audio"]), ("mm_processor_kwargs", {"sampling_rate": 16_000})],
)
async def test_client_rejects_unsupported_audio_arguments(unsupported_argument, value) -> None:
    server = _FakeServer()
    manager = DynamoServerManager([("frontend:8000", server)], thunderagent_enabled=False)

    with pytest.raises(RuntimeError, match="does not support audio inputs"):
        await manager.generate(**_generate_kwargs(1), **{unsupported_argument: value})

    assert server.generate_calls == []


@pytest.mark.asyncio
async def test_agent_loop_worker_wraps_parent_run_in_program_scope(monkeypatch) -> None:
    class ScopeClient:
        active = False
        exits = 0

        @asynccontextmanager
        async def program_scope(self):
            self.active = True
            try:
                yield
            finally:
                self.active = False
                self.exits += 1

    client = ScopeClient()
    worker = object.__new__(DynamoAgentLoopWorker)
    worker.llm_client = client

    async def parent_run(self, *args, **kwargs):
        assert self.llm_client.active
        assert args == ("sampling", "trajectory")
        assert kwargs == {"agent_name": "tool_agent"}
        return "output"

    monkeypatch.setattr(AgentLoopWorker, "_run_agent_loop", parent_run)

    assert await worker._run_agent_loop("sampling", "trajectory", agent_name="tool_agent") == "output"
    assert not client.active
    assert client.exits == 1


@pytest.mark.asyncio
async def test_agent_loop_worker_closes_scope_when_parent_fails(monkeypatch) -> None:
    class ScopeClient:
        exits = 0

        @asynccontextmanager
        async def program_scope(self):
            try:
                yield
            finally:
                self.exits += 1

    client = ScopeClient()
    worker = object.__new__(DynamoAgentLoopWorker)
    worker.llm_client = client

    async def parent_run(_self, *_args, **_kwargs):
        raise ValueError("agent failed")

    monkeypatch.setattr(AgentLoopWorker, "_run_agent_loop", parent_run)

    with pytest.raises(ValueError, match="agent failed"):
        await worker._run_agent_loop("sampling", "trajectory", agent_name="tool_agent")
    assert client.exits == 1


@pytest.mark.asyncio
async def test_server_manager_returns_direct_thunderagent_client() -> None:
    assert "_init_global_load_balancer" in DynamoLLMServerManager.__dict__
    manager = object.__new__(DynamoLLMServerManager)
    manager.server_addresses = ["frontend:8000"]
    manager.server_handles = [_FakeServer()]
    manager.rollout_config = SimpleNamespace(engine_kwargs={"dynamo": {"thunderagent": {"enabled": True}}})

    await manager._init_global_load_balancer()
    client = manager.get_client()

    assert isinstance(client, DynamoServerManager)
    assert client.thunderagent_enabled is True
    assert not hasattr(manager, "global_load_balancer")


def test_thunderagent_command_derives_endpoint_model_and_block_size() -> None:
    server = _make_http_server()

    assert server._build_thunderagent_cmd() == [
        sys.executable,
        "-m",
        "dynamo.thunderagent_router",
        "--endpoint",
        "verl_dynamo.backend.generate",
        "--model-name",
        "test-model",
        "--model-path",
        "/models/test-model",
        "--router-block-size",
        "16",
        "--router-reset-states",
    ]


def test_thunderagent_command_passes_configured_router_options() -> None:
    server = _make_http_server(
        {
            "enabled": True,
            "router_block_size": 32,
            "extra_args": ["--pause-threshold", "0.9"],
        }
    )

    command = server._build_thunderagent_cmd()

    assert command[command.index("--router-block-size") + 1] == "32"
    assert command[-2:] == ["--pause-threshold", "0.9"]


def test_thunderagent_extra_args_must_be_a_list() -> None:
    server = _make_http_server({"enabled": True, "extra_args": "--pause-threshold 0.9"})

    with pytest.raises(TypeError, match="extra_args must be a list"):
        server._build_thunderagent_cmd()


def test_vllm_and_thunderagent_share_block_size(monkeypatch) -> None:
    server = _make_http_server({"enabled": True, "router_block_size": 32})
    monkeypatch.setattr(
        DynamoHttpServer,
        "_build_vllm_cmd",
        lambda _self, _model, _tp, kv_events_config_json: ["python", "-m", "dynamo.vllm"],
    )

    command = server._build_vllm_cmd("test-model", 1, "{}")

    assert command[-2:] == ["--block-size", "32"]


def test_thunderagent_backend_workers_use_internal_model_name(monkeypatch) -> None:
    server = _make_http_server()
    base_calls = []

    def build_base_command(_self, served_model_name, _tp, _kv_events_config_json):
        base_calls.append(served_model_name)
        return ["python", "-m", "dynamo.vllm", "--served-model-name", served_model_name]

    monkeypatch.setattr(DynamoHttpServer, "_build_vllm_cmd", build_base_command)

    command = server._build_vllm_cmd("test-model", 1, "{}")
    thunderagent_command = server._build_thunderagent_cmd()

    assert base_calls == ["test-model--verl-thunderagent-backend"]
    assert command[command.index("--served-model-name") + 1] == "test-model--verl-thunderagent-backend"
    assert thunderagent_command[thunderagent_command.index("--model-name") + 1] == "test-model"
    assert server._served_model_name == "test-model"


def test_thunderagent_payload_supplies_worker_dp_rank(monkeypatch) -> None:
    server = _make_http_server()
    monkeypatch.setattr(
        DynamoHttpServer,
        "_build_frontend_completion_payload",
        lambda _self, _prompt, _sampling, _request_id: {
            "model": _self._served_model_name,
            "nvext": {"extra_fields": ["engine_data"]},
        },
    )

    payload = server._build_frontend_completion_payload([1], {}, "request-a")

    assert payload["model"] == "test-model"
    assert payload["nvext"] == {"extra_fields": ["engine_data"], "dp_rank": 0}


def test_frontend_start_starts_thunderagent_first(monkeypatch) -> None:
    server = _make_http_server()
    events = []
    monkeypatch.setattr(server, "_start_thunderagent", lambda: events.append("thunderagent"))
    monkeypatch.setattr(DynamoHttpServer, "_start_frontend", lambda _self: events.append("frontend"))

    server._start_frontend()

    assert events == ["thunderagent", "frontend"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("enabled", "expected_workers"), [(True, 5), (False, 4)])
async def test_frontend_health_waits_for_router_and_workers(monkeypatch, enabled, expected_workers) -> None:
    server = _make_http_server({"enabled": enabled})
    observed = []

    async def healthcheck(_self, workers):
        observed.append(workers)

    async def finalize(_session_id):
        pass

    monkeypatch.setattr(DynamoHttpServer, "_healthcheck_frontend", healthcheck)
    monkeypatch.setattr(server, "finalize_program", finalize)

    await server._healthcheck_frontend(4)

    assert observed == [expected_workers]


@pytest.mark.asyncio
async def test_generation_adds_program_header(monkeypatch) -> None:
    server = _make_http_server()
    monkeypatch.setattr(server, "_use_direct_generate", lambda: False)

    async def base_generate(self, *_args, **_kwargs):
        return self._frontend_headers("request-a")

    monkeypatch.setattr(DynamoHttpServer, "generate", base_generate)

    headers = await server.generate(
        prompt_ids=[1],
        sampling_params={"max_tokens": 1},
        request_id="request-a",
        thunderagent_session_id="program-a",
    )

    assert headers == {
        "X-Request-Id": "request-a",
        "X-Dynamo-Session-ID": "program-a",
    }


@pytest.mark.asyncio
async def test_generation_requires_program_and_rejects_direct_bypass(monkeypatch) -> None:
    server = _make_http_server()

    with pytest.raises(RuntimeError, match="session ID"):
        await server.generate(prompt_ids=[1], sampling_params={}, request_id="request-a")

    monkeypatch.setattr(server, "_use_direct_generate", lambda: True)
    with pytest.raises(RuntimeError, match="direct_generate"):
        await server.generate(
            prompt_ids=[1],
            sampling_params={},
            request_id="request-a",
            thunderagent_session_id="program-a",
        )


@pytest.mark.asyncio
async def test_finalize_retries_and_accepts_empty_choices(monkeypatch) -> None:
    server = _make_http_server(
        {
            "enabled": True,
            "finalize_max_attempts": 2,
            "finalize_retry_delay_s": 0,
        }
    )
    responses = [(503, "temporarily unavailable"), (200, json.dumps({"choices": []}))]
    calls = []

    async def frontend_post(_payload, request_id):
        calls.append(server._frontend_headers(request_id))
        return responses.pop(0)

    monkeypatch.setattr(server, "_frontend_post", frontend_post)

    await server.finalize_program("program-a")

    assert len(calls) == 2
    assert calls[-1]["X-Dynamo-Session-ID"] == "program-a"
    assert calls[-1]["X-Dynamo-Session-Final"] == "true"


@pytest.mark.asyncio
async def test_finalize_rejects_nonempty_choices_as_router_bypass(monkeypatch) -> None:
    server = _make_http_server({"enabled": True, "finalize_max_attempts": 1})

    async def frontend_post(_payload, _request_id):
        return 200, json.dumps({"choices": [{"text": "model answered"}]})

    monkeypatch.setattr(server, "_frontend_post", frontend_post)

    with pytest.raises(RuntimeError, match="bypassed"):
        await server.finalize_program("program-a")


def test_watchdog_detects_thunderagent_exit(monkeypatch) -> None:
    server = _make_http_server()
    server._thunderagent_process = SimpleNamespace(poll=lambda: 9, returncode=9)
    monkeypatch.setattr(DynamoHttpServer, "_raise_if_subprocess_died", lambda _self: None)

    with pytest.raises(RuntimeError, match="ThunderAgent.*rc=9"):
        server._raise_if_subprocess_died()


@pytest.mark.asyncio
async def test_shutdown_stops_frontend_then_thunderagent_then_base(monkeypatch) -> None:
    server = _make_http_server()
    server._frontend_process = object()
    server._thunderagent_process = object()
    events = []

    monkeypatch.setattr(server, "_stop_one", lambda _process, name, _timeout: events.append(name))

    async def base_shutdown(_self):
        events.append("base-workers-and-infra")

    monkeypatch.setattr(DynamoHttpServer, "shutdown", base_shutdown)

    await server.shutdown()

    assert events == ["frontend", "ThunderAgent", "base-workers-and-infra"]


def test_registered_replica_uses_thunderagent_server_class() -> None:
    assert DynamoThunderAgentReplica.__name__ == "DynamoThunderAgentReplica"


def test_recipe_registry_loads_thunderagent_replica() -> None:
    assert register._load_dynamo() is DynamoThunderAgentReplica


@pytest.mark.asyncio
async def test_llm_server_manager_constructs_thunderagent_replica(monkeypatch) -> None:
    constructed = []

    class FakeReplica:
        def __init__(self, **kwargs):
            constructed.append(kwargs)
            self._server_handle = object()
            self._server_address = "frontend:8000"

        async def init_hybrid_worker_pool(self, worker_group):
            assert worker_group == "worker-group"

    def reject_base_replica(**_kwargs):
        raise AssertionError("base PR #110 replica bypasses ThunderAgent")

    monkeypatch.setattr(dynamo_thunderagent, "DynamoThunderAgentReplica", FakeReplica)
    monkeypatch.setattr(dynamo_async_server, "DynamoReplica", reject_base_replica)
    manager = object.__new__(DynamoLLMServerManager)
    manager.worker_group = "worker-group"
    manager.rollout_config = SimpleNamespace(
        n_gpus_per_node=8,
        prometheus=SimpleNamespace(enable=False),
        name="dynamo",
    )
    manager.model_config = SimpleNamespace(local_path="test-model")

    await manager._initialize_llm_servers()

    assert constructed == [
        {
            "replica_rank": 0,
            "config": manager.rollout_config,
            "model_config": manager.model_config,
            "gpus_per_node": 8,
        }
    ]


def test_recipe_config_enables_thunderagent_agent_loop() -> None:
    config = yaml.safe_load((RECIPE_ROOT / "config" / "dynamo_trainer.yaml").read_text())
    rollout = config["actor_rollout_ref"]["rollout"]

    assert rollout["agent"]["agent_loop_manager_class"] == ("recipe.dynamo.dynamo_agent_loop.DynamoAgentLoopManager")
    assert rollout["engine_kwargs"]["dynamo"]["thunderagent"] == {
        "enabled": True,
        "router_block_size": 16,
    }


def test_recipe_pins_tested_verl_and_dynamo_revisions() -> None:
    required_verl = (RECIPE_ROOT / "REQUIRED_VERL.txt").read_text()
    readme = (RECIPE_ROOT / "README.md").read_text()
    repository_readme = (REPOSITORY_ROOT / "README.md").read_text()

    assert "MODE=pinned_commit" in required_verl
    assert "COMMIT=d82d2777b5dc3e96a8a45168d02660312707ab98" in required_verl
    assert "59d614641837e593f0567b79d75394aae5f864e0" in readme
    assert "dynamo/REQUIRED_VERL.txt" in repository_readme


def test_training_entrypoint_supports_pinned_and_current_verl_runner(monkeypatch) -> None:
    from recipe.dynamo import main_dynamo

    calls = []
    monkeypatch.setattr(main_dynamo, "auto_set_device", lambda _config: None)
    monkeypatch.setattr(main_dynamo, "migrate_legacy_reward_impl", lambda config: config)
    monkeypatch.setattr(main_dynamo, "run_ppo", lambda config, **kwargs: calls.append((config, kwargs)))

    main_dynamo.main.__wrapped__("config")

    expected_kwargs = {} if main_dynamo.TaskRunner is None else {"task_runner_class": main_dynamo.TaskRunner}
    assert calls == [("config", expected_kwargs)]
