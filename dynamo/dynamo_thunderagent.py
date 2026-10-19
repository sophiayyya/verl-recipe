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
"""ThunderAgent extension for the PR #110 Dynamo rollout backend."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import ray

from .dynamo_async_server import DynamoHttpServer, DynamoReplica

logger = logging.getLogger(__file__)

_REQUEST_PROGRAM: ContextVar[tuple[str, bool] | None] = ContextVar(
    "dynamo_thunderagent_request_program",
    default=None,
)
_THUNDERAGENT_BACKEND_MODEL_SUFFIX = "--verl-thunderagent-backend"
_REQUEST_PARENT: ContextVar[str | None] = ContextVar("dynamo_thunderagent_parent", default=None)


class DynamoThunderAgentHttpServer(DynamoHttpServer):
    """Add program-aware routing while preserving PR #110's server stack.

    Native mode installs the Rust classifier and worker selector in the
    frontend. Python mode retains the separate ThunderAgent router endpoint.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._thunderagent_process: Optional[subprocess.Popen] = None
        self._thunderagent_log_fp = None
        if self._thunderagent_enabled():
            self._router_mode = "kv" if self._native_thunderagent() else "round-robin"

    def _native_thunderagent(self) -> bool:
        implementation = self._thunderagent_config().get("implementation", "python")
        if implementation not in ("python", "native"):
            raise ValueError("thunderagent.implementation must be python or native")
        return self._thunderagent_enabled() and implementation == "native"

    def _dynamo_env_vars(self) -> dict[str, str]:
        env = super()._dynamo_env_vars()
        trace_path = os.environ.get("DYN_REQUEST_TRACE_OUTPUT_PATH")
        if self._native_thunderagent() and trace_path:
            # Frontends have independent writers and must not truncate each
            # other's evidence when multiple replicas share a filesystem.
            env["DYN_REQUEST_TRACE_OUTPUT_PATH"] = f"{trace_path}-replica{self.replica_rank}-node{self.node_rank}"
        return env

    def _frontend_extra_args(self) -> list[str]:
        args = super()._frontend_extra_args()
        if not self._native_thunderagent():
            return args
        if any(arg.split("=", 1)[0] in ("--router-policy-config", "--router-session-affinity-mode") for arg in args):
            raise ValueError("Native ThunderAgent owns router policy and soft affinity arguments")
        import yaml

        directory = Path(os.environ.get("VERL_DYNAMO_LOG_DIR", "/tmp"))
        directory.mkdir(parents=True, exist_ok=True)
        policy = directory / f"thunderagent-native-{self.replica_rank}.yaml"
        policy.write_text(
            yaml.safe_dump(
                {
                    "request_classifier": {
                        "type": "thunderagent",
                        "parameters": self._thunderagent_config().get("native_parameters", {}),
                    },
                    "worker_selection": {
                        "aggregated": "thunderagent",
                        "instances": [{"name": "thunderagent", "type": "thunderagent"}],
                    },
                }
            )
        )
        return args + ["--router-policy-config", str(policy), "--router-session-affinity-mode", "soft"]

    def _thunderagent_config(self) -> dict[str, Any]:
        config = self._dynamo_cfg().get("thunderagent", {}) or {}
        if not isinstance(config, dict):
            raise TypeError("rollout.engine_kwargs.dynamo.thunderagent must be a mapping")
        return config

    def _thunderagent_enabled(self) -> bool:
        value = self._thunderagent_config().get("enabled", False)
        if not isinstance(value, bool):
            raise TypeError("rollout.engine_kwargs.dynamo.thunderagent.enabled must be a boolean")
        return value

    def _thunderagent_router_block_size(self) -> int:
        value = int(self._thunderagent_config().get("router_block_size", 16))
        if value <= 0:
            raise ValueError("rollout.engine_kwargs.dynamo.thunderagent.router_block_size must be positive")
        return value

    def _thunderagent_extra_args(self) -> list[str]:
        value = self._thunderagent_config().get("extra_args", []) or []
        if not isinstance(value, list):
            raise TypeError("rollout.engine_kwargs.dynamo.thunderagent.extra_args must be a list")
        return [str(arg) for arg in value]

    def _build_thunderagent_cmd(self) -> list[str]:
        model_name = self._served_model_name or self.model_config.local_path
        return [
            sys.executable,
            "-m",
            "dynamo.thunderagent_router",
            "--endpoint",
            f"{self._namespace}.backend.generate",
            "--model-name",
            str(model_name),
            "--model-path",
            str(self.model_config.local_path),
            "--router-block-size",
            str(self._thunderagent_router_block_size()),
            *self._thunderagent_extra_args(),
        ]

    def _build_vllm_cmd(
        self,
        served_model_name: str,
        tp: int,
        kv_events_config_json: Optional[str] = None,
    ) -> list[str]:
        thunderagent_enabled = self._thunderagent_enabled() and not self._native_thunderagent()
        worker_model_name = (
            f"{served_model_name}{_THUNDERAGENT_BACKEND_MODEL_SUFFIX}" if thunderagent_enabled else served_model_name
        )
        command = super()._build_vllm_cmd(worker_model_name, tp, kv_events_config_json=kv_events_config_json)
        if not thunderagent_enabled:
            return command

        block_size = str(self._thunderagent_router_block_size())
        configured_block_size = None
        for index, arg in enumerate(command):
            if arg == "--block-size" and index + 1 < len(command):
                configured_block_size = command[index + 1]
            elif arg.startswith("--block-size="):
                configured_block_size = arg.partition("=")[2]
        if configured_block_size is not None and configured_block_size != block_size:
            raise ValueError(
                f"Dynamo vLLM and ThunderAgent router block sizes must match: {configured_block_size} != {block_size}"
            )
        if configured_block_size is None:
            command.extend(["--block-size", block_size])
        return command

    def _build_sglang_cmd(self, served_model_name: str, tp: int, nccl_port=None, **kwargs) -> list[str]:
        """TA worker-side glue for the sglang engine, mirroring _build_vllm_cmd.

        Extra keyword arguments (kv_events_config_json, ...) belong to the base builder and
        pass through untouched.

        Rename the worker's served model so the frontend cannot route requests
        (or session-final probes) directly to it — with ThunderAgent on, the
        ONLY handler advertising the public model name must be the TA router,
        otherwise generation silently bypasses program affinity. Page-size
        alignment is handled by the base builder (it falls back to
        thunderagent.router_block_size); an explicit sglang.page_size that
        contradicts the router block size fails fast here, same as the vLLM
        --block-size check.
        """
        if not self._thunderagent_enabled() or self._native_thunderagent():
            return super()._build_sglang_cmd(served_model_name, tp, nccl_port=nccl_port, **kwargs)
        explicit_page_size = self._sglang_cfg().get("page_size")
        router_block_size = self._thunderagent_router_block_size()
        if explicit_page_size is not None and int(explicit_page_size) != int(router_block_size):
            raise ValueError(
                "Dynamo sglang and ThunderAgent router block sizes must match: "
                f"sglang.page_size={explicit_page_size} != thunderagent.router_block_size={router_block_size}"
            )
        worker_model_name = f"{served_model_name}{_THUNDERAGENT_BACKEND_MODEL_SUFFIX}"
        return super()._build_sglang_cmd(worker_model_name, tp, nccl_port=nccl_port, **kwargs)

    def _start_thunderagent(self) -> None:
        if not self._thunderagent_enabled() or self._native_thunderagent() or self._thunderagent_process is not None:
            return
        env = os.environ.copy()
        env.update(self._dynamo_env_vars())
        log_root = os.environ.get("VERL_DYNAMO_LOG_DIR", "/tmp")
        os.makedirs(log_root, exist_ok=True)
        log_path = os.path.join(log_root, f"verl_dynamo_replica{self.replica_rank}_thunderagent.log")
        self._thunderagent_log_fp = open(log_path, "w")
        command = self._build_thunderagent_cmd()
        logger.info("[DynamoHttpServer] starting ThunderAgent (log=%s): %s", log_path, " ".join(command))
        self._thunderagent_process = subprocess.Popen(
            command,
            env=env,
            stdout=self._thunderagent_log_fp,
            stderr=subprocess.STDOUT,
        )

    def _start_frontend(self) -> None:
        if self._thunderagent_enabled():
            self._start_thunderagent()
        super()._start_frontend()

    def _raise_if_subprocess_died(self) -> None:
        super()._raise_if_subprocess_died()
        process = self._thunderagent_process
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"dynamo ThunderAgent exited rc={process.returncode}")

    def _frontend_headers(self, request_id: str) -> dict[str, str]:
        # Build on the base headers: register.py always serves through this class, so
        # the base router_session_affinity_ttl_secs header (x-dynamo-session-id keyed on
        # the trajectory request id) would otherwise never reach the wire. An active
        # ThunderAgent program overrides it -- same header, which the frontend matches
        # case-insensitively, so drop the base spelling before adding TA's.
        headers = super()._frontend_headers(request_id)
        request_program = _REQUEST_PROGRAM.get()
        if request_program is not None:
            session_id, is_final = request_program
            headers.pop("x-dynamo-session-id", None)
            headers["X-Dynamo-Session-ID"] = session_id
            parent = _REQUEST_PARENT.get()
            if parent:
                headers["X-Dynamo-Parent-Session-ID"] = parent
            if is_final:
                headers["X-Dynamo-Session-Final"] = "true"
        return headers

    async def generate(
        self,
        *args,
        thunderagent_session_id: Optional[str] = None,
        thunderagent_parent_session_id: Optional[str] = None,
        **kwargs,
    ):
        if not self._thunderagent_enabled():
            return await super().generate(*args, **kwargs)
        if not thunderagent_session_id:
            raise RuntimeError("ThunderAgent generation requires a non-empty session ID")
        if thunderagent_parent_session_id == thunderagent_session_id:
            raise ValueError("ThunderAgent parent session ID must differ from the session ID")
        if self._use_direct_generate():
            raise RuntimeError("direct_generate bypasses ThunderAgent and cannot be enabled")

        token = _REQUEST_PROGRAM.set((thunderagent_session_id, False))
        parent_token = _REQUEST_PARENT.set(thunderagent_parent_session_id)
        try:
            if self._native_thunderagent() and args:
                raise TypeError("Native ThunderAgent generate requires keyword arguments")
            return await super().generate(*args, **kwargs)
        finally:
            _REQUEST_PARENT.reset(parent_token)
            _REQUEST_PROGRAM.reset(token)

    async def _frontend_post(self, payload: dict[str, Any], request_id: str) -> tuple[int, str]:
        if self._native_thunderagent():
            # Allocate at the HTTP boundary, including extension-negotiation
            # and finalization retries. The caller's stable LB key stays local.
            program = _REQUEST_PROGRAM.get()
            prefix = "thunderagent-final" if program and program[1] else "req"
            request_id = f"{prefix}-{uuid4().hex}"
            if "request_id" in payload:
                payload = dict(payload, request_id=request_id)
        return await super()._frontend_post(payload, request_id)

    def _finalize_max_attempts(self) -> int:
        attempts = int(self._thunderagent_config().get("finalize_max_attempts", 3))
        if attempts <= 0:
            raise ValueError("thunderagent.finalize_max_attempts must be positive")
        return attempts

    def _finalize_retry_delay(self) -> float:
        delay = float(self._thunderagent_config().get("finalize_retry_delay_s", 0.1))
        if delay < 0:
            raise ValueError("thunderagent.finalize_retry_delay_s must be non-negative")
        return delay

    @staticmethod
    def _validate_final_response(status: int, body: str) -> None:
        if not 200 <= status < 300:
            raise RuntimeError(f"ThunderAgent final request failed status={status} body={body[:2000]!r}")
        if not body.strip():
            return
        data = json.loads(body)
        if data.get("choices"):
            raise RuntimeError("ThunderAgent final request was bypassed and reached a model worker")

    async def finalize_program(self, session_id: str) -> bool:
        if not self._thunderagent_enabled():
            return True
        if not session_id:
            raise ValueError("session_id must be non-empty")

        if self._native_thunderagent() and not self._generation_resumed.is_set():
            # A native final is a minimal generation. Let the Gateway return
            # completed trajectories and retry cleanup after the trainer resumes.
            return False

        request_id = f"thunderagent-final-{uuid4().hex}"
        payload = {
            "model": self._served_model_name or self.model_config.local_path,
            "prompt": [0],
            "max_tokens": 1,
            "stream": False,
        }
        token = _REQUEST_PROGRAM.set((session_id, True))
        try:
            last_error = None
            attempts = self._finalize_max_attempts()
            for attempt in range(1, attempts + 1):
                try:
                    status, body = await self._frontend_post(payload, request_id)
                except Exception as error:
                    last_error = error
                else:
                    if status not in {408, 429} and status < 500:
                        if self._native_thunderagent():
                            # Native finalization removes program state and then
                            # dispatches the minimal request. A normal completion
                            # is expected; Python's router instead short-circuits.
                            if not 200 <= status < 300:
                                raise RuntimeError(f"Native final request failed: {status} {body[:500]}")
                            data = json.loads(body)
                            if data.get("error") or "choices" not in data:
                                raise RuntimeError(f"Invalid native final response: {body[:500]}")
                        else:
                            self._validate_final_response(status, body)
                        return True
                    last_error = RuntimeError(f"ThunderAgent final request failed status={status} body={body[:2000]!r}")
                if attempt < attempts:
                    await asyncio.sleep(self._finalize_retry_delay())
            raise RuntimeError(f"ThunderAgent final request failed after {attempts} attempts") from last_error
        finally:
            _REQUEST_PROGRAM.reset(token)

    async def _healthcheck_frontend(self, expected_workers: int) -> None:
        await super()._healthcheck_frontend(
            expected_workers + int(self._thunderagent_enabled() and not self._native_thunderagent())
        )
        if self._thunderagent_enabled():
            await self.finalize_program(f"thunderagent-health-{time.time_ns()}")

    async def shutdown(self) -> None:
        frontend = self._frontend_process
        self._frontend_process = None
        if frontend is not None:
            self._stop_one(frontend, "frontend", 15)

        thunderagent = self._thunderagent_process
        self._thunderagent_process = None
        if thunderagent is not None:
            self._stop_one(thunderagent, "ThunderAgent", 15)

        try:
            await super().shutdown()
        finally:
            if self._thunderagent_log_fp is not None:
                self._thunderagent_log_fp.close()
                self._thunderagent_log_fp = None

    def __getstate__(self):
        state = super().__getstate__()
        state["_thunderagent_process"] = None
        state["_thunderagent_log_fp"] = None
        return state


class DynamoThunderAgentReplica(DynamoReplica):
    """Use the ThunderAgent-aware HTTP server for the PR #110 replica."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(DynamoThunderAgentHttpServer)


__all__ = ["DynamoThunderAgentHttpServer", "DynamoThunderAgentReplica"]
