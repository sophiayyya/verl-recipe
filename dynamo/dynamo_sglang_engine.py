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
"""HTTP control-plane client for ``dynamo.sglang`` workers.

Unlike the vLLM path (which needs ``_dynamo_vllm_with_control.py``, a verl-private
ZMQ REP sidecar, because bare ``dynamo.vllm`` exposes no hook onto its AsyncLLM),
``dynamo.sglang`` already registers a full RL control plane of its own:

``components/src/dynamo/sglang/request_handlers/handler_base.py::register_engine_routes``
registers, unconditionally::

    control/start_profile              control/stop_profile
    control/release_memory_occupation  control/resume_memory_occupation
    control/update_weights_from_disk   control/update_weights_from_tensor
    control/update_weights_from_distributed
    control/update_weights_from_ipc    control/update_weight_version

and, when the worker is started with ``--enable-rl``, additionally::

    call_tokenizer_manager    # generic passthrough to any tokenizer_manager method

Those land on the worker's system-status server, which mounts ``/engine/{*path}``
(``lib/runtime/src/system_status_server.rs``) on ``$DYN_SYSTEM_PORT``. So the whole
control plane is plain HTTP POST against
``http://<worker-host>:<DYN_SYSTEM_PORT>/engine/<registered-key>``.

Two consequences drive the design here:

1. ``DYN_SYSTEM_PORT`` is **mandatory** for the sglang backend (it is merely a
   metrics nicety for vLLM). Dynamo's Rust runtime parses it as i16, so the port
   must stay below 32768 — see ``_allocate_stable_node_port`` on the actor side.
2. There is **no** ``control/flush_cache`` route. ``clear_kv_blocks`` exists on the
   handler but is not registered as an engine route, so flushing the radix cache
   after a weight update has to go through ``call_tokenizer_manager``, which means
   ``--enable-rl`` is effectively required for RL use. See :meth:`flush_cache`.

All of the above verified against dynamo 1.3.0 (94accc7389) + sglang 0.5.14 on an
H100 (M0b, job 16214273): ``call_tokenizer_manager``, ``release_memory_occupation``,
``resume_memory_occupation`` and ``update_weight_version`` all answer 200, while
``control/flush_cache`` answers 404 ``"Route not found"``.

**Robustness note.** A ``serialized_named_tensors`` payload that fails to
deserialize does not come back as an error — it kills the worker process outright
(observed as ``RemoteDisconnected``, then connection-refused for everything after;
M0c, job 16215105). So a malformed weight sync is an engine restart, not a retryable
failure, and the caller cannot distinguish "bad payload" from "worker crashed for an
unrelated reason". Keep the wire format exactly as SGLang specifies it.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Optional

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

# Registered engine-route keys. Mirrors handler_base.register_engine_routes;
# kept as constants so a dynamo-side rename fails loudly in one place.
ROUTE_RELEASE_MEMORY = "control/release_memory_occupation"
ROUTE_RESUME_MEMORY = "control/resume_memory_occupation"
ROUTE_UPDATE_WEIGHTS_FROM_TENSOR = "control/update_weights_from_tensor"
ROUTE_UPDATE_WEIGHTS_FROM_IPC = "control/update_weights_from_ipc"
ROUTE_UPDATE_WEIGHT_VERSION = "control/update_weight_version"
ROUTE_START_PROFILE = "control/start_profile"
ROUTE_STOP_PROFILE = "control/stop_profile"
ROUTE_CALL_TOKENIZER_MANAGER = "call_tokenizer_manager"


class DynamoSGLangControlError(RuntimeError):
    """Raised when a control route returns non-200 or a ``status: error`` body."""


class DynamoSGLangControlClient:
    """Async HTTP client for one ``dynamo.sglang`` worker's ``/engine/*`` routes.

    One instance per DP shard. The node-level ``DynamoHttpServer`` actor holds a
    list of them (one per local shard) and fans out; a trainer-side ServerAdapter
    holds exactly the one that matches its own shard, so CUDA-IPC handles are only
    ever posted to the engine that shares its GPUs.
    """

    def __init__(self, base_url: str, timeout_s: float = 600.0):
        """
        Args:
            base_url: ``http://host:port`` of the worker's system-status server.
            timeout_s: default per-request timeout.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._session = None

    # ------------------------------------------------------------------ #
    # transport
    # ------------------------------------------------------------------ #

    async def _get_session(self):
        if self._session is not None and not self._session.closed:
            return self._session
        import aiohttp

        self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    def route_url(self, route: str) -> str:
        return f"{self.base_url}/engine/{route}"

    async def post(
        self,
        route: str,
        body: Optional[dict] = None,
        timeout_s: Optional[float] = None,
        raise_on_error_status: bool = True,
    ) -> dict:
        """POST one control request and return the decoded JSON body.

        ``/engine/{*path}`` is mounted with axum's ``any(...)``, so the verb is
        not load-bearing; POST is used for every route for uniformity.
        """
        import aiohttp

        session = await self._get_session()
        url = self.route_url(route)
        timeout = aiohttp.ClientTimeout(total=timeout_s if timeout_s is not None else self.timeout_s)
        async with session.post(url, json=body or {}, timeout=timeout) as resp:
            text = await resp.text()
            if resp.status != 200:
                raise DynamoSGLangControlError(f"{url} -> HTTP {resp.status}: {text[:2000]!r}")
            try:
                data = await resp.json(content_type=None)
            except Exception as e:
                raise DynamoSGLangControlError(f"{url} -> non-JSON body: {text[:2000]!r}") from e

        if raise_on_error_status and isinstance(data, dict):
            # Handlers answer either {"status": "ok"|"error", ...} or
            # {"success": bool, "message": str}; treat both as failures.
            if data.get("status") == "error":
                raise DynamoSGLangControlError(f"{url} -> {data.get('message')!r}")
            if data.get("success") is False:
                raise DynamoSGLangControlError(f"{url} -> {data.get('message')!r}")
        return data

    # ------------------------------------------------------------------ #
    # memory occupation (sleep / wake)
    # ------------------------------------------------------------------ #

    async def release_memory_occupation(self, tags: Optional[list[str]] = None) -> dict:
        """Release GPU memory for ``tags`` (``["kv_cache"]`` / ``["kv_cache","weights"]``).

        Requires the worker to have been started with ``--enable-memory-saver``;
        without it SGLang's torch_memory_saver is inactive and this is a no-op at
        best. The handler also *unregisters the worker from discovery* first, so a
        released worker leaves the routing pool until ``resume_memory_occupation``.
        """
        body: dict[str, Any] = {}
        if tags is not None:
            body["tags"] = list(tags)
        return await self.post(ROUTE_RELEASE_MEMORY, body)

    async def resume_memory_occupation(self, tags: Optional[list[str]] = None) -> dict:
        body: dict[str, Any] = {}
        if tags is not None:
            body["tags"] = list(tags)
        return await self.post(ROUTE_RESUME_MEMORY, body)

    # ------------------------------------------------------------------ #
    # weight sync
    # ------------------------------------------------------------------ #

    async def update_weights_from_tensor(self, req, timeout_s: Optional[float] = None) -> dict:
        """POST one ``UpdateWeightsFromTensorReqInput`` worth of CUDA-IPC handles.

        ``req.serialized_named_tensors`` is ``list[bytes]`` (one entry per SGLang TP
        rank, each a ``MultiprocessingSerializer`` blob of CUDA-IPC handles). JSON has
        no bytes type, so it goes on the wire base64-encoded.

        Base64 is not a workaround here, it is SGLang's documented contract for this
        field — ``UpdateWeightsFromTensorReqInput.serialized_named_tensors`` is typed
        ``List[Union[str, bytes]]`` and ``MultiprocessingSerializer.deserialize``
        b64-decodes any ``str`` it is handed. Dynamo's engine route passes the JSON
        body through untouched, which is exactly right. Verified end-to-end against an
        unmodified dynamo 1.3.0 + sglang 0.5.14 (M0c, job 16215105).

        Do NOT "fix" this by decoding server-side: a payload that fails to
        deserialize takes the whole worker process down (see the class docstring),
        so the encoding must match what SGLang expects, not what looks symmetric.
        """
        body = {
            "serialized_named_tensors": [
                base64.b64encode(b).decode("utf-8") for b in req.serialized_named_tensors
            ],
            "load_format": req.load_format,
            "flush_cache": req.flush_cache,
        }
        return await self.post(ROUTE_UPDATE_WEIGHTS_FROM_TENSOR, body, timeout_s=timeout_s)

    async def update_weight_version(self, new_version: str, abort_all_requests: bool = False) -> dict:
        return await self.post(
            ROUTE_UPDATE_WEIGHT_VERSION,
            {"new_version": str(new_version), "abort_all_requests": bool(abort_all_requests)},
        )

    # ------------------------------------------------------------------ #
    # tokenizer_manager passthrough (needs --enable-rl)
    # ------------------------------------------------------------------ #

    async def call_tokenizer_manager(
        self,
        method: str,
        args: Optional[list] = None,
        kwargs: Optional[dict] = None,
        timeout_s: Optional[float] = None,
    ) -> dict:
        """Invoke an arbitrary ``tokenizer_manager`` method.

        Only registered when the worker runs with ``--enable-rl``. Args/kwargs are
        plain JSON values, or ``{"io_struct.ClassName": {...}}`` for a typed
        ``sglang.srt.managers.io_struct`` constructor.
        """
        return await self.post(
            ROUTE_CALL_TOKENIZER_MANAGER,
            {"method": method, "args": args or [], "kwargs": kwargs or {}},
            timeout_s=timeout_s,
        )

    async def flush_cache(self) -> dict:
        """Drop the radix/prefix cache.

        There is no ``control/flush_cache`` engine route — ``clear_kv_blocks``
        exists on the handler but is never registered — so this goes through
        ``call_tokenizer_manager``. **A stale prefix cache after a weight update
        silently serves tokens from the old policy**, so this is not optional in
        RL; it is the main reason the recipe forces ``--enable-rl``.
        """
        return await self.call_tokenizer_manager("flush_cache")

    async def abort_request(self, rid: str = "", abort_all: bool = False) -> dict:
        return await self.call_tokenizer_manager(
            "abort_request", kwargs={"rid": rid, "abort_all": bool(abort_all)}
        )

    # ------------------------------------------------------------------ #
    # profiling
    # ------------------------------------------------------------------ #

    async def start_profile(self, **kwargs) -> dict:
        return await self.post(ROUTE_START_PROFILE, dict(kwargs))

    async def stop_profile(self) -> dict:
        return await self.post(ROUTE_STOP_PROFILE, {})

    # ------------------------------------------------------------------ #
    # health
    # ------------------------------------------------------------------ #

    async def wait_ready(self, timeout_s: float = 1800.0, poll_s: float = 5.0) -> bool:
        """Block until this worker's ``/engine/*`` routes are actually serving.

        The obvious readiness check — "does the port answer" — is wrong twice over,
        and both failure modes were observed on the first real run (M0b, job 16213723):

        1. The system-status server binds ``DYN_SYSTEM_PORT`` within ~20s of process
           start, but the Python engine routes are only registered once the model has
           finished loading. In between, every ``/engine/*`` path 404s while the port
           happily accepts connections. Probing too early "succeeds" against a server
           that cannot do anything.
        2. A 404 is a perfectly good HTTP response, so any check that only asks
           "did I get a reply" passes during that window.

        So gate on a **200** from ``call_tokenizer_manager``: it proves the engine
        routes are registered, the tokenizer_manager is alive, and ``--enable-rl``
        actually took — all three of which the weight-sync path needs.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout_s
        last: str = "no attempt made"
        while loop.time() < deadline:
            try:
                await self.call_tokenizer_manager("flush_cache", timeout_s=15.0)
                return True
            except Exception as e:  # noqa: BLE001 - 404/transport both mean not-ready-yet
                last = f"{type(e).__name__}: {str(e)[:200]}"
                await asyncio.sleep(poll_s)
        raise DynamoSGLangControlError(
            f"dynamo.sglang engine routes at {self.base_url} not serving after {timeout_s}s. "
            f"Last: {last}. If this is a 404 the worker is up but never finished loading the "
            f"model; if it mentions call_tokenizer_manager the worker is missing --enable-rl."
        )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"DynamoSGLangControlClient({self.base_url!r})"


__all__ = [
    "DynamoSGLangControlClient",
    "DynamoSGLangControlError",
]
