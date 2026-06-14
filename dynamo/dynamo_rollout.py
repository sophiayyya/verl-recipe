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
"""ServerAdapter for the dynamo backend.

Inherits the vLLM ServerAdapter (HTTP path is identical: trainer rank reads
``replica.server_address`` and POSTs chat completions to it) and only
overrides the Ray actor name prefix used for sleep/wake/update_weights RPC,
so it lands on ``dynamo_server_*`` (created by DynamoReplica.launch_servers)
rather than ``vllm_server_*``.
"""

import logging
import os
import time
from typing import Generator

import ray
import torch

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender
from verl.workers.rollout.vllm_rollout.vllm_rollout import (
    ServerAdapter as _VllmServerAdapter,
)


class ServerAdapter(_VllmServerAdapter):
    """Per-rank dynamo client.

    All HTTP-based generation goes through the frontend URL stored in
    ``RolloutReplica.server_address``; weight-update / wake-up / sleep
    requests go to the per-node shared Ray actor named
    ``dynamo_server_{shared_pool_replica_rank}_{node_rank}``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Absolute node index from torchrun env. Parent's self.node_rank is
        # rollout_rank // local_world_size, which is always 0 when
        # rollout_world_size <= local_gpus and therefore misroutes RPCs to
        # dynamo_server_*_0 in multi-node deployments.
        rank = int(os.environ["RANK"])
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        self._dynamo_node_rank = rank // local_world_size
        self._dynamo_node_local_rank = rank % local_world_size
        # vLLM's wake_up('weights') is a silent no-op when sleep_level >= 2
        # under the dynamo bridge, so we keep weights GPU-resident through
        # every sleep cycle. update_weights_from_ipc then writes into the
        # live tensors directly.
        self.sleep_level = 1

    def _get_server_name_prefix(self) -> str:
        return "dynamo_"

    def _get_control_actor_name(self) -> str:
        """Return the shared Dynamo server actor name for this node."""
        dynamo_cfg = (self.config.engine_kwargs or {}).get("dynamo", {}) or {}
        shared_replica_rank = int(dynamo_cfg.get("shared_pool_replica_rank", 0))
        return f"{self._get_server_name_prefix()}server_{shared_replica_rank}_{self._dynamo_node_rank}"

    def _is_node_control_rank(self) -> bool:
        """True for the one trainer rank that controls Dynamo on this node."""
        return self._dynamo_node_local_rank == 0

    def _ensure_server_handle(self) -> bool:
        """Lazy-init the shared Dynamo control actor handle for this node.

        Overrides the parent gate (``rollout_rank != 0``) to fire exactly
        once per physical node. The parent shared DynamoHttpServer actor's
        ``collective_rpc`` already broadcasts to all node-local sidecars, so
        firing once per logical replica would duplicate sidecar RPCs while
        firing only on global rank 0 would miss non-master nodes. All
        parent methods (``_execute_method``, ``resume``, ``release``) reach
        this override transparently.
        """
        if not self._is_node_control_rank():
            return False
        if self.server_handle is None:
            self.server_handle = ray.get_actor(self._get_control_actor_name())
        return True

    @torch.no_grad()
    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int = None,
        **kwargs,
    ):
        """Push refreshed weights into the inference engine via CUDA IPC.

        The parent ``update_weights`` gates ``clear_kv_cache`` on
        ``rollout_rank == 0`` (one fire per logical replica), but our
        ``_ensure_server_handle`` only binds ``server_handle`` on the
        per-node control rank. The override keeps every ``server_handle``
        access on the same per-node gate so non-control ranks never reach
        a ``None`` handle.
        """
        start_time = time.time()

        # The naive path resumes vLLM inside ``WorkerDict.update_weights``;
        # the NCCL/NIXL path returns early before that resume runs, so we
        # wake the engine here (weights + kv_cache) before pushing weights
        # via CUDA IPC. Without this the IPC handshake finds no GPU buffers
        # and the next generation request after refit fails on missing KV.
        if self.config.free_cache_engine and self._is_node_control_rank():
            if self.server_handle is None:
                self.server_handle = ray.get_actor(self._get_control_actor_name())
            await self.server_handle.wake_up.remote(tags=["weights", "kv_cache"])

        future = await self._execute_method(
            "update_weights_from_ipc",
            non_block=True,
            kwargs={**kwargs, "use_shm": self.use_shm},
        )

        bucket_size_mb = self.config.checkpoint_engine.update_weights_bucket_megabytes
        sender = BucketedWeightSender(
            zmq_handle=self.zmq_handle,
            bucket_size_mb=bucket_size_mb,
            use_shm=self.use_shm,
        )
        await sender.async_send_weights(weights)

        if future is not None:
            await future

        if self._is_node_control_rank():
            await self.server_handle.clear_kv_cache.remote()
            if global_steps is not None:
                await self.server_handle.set_global_steps.remote(global_steps)

        if self.replica_rank == 0 and self.rollout_rank == 0:
            logger.info("update_weights done, time cost: %.2fs", time.time() - start_time)

    async def resume(self, tags: list[str]):
        """Wake the engine (weights and/or KV cache) before generation."""
        if not self.config.free_cache_engine:
            return None
        if not self._ensure_server_handle():
            return None
        t0 = time.perf_counter()
        await self.server_handle.wake_up.remote(tags=tags)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info("resume tags=%s elapsed_ms=%.2f", tags, elapsed_ms)

    async def release(self):
        """Put the engine to sleep at ``self.sleep_level`` between training steps."""
        if not self.config.free_cache_engine:
            return None
        if not self._ensure_server_handle():
            return None
        t0 = time.perf_counter()
        await self.server_handle.sleep.remote(level=self.sleep_level)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info("release level=%s elapsed_ms=%.2f", self.sleep_level, elapsed_ms)


__all__ = ["ServerAdapter"]
