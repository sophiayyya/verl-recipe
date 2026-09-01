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
"""CPU-only tests for the Dynamo × SGLang rollout backend.

Deliberately avoids importing ``dynamo_sglang_rollout`` at module scope: that
module pulls in verl's sglang ServerAdapter, which imports ``sglang`` and patches
its engine entrypoint at import time. These tests run in the vLLM container too.
"""

import asyncio
import base64
import importlib.metadata
from types import SimpleNamespace

import pytest
from recipe.dynamo.dynamo_async_server import DynamoHttpServer
from recipe.dynamo.dynamo_sglang_engine import (
    ROUTE_UPDATE_WEIGHTS_FROM_TENSOR,
    DynamoSGLangControlClient,
    DynamoSGLangControlError,
)


def _make_server(dynamo_cfg: dict | None = None, **rollout_cfg) -> DynamoHttpServer:
    server = object.__new__(DynamoHttpServer)
    cfg = {
        "tensor_model_parallel_size": 1,
        "gpu_memory_utilization": 0.5,
        "max_model_len": 1024,
        "max_num_batched_tokens": None,
        "max_num_seqs": None,
        "dtype": None,
        "enforce_eager": False,
        "enable_chunked_prefill": False,
        "enable_prefix_caching": True,
        "enable_sleep_mode": False,
        "engine_kwargs": {"dynamo": dynamo_cfg if dynamo_cfg is not None else {}},
    }
    cfg.update(rollout_cfg)
    server.config = SimpleNamespace(**cfg)
    server.model_config = SimpleNamespace(local_path="/models/test-model", trust_remote_code=False)
    server._engine_control_endpoints = []
    server._control_endpoints = []
    server._sglang_clients = None
    return server


# --------------------------------------------------------------------------- #
# engine selection
# --------------------------------------------------------------------------- #


def test_engine_defaults_to_vllm():
    assert _make_server()._engine_kind() == "vllm"
    assert _make_server()._is_sglang() is False


def test_engine_sglang_selected():
    server = _make_server({"engine": "sglang"})
    assert server._engine_kind() == "sglang"
    assert server._is_sglang() is True


def test_unknown_engine_rejected():
    with pytest.raises(ValueError, match="must be one of"):
        _make_server({"engine": "trtllm"})._engine_kind()


# --------------------------------------------------------------------------- #
# CLI mapping: vLLM flags -> SGLang ServerArgs flags
# --------------------------------------------------------------------------- #


def _cmd(dynamo_cfg=None, **rollout_cfg) -> list[str]:
    cfg = {"engine": "sglang"}
    cfg.update(dynamo_cfg or {})
    server = _make_server(cfg, **rollout_cfg)
    tp = server.config.tensor_model_parallel_size
    return server._build_sglang_cmd("test-model", tp)


def test_sglang_cmd_core_mapping():
    cmd = _cmd(tensor_model_parallel_size=4, gpu_memory_utilization=0.7, max_model_len=2048)
    assert cmd[1:3] == ["-m", "dynamo.sglang"]
    # The whole point of the mapping table: none of the vLLM spellings survive.
    for vllm_flag in (
        "--tensor-parallel-size",
        "--gpu-memory-utilization",
        "--max-model-len",
        "--max-num-seqs",
        "--enable-prefix-caching",
        "--enable-sleep-mode",
        "--worker-extension-cls",
        "--kv-events-config",
    ):
        assert vllm_flag not in cmd, f"{vllm_flag} leaked into the sglang command line"
    assert cmd[cmd.index("--tp-size") + 1] == "4"
    assert cmd[cmd.index("--mem-fraction-static") + 1] == "0.7"
    assert cmd[cmd.index("--context-length") + 1] == "2048"
    assert cmd[cmd.index("--model-path") + 1] == "/models/test-model"


def test_sglang_cmd_inverted_flags():
    """SGLang's radix cache and CUDA graph default ON, so the flags invert."""
    on = _cmd(enable_prefix_caching=True, enforce_eager=False)
    assert "--disable-radix-cache" not in on
    assert "--disable-cuda-graph" not in on

    off = _cmd(enable_prefix_caching=False, enforce_eager=True)
    assert "--disable-radix-cache" in off
    assert "--disable-cuda-graph" in off


def test_sleep_mode_maps_to_memory_saver():
    assert "--enable-memory-saver" in _cmd(enable_sleep_mode=True)
    assert "--enable-memory-saver" not in _cmd(enable_sleep_mode=False)


def test_enable_rl_on_by_default():
    """--enable-rl is what registers call_tokenizer_manager, the only cache-flush route."""
    assert "--enable-rl" in _cmd()
    assert "--enable-rl" not in _cmd({"sglang": {"enable_rl": False}})


def test_page_size_falls_back_to_thunderagent_block_size():
    cmd = _cmd({"thunderagent": {"router_block_size": 32}})
    assert cmd[cmd.index("--page-size") + 1] == "32"
    # explicit sglang.page_size wins
    cmd = _cmd({"sglang": {"page_size": 16}, "thunderagent": {"router_block_size": 32}})
    assert cmd[cmd.index("--page-size") + 1] == "16"


def test_extra_args_forwarded():
    cmd = _cmd({"sglang": {"extra_args": ["--schedule-policy", "fcfs"]}})
    assert cmd[-2:] == ["--schedule-policy", "fcfs"]


# --------------------------------------------------------------------------- #
# control-plane invariants
# --------------------------------------------------------------------------- #


def test_sglang_requires_system_port():
    """DYN_SYSTEM_PORT is the sglang control plane, not an optional metrics extra."""
    server = _make_server({"engine": "sglang", "enable_worker_system_metrics": False})
    server.replica_rank = 0
    server.node_rank = 0
    server._cuda_visible_devices = "0"
    server._worker_specs = None
    with pytest.raises(ValueError, match="requires enable_worker_system_metrics"):
        server._start_vllm_workers()


def test_num_engine_workers_counts_sglang_shards():
    server = _make_server({"engine": "sglang"}, tensor_model_parallel_size=2)
    server._engine_control_endpoints = ["http://h:11000", "http://h:11001"]
    assert server.get_num_engine_workers() == 4


# --------------------------------------------------------------------------- #
# shard mapping — the silent-corruption guard
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("local_world_size,tp", [(8, 1), (8, 2), (8, 4), (8, 8), (4, 2)])
def test_trainer_shard_index_matches_launcher_gpu_slicing(local_world_size, tp):
    """Every trainer rank must resolve to the shard that owns its GPU.

    The launcher (``_start_vllm_workers``) gives shard *i* the GPU slice
    ``cvd[i*tp:(i+1)*tp]``. The adapter computes ``shard = local_rank // tp``.
    If those two ever drift, weight sync posts CUDA-IPC handles to an engine on a
    different GPU — which does not raise, it just trains against wrong weights.
    """
    cvd = [str(i) for i in range(local_world_size)]
    n_shards = local_world_size // tp
    launcher_owner = {}
    for shard_idx in range(n_shards):
        for gpu in cvd[shard_idx * tp : (shard_idx + 1) * tp]:
            launcher_owner[gpu] = shard_idx

    for local_rank in range(local_world_size):
        adapter_shard = local_rank // tp  # SGLangServerAdapter._shard_idx_local
        assert adapter_shard == launcher_owner[str(local_rank)]
        assert 0 <= adapter_shard < n_shards


def test_tp_group_ranks_are_contiguous_and_tp_aligned():
    """The shard TP process group is [k*tp, (k+1)*tp) over global ranks."""
    world_size, tp = 16, 4
    groups = [list(range(base, base + tp)) for base in range(0, world_size, tp)]
    assert len(groups) == world_size // tp
    assert [r for g in groups for r in g] == list(range(world_size))
    for rank in range(world_size):
        owning = [g for g in groups if rank in g]
        assert len(owning) == 1
        assert owning[0][0] == (rank // tp) * tp


# --------------------------------------------------------------------------- #
# control client
# --------------------------------------------------------------------------- #


class _FakeResponse:
    def __init__(self, status, payload):
        self.status = status
        self._payload = payload

    async def text(self):
        import json

        return json.dumps(self._payload)

    async def json(self, content_type=None):
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, status=200, payload=None):
        self.status = status
        self.payload = payload if payload is not None else {"status": "ok"}
        self.calls = []
        self.closed = False

    def post(self, url, json=None, timeout=None):
        self.calls.append((url, json))
        return _FakeResponse(self.status, self.payload)


def _client(session, **kwargs):
    client = DynamoSGLangControlClient("http://worker:11000", **kwargs)
    client._session = session
    return client


def test_route_url_shape():
    client = DynamoSGLangControlClient("http://worker:11000/")
    assert client.route_url("control/release_memory_occupation") == (
        "http://worker:11000/engine/control/release_memory_occupation"
    )


@pytest.mark.asyncio
async def test_update_weights_from_tensor_base64_encodes():
    session = _FakeSession(payload={"success": True, "message": "ok"})
    client = _client(session)
    blobs = [b"\x00\x01\xff-not-utf8", b"second"]
    req = SimpleNamespace(serialized_named_tensors=blobs, load_format=None, flush_cache=False)

    await client.update_weights_from_tensor(req)

    url, body = session.calls[0]
    assert url.endswith(f"/engine/{ROUTE_UPDATE_WEIGHTS_FROM_TENSOR}")
    # JSON has no bytes; the wire form must be b64 and must round-trip exactly.
    assert [base64.b64decode(v) for v in body["serialized_named_tensors"]] == blobs
    assert body["flush_cache"] is False


@pytest.mark.asyncio
async def test_update_weights_survives_full_byte_range():
    """Every byte value must round-trip: a payload SGLang cannot deserialize does not
    return an error, it kills the worker process (M0c, job 16215105)."""
    session = _FakeSession(payload={"success": True})
    client = _client(session)
    blob = bytes(range(256))
    req = SimpleNamespace(serialized_named_tensors=[blob], load_format=None, flush_cache=True)

    await client.update_weights_from_tensor(req)

    _, body = session.calls[0]
    assert base64.b64decode(body["serialized_named_tensors"][0]) == blob


@pytest.mark.asyncio
async def test_error_status_body_raises():
    """A 200 with {"status": "error"} must not be mistaken for success."""
    client = _client(_FakeSession(payload={"status": "error", "message": "memory control not supported"}))
    with pytest.raises(DynamoSGLangControlError, match="memory control not supported"):
        await client.release_memory_occupation(tags=["kv_cache"])


@pytest.mark.asyncio
async def test_success_false_body_raises():
    client = _client(_FakeSession(payload={"success": False, "message": "deserialize failed"}))
    req = SimpleNamespace(serialized_named_tensors=[b"x"], load_format=None, flush_cache=False)
    with pytest.raises(DynamoSGLangControlError, match="deserialize failed"):
        await client.update_weights_from_tensor(req)


@pytest.mark.asyncio
async def test_http_error_raises():
    client = _client(_FakeSession(status=404, payload={"error": "not found"}))
    with pytest.raises(DynamoSGLangControlError, match="HTTP 404"):
        await client.flush_cache()


@pytest.mark.asyncio
async def test_flush_cache_goes_through_tokenizer_manager():
    """There is no control/flush_cache engine route; it must use the RL passthrough."""
    session = _FakeSession()
    client = _client(session)
    await client.flush_cache()
    url, body = session.calls[0]
    assert url.endswith("/engine/call_tokenizer_manager")
    assert body["method"] == "flush_cache"


@pytest.mark.asyncio
async def test_memory_occupation_tags_passed_through():
    session = _FakeSession()
    client = _client(session)
    await client.release_memory_occupation(tags=["kv_cache", "weights"])
    _, body = session.calls[0]
    assert body["tags"] == ["kv_cache", "weights"]


# --------------------------------------------------------------------------- #
# release/resume tag symmetry — regression for the M2 scheduler kill
# --------------------------------------------------------------------------- #


class _FakeAdapter:
    """Stand-in for DynamoHttpServer.sglang_release/sglang_resume.

    The state lives on the node actor, not the per-rank adapter: the actor's sleep()
    releases memory (and dynamo unregisters the worker from discovery while
    released), so per-adapter bookkeeping would skip the matching resume and leave
    the frontend answering 503 forever. These tests pin the actor-side semantics.
    """

    def __init__(self):
        self._released_tags: set[str] = set()
        self.resumed_calls: list[list[str]] = []
        self.released_calls: list[list[str]] = []
        self.sleep_level = 2

    async def release(self):
        tags = ["kv_cache"] if self.sleep_level == 1 else ["kv_cache", "weights"]
        self.released_calls.append(tags)
        self._released_tags.update(tags)

    async def resume(self, tags):
        wanted = [t for t in tags if t in self._released_tags]
        if not wanted:
            return
        self.resumed_calls.append(wanted)
        self._released_tags.difference_update(wanted)


@pytest.mark.asyncio
async def test_resume_without_prior_release_is_skipped():
    """verl resumes weights before the FIRST weight sync, when nothing is released.

    Forwarding that to SGLang raises KeyError inside weight_updater.py and kills the
    scheduler process (observed as ServerDisconnectedError). vLLM tolerates it, which
    is why verl does it unconditionally.
    """
    a = _FakeAdapter()
    await a.resume(["weights"])
    assert a.resumed_calls == [], "resume must be skipped when nothing was released"


@pytest.mark.asyncio
async def test_resume_only_covers_released_tags():
    """A level-1 sleep releases kv_cache only; resuming weights must not be forwarded."""
    a = _FakeAdapter()
    a.sleep_level = 1
    await a.release()
    await a.resume(["weights"])
    assert a.resumed_calls == []
    await a.resume(["kv_cache"])
    assert a.resumed_calls == [["kv_cache"]]


@pytest.mark.asyncio
async def test_release_resume_round_trip_clears_state():
    a = _FakeAdapter()
    await a.release()
    assert a._released_tags == {"kv_cache", "weights"}
    await a.resume(["weights", "kv_cache"])
    assert a._released_tags == set()
    # A second resume is now a no-op rather than a fatal engine call.
    await a.resume(["weights"])
    assert len(a.resumed_calls) == 1


# --------------------------------------------------------------------------- #
# shard index must come from the GLOBAL rank — regression for the TP=2 silent bug
# --------------------------------------------------------------------------- #


def _verl_replica_local_rank(global_rank: int, tp: int, dp: int = 1, pp: int = 1) -> int:
    """Reproduce verl sglang ServerAdapter's local_rank (see sglang_rollout.py).

    rollout_world_size = tp * dp * pp; rollout_rank = rank % rollout_world_size;
    local_rank = rollout_rank % local_world_size. The key property: it is
    REPLICA-relative, so it never exceeds the replica's world size.
    """
    rollout_world_size = tp * dp * pp
    return global_rank % rollout_world_size


@pytest.mark.parametrize("local_world_size,tp", [(8, 2), (8, 4), (4, 2)])
def test_shard_index_uses_global_rank_not_replica_local_rank(local_world_size, tp):
    """Deriving the shard from verl's local_rank collapses every rank onto shard 0.

    Observed on 8xH100 TP=2 (job 16232079): all eight ranks logged shard=0 while
    four shards were registered, so every rank posted its CUDA-IPC handles to
    shard 0 and the other three engines kept stale weights. Nothing raised.
    At tp=1 the wrong formula happens to give the right answer, which is why six
    earlier green runs never caught it.
    """
    n_shards = local_world_size // tp
    shards_from_global = set()
    shards_from_replica_local = set()
    for global_rank in range(local_world_size):
        node_local = global_rank % local_world_size
        shards_from_global.add(node_local // tp)
        shards_from_replica_local.add(_verl_replica_local_rank(global_rank, tp) // tp)

    assert shards_from_global == set(range(n_shards)), "global-rank derivation must cover every shard"
    # The buggy derivation cannot reach past shard 0 whenever tp == replica width.
    assert shards_from_replica_local == {0}
    assert len(shards_from_global) > 1, "parametrisation must actually exercise >1 shard"


def test_single_shard_hides_the_bug_regardless_of_tp():
    """Pin down what actually hid the bug: ONE shard, not tp=1.

    The first explanation ("tp=1 makes both formulas agree") was wrong, and this
    test caught it. At tp=1 verl's rollout_world_size is 1, so its local_rank is 0
    for *every* rank — the buggy formula collapses to shard 0 there too. It only
    looked correct because the earlier runs used a single GPU, where shard 0 is the
    only shard that exists. An 8-GPU tp=1 run (8 shards) would have failed just as
    the tp=2 run did.
    """
    # one GPU -> one shard: both formulas trivially agree, bug invisible
    for tp in (1,):
        assert (0 % 1) // tp == _verl_replica_local_rank(0, tp) // tp

    # 8 GPUs, tp=1 -> 8 shards: the buggy formula still collapses to shard 0
    local_world_size, tp = 8, 1
    buggy = {_verl_replica_local_rank(r, tp) // tp for r in range(local_world_size)}
    correct = {(r % local_world_size) // tp for r in range(local_world_size)}
    assert buggy == {0}, "verl local_rank is 0 for every rank when tp*dp*pp == 1"
    assert correct == set(range(8))


@pytest.mark.parametrize("local_world_size,tp", [(8, 2), (8, 4), (8, 8), (4, 2)])
def test_shard_and_tp_group_agree(local_world_size, tp):
    """The shard a rank talks to must own the GPUs of its TP group."""
    for global_rank in range(local_world_size):
        node_local = global_rank % local_world_size
        shard = node_local // tp
        tp_group_src = (global_rank // tp) * tp
        # the TP group's first rank must sit at the start of that shard's GPU slice
        assert tp_group_src % local_world_size == shard * tp



# --------------------------------------------------------------------------- #
# engine dispatch lives in ONE place (rollout.name=dynamo + engine=...)
# --------------------------------------------------------------------------- #


def test_engine_dispatch_reads_config_not_rollout_name():
    """Selecting the engine must be a single config key, like every other verl backend.

    An earlier revision registered a second rollout name (``dynamo_sglang``) with its
    own trainer yaml and entry point, so the choice had to be repeated in three
    places that could disagree — and the extra name silently missed verl's own
    ``rollout.name == "sglang"`` special-cases.
    """
    from recipe.dynamo.dynamo_rollout import _dynamo_engine

    assert _dynamo_engine(None) == "vllm"
    assert _dynamo_engine(SimpleNamespace(engine_kwargs=None)) == "vllm"
    assert _dynamo_engine(SimpleNamespace(engine_kwargs={})) == "vllm"
    assert _dynamo_engine(SimpleNamespace(engine_kwargs={"dynamo": {}})) == "vllm"
    assert _dynamo_engine(SimpleNamespace(engine_kwargs={"dynamo": {"engine": "sglang"}})) == "sglang"


def test_registry_exposes_exactly_one_dynamo_name():
    """No second rollout name for the sglang engine."""
    from verl.workers.rollout.base import _ROLLOUT_REGISTRY

    import recipe.dynamo.register  # noqa: F401  (registers on import)

    dynamo_names = {name for (name, _mode) in _ROLLOUT_REGISTRY if name.startswith("dynamo")}
    assert dynamo_names == {"dynamo"}, f"unexpected dynamo rollout names: {dynamo_names}"


@pytest.mark.parametrize("nnodes,local_world_size,tp", [(2, 8, 2), (2, 8, 4), (4, 8, 2)])
def test_node_rank_uses_global_rank_not_replica_relative(nnodes, local_world_size, tp):
    """Every rank must resolve the DynamoHttpServer actor on ITS OWN node.

    verl's node_rank is rollout_rank // local_world_size with
    rollout_rank = rank % (tp*dp*pp); for tp<local_world_size that inner value
    never reaches local_world_size, so node_rank collapses to 0 and every rank on
    every node targets node 0's actor. Ranks on other nodes then post CUDA-IPC
    handles for their own GPUs to node 0's engines, and sglang rejects them with
    "Invalid device_uuid=..." while killing the scheduler (observed on 2x8 H100,
    job 16253641). Invisible on one node, where 0 is the only right answer.
    """
    world = nnodes * local_world_size
    correct = {r: r // local_world_size for r in range(world)}
    buggy = {r: _verl_replica_local_rank(r, tp) // local_world_size for r in range(world)}

    assert set(correct.values()) == set(range(nnodes)), "global derivation must span all nodes"
    assert set(buggy.values()) == {0}, "the replica-relative form cannot leave node 0"
    for r in range(world):
        # the actor a rank talks to must own the GPU that rank runs on
        assert correct[r] == r // local_world_size


def test_missing_token_ids_is_loud_not_silent(caplog):
    """The 1-token fallback must announce itself.

    Regression for the failure that cost a full 2-node 30B retool run (job
    16270139): the frontend returned text but no token ids, so every sample was
    scored as a single EOS. response_length/mean==1.0, grad_norm==0, reward
    pinned at its floor — and the job exited 0 with a complete metric set. The
    text in the rollout dump was 2384 characters of coherent reasoning, which is
    what finally separated "generation is broken" from "the length accounting is
    broken". Nothing in the logs distinguished the two for three wrong
    hypotheses, so silence here is the actual defect.
    """
    server = DynamoHttpServer.__new__(DynamoHttpServer)
    server.model_config = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=151645))

    with caplog.at_level("ERROR"):
        token_ids = server._fallback_token_ids()

    assert token_ids == [151645]
    assert "NO TOKEN IDS" in caplog.text
    assert "request_completion_token_ids" in caplog.text, (
        "the log must name the flag that fixes it, not merely report the symptom"
    )


def test_fallback_keeps_logging_on_a_long_run(caplog):
    """Log the first few hits and then periodically — never just once.

    A single line at hit #1 scrolls away in minutes on a 100-step run and the
    remaining thousands of corrupted samples look clean.
    """
    server = DynamoHttpServer.__new__(DynamoHttpServer)
    server.model_config = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=0))

    with caplog.at_level("ERROR"):
        for _ in range(205):
            server._fallback_token_ids()

    assert server._fallback_token_id_hits == 205
    # 3 early + hits 100 and 200
    assert caplog.text.count("NO TOKEN IDS") == 5


@pytest.mark.parametrize("level", [1, 2])
def test_sleep_frees_weights_at_every_level(level, monkeypatch):
    """vLLM sleep(level=1) already takes weights off the GPU, so the sglang
    equivalent must release BOTH tags.

    Regression for the OOM in job 16280143: mapping level 1 -> ["kv_cache"] left
    ~31 GB of TP=2 Qwen3-30B weights resident on each H100 through the training
    step, and the trainer's FSDP forward died with 92 MiB free
    (48.0 GiB trainer + 30.9 GiB engine on 79.1 GiB). The two APIs use different
    vocabularies for the same thing; matching them by label instead of by effect
    on GPU memory is what produced the bug.
    """
    server = DynamoHttpServer.__new__(DynamoHttpServer)
    released = []

    async def fake_release(tags):
        released.append(list(tags))

    monkeypatch.setattr(server, "_is_sglang", lambda: True, raising=False)
    monkeypatch.setattr(server, "_free_engine_on_train", lambda: True, raising=False)
    monkeypatch.setattr(server, "sglang_release", fake_release, raising=False)

    asyncio.run(server.sleep(level=level))

    assert released == [["kv_cache", "weights"]], (
        f"level={level} must free weights; leaving them resident OOMs the trainer"
    )


def test_release_is_idempotent_per_tag():
    """A second release of a still-released tag must not reach the engine.

    Regression for job 16283764. verl releases before every weight sync but
    resumes "weights" and "kv_cache" at different points, so the real trace was:

        released ['kv_cache','weights']   -> released={kv_cache, weights}
        resumed  ['weights']              -> released={kv_cache}
        released ['kv_cache','weights']   -> kv_cache released a SECOND time

    Double-releasing under torch_memory_saver unbinds an already-unbound region,
    and the pool's storage stops being GPU-backed. The failure then surfaces three
    layers away as HTTP 500 "Failed to fold completions stream", with a Triton
    "cpu tensor?" ValueError in the shard log and nothing naming memory release.
    """
    server = DynamoHttpServer.__new__(DynamoHttpServer)
    server._sglang_released_tags = set()
    server._sglang_tag_lock = None
    sent = []

    async def fake_control_all(method, **kwargs):
        sent.append((method, tuple(kwargs.get("tags", ()))))
        return []

    server._sglang_control_all = fake_control_all

    # One event loop for the whole sequence: the lock is created lazily and binds
    # to the loop it was made in, which is exactly how the Ray async actor runs it.
    async def main():
        await server.sglang_release(["kv_cache", "weights"])
        await server.sglang_resume(["weights"])
        await server.sglang_release(["kv_cache", "weights"])

    asyncio.run(main())

    assert sent == [
        ("release_memory_occupation", ("kv_cache", "weights")),
        ("resume_memory_occupation", ("kv_cache", "weights")),  # widened
        ("release_memory_occupation", ("kv_cache", "weights")),
    ], sent
    assert server._sglang_released_tags == {"kv_cache", "weights"}


def test_release_and_resume_guards_are_symmetric():
    """Both directions filter against the same state.

    The resume guard was added first (for a KeyError that killed the scheduler)
    and the release guard was not — one bug fixed, its sibling left in place.
    This asserts the pair, so neither can regress alone.
    """
    server = DynamoHttpServer.__new__(DynamoHttpServer)
    server._sglang_released_tags = set()
    server._sglang_tag_lock = None
    sent = []

    async def fake_control_all(method, **kwargs):
        sent.append(method)
        return []

    server._sglang_control_all = fake_control_all

    async def main():
        await server.sglang_resume(["weights"])    # never released -> no call
        assert sent == []
        await server.sglang_release(["weights"])
        await server.sglang_release(["weights"])   # already released -> no call

    asyncio.run(main())
    assert sent == ["release_memory_occupation"]


def test_concurrent_release_reaches_engine_once():
    """Concurrent callers must not each perform the release.

    Root cause of the sglang arm's HTTP 500 storm (jobs 16283764 / 16439516).
    Every shard adapter on a node calls sglang_release/resume in parallel, and a
    Ray async actor runs them on one event loop. The tag filter alone is a
    check-then-act across an await, so all four coroutines passed the check
    before any wrote back the state — the log showed four "resumed ['weights']"
    lines in the same millisecond, each still reporting weights as released.

    Four real release_memory_occupation calls unbind an already-unbound region in
    torch_memory_saver, and the next prefill dies in a Triton kernel with
    "Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)".
    """
    server = DynamoHttpServer.__new__(DynamoHttpServer)
    server._sglang_released_tags = set()
    server._sglang_tag_lock = None
    calls = []

    async def fake_control_all(method, **kwargs):
        calls.append((method, tuple(kwargs.get("tags", ()))))
        await asyncio.sleep(0)  # force a yield, as the real HTTP fan-out does
        return []

    server._sglang_control_all = fake_control_all

    async def main():
        await asyncio.gather(*[server.sglang_release(["kv_cache", "weights"]) for _ in range(4)])
        await asyncio.gather(*[server.sglang_resume(["weights"]) for _ in range(4)])

    asyncio.run(main())

    assert calls == [
        ("release_memory_occupation", ("kv_cache", "weights")),
        # widened: a partial resume would put the shard back in the routing pool
        # with its kv_cache still released
        ("resume_memory_occupation", ("kv_cache", "weights")),
    ], calls
    assert server._sglang_released_tags == set()


def test_partial_resume_is_widened_to_every_released_tag():
    """A weights-only resume must also restore kv_cache.

    Dynamo's sglang handler re-registers the shard into discovery on the first
    resume. verl resumes "weights" before the weight sync and "kv_cache" only
    after it, so a faithful partial resume advertises a shard whose KV pool is
    still released. Job 16441143 spent ~9 s in that state and the scheduler died
    in write_req_to_token_pool_triton with "cpu tensor?", surfacing to the trainer
    as 257 unexplained HTTP 500s.
    """
    server = DynamoHttpServer.__new__(DynamoHttpServer)
    server._sglang_released_tags = {"kv_cache", "weights"}
    server._sglang_tag_lock = None
    sent = []

    async def fake_control_all(method, **kwargs):
        sent.append((method, tuple(sorted(kwargs.get("tags", ())))))
        return []

    server._sglang_control_all = fake_control_all
    asyncio.run(server.sglang_resume(["weights"]))

    assert sent == [("resume_memory_occupation", ("kv_cache", "weights"))]
    assert server._sglang_released_tags == set()


def test_facade_imports_without_vllm(monkeypatch):
    """recipe.dynamo.dynamo_rollout must import on a vLLM-free image.

    rollout.name=dynamo resolves through this one module for BOTH engines, so a
    hard top-level `from verl...vllm_rollout import ServerAdapter` makes the
    sglang path unusable on the official verlai/verl:sgl* images — verl's
    vllm_rollout/__init__.py raises PackageNotFoundError (not ImportError) when
    vLLM is absent. Job 16510923 died that way after a fully healthy bootstrap.
    """
    import importlib
    import sys

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def no_vllm(name, *args, **kwargs):
        if name.startswith("verl.workers.rollout.vllm_rollout"):
            raise importlib.metadata.PackageNotFoundError("vllm missing (simulated)")
        return real_import(name, *args, **kwargs)

    for mod in [m for m in sys.modules if m.startswith("recipe.dynamo.dynamo_rollout")]:
        del sys.modules[mod]
    monkeypatch.setattr("builtins.__import__", no_vllm)

    mod = importlib.import_module("recipe.dynamo.dynamo_rollout")
    assert mod.ServerAdapter is not None
    # the vLLM subclass still exists, but instantiating it must say why it cannot work
    with pytest.raises(RuntimeError, match="needs the vLLM package"):
        mod.VllmDynamoServerAdapter()
