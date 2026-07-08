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

import pytest

try:
    from recipe.dynamo.thunderagent import ProgramScope, current_program, program_scope
except ModuleNotFoundError:
    from dynamo.thunderagent import ProgramScope, current_program, program_scope


async def _hold_request(scope: ProgramScope, entered: asyncio.Event, release: asyncio.Event) -> None:
    async with scope.request():
        entered.set()
        await release.wait()


@pytest.mark.asyncio
async def test_program_scope_binds_id_and_finalizes_once() -> None:
    finalized = []

    async with program_scope("program-a", finalized.append):
        assert current_program() is not None
        assert current_program().session_id == "program-a"

    assert current_program() is None
    assert finalized == ["program-a"]


@pytest.mark.asyncio
async def test_program_scope_finalizes_after_body_error() -> None:
    finalized = []

    with pytest.raises(ValueError, match="agent failed"):
        async with program_scope("program-a", finalized.append):
            raise ValueError("agent failed")

    assert finalized == ["program-a"]


@pytest.mark.asyncio
async def test_close_waits_for_inflight_request_and_is_idempotent() -> None:
    scope = ProgramScope("program-a")
    entered = asyncio.Event()
    release = asyncio.Event()
    finalized = []
    request = asyncio.create_task(_hold_request(scope, entered, release))
    await entered.wait()

    scope.stop_accepting_requests()
    first_close = asyncio.create_task(scope.close(finalized.append))
    second_close = asyncio.create_task(scope.close(finalized.append))
    await asyncio.sleep(0)
    assert finalized == []

    release.set()
    await request
    await asyncio.gather(first_close, second_close)
    assert finalized == ["program-a"]


@pytest.mark.asyncio
async def test_closing_scope_rejects_late_request() -> None:
    scope = ProgramScope("program-a")
    scope.stop_accepting_requests()

    with pytest.raises(RuntimeError, match="closing"):
        async with scope.request():
            pass


@pytest.mark.asyncio
async def test_contextvars_isolate_concurrent_programs() -> None:
    finalized = []
    both_entered = asyncio.Barrier(2)

    async def run(session_id: str) -> str:
        async with program_scope(session_id, finalized.append):
            await both_entered.wait()
            assert current_program() is not None
            return current_program().session_id

    assert await asyncio.gather(run("program-a"), run("program-b")) == ["program-a", "program-b"]
    assert sorted(finalized) == ["program-a", "program-b"]


@pytest.mark.asyncio
async def test_cancellation_waits_for_finalizer() -> None:
    body_entered = asyncio.Event()
    finalizer_entered = asyncio.Event()
    allow_finalizer = asyncio.Event()

    async def finalize(_session_id: str) -> None:
        finalizer_entered.set()
        await allow_finalizer.wait()

    async def run() -> None:
        async with program_scope("program-a", finalize):
            body_entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(run())
    await body_entered.wait()
    task.cancel()
    await finalizer_entered.wait()
    assert not task.done()

    allow_finalizer.set()
    with pytest.raises(asyncio.CancelledError):
        await task
