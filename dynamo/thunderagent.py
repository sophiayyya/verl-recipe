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
"""Task-local lifecycle for one Dynamo ThunderAgent program."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar

Finalize = Callable[[str], Awaitable[None] | None]


class ProgramScope:
    """Track requests belonging to one agent trajectory."""

    def __init__(self, session_id: str):
        if not session_id:
            raise ValueError("session_id must be non-empty")
        self.session_id = session_id
        self._accepting_requests = True
        self._inflight = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self._finalized = False
        self._finalize_lock = asyncio.Lock()

    @asynccontextmanager
    async def request(self) -> AsyncIterator[None]:
        """Admit one request, or reject it once program shutdown starts."""
        if not self._accepting_requests:
            raise RuntimeError(f"ThunderAgent program {self.session_id!r} is closing")
        self._inflight += 1
        self._idle.clear()
        try:
            yield
        finally:
            self._inflight -= 1
            if self._inflight == 0:
                self._idle.set()

    def stop_accepting_requests(self) -> None:
        self._accepting_requests = False

    async def close(self, finalize: Finalize) -> None:
        """Drain admitted requests and finalize the program exactly once."""
        self.stop_accepting_requests()
        await self._idle.wait()
        async with self._finalize_lock:
            if self._finalized:
                return
            result = finalize(self.session_id)
            if inspect.isawaitable(result):
                await result
            self._finalized = True


_CURRENT_PROGRAM: ContextVar[ProgramScope | None] = ContextVar("dynamo_program", default=None)


def current_program() -> ProgramScope | None:
    return _CURRENT_PROGRAM.get()


@asynccontextmanager
async def program_scope(session_id: str, finalize: Finalize) -> AsyncIterator[ProgramScope]:
    """Bind a program to this task and finalize it on every exit path."""
    scope = ProgramScope(session_id)
    token = _CURRENT_PROGRAM.set(scope)
    try:
        yield scope
    finally:
        scope.stop_accepting_requests()
        _CURRENT_PROGRAM.reset(token)
        close_task = asyncio.create_task(scope.close(finalize))
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError:
            await close_task
            raise


__all__ = ["ProgramScope", "current_program", "program_scope"]
