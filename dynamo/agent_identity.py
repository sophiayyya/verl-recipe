"""Coding-agent identity derived from request headers.

The shape is Uni-Agent's ``AgentContext`` (gateway/session/types.py): the harness session, the agent that
sent the turn (``None`` for the main agent) and that agent's parent (``None`` when not stated). The Gateway
terminates these headers on the training path, so the recipe reads them itself until it receives the context.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

_SESSION = "x-claude-code-session-id"
_AGENT = "x-claude-code-agent-id"
_PARENT = "x-claude-code-parent-agent-id"


@dataclass(frozen=True)
class AgentIdentity:
    session_id: str
    agent_id: str | None = None
    parent_agent_id: str | None = None

    @classmethod
    def from_context(cls, context: Mapping[str, str | None]) -> AgentIdentity:
        """From Uni-Agent's ``AgentContext``."""
        return cls(context["session_id"], context.get("agent_id"), context.get("parent_agent_id"))


def _value(headers: Mapping[str, str], name: str) -> str | None:
    value = headers.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def agent_identity_from_headers(headers: Mapping[str, str]) -> AgentIdentity | None:
    """The identity a Claude Code turn carries, or None without the session header. Lookups are lower-case,
    so ``headers`` must be case-insensitive. A subagent repeats its agent id on every turn and names its
    parent only when nested; a parent header on a main-agent turn or naming the agent itself is stale."""
    session_id = _value(headers, _SESSION)
    if session_id is None:
        return None
    agent_id = _value(headers, _AGENT)
    if agent_id is None:
        return AgentIdentity(session_id)
    parent = _value(headers, _PARENT)
    return AgentIdentity(session_id, agent_id, None if parent == agent_id else parent)


__all__ = ["AgentIdentity", "agent_identity_from_headers"]
