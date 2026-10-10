"""Coding-agent program identity derived from request headers.

Dynamo's HTTP frontend normalizes coding-agent identity headers into
``agent_context`` (``lib/llm/src/protocols/agents.rs``), and ThunderAgent keys
its programs off that session id. On the training path Claude Code talks to the
Uni-Agent Gateway instead, so the headers terminate there. This module applies
the same precedence, so the program tree the recipe reports is the one Dynamo
would have derived: a Claude Code subagent turn carries ``x-claude-code-agent-id``
and becomes a program of its own, with the run's session as its parent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class _HeaderMapping:
    root_session: str
    child_session: str | None
    parent_session: str | None
    # A child turn with no explicit parent inherits the root session as its
    # parent; Claude Code's top-level subagents rely on this.
    infer_parent_for_child: bool


_DYNAMO_SESSION = "x-dynamo-session-id"
_DYNAMO_PARENT = "x-dynamo-parent-session-id"
_DYNAMO_FINAL = "x-dynamo-session-final"

_AGENT_HEADERS = (
    _HeaderMapping("x-claude-code-session-id", "x-claude-code-agent-id", "x-claude-code-parent-agent-id", True),
    _HeaderMapping("thread-id", None, "x-codex-parent-thread-id", False),
    _HeaderMapping("x-session-id", None, "x-parent-session-id", False),
)


@dataclass(frozen=True)
class AgentIdentity:
    """One reasoning chain's identity as the agent reported it.

    ``session_id`` is the chain's own id: for a Claude Code subagent turn the
    agent id, not the run's session. ``parent_session_id`` names the chain that
    dispatched it, when known.
    """

    session_id: str
    parent_session_id: str | None = None
    session_final: bool | None = None


def _value(headers: Mapping[str, str], name: str) -> str | None:
    value = headers.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def agent_identity_from_headers(headers: Mapping[str, str]) -> AgentIdentity | None:
    """Derive the program identity from request headers, or None when none are set.

    Canonical ``x-dynamo-*`` headers win; otherwise the first coding agent whose
    root header is present decides. A parent equal to the session itself is
    dropped. Lookups use lower-case names, so ``headers`` must be
    case-insensitive, as Starlette's ``Headers`` is.
    """
    final = _value(headers, _DYNAMO_FINAL)
    session_final = None if final is None else final.lower() == "true"

    session_id = _value(headers, _DYNAMO_SESSION)
    if session_id is not None:
        parent = _value(headers, _DYNAMO_PARENT)
        return AgentIdentity(session_id, parent if parent != session_id else None, session_final)

    for mapping in _AGENT_HEADERS:
        root = _value(headers, mapping.root_session)
        if root is None:
            continue
        child = _value(headers, mapping.child_session) if mapping.child_session else None
        session_id = child or root
        is_child = session_id != root
        parent = _value(headers, mapping.parent_session) if mapping.parent_session else None
        if parent == session_id:
            parent = None
        if parent is not None and mapping.infer_parent_for_child and not is_child:
            parent = None  # a root turn has no parent; a stale header must not invent one
        if parent is None and mapping.infer_parent_for_child and is_child:
            parent = root
        return AgentIdentity(session_id, parent, session_final)
    return None


__all__ = ["AgentIdentity", "agent_identity_from_headers"]
