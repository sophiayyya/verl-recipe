"""Header-derived program identity, mirroring Dynamo's ``lib/llm/src/protocols/agents.rs``."""

from recipe.dynamo.agent_identity import AgentIdentity, agent_identity_from_headers
from starlette.datastructures import Headers


def test_no_recognized_headers_yields_none():
    assert agent_identity_from_headers(Headers({"content-type": "application/json"})) is None


def test_claude_code_root_turn_has_no_parent():
    identity = agent_identity_from_headers(Headers({"x-claude-code-session-id": "root-1"}))
    assert identity == AgentIdentity("root-1", None)


def test_claude_code_subagent_turn_infers_parent_from_session():
    # agents.rs: infer_parent_from_session_for_child = true for Claude Code.
    identity = agent_identity_from_headers(
        Headers({"x-claude-code-session-id": "root-1", "x-claude-code-agent-id": "sub-a"})
    )
    assert (identity.session_id, identity.parent_session_id) == ("sub-a", "root-1")


def test_nested_subagent_prefers_the_explicit_parent_agent():
    identity = agent_identity_from_headers(
        Headers(
            {
                "x-claude-code-session-id": "root-1",
                "x-claude-code-agent-id": "sub-b",
                "x-claude-code-parent-agent-id": "sub-a",
            }
        )
    )
    assert (identity.session_id, identity.parent_session_id) == ("sub-b", "sub-a")


def test_parent_agent_header_is_ignored_on_a_root_turn():
    identity = agent_identity_from_headers(
        Headers({"x-claude-code-session-id": "root-1", "x-claude-code-parent-agent-id": "stale"})
    )
    assert (identity.session_id, identity.parent_session_id) == ("root-1", None)


def test_self_referential_parent_is_dropped():
    identity = agent_identity_from_headers(
        Headers(
            {
                "x-claude-code-session-id": "root-1",
                "x-claude-code-agent-id": "sub-a",
                "x-claude-code-parent-agent-id": "sub-a",
            }
        )
    )
    assert identity.parent_session_id == "root-1"


def test_canonical_dynamo_headers_take_precedence():
    identity = agent_identity_from_headers(
        Headers(
            {
                "x-dynamo-session-id": "canonical",
                "x-dynamo-parent-session-id": "canonical-parent",
                "x-claude-code-session-id": "root-1",
                "x-claude-code-agent-id": "sub-a",
            }
        )
    )
    assert (identity.session_id, identity.parent_session_id) == ("canonical", "canonical-parent")


def test_header_lookup_is_case_insensitive_and_trims():
    assert agent_identity_from_headers(Headers({"X-Claude-Code-Session-Id": "  root-1  "})).session_id == "root-1"


def test_blank_header_values_are_treated_as_absent():
    assert agent_identity_from_headers(Headers({"x-claude-code-session-id": "   "})) is None
    identity = agent_identity_from_headers(
        Headers({"x-claude-code-session-id": "root-1", "x-claude-code-agent-id": "  "})
    )
    assert identity.session_id == "root-1"


def test_codex_and_opencode_mappings():
    codex = agent_identity_from_headers(Headers({"thread-id": "t1", "x-codex-parent-thread-id": "t0"}))
    assert (codex.session_id, codex.parent_session_id) == ("t1", "t0")
    opencode = agent_identity_from_headers(Headers({"x-session-id": "s1", "x-parent-session-id": "s0"}))
    assert (opencode.session_id, opencode.parent_session_id) == ("s1", "s0")


def test_claude_code_headers_as_sent_by_the_cli():
    # A subagent repeats its agent id on every turn and sends no parent header.
    root = Headers({"x-claude-code-session-id": "c6602f61"})
    subagent = Headers({"x-claude-code-session-id": "c6602f61", "x-claude-code-agent-id": "a5a8b926f8"})
    assert agent_identity_from_headers(root) == AgentIdentity("c6602f61", None)
    for _turn in range(2):
        assert agent_identity_from_headers(subagent) == AgentIdentity("a5a8b926f8", "c6602f61")
