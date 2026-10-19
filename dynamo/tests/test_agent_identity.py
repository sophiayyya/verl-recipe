"""Header-derived agent identity, in the shape of Uni-Agent's ``AgentContext``."""

from recipe.dynamo.agent_identity import AgentIdentity, agent_identity_from_headers
from starlette.datastructures import Headers


def test_no_recognized_headers_yields_none():
    assert agent_identity_from_headers(Headers({"content-type": "application/json"})) is None


def test_claude_code_headers_as_sent_by_the_cli():
    # The main agent sends only the session; a subagent repeats its agent id on every turn and no parent.
    root = Headers({"x-claude-code-session-id": "c6602f61"})
    subagent = Headers({"x-claude-code-session-id": "c6602f61", "x-claude-code-agent-id": "a5a8b926f8"})
    assert agent_identity_from_headers(root) == AgentIdentity("c6602f61")
    for _turn in range(2):
        assert agent_identity_from_headers(subagent) == AgentIdentity("c6602f61", "a5a8b926f8")


def test_nested_subagent_names_its_parent_agent():
    identity = agent_identity_from_headers(
        Headers(
            {
                "x-claude-code-session-id": "root-1",
                "x-claude-code-agent-id": "sub-b",
                "x-claude-code-parent-agent-id": "sub-a",
            }
        )
    )
    assert identity == AgentIdentity("root-1", "sub-b", "sub-a")


def test_stale_parent_headers_are_dropped():
    on_root = Headers({"x-claude-code-session-id": "root-1", "x-claude-code-parent-agent-id": "stale"})
    assert agent_identity_from_headers(on_root) == AgentIdentity("root-1")
    on_self = Headers(
        {"x-claude-code-session-id": "root-1", "x-claude-code-agent-id": "sub-a", "x-claude-code-parent-agent-id": "sub-a"}
    )
    assert agent_identity_from_headers(on_self) == AgentIdentity("root-1", "sub-a")


def test_header_values_are_trimmed_and_blank_ones_are_absent():
    assert agent_identity_from_headers(Headers({"X-Claude-Code-Session-Id": "  root-1  "})).session_id == "root-1"
    assert agent_identity_from_headers(Headers({"x-claude-code-session-id": "   "})) is None
    blank_agent = Headers({"x-claude-code-session-id": "root-1", "x-claude-code-agent-id": "  "})
    assert agent_identity_from_headers(blank_agent) == AgentIdentity("root-1")


def test_identity_is_built_from_uni_agent_agent_context():
    context = {"provider": "claude_code", "session_id": "root-1", "agent_id": "sub-a", "parent_agent_id": None}
    assert AgentIdentity.from_context(context) == AgentIdentity("root-1", "sub-a")
