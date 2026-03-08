"""
TDD tests for the Claude Haiku agentic loop.

Steps 1-2 run without external services.
Steps 3-5 are marked integration and require a live DB and ANTHROPIC_API_KEY.
"""

import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Step 1 — Module importable
# ---------------------------------------------------------------------------
def test_claude_agent_importable():
    from indexer.claude_agent import run_agent
    assert callable(run_agent)


# ---------------------------------------------------------------------------
# Step 2 — Tool schemas valid
# ---------------------------------------------------------------------------
def test_tool_schemas_have_required_keys():
    from indexer.claude_agent import TOOL_SCHEMAS
    assert len(TOOL_SCHEMAS) > 0
    for schema in TOOL_SCHEMAS:
        assert "name" in schema, f"Schema missing 'name': {schema}"
        assert "description" in schema, f"Schema missing 'description': {schema}"
        assert "input_schema" in schema, f"Schema missing 'input_schema': {schema}"
        assert "properties" in schema["input_schema"], (
            f"input_schema missing 'properties': {schema['input_schema']}"
        )


# ---------------------------------------------------------------------------
# Step 2b — Agentic loop survives two iterations (no API key, no DB)
# ---------------------------------------------------------------------------
def test_run_agent_two_iterations():
    """Force the tool_use → end_turn path using real SDK Pydantic types.

    Using real TextBlock / ToolUseBlock objects (not plain MagicMocks) is
    deliberate: it reproduces the serialization bug where Pydantic model
    instances passed back into messages.create() caused a by_alias=None crash.
    """
    import anthropic.types

    text_block = anthropic.types.TextBlock(
        type="text", text="I'll search for minimum wage laws."
    )
    tool_use_block = anthropic.types.ToolUseBlock(
        type="tool_use",
        id="tu_123",
        name="search_laws",
        input={"query": "minimum wage"},
    )
    first_response = MagicMock()
    first_response.stop_reason = "tool_use"
    first_response.content = [text_block, tool_use_block]

    final_block = anthropic.types.TextBlock(
        type="text", text="The minimum wage is €13.50 per hour."
    )
    second_response = MagicMock()
    second_response.stop_reason = "end_turn"
    second_response.content = [final_block]

    mock_create = MagicMock(side_effect=[first_response, second_response])
    tool_result = json.dumps([{"id": 1, "name": "National Minimum Wage Act 2000"}])

    with patch("indexer.claude_agent.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create = mock_create
        with patch("indexer.claude_agent.dispatch_tool", return_value=tool_result):
            from indexer.claude_agent import run_agent
            result = run_agent("What is the minimum wage?")

    assert mock_create.call_count == 2
    assert result.response == "The minimum wage is €13.50 per hour."


# ---------------------------------------------------------------------------
# Step 2c — run_agent survives when root logger is at DEBUG (regression)
# ---------------------------------------------------------------------------
def test_run_agent_survives_debug_logging():
    """Regression test for the by_alias=None Pydantic crash.

    setup_logger calls logging.basicConfig(level=DEBUG), which makes the
    Anthropic SDK's internal logger DEBUG-enabled.  The SDK's _build_request
    then calls model_dump(options, exclude_unset=True) — without by_alias —
    which lands in _compat.model_dump(by_alias=None).  Pydantic v2 rejects
    None for by_alias:
        TypeError: argument 'by_alias': 'NoneType' object cannot be
        converted to 'PyBool'

    The fix is to silence the 'anthropic' logger in setup_logger so the
    debug guard in the SDK never fires.

    This test uses a real anthropic.Anthropic client (so _build_request is
    actually called) backed by an httpx.MockTransport (no real network call).
    It sets the root logger to DEBUG to reproduce the crash, then tears it
    down afterwards.
    """
    import anthropic as anthropic_sdk

    fake_body = json.dumps({
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "The minimum wage is €13.50."}],
        "model": "claude-haiku-4-5-20251001",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=fake_body, headers={"content-type": "application/json"}
        )

    real_client = anthropic_sdk.Anthropic(
        api_key="fake-test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    root = logging.getLogger()
    old_level = root.level
    root.setLevel(logging.DEBUG)  # Reproduces what setup_logger does

    try:
        with patch("indexer.claude_agent.anthropic.Anthropic", return_value=real_client):
            from indexer.claude_agent import run_agent
            result = run_agent("What is the minimum wage?")
        assert "13.50" in result.response
    finally:
        root.setLevel(old_level)


# ---------------------------------------------------------------------------
# Step 3 — Tool dispatch works (requires DB)
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_dispatch_search_laws_returns_json():
    from indexer.claude_agent import dispatch_tool
    result = dispatch_tool("search_laws", {"query": "data protection"})
    parsed = json.loads(result)
    assert isinstance(parsed, list)


# ---------------------------------------------------------------------------
# Step 4 — Agent returns QueryResponse (requires ANTHROPIC_API_KEY + DB)
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.requires_api_key
def test_run_agent_returns_query_response():
    from indexer.claude_agent import run_agent
    from indexer.eval_queries import QueryResponse
    result = run_agent("What is the minimum wage?")
    assert isinstance(result, QueryResponse)
    assert len(result.response) > 0


@pytest.mark.integration
@pytest.mark.requires_api_key
def test_run_agent_makes_tool_calls():
    """Verify the agent makes at least one tool call, proving the loop went
    around at least twice (tool_use → end_turn).  If the Pydantic serialization
    bug is still present the second messages.create() call will raise a
    TypeError before we ever get a result.
    """
    from indexer.claude_agent import run_agent
    result = run_agent("What is the minimum wage?")
    assert len(result.source_nodes) > 0, (
        "Agent returned no source nodes — it never made a tool call, "
        "so the second-iteration serialization path was not exercised."
    )


# ---------------------------------------------------------------------------
# Step 5 — source_nodes compatible with save_eval_result
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.requires_api_key
def test_source_nodes_compatible_with_save_eval_result():
    from indexer.claude_agent import run_agent
    result = run_agent("What is the minimum wage?")
    for node in result.source_nodes:
        assert hasattr(node, "text"), "SourceNode missing .text"
        assert hasattr(node, "score"), "SourceNode missing .score"
        assert hasattr(node, "metadata"), "SourceNode missing .metadata"
        # Must be JSON-serialisable (mirrors what save_eval_result does)
        json.dumps({"text": node.text, "score": node.score, "metadata": node.metadata})
