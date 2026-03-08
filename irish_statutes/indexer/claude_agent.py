"""
Agentic loop using Claude Haiku for the Irish Statute Book query pipeline.

Claude receives a natural-language query, decides which DB-lookup tools to
call, iterates until it has enough information, then synthesises a grounded
answer citing specific sections.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import anthropic

# The Anthropic SDK calls model_dump(by_alias=None) in its debug-logging path
# (_base_client.py _build_request), which crashes in Pydantic v2 with:
#   TypeError: argument 'by_alias': 'NoneType' object cannot be converted to 'PyBool'
# Silence the SDK's own loggers so the debug guard never fires, regardless of
# how the root logger is configured (e.g. setup_logger sets root to DEBUG).
logging.getLogger("anthropic").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from .db import get_law_by_name, get_law_sections, get_section_by_ref, search_laws
from .eval_queries import QueryResponse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """You are a legal research assistant specialising in Irish statute law.

You have access to a database of Irish statutes. Use the provided tools to look up
relevant laws and their specific sections before answering. Always cite the specific
sections you rely on. Be precise and grounded in the actual statutory text.

Strategy:
1. Use search_laws to find candidate laws by keyword.
2. Use get_law_by_name to confirm and retrieve the law's database ID.
3. Use get_law_sections or get_section_by_ref to retrieve the relevant text.
4. Synthesise a clear, cited answer from the retrieved text."""

TOOL_SCHEMAS = [
    {
        "name": "search_laws",
        "description": (
            "Search for Irish laws by keyword. Returns a list of matching laws "
            "with their IDs, names, and years. Use this to discover which laws "
            "are relevant to the query."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keyword(s) to search for in law names",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_law_by_name",
        "description": (
            "Look up a specific law by name (and optionally year). Returns the "
            "law record including its database ID, which is needed for section lookups."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The name or partial name of the law",
                },
                "year": {
                    "type": "integer",
                    "description": "Optional year to narrow the search",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "get_law_sections",
        "description": (
            "Get all sections of a law by its database ID. Returns a list of "
            "sections with their text content. Use this to browse a law's structure."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "law_id": {
                    "type": "integer",
                    "description": "The database ID of the law",
                }
            },
            "required": ["law_id"],
        },
    },
    {
        "name": "get_section_by_ref",
        "description": (
            "Get a specific section of a law by its reference (e.g. '3', '3A', '3(1)'). "
            "Returns the section text. Use this when you know the exact section needed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "law_id": {
                    "type": "integer",
                    "description": "The database ID of the law",
                },
                "section_ref": {
                    "type": "string",
                    "description": "The section reference string",
                },
            },
            "required": ["law_id", "section_ref"],
        },
    },
]


# ---------------------------------------------------------------------------
# SourceNode dataclass
# ---------------------------------------------------------------------------


@dataclass
class SourceNode:
    """A source node compatible with save_eval_result expectations.

    save_eval_result iterates source_nodes and accesses .text, .score,
    and .metadata as attributes.
    """

    text: str
    score: float
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


def dispatch_tool(name: str, args: dict) -> str:
    """Execute a DB tool call and return the result as a JSON string."""
    if name == "search_laws":
        result = search_laws(args["query"])
    elif name == "get_law_by_name":
        result = get_law_by_name(args["name"], args.get("year"))
    elif name == "get_law_sections":
        result = get_law_sections(int(args["law_id"]))
    elif name == "get_section_by_ref":
        result = get_section_by_ref(int(args["law_id"]), args["section_ref"])
    else:
        result = {"error": f"Unknown tool: {name}"}
    return json.dumps(result, default=str)


# ---------------------------------------------------------------------------
# Source node extraction
# ---------------------------------------------------------------------------


def _extract_source_nodes(messages: list) -> list[SourceNode]:
    """Walk the message history and build SourceNode objects from tool results."""
    nodes: list[SourceNode] = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_result":
                continue
            raw = block.get("content", "")
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            items = data if isinstance(data, list) else ([data] if data else [])
            for item in items:
                if not isinstance(item, dict):
                    continue
                # Prefer text_content (law_sections), fall back to name or repr
                text = item.get("text_content") or item.get("name") or str(item)
                metadata = {k: v for k, v in item.items() if k != "text_content"}
                nodes.append(SourceNode(text=str(text), score=0.0, metadata=metadata))
    return nodes


# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------


def run_agent(query: str) -> QueryResponse:
    """Run the agentic loop and return a QueryResponse.

    Claude calls DB tools until it has enough context, then generates a
    final grounded answer.
    """
    client = anthropic.Anthropic()
    messages: list[dict[str, Any]] = [{"role": "user", "content": query}]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            text_blocks = [b for b in response.content if b.type == "text"]
            final_text = text_blocks[0].text if text_blocks else ""
            break

        # stop_reason == "tool_use": execute each tool and feed results back
        # Serialize to plain dicts — passing Pydantic model objects back into
        # the SDK triggers a by_alias=None Pydantic serialization bug.
        content_dicts = []
        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"\n[thinking] {block.text.strip()}")
            if block.type == "text":
                content_dicts.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content_dicts.append(
                    {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
                )
        messages.append({"role": "assistant", "content": content_dicts})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\n[tool_call] {block.name}({json.dumps(block.input)})")
                result_str = dispatch_tool(block.name, block.input)
                try:
                    parsed = json.loads(result_str)
                    print(f"[tool_result] {json.dumps(parsed, indent=2, default=str)}")
                except (json.JSONDecodeError, TypeError):
                    print(f"[tool_result] {result_str}")
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    }
                )

        messages.append({"role": "user", "content": tool_results})

    source_nodes = _extract_source_nodes(messages)

    return QueryResponse(
        query=query,
        response=final_text,
        top_k=len(source_nodes),
        scores=[n.score for n in source_nodes],
        source_nodes=source_nodes,
    )
