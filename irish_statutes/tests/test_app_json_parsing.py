"""
Tests for parse_jsonb_list in indexer.db.

psycopg2 deserialises JSONB columns to Python lists automatically.
parse_jsonb_list must handle both already-deserialised lists (psycopg2)
and JSON strings (SQLAlchemy / future use).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from indexer.db import parse_jsonb_list


def test_list_input_is_returned_unchanged():
    """psycopg2 gives us a list; it must come back as-is."""
    data = [{"text": "some law text", "score": 0.72, "metadata": {}}]
    assert parse_jsonb_list(data) == data


def test_json_string_is_deserialised():
    """A JSON string (e.g. from SQLAlchemy) must still work."""
    data = [{"text": "foo", "score": 0.5, "metadata": {}}]
    assert parse_jsonb_list(json.dumps(data)) == data


def test_none_returns_empty_list():
    assert parse_jsonb_list(None) == []


def test_empty_list_returns_empty_list():
    assert parse_jsonb_list([]) == []
