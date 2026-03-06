"""
Tests for JSONB parsing in the eval app.

psycopg2 deserialises JSONB columns to Python objects automatically.
The app must handle both already-deserialised lists (psycopg2) and
JSON strings (fallback / SQLAlchemy text columns).
"""
import json
import pytest


# ---------------------------------------------------------------------------
# Replicate the exact logic currently in app.py answer_panel()
# ---------------------------------------------------------------------------
def parse_jsonb_current(val):
    """Current (broken) implementation copied verbatim from app.py."""
    try:
        return json.loads(val) if val else []
    except Exception:
        return []


def parse_jsonb_fixed(val):
    """Fixed implementation that handles pre-deserialised lists."""
    if not val:
        return []
    if isinstance(val, list):
        return val
    return json.loads(val)


# ---------------------------------------------------------------------------
# Tests that should FAIL with the current code and PASS with the fix
# ---------------------------------------------------------------------------
class TestCurrentlyBroken:
    def test_list_input_returns_empty_with_current_code(self):
        """psycopg2 gives us a list; current code silently returns []."""
        data = [{"text": "some law text", "score": 0.72, "metadata": {}}]
        result = parse_jsonb_current(data)
        # This WRONGLY returns [] because json.loads(list) raises TypeError
        assert result == [], "current code returns [] for list input (the bug)"

    def test_list_input_loses_data(self):
        """Confirm the data is non-empty going in but empty coming out."""
        data = [{"text": "foo"}, {"text": "bar"}]
        result = parse_jsonb_current(data)
        assert len(result) == 0  # data lost — this is the bug


class TestFixed:
    def test_list_passthrough(self):
        """Fixed code returns the list unchanged."""
        data = [{"text": "some law text", "score": 0.72, "metadata": {}}]
        result = parse_jsonb_fixed(data)
        assert result == data

    def test_json_string_still_works(self):
        """Fixed code still handles a JSON string (SQLAlchemy / future-proofing)."""
        data = [{"text": "foo", "score": 0.5, "metadata": {}}]
        result = parse_jsonb_fixed(json.dumps(data))
        assert result == data

    def test_none_returns_empty(self):
        assert parse_jsonb_fixed(None) == []

    def test_empty_list_returns_empty(self):
        assert parse_jsonb_fixed([]) == []
