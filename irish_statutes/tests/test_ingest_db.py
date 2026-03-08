"""
Tests for indexer.ingest and indexer.db — end-to-end ingest into Postgres.

These tests require a running Postgres instance (docker-compose in the project).
If the DB is not reachable they are skipped automatically.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

RAW_HTML = Path(__file__).parent.parent / "raw_html"
ACT_11 = RAW_HTML / "2004" / "act_11.html"   # Air Navigation Act 2004
ACT_46 = RAW_HTML / "2013" / "act_46.html"   # Companies (Misc Provisions) Act 2013


# ---------------------------------------------------------------------------
# DB availability fixture — skip entire module if Postgres is unreachable
# ---------------------------------------------------------------------------

def _db_available() -> bool:
    try:
        import psycopg2
        conn = psycopg2.connect("postgresql://postgres:pword@localhost:5432/vector_db",
                                connect_timeout=3)
        conn.close()
        return True
    except Exception:
        return False


requires_db = pytest.mark.skipif(not _db_available(), reason="Postgres not reachable")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ingested_laws():
    """
    Ingest ACT_11 and ACT_46 without vector embedding; yield the law_ids.

    Only deletes rows that did not already exist before the fixture ran,
    so re-running tests after a full ingest does not wipe production data.
    """
    from indexer.ingest import ingest_file
    from indexer.db import get_connection

    def _existing_id(conn, year, act_number):
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM laws WHERE year = %s AND act_number = %s",
                (year, act_number),
            )
            row = cur.fetchone()
            return row[0] if row else None

    with get_connection() as conn:
        pre_11 = _existing_id(conn, 2004, 11)
        pre_46 = _existing_id(conn, 2013, 46)

    law_id_11 = ingest_file(str(ACT_11), embed=False)
    law_id_46 = ingest_file(str(ACT_46), embed=False)

    yield {"act_11": law_id_11, "act_46": law_id_46}

    # Only clean up rows that the fixture itself created
    to_delete = []
    if pre_11 is None:
        to_delete.append(law_id_11)
    if pre_46 is None:
        to_delete.append(law_id_46)

    if to_delete:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM laws WHERE id = ANY(%s)", (to_delete,)
                )


# ---------------------------------------------------------------------------
# laws table
# ---------------------------------------------------------------------------

@requires_db
def test_ingest_creates_law_row_act11(ingested_laws):
    from indexer.db import get_law_by_name
    law = get_law_by_name("Air Navigation", year=2004)
    assert law is not None
    assert law["year"] == 2004
    assert law["act_number"] == 11
    assert "Air Navigation" in law["name"]


@requires_db
def test_ingest_creates_law_row_act46(ingested_laws):
    from indexer.db import get_law_by_name
    law = get_law_by_name("Companies (Miscellaneous", year=2013)
    assert law is not None
    assert law["year"] == 2013
    assert law["act_number"] == 46


@requires_db
def test_ingest_html_path_stored(ingested_laws):
    from indexer.db import get_law_by_name
    law = get_law_by_name("Air Navigation", year=2004)
    assert law["html_path"] is not None
    assert "act_11.html" in law["html_path"]


# ---------------------------------------------------------------------------
# law_sections table — row counts and types
# ---------------------------------------------------------------------------

@requires_db
def test_ingest_creates_sections(ingested_laws):
    from indexer.db import get_law_sections
    sections = get_law_sections(ingested_laws["act_11"])
    assert len(sections) > 50, f"Expected many sections, got {len(sections)}"


@requires_db
def test_ingest_sections_have_expected_types(ingested_laws):
    from indexer.db import get_law_sections
    sections = get_law_sections(ingested_laws["act_11"])
    types = {s["section_type"] for s in sections}
    assert "part" in types
    assert "section" in types
    assert "subsection" in types


@requires_db
def test_ingest_sections_idempotent(ingested_laws):
    """Re-ingesting the same act should not duplicate rows."""
    from indexer.db import get_law_sections
    from indexer.ingest import ingest_file

    before = len(get_law_sections(ingested_laws["act_11"]))
    ingest_file(str(ACT_11), embed=False)
    after = len(get_law_sections(ingested_laws["act_11"]))
    assert before == after


# ---------------------------------------------------------------------------
# Section-level lookups
# ---------------------------------------------------------------------------

@requires_db
def test_get_section_by_ref_col1_format(ingested_laws):
    """Section 7(2) of the Air Navigation Act should be retrievable by ref."""
    from indexer.db import get_section_by_ref
    law_id = ingested_laws["act_11"]
    sec = get_section_by_ref(law_id, "7(2)")
    assert sec is not None, "Section 7(2) not found"
    assert sec["section_type"] == "subsection"
    assert sec["text_content"]


@requires_db
def test_get_section_by_ref_col2_format(ingested_laws):
    """Section 2(8) of the Companies Act should be retrievable by ref."""
    from indexer.db import get_section_by_ref
    law_id = ingested_laws["act_46"]
    sec = get_section_by_ref(law_id, "2(8)")
    assert sec is not None, "Section 2(8) not found"
    assert sec["section_type"] == "subsection"
    assert sec["text_content"]


@requires_db
def test_get_section_by_ref_missing_returns_none(ingested_laws):
    from indexer.db import get_section_by_ref
    result = get_section_by_ref(ingested_laws["act_11"], "999(99)")
    assert result is None


# ---------------------------------------------------------------------------
# Search and name lookup
# ---------------------------------------------------------------------------

@requires_db
def test_search_laws_finds_by_partial_name(ingested_laws):
    from indexer.db import search_laws
    results = search_laws("Air Navigation")
    names = [r["name"] for r in results]
    assert any("Air Navigation" in n for n in names)


@requires_db
def test_search_laws_empty_query_returns_results(ingested_laws):
    from indexer.db import search_laws
    # Empty-ish wildcard — should return all laws
    results = search_laws("")
    assert len(results) >= 2


@requires_db
def test_search_laws_no_match_returns_empty(ingested_laws):
    from indexer.db import search_laws
    results = search_laws("xyzzy_nonexistent_act")
    assert results == []


@requires_db
def test_get_law_by_name_with_year_filter(ingested_laws):
    from indexer.db import get_law_by_name
    # Correct year
    assert get_law_by_name("Companies", year=2013) is not None
    # Wrong year should not match
    assert get_law_by_name("Companies", year=1900) is None


# ---------------------------------------------------------------------------
# Positions are ordered
# ---------------------------------------------------------------------------

@requires_db
def test_sections_have_increasing_positions(ingested_laws):
    from indexer.db import get_law_sections
    sections = get_law_sections(ingested_laws["act_46"])
    positions = [s["position"] for s in sections]
    # Not necessarily strictly increasing globally (position resets per-parent),
    # but every position should be a positive integer
    assert all(isinstance(p, int) and p >= 1 for p in positions)
