"""
Tests for indexer.parse_statute — HTML parsing, title extraction, structure detection.
No database required.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from indexer.parse_statute import parse_html, flatten, debug_structure

RAW_HTML = Path(__file__).parent.parent / "raw_html"

# Two sample acts that exercise different HTML formats:
#   act_11.html  — col[1]-title format (section title in col[1], text in col[2])
#   act_46.html  — all-in-col[2] format (everything in col[2])
ACT_11 = RAW_HTML / "2004" / "act_11.html"   # Air Navigation Act 2004
ACT_46 = RAW_HTML / "2013" / "act_46.html"   # Companies (Misc Provisions) Act 2013


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Title extraction
# ---------------------------------------------------------------------------

def test_parse_extracts_title_col1_format():
    root = parse_html(_read(ACT_11))
    assert "Air Navigation" in root.section_title
    assert "2004" in root.section_title


def test_parse_extracts_title_col2_format():
    root = parse_html(_read(ACT_46))
    assert "Companies" in root.section_title
    assert "2013" in root.section_title


def test_explicit_law_name_overrides_html_title():
    root = parse_html(_read(ACT_11), law_name="Custom Name")
    assert root.section_title == "Custom Name"


# ---------------------------------------------------------------------------
# Table-of-contents deduplication (BE IT ENACTED filter)
# ---------------------------------------------------------------------------

def test_no_duplicate_parts_col1_format():
    """TOC rows before 'BE IT ENACTED' must not produce duplicate part nodes."""
    root = parse_html(_read(ACT_11))
    parts = [n for n in root.children if n.section_type == "part"]
    part_refs = [p.section_ref for p in parts]
    # PART 1, PART 2, PART 3 appear in TOC and body — only body ones should survive
    assert len(part_refs) == len(set(part_refs)), f"Duplicate part refs: {part_refs}"


# ---------------------------------------------------------------------------
# col[1]-title format (Act 11 / 2004)
# ---------------------------------------------------------------------------

def test_col1_format_has_parts():
    root = parse_html(_read(ACT_11))
    parts = [n for n in root.children if n.section_type == "part"]
    assert len(parts) >= 3


def test_col1_format_section_has_title():
    root = parse_html(_read(ACT_11))
    sections = flatten(root)
    sec7 = next((s for s in sections if s["section_ref"] == "7"), None)
    assert sec7 is not None, "Section 7 not found"
    assert sec7["section_title"]  # col[1] title should be populated


def test_col1_format_compound_subsection_refs():
    sections = flatten(parse_html(_read(ACT_11)))
    refs = {s["section_ref"] for s in sections if s["section_type"] == "subsection"}
    assert "7(2)" in refs, f"Expected '7(2)' in subsection refs, got sample: {list(refs)[:10]}"
    assert "3(2)" in refs


def test_col1_format_subsection_text():
    sections = flatten(parse_html(_read(ACT_11)))
    s72 = next(s for s in sections if s["section_ref"] == "7(2)")
    assert "subsection" in s72["text_content"].lower() or len(s72["text_content"]) > 20


# ---------------------------------------------------------------------------
# col[2]-only format (Act 46 / 2013)
# ---------------------------------------------------------------------------

def test_col2_format_detects_sections():
    sections = flatten(parse_html(_read(ACT_46)))
    sec_refs = {s["section_ref"] for s in sections if s["section_type"] == "section"}
    assert "1" in sec_refs
    assert "2" in sec_refs
    assert "3" in sec_refs


def test_col2_format_compound_subsection_refs():
    sections = flatten(parse_html(_read(ACT_46)))
    refs = {s["section_ref"] for s in sections if s["section_type"] == "subsection"}
    # Section 2's subsections should have compound refs like "2(8)", "2(9)"
    assert any(r.startswith("2(") for r in refs), f"No '2(...)' refs found. Sample: {list(refs)[:10]}"


def test_col2_format_no_bare_subsection_refs():
    """After the fix, subsections must not appear as bare '(8)' — they need a section prefix."""
    sections = flatten(parse_html(_read(ACT_46)))
    bare = [s["section_ref"] for s in sections
            if s["section_type"] == "subsection" and s["section_ref"].startswith("(")]
    assert bare == [], f"Found bare subsection refs: {bare}"


# ---------------------------------------------------------------------------
# flatten() structure
# ---------------------------------------------------------------------------

def test_flatten_returns_list_of_dicts():
    sections = flatten(parse_html(_read(ACT_11)))
    assert isinstance(sections, list)
    assert len(sections) > 0
    expected_keys = {"section_type", "section_ref", "section_title", "text_content", "position", "parent_ref"}
    assert expected_keys.issubset(sections[0].keys())


def test_flatten_position_is_positive_int():
    sections = flatten(parse_html(_read(ACT_11)))
    for s in sections:
        assert isinstance(s["position"], int)
        assert s["position"] >= 1


def test_flatten_section_types_are_valid():
    from indexer.parse_statute import SECTION_TYPES
    sections = flatten(parse_html(_read(ACT_11)))
    for s in sections:
        assert s["section_type"] in SECTION_TYPES, f"Unknown type: {s['section_type']}"


# ---------------------------------------------------------------------------
# debug_structure helper
# ---------------------------------------------------------------------------

def test_debug_structure_runs_without_error(capsys):
    debug_structure(str(ACT_11), max_nodes=10)
    captured = capsys.readouterr()
    assert "Total nodes:" in captured.out


# ---------------------------------------------------------------------------
# raw_html directory (backfill verification)
# ---------------------------------------------------------------------------

def test_raw_html_year_dirs_exist():
    assert RAW_HTML.is_dir(), f"raw_html/ not found at {RAW_HTML}"
    year_dirs = [d for d in RAW_HTML.iterdir() if d.is_dir() and d.name.isdigit()]
    assert len(year_dirs) >= 10, "Expected at least 10 year directories in raw_html/"


def test_raw_html_contains_html_files():
    html_files = list(RAW_HTML.rglob("act_*.html"))
    assert len(html_files) >= 100, f"Expected many .html files, found {len(html_files)}"


def test_raw_html_filenames_match_pattern():
    import re
    html_files = list(RAW_HTML.rglob("act_*.html"))
    for f in html_files[:20]:  # spot-check first 20
        assert re.match(r"act_\d+\.html", f.name), f"Unexpected filename: {f.name}"
        assert f.parent.name.isdigit(), f"Parent dir not a year: {f.parent.name}"
