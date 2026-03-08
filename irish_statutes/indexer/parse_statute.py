"""
Parse Irish Statute Book HTML into a structured section tree.

HTML structure (table-based, 3-column rows):
  col[0]  col[1]           col[2]
  ------  ---------------  ---------------------------------------------------------
  ''      ''               'PART 1Preliminary Matters'   ← part heading
  ''      ''               'SCHEDULE 1THE WARSAW...'     ← schedule heading
  ''      'Short title.'   '1.—This Act may be cited...' ← section (title in col[1])
  ''      ''               '(2) In this Act...'          ← subsection
  ''      ''               '(a) the provisions of...'    ← paragraph
  ''      ''               '(i) the amount specified...' ← sub-paragraph
  ''      ''               'regular continuation text'   ← body text appended to current node
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

SECTION_TYPES = ("act", "part", "section", "subsection", "paragraph", "subparagraph", "schedule", "chapter", "article")


@dataclass
class StatuteNode:
    section_type: str       # one of SECTION_TYPES
    section_ref: str        # e.g. '7', '7(2)', 'I', 'Schedule 1'
    section_title: str      # e.g. 'Short title.'
    text_content: str       # full text of this node (may be multi-line)
    children: list[StatuteNode] = field(default_factory=list)
    position: int = 0


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

RE_PART      = re.compile(r"^PART\s+(\w+)(.*)", re.DOTALL)
RE_SCHEDULE  = re.compile(r"^SCHEDULE\s+(\w+)(.*)", re.DOTALL)
RE_CHAPTER   = re.compile(r"^CHAPTER\s+(\w+)(.*)", re.DOTALL)
RE_ARTICLE   = re.compile(r"^Article\s+(\d+)(.*)", re.DOTALL)
RE_SECTION   = re.compile(r"^(\d+)\s*[.—–]\s*(.*)", re.DOTALL)
RE_SUBSECTION = re.compile(r"^\((\d+)\)\s*(.*)", re.DOTALL)
RE_PARAGRAPH  = re.compile(r"^\(([a-z]{1,3})\)\s*(.*)", re.DOTALL)
RE_SUBPARAGRAPH = re.compile(r"^\(([ivxlcdm]+)\)\s*(.*)", re.DOTALL)


def _classify_row(c2: str, c1: str = "") -> Optional[tuple[str, str, str]]:
    """
    Classify the text in col[2].
    Returns (section_type, ref, remaining_text) or None if just body text.
    c1 is the section-title column (col[1]); when non-empty it signals a section row.
    """
    text = c2.strip()

    m = RE_PART.match(text)
    if m:
        return ("part", m.group(1).strip(), m.group(2).strip())

    m = RE_SCHEDULE.match(text)
    if m:
        return ("schedule", f"Schedule {m.group(1).strip()}", m.group(2).strip())

    m = RE_CHAPTER.match(text)
    if m:
        return ("chapter", f"Chapter {m.group(1).strip()}", m.group(2).strip())

    m = RE_ARTICLE.match(text)
    if m:
        return ("article", f"Article {m.group(1)}", m.group(2).strip())

    # Section: col[1] has a title, OR col[2] starts with "N." / "N.—" / "N ."
    if c1:
        m = RE_SECTION.match(text)
        if m:
            return ("section", m.group(1), m.group(2).strip())
    else:
        # No title in col[1] — only treat as a new section if it starts with "N ." or "N.—"
        # (i.e., digit + optional space + [.—–]) to avoid matching inline "3(7)" references
        m = RE_SECTION.match(text)
        if m and re.match(r"^\d+\s*[.—–]", text):
            return ("section", m.group(1), m.group(2).strip())

    m = RE_SUBSECTION.match(text)
    if m:
        return ("subsection", m.group(1), m.group(2).strip())

    # paragraph: (a), (b), ... but not (i), (ii), (iii) which are subparagraphs
    m = RE_PARAGRAPH.match(text)
    if m:
        ref = m.group(1)
        # Roman numeral check: single letters i, v, x, l, c could be ambiguous;
        # treat pure roman-numeral refs that look like (i)/(ii)/(iii) as subparagraph
        if re.fullmatch(r"[ivxlcdm]+", ref):
            return ("subparagraph", ref, m.group(2).strip())
        return ("paragraph", ref, m.group(2).strip())

    return None


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_html(html: str, law_name: str = "", year: int = 0) -> StatuteNode:
    """
    Parse statute HTML and return a StatuteNode tree rooted at the act level.
    """
    soup = BeautifulSoup(html, "lxml")
    tables = soup.find_all("table")
    # breakpoint()
    if not tables:
        # Fallback: grab all body text
        return StatuteNode("act", "", law_name, soup.get_text(separator="\n", strip=True))

    main_table = tables[0]
    rows = main_table.find_all("tr")
    # extract_act_title(rows)
    # Extract act title from the HTML (all-caps line containing "ACT", before "BE IT ENACTED")
    if not law_name:
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 3:
                continue
            c2 = cells[2].get_text(separator=" ", strip=True)
            if "ENACTED" in c2.upper():
                break
            if c2.isupper() and "ACT" in c2:
                law_name = c2.title()  # convert ALL-CAPS to Title Case
                break

    root = StatuteNode("act", "", law_name, "")
    # create_stack_and_position_counters(root) -> (stack, position_counters)
    # Stack tracks current nesting:  list of (node, depth_rank)
    # depth_rank:  act=0, part/schedule=1, section=2, subsection=3, paragraph=4, subparagraph=5
    RANKS = {"act": 0, "part": 1, "schedule": 1, "chapter": 1, "section": 2,
              "article": 2, "subsection": 3, "paragraph": 4, "subparagraph": 5}

    stack: list[StatuteNode] = [root]
    position_counters: dict[int, int] = {0: 0}  # rank → counter

    def current_parent_for(rank: int) -> StatuteNode:
        # Pop stack until top has rank < new rank
        while len(stack) > 1 and RANKS.get(stack[-1].section_type, 0) >= rank:
            stack.pop()
        return stack[-1]

    def append_node(node: StatuteNode, rank: int):
        parent = current_parent_for(rank)
        pos = position_counters.get(id(parent), 0) + 1
        position_counters[id(parent)] = pos
        node.position = pos
        parent.children.append(node)
        stack.append(node)

    current_section: Optional[StatuteNode] = None  # track for section_ref like "7(2)"

    past_enactment = False  # skip table-of-contents rows before "BE IT ENACTED"
    #also extract_act_title, for some reason???
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 3:
            continue

        c0 = cells[0].get_text(strip=True)
        c1 = cells[1].get_text(strip=True)
        c2 = cells[2].get_text(separator=" ", strip=True)

        if not c2:
            continue

        if not past_enactment:
            if "BE IT ENACTED" in c2.upper() or "HEREBY ENACTED" in c2.upper():
                past_enactment = True
                # Capture the act title from the "Number N of YEAR" line if seen before
            continue

        # Classify row — pass c1 so section detection knows whether a title is present
        classification = _classify_row(c2, c1)

        if classification:
            stype, ref, text = classification

            if stype == "section":
                node = StatuteNode("section", ref, c1, text)
                append_node(node, RANKS["section"])
                current_section = node

            elif stype == "subsection":
                # Build compound ref: "7(2)"
                if current_section:
                    compound_ref = f"{current_section.section_ref}({ref})"
                else:
                    compound_ref = f"({ref})"
                node = StatuteNode("subsection", compound_ref, "", text)
                append_node(node, RANKS["subsection"])

            elif stype in ("paragraph", "subparagraph"):
                node = StatuteNode(stype, ref, "", text)
                append_node(node, RANKS[stype])

            elif stype in ("part", "schedule", "chapter", "article"):
                node = StatuteNode(stype, ref, text, "")
                append_node(node, RANKS[stype])
                if stype in ("part", "schedule"):
                    current_section = None  # reset section tracking across parts
                elif stype == "article":
                    current_section = node  # track article as parent for subsection refs

        else:
            # Plain continuation text — append to the most recent node on stack
            if len(stack) > 1:
                target = stack[-1]
                if target.text_content:
                    target.text_content += "\n" + c2
                else:
                    target.text_content = c2

    return root


# ---------------------------------------------------------------------------
# Flatten tree to list of dicts for DB insertion
# ---------------------------------------------------------------------------

def flatten(root: StatuteNode) -> list[dict]:
    """
    Depth-first walk of the tree; returns list of dicts matching law_sections schema.
    parent_ref is the section_ref of the immediate parent (or None for top-level).
    """
    results: list[dict] = []

    def walk(node: StatuteNode, parent_ref: Optional[str]):
        results.append({
            "section_type":  node.section_type,
            "section_ref":   node.section_ref,
            "section_title": node.section_title,
            "text_content":  node.text_content,
            "position":      node.position,
            "parent_ref":    parent_ref,
        })
        for child in node.children:
            walk(child, node.section_ref)

    for child in root.children:
        walk(child, None)
    return results


# ---------------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------------

def debug_structure(html_path: str, max_nodes: int = 80) -> None:
    """Print the detected section tree of an act HTML file."""
    with open(html_path, encoding="utf-8") as f:
        html = f.read()

    root = parse_html(html)
    rows = flatten(root)

    print(f"Act: {root.section_ref!r} — {root.section_title!r}")
    print(f"Total nodes: {len(rows)}\n")

    indent_map = {"part": 0, "schedule": 0, "chapter": 1, "section": 1,
                  "article": 1, "subsection": 2, "paragraph": 3, "subparagraph": 4}

    for r in rows[:max_nodes]:
        indent = "  " * indent_map.get(r["section_type"], 0)
        title = f" — {r['section_title']}" if r["section_title"] else ""
        text_preview = (r["text_content"] or "")[:60].replace("\n", " ")
        print(f"{indent}[{r['section_type']}] {r['section_ref']}{title}  |  {text_preview!r}")

    if len(rows) > max_nodes:
        print(f"  ... ({len(rows) - max_nodes} more nodes)")
