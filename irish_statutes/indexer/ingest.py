"""
Ingest pipeline: raw_html/ → parse → laws + law_sections tables → vector store.

Usage:
    # Ingest all acts in raw_html/
    uv run python -m indexer.ingest

    # Ingest only a specific file (for testing)
    uv run python -m indexer.ingest --file raw_html/2004/act_11.html

    # Ingest all acts but skip vector indexing
    uv run python -m indexer.ingest --no-vectors
"""
from __future__ import annotations

import argparse
import os
import re
import sys

from indexer.db import upsert_law, insert_sections
from indexer.parse_statute import parse_html, flatten

RAW_HTML_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw_html")


def _year_and_number_from_path(html_path: str) -> tuple[int, int] | None:
    """Extract (year, act_number) from path like raw_html/2004/act_11.html"""
    m = re.search(r"[/\\](\d{4})[/\\]act_(\d+)\.html$", html_path)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def ingest_file(html_path: str, law_name: str = "", url: str = "", embed: bool = True) -> int:
    """
    Parse one HTML file, insert into DB, optionally embed sections.
    Returns the law_id.
    """
    parsed = _year_and_number_from_path(html_path)
    if not parsed:
        raise ValueError(f"Cannot determine year/act_number from path: {html_path}")
    year, act_number = parsed

    with open(html_path, encoding="utf-8") as f:
        html = f.read()

    root = parse_html(html, law_name=law_name, year=year)
    if not law_name:
        law_name = root.section_title or f"Act {act_number} of {year}"

    law_id = upsert_law(
        name=law_name,
        year=year,
        act_number=act_number,
        url=url or None,
        html_path=html_path,
    )
    print(law_id)
    sections = flatten(root)
    print(f"{len(sections)=}")
    insert_sections(law_id, sections)

    if embed and sections:
        _embed_sections(law_id, law_name, year, sections)

    return law_id


def _embed_sections(law_id: int, law_name: str, year: int, sections: list[dict]) -> None:
    """Embed section-level documents into PGVectorStore."""
    from llama_index.core import VectorStoreIndex, StorageContext, Settings
    from llama_index.core.schema import Document
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.ollama import Ollama
    from indexer.vstore import get_vector_store

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    Settings.llm = Ollama(model="llama3", request_timeout=180.0)

    embeddable_types = {"section", "subsection"}
    docs = []
    for s in sections:
        if s["section_type"] not in embeddable_types:
            continue
        text = s["text_content"] or ""
        if not text.strip():
            continue
        title_part = f"{s['section_title']}: " if s["section_title"] else ""
        full_text = f"{law_name} — Section {s['section_ref']}\n{title_part}{text}"
        doc = Document(
            text=full_text,
            metadata={
                "law_id": law_id,
                "law_name": law_name,
                "year": year,
                "section_ref": s["section_ref"],
                "section_type": s["section_type"],
                "section_title": s["section_title"] or "",
            },
        )
        docs.append(doc)

    if not docs:
        return

    vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(docs, storage_context=storage_context, show_progress=False)


def ingest_all(raw_html_dir: str = RAW_HTML_DIR, embed: bool = True) -> None:
    """Walk raw_html/ and ingest every act HTML file."""
    total = skipped = errors = 0

    for year_dir in sorted(os.listdir(raw_html_dir)):
        year_path = os.path.join(raw_html_dir, year_dir)
        if not os.path.isdir(year_path):
            continue
        for fname in sorted(os.listdir(year_path)):
            if not fname.endswith(".html"):
                continue
            html_path = os.path.join(year_path, fname)
            try:
                law_id = ingest_file(html_path, embed=embed)
                total += 1
                if total % 20 == 0:
                    print(f"  Ingested {total} acts...")
            except Exception as e:
                print(f"  ERROR {html_path}: {e}", file=sys.stderr)
                errors += 1

    print(f"Done: {total} ingested, {skipped} skipped, {errors} errors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Ingest a single HTML file")
    parser.add_argument("--raw-html-dir", default=RAW_HTML_DIR)
    parser.add_argument("--no-vectors", action="store_true", help="Skip vector embedding")
    args = parser.parse_args()

    embed = not args.no_vectors

    if args.file:
        law_id = ingest_file(args.file, embed=embed)
        print(f"Ingested law_id={law_id}")
    else:
        ingest_all(raw_html_dir=args.raw_html_dir, embed=embed)
