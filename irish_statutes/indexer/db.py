import json
from datetime import datetime

import pandas as pd
import psycopg2

CONNECTION_STRING = "postgresql://postgres:pword@localhost:5432/vector_db"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS eval_results (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    llm_response TEXT,
    scores JSONB,
    source_nodes JSONB,
    rating VARCHAR(10) DEFAULT 'unrated',
    top_k INTEGER DEFAULT 5,
    created_at TIMESTAMP DEFAULT NOW(),
    rated_at TIMESTAMP
);
"""


def get_connection():
    return psycopg2.connect(CONNECTION_STRING)


def create_eval_table():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)


def save_eval_result(query: str, response: str, scores: list, source_nodes: list, top_k: int = 5) -> int:
    scores_json = json.dumps(scores)
    nodes_json = json.dumps([
        {"text": n.text, "score": n.score, "metadata": n.metadata}
        for n in source_nodes
    ])
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO eval_results (query_text, llm_response, scores, source_nodes, top_k)
                   VALUES (%s, %s, %s, %s, %s) RETURNING id""",
                (query, response, scores_json, nodes_json, top_k)
            )
            row_id = cur.fetchone()[0]
    return row_id


def update_rating(row_id: int, rating: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE eval_results SET rating = %s, rated_at = %s WHERE id = %s",
                (rating, datetime.now(), row_id)
            )


def load_all_results() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql("SELECT * FROM eval_results ORDER BY created_at DESC", conn)


def load_unrated() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql(
            "SELECT * FROM eval_results WHERE rating = 'unrated' ORDER BY id",
            conn
        )


def parse_jsonb_list(val) -> list:
    if not val:
        return []
    if isinstance(val, list):
        return val
    return json.loads(val)


def query_exists(query_text: str) -> bool:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM eval_results WHERE query_text = %s LIMIT 1",
                (query_text,)
            )
            return cur.fetchone() is not None


# ---------------------------------------------------------------------------
# Laws / law_sections helpers
# ---------------------------------------------------------------------------

def upsert_law(name: str, year: int, act_number: int, url: str = None, html_path: str = None) -> int:
    """Insert or update a law row; return its id."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO laws (name, year, act_number, url, html_path)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (year, act_number) DO UPDATE
                       SET name = EXCLUDED.name,
                           url = COALESCE(EXCLUDED.url, laws.url),
                           html_path = COALESCE(EXCLUDED.html_path, laws.html_path)
                   RETURNING id""",
                (name, year, act_number, url, html_path)
            )
            return cur.fetchone()[0]


def insert_sections(law_id: int, sections: list[dict]) -> None:
    """
    Bulk-insert parsed section rows for a law.
    Each dict must have: section_type, section_ref, section_title, text_content,
                         position, parent_ref (used to resolve parent_id).

    Existing sections for this law are deleted first (idempotent re-ingest).
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM law_sections WHERE law_id = %s", (law_id,))

            # First pass: insert all rows, collect (section_ref → row_id) and (index → row_id)
            ref_to_id: dict[str, int] = {}
            inserted_ids: list[int] = []

            for s in sections:
                cur.execute(
                    """INSERT INTO law_sections
                           (law_id, section_type, section_ref, section_title, text_content, position)
                       VALUES (%s, %s, %s, %s, %s, %s)
                       RETURNING id""",
                    (law_id, s["section_type"], s["section_ref"],
                     s["section_title"], s["text_content"], s["position"])
                )
                row_id = cur.fetchone()[0]
                inserted_ids.append(row_id)
                if s["section_ref"]:
                    ref_to_id[s["section_ref"]] = row_id

            # Second pass: update parent_id where we can resolve parent_ref
            for row_id, s in zip(inserted_ids, sections):
                parent_ref = s.get("parent_ref")
                if parent_ref and parent_ref in ref_to_id:
                    cur.execute(
                        "UPDATE law_sections SET parent_id = %s WHERE id = %s",
                        (ref_to_id[parent_ref], row_id)
                    )


def get_law_by_name(name: str, year: int = None) -> dict | None:
    sql = "SELECT * FROM laws WHERE name ILIKE %s"
    params: list = [f"%{name}%"]
    if year:
        sql += " AND year = %s"
        params.append(year)
    sql += " LIMIT 1"
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))


def get_law_sections(law_id: int) -> list[dict]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM law_sections WHERE law_id = %s ORDER BY position",
                (law_id,)
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_section_by_ref(law_id: int, section_ref: str) -> dict | None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM law_sections WHERE law_id = %s AND section_ref = %s LIMIT 1",
                (law_id, section_ref)
            )
            row = cur.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))


def search_laws(query: str) -> list[dict]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM laws WHERE name ILIKE %s ORDER BY year, act_number",
                (f"%{query}%",)
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
