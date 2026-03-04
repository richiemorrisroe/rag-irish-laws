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


def query_exists(query_text: str) -> bool:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM eval_results WHERE query_text = %s LIMIT 1",
                (query_text,)
            )
            return cur.fetchone() is not None
