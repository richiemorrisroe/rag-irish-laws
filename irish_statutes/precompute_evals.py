"""
Batch script to precompute eval results for all QUERIES.
Run with: uv run python irish_statutes/precompute_evals.py

Idempotent — skips queries already in the eval_results table.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from indexer.db import create_eval_table, save_eval_result, query_exists
from indexer.eval_queries import QUERIES, query_llm, setup_llm
from indexer.utils import setup_logger
from indexer.vstore import get_index_from_database

logger = setup_logger(__file__)


def main():
    logger.warning("Setting up embedding model...")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    Settings.llm = setup_llm()

    logger.warning("Creating eval_results table if needed...")
    create_eval_table()

    logger.warning("Loading vector index...")
    index = get_index_from_database()

    skipped = 0
    saved = 0

    for i, query in enumerate(QUERIES, 1):
        query = query.strip()
        if query_exists(query):
            logger.warning(f"[{i}/{len(QUERIES)}] SKIP (already in DB): {query[:60]}")
            skipped += 1
            continue

        logger.warning(f"[{i}/{len(QUERIES)}] Running query: {query[:60]}")
        result = query_llm(index, query, top_k=5, logger=logger)

        row_id = save_eval_result(
            query=query,
            response=result.response,
            scores=result.scores,
            source_nodes=result.source_nodes,
            top_k=result.top_k,
        )
        logger.warning(f"  -> Saved with id={row_id}")
        saved += 1

    logger.warning(f"Done. {saved} saved, {skipped} skipped.")


if __name__ == "__main__":
    main()
