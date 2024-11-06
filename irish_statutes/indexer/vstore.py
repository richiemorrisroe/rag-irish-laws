import os

import psycopg2
from sqlalchemy import make_url

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.vector_stores.postgres import PGVectorStore

def get_index_from_database(table_name="irish_laws"):
    connection_string = "postgresql://postgres:pword@localhost:5432"
    db_name = "vector_db"
    conn = psycopg2.connect(connection_string)
    conn.autocommit = True
    url = make_url(connection_string)

    vector_store = PGVectorStore.from_params(
        database=db_name,
        host=url.host,
        password=url.password,
        port=url.port,
        user=url.username,
        table_name=table_name,
        embed_dim=768,  # openai embedding dimension
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return index


def get_index_from_filesystem(data_dir, storage_format):
    if data_dir and storage_format == 'filetype':
        PERSIST_DIR = "./full_storage"
    else:
        PERSIST_DIR = data_dir
    if not os.path.exists(PERSIST_DIR) and storage_format == 'filetype':
        raise ValueError("please index some files first")
    elif storage_format == 'filetype':
        # load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        return index
