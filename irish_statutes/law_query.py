import argparse
import fnmatch
import glob
import logging
import os
import sys

from pprint import pprint

from IPython.display import display

import psycopg2
from sqlalchemy import make_url

from llama_index.core import (SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext,
                              load_index_from_storage, get_response_synthesizer)
from llama_index.readers.file import FlatReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.llms.ollama import Ollama


from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

parser = argparse.ArgumentParser()

parser.add_argument("--query")
parser.add_argument("--data_dir")
parser.add_argument('--storage-format', choices=['postgres', 'file'])

args = parser.parse_args()

# DATA_DIR = './laws_test'


logger = logging.getLogger()

logging.basicConfig(filename='law_query.log',
                    encoding='utf-8', level=logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.WARNING)
formatter = logging.Formatter(
    '%(asctime)s - %(filename)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logging.warning("extracted all plain text")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
logging.warning("downloaded embedding model")
# ollama
Settings.llm = Ollama(model="llama3", temperature=0.5, request_timeout=90.0)
logging.warning("set up Ollama")


def display_prompt_dict(prompts_dict):
    for k, p in prompts_dict.items():
        text_md = f"**Prompt Key**: {k}<br>" f"**Text:** <br>"
        pprint(text_md)
        print(p.get_template())
        # display(Markdown("<br><br>"))


if args.storage_format == 'postgres':

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
        table_name="irish_laws",
        embed_dim=768,  # openai embedding dimension
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)


# if not args.data_dir and args.storage_format == 'filetype':
#     PERSIST_DIR = "./full_storage"
# else:
#     PERSIST_DIR = args.data_dir

# if not os.path.exists(PERSIST_DIR) and args.storage_format == 'filetype':

#     logging.warning("you should run `law_index.py` to generate an index first")
#     raise ValueError("please run law_index.py first")
# elif args.storage_format == 'filetype':
#     # load the existing index
#     storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
#     index = load_index_from_storage(storage_context)


if not args.query:
    query = "What are the requirements for data protection in ireland?"
else:
    query = args.query
query_engine = index.as_query_engine(similarity_top_k=2)
prompts_dict = query_engine.get_prompts()
logger.info(f"{prompts_dict=}")
# display_prompt_dict(prompts_dict)
response = query_engine.query(query)
scores = [x.score for x in response.source_nodes]
logger.warning(f"scores for matched documents were {scores=}")
logger.info(f"{response.source_nodes}")

logger.info(f"{response.source_nodes=}")
print(response)


QUERIES = ["""What are the requirements for data protection under the 2018 act?""",
           """What is the procedure for firing an employee in Irish law?""",
           """what are the functions of a data protection officer?""",
           """what information must motor insurance providers supply to policyholders?""",
           """What constitutes a breach of equality law?""",
           """How many breaks are employees entitled to during an eight hour shift?""",
           """What are the legal requirements for an employment contract?""",
           """What constitutes unfair dismissal under irish law?""",
           """What grounds does the Equality act cover?""",
           """Is it legal to import gasoline for personal use?""",
           """What requirements must be met to transport gasoline?""",
           """Under what grounds may a residential tenancy be terminated?""",
           """May a joint residential tenancy be terminated if one of the parties leaves the property?""",
           """What are the conditions required for irish citizenship?""",
           """What are the conditions required for a claim of contructive dismissal to succeed?""",
           """How is land ownership determined?""",
           """What happens if someone dies without making a will?""",
           """What is intestate succession in the context of inheritance?""",
           """What is intestate succession?""",
           """What is the minimum wage for full time employees?""",
           """What is the national minimum hourly rate of pay?""",
           """Is it legal to drive without insurance?"""
           """What are the penalities for driving without insurance?""",
           """Can i be prosecuted for allowing my friend to drive a vehicle on which they are not insured?""",
           """What are the requirements for setting up a limited company in Ireland?""",
           """What are the differences between a private limited company and a private unlimited company?""",
           """Does a private limited company need to file accounts, and if so, how often?""",
           
           
           ]
