import argparse
import fnmatch
import glob
import logging
import os
import sys

from pprint import pprint

from IPython.display import display



from llama_index.core import (SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext,
                              load_index_from_storage, get_response_synthesizer)
from llama_index.readers.file import FlatReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding




from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

from indexer.utils import setup_logger
from indexer.eval_queries import query_llm, setup_llm
from indexer.vstore import get_index_from_database

parser = argparse.ArgumentParser()

parser.add_argument("--query")
parser.add_argument("--data_dir")
parser.add_argument('--storage-format', choices=['postgres', 'file'])

args = parser.parse_args()

# DATA_DIR = './laws_test'

logger = setup_logger(__file__)


logger.warning("extracted all plain text")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
logger.warning("downloaded embedding model")
# ollama
Settings.llm = setup_llm()
logger.warning("set up Ollama")

if args.storage_format == 'postgres':
    index = get_index_from_database()

if not args.query:
    query = "What are the requirements for data protection in ireland?"
else:
    query = args.query
response = query_llm(index, query)
logger.warning(f"scores for matched documents were {response.scores=}")
logger.info(f"{response.source_nodes=}")
print(response)
