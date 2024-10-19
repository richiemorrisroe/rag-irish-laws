import argparse
import os
import logging
import sys

from llama_index.core import (SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext,
                              load_index_from_storage)
from llama_index.readers.file import FlatReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

parser = argparse.ArgumentParser()

parser.add_argument("--query")

args = parser.parse_args()

DATA_DIR = './laws_test'


logger = logging.getLogger()

logging.basicConfig(filename='law_index.log',
                    encoding='utf-8', level=logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logging.warning("extracted all plain text")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
logging.warning("downloaded embedding model")
# ollama
Settings.llm = Ollama(model="llama3", request_timeout=180.0)
logging.warning("set up Ollama")

PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    logging.warning("got to reading files")
    parser = FlatReader()
    file_extractor = {".txt": parser} 
    documents = SimpleDirectoryReader(
        DATA_DIR, file_extractor=file_extractor
    ).load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)


if not args.query:
    query = "What are the requirements for data protection in ireland?"
else:
    query = args.query
query_engine = index.as_query_engine()
response = query_engine.query(query)
print(response)    
