import argparse
import fnmatch
import glob
import logging
import os
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


def get_files_from_directory(directory):
    res = glob.glob(directory + '/*')
    return res


def indexing(input_files):
    documents = SimpleDirectoryReader(
        input_files=input_files,
        recursive=True,
        filename_as_id=True,
    ).load_data()


def batch_files(directory, batch_size=None, included_exts=None):
    if included_exts is None:
        included_exts = ['.txt']
    if batch_size is None:
        batch_size = 10

    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(fnmatch.fnmatch(file, "*" + ext) for ext in included_exts):
                all_files.append(os.path.join(root, file))

    batches = [
        all_files[i: i + batch_size] for i in range(0, len(all_files), batch_size)
    ]

    return batches, len(all_files)


files = get_files_from_directory(DATA_DIR)

all_batches, total_count = batch_files(DATA_DIR, 10)
logger.warning(f"{all_batches=}, {total_count=}")

PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    
    logging.warning("you should run `law_index.py` to generate an index first")
    # parser = FlatReader()
    # file_extractor = {".txt": parser}
    # # documents = SimpleDirectoryReader(
    # #     DATA_DIR, file_extractor=file_extractor
    # # ).load_data()
    # documents = indexing(files)
    # index = VectorStoreIndex.from_documents(documents)
    # # store it for later
    # index.storage_context.persist(persist_dir=PERSIST_DIR)
    raise ValueError("please run law_index.py first")
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
