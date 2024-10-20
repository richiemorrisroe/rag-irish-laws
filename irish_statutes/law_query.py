import argparse
import fnmatch
import glob
import logging
import os
import sys

from pprint import pprint

from IPython.display import display

from llama_index.core import (SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext,
                              load_index_from_storage)
from llama_index.readers.file import FlatReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

parser = argparse.ArgumentParser()

parser.add_argument("--query")
parser.add_argument("--data_dir")

args = parser.parse_args()

# DATA_DIR = './laws_test'


logger = logging.getLogger()

logging.basicConfig(filename='law_query.log',
                    encoding='utf-8', level=logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.WARNING)
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

def display_prompt_dict(prompts_dict):
    for k, p in prompts_dict.items():
        text_md = f"**Prompt Key**: {k}<br>" f"**Text:** <br>"
        pprint(text_md)
        print(p.get_template())
        # display(Markdown("<br><br>"))

        


if not args.data_dir:
    PERSIST_DIR = "./full_storage"
else:
    PERSIST_DIR = args.data_dir
    
if not os.path.exists(PERSIST_DIR):
    
    logging.warning("you should run `law_index.py` to generate an index first")
    raise ValueError("please run law_index.py first")
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)


if not args.query:
    query = "What are the requirements for data protection in ireland?"
else:
    query = args.query
query_engine = index.as_query_engine(similarity_top_k=10)
prompts_dict = query_engine.get_prompts()
display_prompt_dict(prompts_dict)
response = query_engine.query(query)
logger.info(f"{response.source_nodes=}")
print(response)
