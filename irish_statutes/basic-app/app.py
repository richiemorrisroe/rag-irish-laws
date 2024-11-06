from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from indexer.eval_queries import query_llm, setup_llm
from indexer.vstore import get_index_from_database

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

Settings.llm = setup_llm()

index = get_index_from_database()

from shiny import render, ui, reactive
from shiny.express import input


query = None
ui.panel_title("Lawbot")
ui.input_text("query", "Enter a query")

response = index.query(query)

@reactive.effect
def _():
    print(input.query())

@render.ui
def txt():
    return f"query is {input.query()}; response is {response}"
