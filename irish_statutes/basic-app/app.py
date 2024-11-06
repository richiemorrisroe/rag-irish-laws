import os
from pathlib import Path
import re
import sys
from time import sleep


from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
root_dir = Path(__file__).parent
print(root_dir)
sys.path.append(str(root_dir))

from indexer.eval_queries import query_llm, setup_llm
from indexer.vstore import get_index_from_database

from shiny import render, ui, reactive
from shiny.express import input

chat = ui.Chat(id='chat')


Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

Settings.llm = setup_llm()

index = get_index_from_database()




query = """Does a private limited company need to file accounts,
           and if so, how often?"""
ui.panel_title("Lawbot")
ui.input_text("query", "Enter a query")



# @reactive.effect
# def _():
#     print(input.query())

@render.text
def query_txt():
    sleep(1)
    return f"query is {input.query()}"


@render.ui
def answer_query():
    sleep(2)
    query_res = query_llm(index, input.query())
    resp = query_res.response
    source_nodes = query_res.source_nodes
    node_text = [node.text for node in source_nodes]
    scores = query_res.scores
    node_text_full = """\n\n** SECOND_DOCUMENT\n\n""".join(text for text in node_text)
    # top_k = query_res.top_k
    return ui.h1('Answer'), ui.markdown(resp), ui.h1('Sources'), ui.markdown(node_text_full),

# answer = answer_query()

# @render.ui
# def answer(answer):
#     resp, _, _ = answer_query()
#     return ui.markdown(resp)

# @render.ui
# def sources(answer):
#     _, sources, _ = answer_query()
#     sources = [re.split(r'\s{2,}', string=x.text) for x in sources]
#     print(sources)
#     return ui.markdown(sources[0][0])
