import os
from pathlib import Path
import re
import sys
from time import sleep

import pandas as pd

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
ui.input_select(  
    "select",  
    "Is this a good answer?:",  
    {"1A": "Yes", "1B": "No"},  
)

# @reactive.effect
# def _():
#     print(input.query())

@render.text
def query_txt():
    sleep(1)
    return f"query is {input.query()}"


results_dict = {}
@render.ui
def answer_query():
    sleep(2)
    query_res = query_llm(index, input.query(), top_k=5)
    
    resp = query_res.response
    source_nodes = query_res.source_nodes
    print(len(source_nodes))
    node_text = [node.text for node in source_nodes]
    scores = query_res.scores
    scores_text = "Score2:".join(str(s) for s in scores)
    node_text_full = "\n\n".join(text for text in node_text)
    # top_k = query_res.top_k
    results_dict[query] = [resp, source_nodes, scores]
    return (ui.h1('Answer'), ui.markdown(resp),
            ui.h1('Scores'), ui.markdown(scores_text),
            ui.h1('Sources'), ui.markdown(node_text_full)
            )

result_df = None
if not result_df:
    result_df = pd.DataFrame.from_dict(results_dict)
else:
    new_results = pd.DataFrame.from_dict(results_dict)
    result_df = pd.concat([result_df, new_results])

print(f"{result_df.head()=}")
