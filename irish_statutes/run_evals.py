
from dataclasses import asdict
import pickle
from llama_index.core import Settings


from vstore import get_index_from_database
from eval_queries import setup_llm, query_llm, QUERIES, setup_embedding

Settings.llm = setup_llm(temperature=0.1)
Settings.embed_model = setup_embedding()
index = get_index_from_database()
results = {}
total_cnt = 0
for query in QUERIES:
    param_cnt = 0
    for top_k in range(2, 10):
        print(f"{top_k=}")
        response = query_llm(index, query, top_k=top_k)
        results[query] = asdict(response)
    total_cnt+= 1
    print(f'{total_cnt=}')
with open('query_eval_top_k_2_10.pkl', 'wb') as f:
    pickle.dump(results, f)
