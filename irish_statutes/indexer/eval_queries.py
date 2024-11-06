from dataclasses import dataclass
from typing import Any

from llama_index.llms.ollama import Ollama

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

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
           """May a joint residential tenancy be terminated if one of
           the parties leaves the property?""",
           """What are the conditions required for irish citizenship?""",
           """What are the conditions required for a
           claim of contructive dismissal to succeed?""",
           """How is land ownership determined?""",
           """What happens if someone dies without making a will?""",
           """What is intestate succession in the context of inheritance?""",
           """What is intestate succession?""",
           """What is the minimum wage for full time employees?""",
           """What is the national minimum hourly rate of pay?""",
           """Is it legal to drive without insurance?"""
           """What are the penalities for driving without insurance?""",
           """Can i be prosecuted for allowing my friend to drive a vehicle
           on which they are not insured?""",
           """What are the requirements for setting up a limited
           company in Ireland?""",
           """What are the differences between a private limited company
           and a private unlimited company?""",
           """Does a private limited company need to file accounts,
           and if so, how often?""",
           """Are individuals required to have health insurance?""",
           """Under what conditions can adults in Ireland be charged different
           prices for health insurance?"""
           """What are special categories of protected data?""",
           """What is the purpose of data protection law?"""
           ]

@dataclass
class QueryResponse():
    query: str
    response: str
    top_k: int
    scores :list[float]
    source_nodes: Any

    def __repr__(self):
        return str(self.response)


def setup_llm(temperature=0.5, timeout_secs=90):
    llm = Ollama(model="llama3", temperature=temperature, request_timeout=timeout_secs)
    return llm

def setup_embedding():
    embedding = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    return embedding


def query_llm(index, query, top_k=2):
    query_engine = index.as_query_engine(top_k=top_k)
    response = query_engine.query(query)
    scores = [x.score for x in response.source_nodes]
    source_nodes = response.source_nodes
    return QueryResponse(query=query,
                         response=response.response,
                         top_k=top_k,
                         scores=scores,
                         source_nodes=source_nodes)
