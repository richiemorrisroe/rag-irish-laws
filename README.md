# Introduction

This is a simple test of RAG (Retrieval Augmented Generation) and
using LLMs to do question answering.


The goal here is to more deeply understand tradeoffs around RaG
and investigate how I can ship apps with this functionality. 

# Setup

You'll need Python (I'm using 3.12), and conda. 

If you already have conda, then running `conda env create -f environment.yml` should
get you up and running. 

However, you'll additionally need a postgres DB with pgvector. 

I've been using `pgvector/pgvector:pg17` for my tests, so you probably want
to use that also. 

This is encapsulated in the `docker-compose.yml` file (to which I need
to add the evaluation app). For now, this file just gets pgvector
and sets it running with a datadirectory of `pgdata` wherever you ran this command. 

# Details

Most of the important code is in the `indexer` subdirectory. 

If you just want to test out the approach, then you can get pretty far
by firstly running `law_index.py` (follow the help instructions
to load all the data into a PG database.

You can then run `law_query.py` with a query argument, which
will then provide a response to your query (often quite poor). 

If you'd like an easier way to search over a set of questions
then run the shiny app as follows. 

`shiny run --reload evals-app/app.py`.

Note that this doesn't allow you to select a question, as I
just built it to run evaluations so I can measure performance
of the overall system. 

