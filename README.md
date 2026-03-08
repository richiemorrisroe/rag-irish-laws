# Introduction

The is a simple RAG and agentic app designed to answer questions
about Irish legislation, from [here](https://www.irishstatutebook.ie/).

It started off as an attempt to use local LLMs (llama3, gemma etc),
but I ended up switching to Claude given how irritating and annoying 
it is to run ollama, postgres and an app on any rented hardware. 


The goal here is to more deeply understand tradeoffs around RaG and agentic search
and investigate how I can ship better apps with this functionality. 

# Setup

You'll need Python (I'm using 3.12), and uv. 

Just run `uv sync` and it all should just work. 

However, you'll additionally need a postgres DB with pgvector. 

I've been using `pgvector/pgvector:pg17` for my tests, so you probably want
to use that also. 

The `docker-compose.yml` file exists and pulls the appropriate image,
but in general, containerisation is a work in progress. 


# Details

Most of the important code is in the `indexer` subdirectory, the 
HTML files are pulled from a separate scrapy app.

The `db` file contains the logic for interacting with the db
and assumes that postgres is available as per the container. 

The `parse_statute` file handles turning the HTML into smaller portions
(sections, subsections, paragraphs etc)

The `ingest` file just wraps the `parse_statute` code and 
sends the resulting data to the DB.

The `claude_agent` file defines the agent loop and sets up
the functions to act as tools to the agent.

If you look closely, you may note that there's no actual similarity 
search here. This is a transient phase, I plan to create embeddings
at various levels to aid in the searching process.

The trouble with the current approach (just text search attached to an LLM)
is that it generates far too much context, and often overflows the max
window, resulting in no results (as well as taking a long time). 


If you'd like an easier way to search over a set of questions
then run the shiny app as follows. 

`shiny run --reload evals-app/app.py`.

Here you should be able to ask questions and rate the answers. 

