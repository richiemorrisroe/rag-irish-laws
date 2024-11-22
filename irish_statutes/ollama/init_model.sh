#!/bin/bash
set -e

# Pull the model you're using
ollama pull llama3

# Optionally warm up the model with a simple query
echo "Warming up model..."
curl -X POST http://localhost:11434/api/chat -d '{
    "model": "llama3",
    "messages": [{"role": "user", "content": "hello"}]
}'
