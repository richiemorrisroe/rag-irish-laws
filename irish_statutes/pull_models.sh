#!/bin/bash
# Wait for Ollama to be ready
until curl -s http://ollama:11434 > /dev/null; do
    echo "Waiting for Ollama to be ready..."
    sleep 2
done

# Pull your models
curl http://ollama:11434/api/pull -d '{"name":"llama3"}'
# Add more models as needed
