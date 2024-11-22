#!/bin/bash
set -e

# Initialize the model in the background
/init-model.sh &

# Start Ollama server
exec ollama serve
