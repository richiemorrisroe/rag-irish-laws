FROM ollama/ollama

# Install curl and wget
RUN apt-get update && \
    apt-get install -y curl wget && \
    rm -rf /var/lib/apt/lists/*

# Copy a script to pull and initialize the model
COPY init_model.sh /init_model.sh
RUN chmod +x /init_model.sh

# Use an entrypoint script to ensure the model is loaded
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
