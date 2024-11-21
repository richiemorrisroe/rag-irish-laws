# Use the official conda image as the base
FROM continuumio/miniconda3

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY environment.yml requirements.txt ./

# Create conda environment without pip packages first
RUN conda env create -f environment.yml

# Activate environment and install pip requirements separately
RUN conda run -n irish_statutes pip install -r requirements.txt

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "irish_statutes", "/bin/bash", "-c"]

# Copy application code
COPY . .

# Install the local package in development mode
RUN conda run -n irish_statutes pip install -e .

# COPY pull_models.sh /app/
# RUN chmod +x /app/pull_models.sh
# Expose the port that Shiny will run on
EXPOSE 8050

# The code to run when container is started:
ENTRYPOINT [ "/bin/bash", "-c", "/app/pull_models.sh", "&&", "conda", "run", "-n", "irish_statutes", "python", "-m", "shiny", "run", "/app/evals-app/app.py", "--host", "0.0.0.0", "--port", "8050"]
