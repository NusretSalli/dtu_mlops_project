# Change from latest to a specific version if your requirements.txt
FROM python:3.12-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY models models/
COPY api_default_data api_default_data/
COPY requirements_api.txt requirements_api.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

# Copy the service account key (ensure this is temporarily added and managed securely)
COPY best-mlops-project-de71c25e08be.json key.json
# Set environment variables for Google Cloud authentication
ENV GOOGLE_APPLICATION_CREDENTIALS="key.json"

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_api.txt

EXPOSE 8000

ENTRYPOINT ["uvicorn", "src.exam_project.api:app", "--host", "0.0.0.0", "--port", "8000"]
