# Change from latest to a specific version if your requirements.txt
FROM python:3.12-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY models models/
COPY requirements_api.txt requirements_api.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_api.txt

ENTRYPOINT ["uvicorn", "src.exam_project.api:app", "--host", "0.0.0.0", "--port", "8000"]
