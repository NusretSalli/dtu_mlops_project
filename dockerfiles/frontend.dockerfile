# Change from latest to a specific version if your requirements.txt
FROM python:3.12-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*


COPY requirements_frontend requirements_frontend
COPY pyproject.toml pyproject.toml

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_frontend.txt

EXPOSE $PORT

CMD ["streamlit", "run", "frontend.py", "--server.port", "$PORT"]
