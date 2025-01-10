import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "exam_project"
PYTHON_VERSION = "3.12"

# Setup commands
@task
def createenvironment(ctx: Context) -> None:
    """Create a new conda environment for project."""
    ctx.run(
        f"conda create --name {PROJECT_NAME} python={PYTHON_VERSION} pip --no-default-packages --yes",
        echo=True,
        pty=not WINDOWS,
    )

@task
def requirements(ctx: Context, installtype: str = "pip") -> None:
    if installtype == "pip":
        ctx.run("pip install -U pip setuptools wheel", echo=True, pty=not WINDOWS)
        ctx.run("pip install -r requirements.txt", echo=True, pty=not WINDOWS)
        ctx.run("pip install -e .", echo=True, pty=not WINDOWS)
    if installtype == "uv":
        ctx.run("uv pip install -U pip setuptools wheel", echo=True, pty=not WINDOWS)
        ctx.run("uv pip install -r requirements.txt", echo=True, pty=not WINDOWS)
        ctx.run("uv pip install -e .", echo=True, pty=not WINDOWS)
    


@task(requirements)
def devrequirements(ctx: Context) -> None:
    """Install development requirements."""
    ctx.run('pip install -e .["dev"]', echo=True, pty=not WINDOWS)

# Project commands
@task
def preprocessdata(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"python src/{PROJECT_NAME}/data.py data/raw data/processed", echo=True, pty=not WINDOWS)

@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"python src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)

@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("coverage report -m", echo=True, pty=not WINDOWS)

@task
def dockerbuild(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )

# Documentation commands
@task(devrequirements)
def builddocs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task(devrequirements)
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)


# formatting and linting task code
@task
def lint(ctx) -> None:
    """Run linters."""
    ctx.run("ruff check . --fix", echo=True, pty=not WINDOWS, warn=True)
    ctx.run("ruff format .", echo=True, pty=not WINDOWS, warn=True)
    ctx.run("mypy .", echo=True, pty=not WINDOWS)


# git commands task code
@task
def gitupload(ctx, message):
    ctx.run("git add .")
    ctx.run(f'git commit -m "{message}"')
    ctx.run("git push")
