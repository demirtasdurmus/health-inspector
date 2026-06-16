# Health Inspector

## Description

Health Inspector is a tool that helps you inspect the health of your food and environment.

## Installation And Ingestion(One Time)

```bash
uv sync
cp .env.example .env # Fill the .env file with your OpenAI API key
uv run ingest.py # Ingest the regulations
```

## Run the API

```bash
uv run uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Check the docs for more [information](docs/HEALTH_INSPECTOR.md).
