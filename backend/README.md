# Open Voice Backend

Open Voice backend is a FastAPI application that handles WebRTC connections and provides real-time voice AI capabilities.

## Requirements

- Python 3.14+
- [uv](https://docs.astral.sh/uv/)
- API Keys for various services (see `src/env.py`)

## Dev Server

Run the dev server (with reload on file changes) on port 8000:

```bash
make dev
```

## Type checking

```bash
make type-check
```