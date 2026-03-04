# Deployment

The dashboard is a FastAPI web app that serves both the backend APIs and the static frontend.

## Recommended: Render

1. Push the repository to GitHub.
2. Create a new Render Web Service from the repository.
3. Render can use `render.yaml` automatically.
4. Confirm the service settings:
   - Build command: `pip install --upgrade pip && pip install .`
   - Start command: `uvicorn enterprise_rag_eval.server:app --host 0.0.0.0 --port $PORT`
   - Health check path: `/api/health`
5. Deploy and share the generated Render URL.

## Docker

```bash
docker build -t enterprise-rag-eval-harness .
docker run --rm -p 8000:8000 enterprise-rag-eval-harness
```

Open:

```text
http://127.0.0.1:8000
```

## Notes

- No API keys are required for the default local retrieval and evaluation flow.
- Uploaded files are written to `data/raw/<domain>/` inside the running service.
- On ephemeral hosts, uploaded files may disappear after redeploys unless persistent storage is
  configured.
- The included benchmark corpus is synthetic and intended for retrieval and evaluation testing.
