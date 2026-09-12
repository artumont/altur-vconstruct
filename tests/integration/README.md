# Integration tests

Black-box HTTP tests for the `POST /detect` API. No app imports, no mocks —
they hit a **running** server over real HTTP (typically the Docker deployment).

## Prerequisites

API reachable over HTTP:

```bash
# from repo root
docker compose up --build -d
```

## Run

```bash
# from repo root
uv run --directory tests/integration pytest -v

# from tests/integration/
uv run pytest -v
```

## Configuration

| Env var        | Default                 | Description                        |
| -------------- | ----------------------- | ---------------------------------- |
| `API_URL`      | `http://localhost:8000` | Base URL of the running API        |
| `TEST_TIMEOUT` | `30`                    | Per-request timeout in seconds     |

```bash
API_URL=http://my-host:8000 uv run --directory tests/integration pytest -v
```

## Coverage

| Class                | What it checks                                              |
| -------------------- | ----------------------------------------------------------- |
| `TestHealth`         | `GET /health` returns 200 with `status`/`device`/`ready`    |
| `TestDetectHappyPath`| Valid stereo/mono WAV → 200, correct field types, 0–1 range |
| `TestDetectErrors`   | 422 on bad payloads, 400 on bad base64/WAV, 404/405 routing |
| `TestContract`       | Exact response keys `{is_synthetic, confidence}`            |

## Notes

- Tests are read-only; they do not modify server state.
- Require a working model bundle — the API returns `503` on `/detect` if
  checkpoints are missing, which will fail the happy-path tests.
