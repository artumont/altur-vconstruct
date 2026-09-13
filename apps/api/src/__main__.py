"""Entry point — launch the API with uvicorn.

Usage:
    uv run python -m            # from apps/api/
    uv run uvicorn app:app
"""

from __future__ import annotations

import os

import uvicorn  # pyright: ignore[reportMissingImports]


def main() -> None:
    """Run the uvicorn server.

    Raises:
        ValueError: If API_PORT is not an integer.
    """
    host = os.environ.get("API_HOST", "0.0.0.0")
    try:
        port = int(os.environ.get("API_PORT", "8000"))
    except ValueError as exc:
        raise ValueError("API_PORT must be an integer") from exc
    uvicorn.run("app:app", host=host, port=port)


if __name__ == "__main__":
    main()
