"""Entry point — launch the API with uvicorn.

Usage:
    uv run python -m            # from apps/api/
    uv run uvicorn app:app
"""

from __future__ import annotations

import uvicorn  # pyright: ignore[reportMissingImports]


def main() -> None:
    """Run the uvicorn server."""
    uvicorn.run("app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

