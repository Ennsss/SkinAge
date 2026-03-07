"""
CLI entry point for the SkinAge API server.

Usage::

    python -m scripts.serve
    python -m scripts.serve --port 8080 --workers 2
    python -m scripts.serve --config config/api_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on the Python path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]  # SkinAge/
sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> None:
    """Parse arguments and start the uvicorn server."""
    parser = argparse.ArgumentParser(
        description="Start the SkinAge API server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_PROJECT_ROOT / "config" / "api_config.yaml"),
        help="Path to api_config.yaml",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (use 1 for development)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Import here so logging is configured first
    import uvicorn

    from src.api.app import create_app

    # Create the app with the specified config
    app = create_app(config_path=args.config)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
