"""
CLI entry point for the SkinAge Streamlit dashboard.

Usage::

    python -m scripts.dashboard
    python -m scripts.dashboard --port 8501
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]  # SkinAge/
_DASHBOARD_APP = _PROJECT_ROOT / "src" / "dashboard" / "app.py"


def main() -> None:
    """Parse arguments and launch the Streamlit dashboard."""
    parser = argparse.ArgumentParser(
        description="Start the SkinAge Streamlit dashboard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for the Streamlit server",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the SkinAge API server",
    )

    args = parser.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(_DASHBOARD_APP),
        "--server.port",
        str(args.port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]

    print(f"Starting SkinAge Dashboard on port {args.port}...")
    print(f"API URL: {args.api_url}")
    print(f"Dashboard: http://localhost:{args.port}")

    subprocess.run(cmd, cwd=str(_PROJECT_ROOT))


if __name__ == "__main__":
    main()
