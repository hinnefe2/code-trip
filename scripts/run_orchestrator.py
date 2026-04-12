"""Manual end-to-end runner for the code-trip orchestrator."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

from code_trip.config import load_config
from code_trip.orchestrator import Orchestrator


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the code-trip voice loop.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.toml"),
        help="Path to TOML config file (default: ./config.toml)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    orchestrator = Orchestrator(config)

    def _handle_signal(signum, frame):  # noqa: ARG001
        print("\nShutting down...", file=sys.stderr)
        orchestrator.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print(f"Orchestrator ready. Hold {config.audio.hotkey.upper()} to talk. Ctrl-C to exit.")
    orchestrator.start()
    return 0


if __name__ == "__main__":
    sys.exit(main())
