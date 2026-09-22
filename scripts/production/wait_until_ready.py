#!/usr/bin/env python3
"""Wait for a health/readiness endpoint before starting a benchmark."""

from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--url",
        default=("http://127.0.0.1:9108/health/ready"),
    )

    parser.add_argument(
        "--timeout-s",
        type=float,
        default=120.0,
    )

    parser.add_argument(
        "--interval-s",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    deadline = time.monotonic() + args.timeout_s

    last_error = "not checked"

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                args.url,
                timeout=min(
                    args.interval_s,
                    5.0,
                ),
            ) as response:
                if 200 <= response.status < 300:
                    print(f"Ready: {args.url}")
                    return

                last_error = f"HTTP {response.status}"

        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
        ) as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        time.sleep(args.interval_s)

    raise SystemExit(
        "Service did not become ready "
        f"within {args.timeout_s}s; "
        f"last_error={last_error}"
    )


if __name__ == "__main__":
    main()
