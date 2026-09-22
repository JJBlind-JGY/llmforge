#!/usr/bin/env python3
"""Run any service/benchmark command with logs and a reproducibility manifest."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

from llmforge.production.lifecycle import (
    GracefulShutdown,
)
from llmforge.production.manifest import (
    RunCompletion,
    build_run_manifest,
    utc_now,
    write_json,
)


def parse_env(
    values: list[str],
) -> dict[str, str]:
    parsed = {}

    for item in values:
        if "=" not in item:
            raise ValueError("--env requires NAME=VALUE.")

        name, value = item.split(
            "=",
            1,
        )

        if not name:
            raise ValueError("Environment variable name must not be empty.")

        parsed[name] = value

    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/runs"),
    )

    parser.add_argument(
        "--config",
        type=Path,
        help=("Optional JSON config copied into the run manifest."),
    )

    parser.add_argument(
        "--timeout-s",
        type=float,
    )

    parser.add_argument(
        "--grace-s",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help=("Environment override NAME=VALUE; may be repeated."),
    )

    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    command = list(args.command)

    if command and command[0] == "--":
        command = command[1:]

    if not command:
        raise SystemExit("A command is required after --.")

    if args.timeout_s is not None and args.timeout_s <= 0:
        raise ValueError("timeout_s must be positive.")

    if args.grace_s <= 0:
        raise ValueError("grace_s must be positive.")

    environment = os.environ.copy()
    environment.update(parse_env(args.env))

    config_payload = None

    if args.config is not None:
        config_payload = json.loads(args.config.read_text(encoding="utf-8"))

    run_id = time.strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]

    run_dir = args.output_root / run_id

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    log_path = run_dir / "process.log"

    manifest = build_run_manifest(
        run_id=run_id,
        command=command,
        cwd=Path.cwd(),
        config=config_payload,
        environment=environment,
    )

    write_json(
        run_dir / "manifest.json",
        manifest.to_dict(),
    )

    shutdown = GracefulShutdown()
    shutdown.install_signal_handlers()

    timed_out = False
    interrupted = False
    started = time.monotonic()

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as log_handle:
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )

        try:
            while process.poll() is None:
                if shutdown.requested:
                    interrupted = True
                    process.terminate()
                    break

                if (
                    args.timeout_s is not None
                    and (time.monotonic() - started) >= args.timeout_s
                ):
                    timed_out = True
                    process.terminate()
                    break

                time.sleep(0.2)

            if process.poll() is None:
                try:
                    process.wait(timeout=args.grace_s)
                except subprocess.TimeoutExpired:
                    process.kill()

            return_code = process.wait()

        finally:
            completion = RunCompletion(
                run_id=run_id,
                finished_at=utc_now(),
                return_code=(process.returncode),
                timed_out=timed_out,
                interrupted=interrupted,
                log_path=str(log_path),
            )

            write_json(
                run_dir / "completion.json",
                completion.to_dict(),
            )

    print(f"Run directory: {run_dir}")

    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
