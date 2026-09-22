#!/usr/bin/env python3
"""Run deterministic chat-completion probes against one server arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def sha256_text(
    text: str,
) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_cases(
    path: Path,
) -> list[dict[str, Any]]:
    cases = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            line = line.strip()

            if line:
                cases.append(json.loads(line))

    return cases


def post_json(
    *,
    url: str,
    payload: dict,
    timeout_s: float,
) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": ("application/json")},
        method="POST",
    )

    with urllib.request.urlopen(
        request,
        timeout=timeout_s,
    ) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-url",
        default=("http://127.0.0.1:8000"),
    )

    parser.add_argument(
        "--model",
        default="qwen3-8b",
    )

    parser.add_argument(
        "--workload",
        type=Path,
        default=Path("configs/optimization/m7_correctness_cases.jsonl"),
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--timeout-s",
        type=float,
        default=120.0,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    cases = load_cases(args.workload)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for case in cases:
            case_id = str(case["case_id"])

            messages = case["messages"]

            prompt_material = json.dumps(
                messages,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )

            record = {
                "case_id": case_id,
                "prompt_sha256": (sha256_text(prompt_material)),
                "response_sha256": "",
                "prompt_tokens": None,
                "completion_tokens": None,
                "finish_reason": None,
                "success": False,
                "error": None,
            }

            payload = {
                "model": args.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": (args.max_tokens),
                "seed": args.seed,
                "stream": False,
            }

            try:
                response = post_json(
                    url=(args.base_url.rstrip("/") + "/v1/chat/completions"),
                    payload=payload,
                    timeout_s=(args.timeout_s),
                )

                choice = response["choices"][0]

                content = choice["message"].get("content") or ""

                usage = response.get(
                    "usage",
                    {},
                )

                record.update(
                    {
                        "response_sha256": (sha256_text(content)),
                        "prompt_tokens": (usage.get("prompt_tokens")),
                        "completion_tokens": (usage.get("completion_tokens")),
                        "finish_reason": (choice.get("finish_reason")),
                        "success": True,
                    }
                )

            except (
                urllib.error.URLError,
                TimeoutError,
                OSError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
            ) as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"

            handle.write(
                json.dumps(
                    record,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            )

            handle.flush()

    print(f"Correctness artifact written to: {args.output}")


if __name__ == "__main__":
    main()
