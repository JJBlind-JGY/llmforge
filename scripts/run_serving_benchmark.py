#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from llmforge.benchmark.guard import validate_formal_benchmark
from llmforge.environment import collect_environment
from llmforge.serving.client import ClientConfig, VLLMOpenAIClient
from llmforge.serving.runner import run_workload
from llmforge.serving.workloads import (
    burst_workload,
    fixed_workload,
    mixed_workload,
    poisson_workload,
    shared_prefix_workload,
)


def parse_shapes(raw: str) -> list[tuple[int, int]]:
    return [tuple(map(int, x.split(":"))) for x in raw.split(",")]


def build_workload(args):
    common = {
        "seed": args.seed,
        "vocab_size": args.vocab_size,
        "token_low": args.token_low,
        "token_high": args.token_high,
    }
    if args.workload == "fixed":
        return fixed_workload(
            num_requests=args.num_requests,
            prompt_tokens=args.prompt_tokens,
            output_tokens=args.output_tokens,
            request_prefix=args.case_name,
            **common,
        )
    if args.workload == "mixed":
        return mixed_workload(
            num_requests=args.num_requests,
            shapes=parse_shapes(args.shapes),
            request_prefix=args.case_name,
            **common,
        )
    if args.workload == "shared_prefix":
        return shared_prefix_workload(
            num_requests=args.num_requests,
            prompt_tokens=args.prompt_tokens,
            output_tokens=args.output_tokens,
            shared_prefix_tokens=args.shared_prefix_tokens,
            request_prefix=args.case_name,
            **common,
        )
    if args.workload == "poisson":
        return poisson_workload(
            num_requests=args.num_requests,
            prompt_tokens=args.prompt_tokens,
            output_tokens=args.output_tokens,
            request_rate=args.request_rate,
            request_prefix=args.case_name,
            **common,
        )
    return burst_workload(
        num_bursts=args.num_bursts,
        burst_size=args.burst_size,
        prompt_tokens=args.prompt_tokens,
        output_tokens=args.output_tokens,
        inter_burst_s=args.inter_burst_s,
        request_prefix=args.case_name,
        **common,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case-name", required=True)
    p.add_argument(
        "--workload",
        required=True,
        choices=["fixed", "mixed", "shared_prefix", "poisson", "burst"],
    )
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="qwen3-8b")
    p.add_argument("--endpoint", default="/v1/completions")
    p.add_argument("--timeout-s", type=float, default=180.0)
    p.add_argument("--num-requests", type=int, default=20)
    p.add_argument("--max-in-flight", type=int, default=1)
    p.add_argument("--prompt-tokens", type=int, default=512)
    p.add_argument("--output-tokens", type=int, default=32)
    p.add_argument("--shapes", default="128:32,512:16,2048:8,4096:4")
    p.add_argument("--shared-prefix-tokens", type=int, default=0)
    p.add_argument("--request-rate", type=float, default=4.0)
    p.add_argument("--num-bursts", type=int, default=4)
    p.add_argument("--burst-size", type=int, default=8)
    p.add_argument("--inter-burst-s", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vocab-size", type=int, default=151936)
    p.add_argument("--token-low", type=int, default=1000)
    p.add_argument("--token-high", type=int, default=10000)
    p.add_argument("--ttft-slo-ms", type=float)
    p.add_argument("--tpot-slo-ms", type=float)
    p.add_argument("--e2e-slo-ms", type=float)
    p.add_argument(
        "--output-root", type=Path, default=Path("artifacts/benchmarks/serving")
    )
    args = p.parse_args()

    env = collect_environment(role="gpu-server")
    validate_formal_benchmark(env)
    requests = build_workload(args)

    result = run_workload(
        client=VLLMOpenAIClient(
            ClientConfig(
                base_url=args.base_url,
                model=args.model,
                endpoint=args.endpoint,
                timeout_s=args.timeout_s,
            )
        ),
        requests=requests,
        max_in_flight=args.max_in_flight,
        ttft_slo_ms=args.ttft_slo_ms,
        tpot_slo_ms=args.tpot_slo_ms,
        e2e_slo_ms=args.e2e_slo_ms,
    )

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.output_root / args.case_name / run_id / "result.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "schema_version": 1,
        "case_name": args.case_name,
        "workload": {
            "kind": args.workload,
            "num_requests": len(requests),
            "max_in_flight": args.max_in_flight,
            "prompt_tokens": args.prompt_tokens,
            "output_tokens": args.output_tokens,
            "shapes": args.shapes,
            "shared_prefix_tokens": args.shared_prefix_tokens,
            "request_rate": args.request_rate,
            "seed": args.seed,
        },
        "client": {
            "base_url": args.base_url,
            "model": args.model,
            "endpoint": args.endpoint,
        },
        "environment": env,
        "summary": result.summary.to_dict(),
        "requests": [r.to_dict() for r in result.results],
    }
    out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact["summary"], indent=2))
    print(f"Result written to: {out}")


if __name__ == "__main__":
    main()
