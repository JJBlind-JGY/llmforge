#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--suite", type=Path, default=Path("configs/serving/m3_suite.json"))
    p.add_argument("--group", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = json.loads(args.suite.read_text(encoding="utf-8"))
    if args.group not in cfg["groups"]:
        raise SystemExit(f"unknown group: {args.group}")
    defaults = cfg.get("defaults", {})

    flag_map = {
        "prompt_tokens": "--prompt-tokens",
        "output_tokens": "--output-tokens",
        "shapes": "--shapes",
        "shared_prefix_tokens": "--shared-prefix-tokens",
        "request_rate": "--request-rate",
        "num_bursts": "--num-bursts",
        "burst_size": "--burst-size",
        "inter_burst_s": "--inter-burst-s",
        "ttft_slo_ms": "--ttft-slo-ms",
        "tpot_slo_ms": "--tpot-slo-ms",
        "e2e_slo_ms": "--e2e-slo-ms",
    }

    for case in cfg["groups"][args.group]:
        cmd = [
            sys.executable,
            "scripts/run_serving_benchmark.py",
            "--case-name",
            case["name"],
            "--workload",
            case["workload"],
            "--base-url",
            case.get("base_url", defaults["base_url"]),
            "--model",
            case.get("model", defaults["model"]),
            "--num-requests",
            str(case.get("num_requests", defaults["num_requests"])),
            "--max-in-flight",
            str(case["max_in_flight"]),
            "--seed",
            str(case.get("seed", defaults["seed"])),
        ]
        for key, flag in flag_map.items():
            if key in case:
                cmd += [flag, str(case[key])]
        print("$ " + " ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
