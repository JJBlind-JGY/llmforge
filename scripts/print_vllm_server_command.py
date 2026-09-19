#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from llmforge.serving.vllm import VLLMServerConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--hf-home", required=True)
    args = p.parse_args()
    config = VLLMServerConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    print(config.shell_command(gpu=args.gpu, hf_home=args.hf_home))


if __name__ == "__main__":
    main()
