#!/usr/bin/env python3
"""NCCL collective benchmark for LLMForge M5.

Run through torchrun, for example:

python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/distributed/benchmark_collective.py \
  --collective all_reduce \
  --message-mib 64 \
  --iterations 50 \
  --output artifacts/distributed/nccl/all_reduce_64mib.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--collective",
        choices=[
            "all_reduce",
            "all_gather",
            "reduce_scatter",
        ],
        required=True,
    )

    parser.add_argument(
        "--message-mib",
        type=int,
        required=True,
        help=(
            "NCCL-tests style payload S in MiB. "
            "AllReduce: tensor size. "
            "AllGather: gathered output size. "
            "ReduceScatter: full input size."
        ),
    )

    parser.add_argument(
        "--dtype",
        choices=[
            "float32",
            "bfloat16",
        ],
        default="float32",
    )

    parser.add_argument(
        "--warmups",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    if args.message_mib <= 0:
        raise ValueError("message_mib must be positive.")

    if args.warmups < 0:
        raise ValueError("warmups must be non-negative.")

    if args.iterations <= 0:
        raise ValueError("iterations must be positive.")

    import torch
    import torch.distributed as dist

    from llmforge.distributed.nccl.bandwidth import (
        calculate_collective_bandwidth,
    )
    from llmforge.distributed.nccl.schema import (
        CollectiveKind,
        CollectiveRun,
    )
    from llmforge.distributed.nccl.stats import (
        summarize_latency,
    )

    dist.init_process_group(
        backend="nccl",
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    device = torch.device(
        "cuda",
        local_rank,
    )

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    bytes_per_element = torch.tensor(
        [],
        dtype=dtype,
    ).element_size()

    payload_bytes = args.message_mib * 1024 * 1024

    if payload_bytes % bytes_per_element:
        raise ValueError("message size must be divisible by dtype element size.")

    payload_elements = payload_bytes // bytes_per_element

    collective = CollectiveKind(args.collective)

    if collective == CollectiveKind.ALL_REDUCE:
        per_rank_input_elements = payload_elements
        per_rank_output_elements = payload_elements

    elif collective == CollectiveKind.ALL_GATHER:
        if payload_elements % world_size:
            raise ValueError("AllGather payload must divide world_size.")

        per_rank_input_elements = payload_elements // world_size

        per_rank_output_elements = payload_elements

    else:
        if payload_elements % world_size:
            raise ValueError("ReduceScatter payload must divide world_size.")

        per_rank_input_elements = payload_elements

        per_rank_output_elements = payload_elements // world_size

    input_tensor = torch.empty(
        per_rank_input_elements,
        dtype=dtype,
        device=device,
    )

    if collective == CollectiveKind.ALL_REDUCE:
        output_tensor = input_tensor

    elif collective == CollectiveKind.ALL_GATHER:
        output_tensor = torch.empty(
            per_rank_output_elements,
            dtype=dtype,
            device=device,
        )

    else:
        output_tensor = torch.empty(
            per_rank_output_elements,
            dtype=dtype,
            device=device,
        )

    expected_sum = world_size * (world_size + 1) / 2

    def reset_input() -> None:
        input_tensor.fill_(float(rank + 1))

    def run_collective() -> None:
        if collective == CollectiveKind.ALL_REDUCE:
            dist.all_reduce(
                input_tensor,
                op=dist.ReduceOp.SUM,
            )

        elif collective == CollectiveKind.ALL_GATHER:
            dist.all_gather_into_tensor(
                output_tensor,
                input_tensor,
            )

        else:
            dist.reduce_scatter_tensor(
                output_tensor,
                input_tensor,
                op=dist.ReduceOp.SUM,
            )

    # One correctness pass.
    reset_input()
    dist.barrier()
    run_collective()
    torch.cuda.synchronize()

    if collective == CollectiveKind.ALL_REDUCE:
        expected = torch.full_like(
            input_tensor[:1],
            expected_sum,
        )

        torch.testing.assert_close(
            input_tensor[:1],
            expected,
        )

    elif collective == CollectiveKind.ALL_GATHER:
        chunk = per_rank_input_elements

        for source_rank in range(world_size):
            begin = source_rank * chunk

            expected = torch.full_like(
                output_tensor[begin : begin + 1],
                float(source_rank + 1),
            )

            torch.testing.assert_close(
                output_tensor[begin : begin + 1],
                expected,
            )

    else:
        expected = torch.full_like(
            output_tensor[:1],
            expected_sum,
        )

        torch.testing.assert_close(
            output_tensor[:1],
            expected,
        )

    # Warmup.
    for _ in range(args.warmups):
        reset_input()
        dist.barrier()
        run_collective()

    torch.cuda.synchronize()

    local_samples_ms: list[float] = []

    for _ in range(args.iterations):
        reset_input()

        dist.barrier()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        run_collective()
        end.record()

        end.synchronize()

        local_samples_ms.append(float(start.elapsed_time(end)))

    # Gather all rank-local timings once after the timed loop. The M5 summary
    # uses the slowest rank for each iteration because a collective completes
    # only when every participant has made progress.
    local_tensor = torch.tensor(
        local_samples_ms,
        dtype=torch.float32,
        device=device,
    )

    gathered = torch.empty(
        world_size * args.iterations,
        dtype=torch.float32,
        device=device,
    )

    dist.all_gather_into_tensor(
        gathered,
        local_tensor,
    )

    torch.cuda.synchronize()

    if rank == 0:
        rank_matrix = (
            gathered.reshape(
                world_size,
                args.iterations,
            )
            .cpu()
            .tolist()
        )

        max_rank_samples = [
            max(rank_matrix[rank_index][iteration] for rank_index in range(world_size))
            for iteration in range(args.iterations)
        ]

        latency = summarize_latency(max_rank_samples)

        bandwidth = calculate_collective_bandwidth(
            collective=collective,
            payload_bytes=(payload_bytes),
            latency_ms=(latency.p50_ms),
            world_size=world_size,
        )

        nccl_version = (
            torch.cuda.nccl.version()
            if hasattr(
                torch.cuda,
                "nccl",
            )
            else None
        )

        run = CollectiveRun(
            collective=collective,
            world_size=world_size,
            dtype=args.dtype,
            payload_bytes=payload_bytes,
            per_rank_input_bytes=(per_rank_input_elements * bytes_per_element),
            per_rank_output_bytes=(per_rank_output_elements * bytes_per_element),
            warmups=args.warmups,
            iterations=args.iterations,
            local_rank_samples_ms=tuple(
                tuple(float(value) for value in samples) for samples in rank_matrix
            ),
            max_rank_samples_ms=tuple(float(value) for value in max_rank_samples),
            latency=latency,
            algorithm_bandwidth_gbps=(bandwidth.algorithm_bandwidth_gbps),
            bus_bandwidth_gbps=(bandwidth.bus_bandwidth_gbps),
            metadata={
                "hostname": (platform.node()),
                "git_commit": (_git_commit()),
                "torch": (torch.__version__),
                "torch_cuda": (torch.version.cuda),
                "nccl_version": (nccl_version),
                "cuda_visible_devices": (os.environ.get("CUDA_VISIBLE_DEVICES")),
                "gpu_names": [
                    torch.cuda.get_device_name(index) for index in range(world_size)
                ],
                "timing": ("CUDA events; per-iteration max across ranks"),
            },
        )

        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.output.write_text(
            json.dumps(
                run.to_dict(),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            json.dumps(
                run.to_dict(),
                indent=2,
                sort_keys=True,
            )
        )

        print(f"Result written to: {args.output}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
