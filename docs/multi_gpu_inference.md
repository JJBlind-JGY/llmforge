# M5 — Distributed Inference

## Core question

Why can adding GPUs make inference slower, and when should we use tensor
parallelism versus model replicas?

The original M5 syllabus requires:

```text
PCIe / NVLink / NVSwitch
nvidia-smi topo -m
NCCL
AllReduce / AllGather / ReduceScatter
tensor parallel
pipeline parallel concept
replica/data-parallel serving
communication/computation overlap
topology-aware placement
```

The experiment matrix is:

```text
Single GPU
vs
TP=2
vs
TP=3 if the model/framework allows it
vs
2 replicas
```

LLMForge adds two topology controls on the primary 4×4090 server:

```text
TP=2 on GPU0,1  → same NUMA / NODE
TP=2 on GPU0,2  → cross NUMA / SYS
```

and an additional:

```text
TP=4 on GPU0,1,2,3
```

to expose scaling across both NUMA domains.

## Why TP=3 is a preflight experiment, not a mandatory run

Qwen3-8B uses 32 attention heads and 8 KV heads. The M5 profile validator checks
head-sharding compatibility before reserving GPUs. A TP size that cannot evenly
partition query heads is rejected before server launch.

This fulfills the syllabus wording: TP=3 is tested only if the framework/model
structure allows it.

## Phase A — topology

Capture:

```bash
python scripts/distributed/capture_topology.py
```

The artifact records:

- GPU index/name/UUID/PCI bus ID/VRAM;
- CPU affinity;
- NUMA node;
- every pairwise `nvidia-smi topo -m` relation;
- candidate placements.

Do not label a pair "fast" from intuition alone. The topology artifact is
descriptive evidence; the NCCL sweep measures the communication behavior.

## Phase B — collective microbenchmarks

Run at least the same-NUMA and cross-NUMA pairs:

```bash
python scripts/distributed/run_collective_sweep.py   --gpus 0,1   --placement-label same_numa_01
```

```bash
python scripts/distributed/run_collective_sweep.py   --gpus 0,2   --placement-label cross_numa_02
```

The benchmark covers:

```text
AllReduce
AllGather
ReduceScatter
```

and message sizes:

```text
1 / 4 / 16 / 64 / 256 MiB
```

### Timing discipline

Each rank uses CUDA events around the collective. Raw rank-local samples are
gathered after the benchmark. The per-iteration latency reported by M5 is the
**maximum rank latency**, because the distributed step is gated by the slowest
participant.

### Bandwidth semantics

The project follows NCCL-tests terminology:

```text
algBW = S / time
```

and normalizes bus bandwidth with the collective-specific correction factor.

For AllReduce:

```text
busBW = algBW × 2(N-1)/N
```

For AllGather and ReduceScatter:

```text
busBW = algBW × (N-1)/N
```

The payload `S` is defined explicitly in every artifact.

## Phase C — distributed serving profiles

Validate profiles before launching:

```bash
python scripts/distributed/validate_inference_profiles.py
```

Print exact server commands:

```bash
python scripts/distributed/print_inference_commands.py   --hf-home "$HF_HOME"
```

Mandatory profiles:

```text
single_gpu0
tp2_same_numa_01
tp2_cross_numa_02
tp4_all
dp2_replicas_01
```

Optional:

```text
pp2_exploratory_01
tp3_probe
```

Use the exact same M3 workload and serving measurement contract for every valid
profile. Do not compare runs with different prompt/output/concurrency settings.

## Phase D — GPU utilization

Record `nvidia-smi` telemetry during each serving benchmark:

```bash
python scripts/distributed/monitor_gpus.py   --gpus 0,1   --duration-s 60   --output artifacts/distributed/telemetry/tp2_same_numa_01.csv
```

This provides descriptive GPU utilization, memory, and power context.

## Phase E — actual model communication profile

NCCL microbenchmarks do **not** equal vLLM model communication time.

For a short profiling run, add a vLLM profiler config. Generate the argument:

```bash
python scripts/distributed/print_profiler_args.py   --output-dir /absolute/path/to/profile
```

Use only a few benchmark requests. Profiling perturbs performance and should not
be used as the throughput/latency benchmark itself.

After the trace is flushed:

```bash
python scripts/distributed/analyze_profiler_trace.py   /path/to/rank0_trace.json.gz   /path/to/rank1_trace.json.gz   --output artifacts/distributed/profiles/tp2_comm.json
```

The parser reports collective-like profiler event duration per worker trace.

## Phase F — scaling analysis

Create a real manifest from:

```text
configs/distributed/m5_result_manifest.example.json
```

Then:

```bash
python scripts/distributed/analyze_multi_gpu_serving.py   --manifest configs/distributed/m5_result_manifest.json   --output artifacts/distributed/m5_analysis.json
```

Finally:

```bash
python scripts/distributed/render_multi_gpu_report.py   artifacts/distributed/m5_analysis.json   --output reports/multi_gpu_inference.md
```

## Interpretation discipline

For TP, throughput scaling efficiency is informative but not the only objective.
TP may be used for capacity or latency even when aggregate throughput does not
scale linearly.

For replicas/DP, the primary expected benefit is independent-request throughput.

Never infer model communication time from the collective microbenchmark alone.
Use profiler evidence for model-level communication.

## Exit questions

You should be able to answer without notes:

1. Why does TP require communication?
2. Why can TP=2 be slower than one GPU?
3. When do replicas serve a different goal than TP?
4. Why does `NODE` versus `SYS` topology matter?
5. What is the difference between algBW and busBW?
6. Why do we use the slowest rank for collective latency?
7. What is your NCCL hang/slow triage order?
