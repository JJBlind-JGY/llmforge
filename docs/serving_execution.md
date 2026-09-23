# M3 Serving Execution Plan

## Status

Code is prepared before GPU execution. Do not add performance claims until the
corresponding experiment has run on the pinned benchmark machine.

## Measurement contract

The client sends exact token-ID prompts to `/v1/completions`.

```text
TTFT = first generated token arrival - HTTP dispatch
ITL  = adjacent generated-token arrival gaps
TPOT = (E2E - TTFT) / (output_tokens - 1)
E2E  = final streamed response completion - HTTP dispatch
```

Client-side semaphore queueing is measured separately as `client_queue_ms` and
is never folded into TTFT.

The request fixes `stream_interval=1`, `ignore_eos=true`,
`min_tokens=max_tokens`, `add_special_tokens=false`, and
`return_token_ids=true`.

## Required workload families

```text
fixed synthetic
mixed prompt/output lengths
shared-prefix
Poisson arrivals
burst arrivals
```

## Run order

```text
shape
  ↓
concurrency
  ↓
mixed
  ↓
shared_prefix
  ↓
poisson / burst
```

Each stage answers a different question. Do not run the whole suite blindly.

## Server profiles

Baseline:

```text
configs/serving/qwen3_8b_vllm_baseline.json
```

Prefix-cache experiment:

```text
configs/serving/qwen3_8b_vllm_prefix_cache.json
```

Print the exact launch command:

```bash
python scripts/print_vllm_server_command.py   --config configs/serving/qwen3_8b_vllm_baseline.json   --gpu 3   --hf-home "$HOME/workspace/.cache/huggingface"
```

## Dry-run a suite group

```bash
python scripts/run_m3_serving_suite.py   --group shape   --dry-run
```

## Execute later

```bash
python scripts/run_m3_serving_suite.py   --group shape
```

Raw results are written under:

```text
artifacts/benchmarks/serving/<case>/<timestamp>/result.json
```

The final `reports/vllm_baseline.md` should be written only after real execution
and interpretation.
