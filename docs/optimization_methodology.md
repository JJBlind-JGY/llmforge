# M7 — Original Optimization

## Why M7 is different

M7 is the first stage where LLMForge is allowed to make a performance change
that may become the project's headline technical contribution.

It is also the stage where the specific optimization is selected **latest**.

The selection rule is:

```text
real workload
    ↓
stable symptom
    ↓
profiler / telemetry evidence
    ↓
mechanism
    ↓
candidate
    ↓
correctness
    ↓
before / after
    ↓
trade-off
```

Code existing in the repository is not evidence that an optimization works.

## 1. Selection gate

A headline candidate must satisfy all eight conditions:

1. reproducible in a real workload;
2. profiler or telemetry evidence exists;
3. the system mechanism can be explained;
4. it is not only changing one default parameter;
5. the baseline is frozen;
6. correctness and regression tests are possible;
7. the experiment is feasible on available hardware;
8. the problem remains relevant to the current upstream ecosystem.

Run:

```bash
python scripts/optimization/evaluate_selection_gate.py   --candidate configs/optimization/m7_selection_evidence.json   --output artifacts/optimization/<experiment>/selection_gate.json
```

If `eligible=false`, do not present the candidate as the project's main
optimization.

## 2. Observation sheet

M7 consumes the evidence produced by earlier milestones:

```text
M3
client TTFT / TPOT / E2E / throughput
       │
M4    queue wait / scheduled prompt+decode / KV / preemption
       │
M5    NCCL / topology / TP / replica / communication
       │
M6    normalized live telemetry / logs / traces
       ▼
M7 Observation Sheet
```

Build it from machine-readable artifacts:

```bash
python scripts/optimization/build_observation_sheet.py   --manifest configs/optimization/m7_observation_manifest.json   --output artifacts/optimization/<experiment>/observation.json
```

Do not use screenshots as the only evidence.

## 3. Upstream-overlap audit

The pinned vLLM line already provides controls such as:

```text
max_num_batched_tokens
max_num_scheduled_tokens
enable_chunked_prefill
long_prefill_token_threshold
scheduling_policy
max_num_queued_reqs
max_num_queued_tokens
scheduler_reserve_full_isl
watermark
```

Those controls are useful baselines.

Changing one of them and reporting a better number is not, by itself, the M7
contribution.

Read:

```text
docs/m7_upstream_gap_audit.md
configs/optimization/m7_candidate_catalog.json
```

before selecting a candidate.

## 4. Candidate catalog

The repository intentionally contains multiple categories.

### Main-candidate class

```text
adaptive_mixed_batch_budget
priority_aging
```

The first has a reference implementation. The second remains design-only until
fairness/starvation evidence exists.

### Baseline/control class

```text
static_token_budget_sweep
```

### Upstream-equivalent class

```text
kv_watermark_tuning
queued_token_cap_tuning
```

These can appear in comparisons but cannot be presented as original
optimizations.

### Supporting-system class

```text
topology_aware_placement
```

This can become important if M5 evidence shows communication is the dominant
bottleneck.

## 5. Reference candidate: adaptive mixed-batch budget

The reference candidate targets one specific hypothesis:

> Under mixed long-prefill/decode traffic, large prefill work can consume a
> large share of the scheduler's per-step token budget and worsen decode tail
> latency.

Let:

```text
B      = upstream/base scheduled-token budget
D      = active decode requests
P      = active/waiting prefill pressure
U_kv   = current KV usage ratio
B_min  = minimum candidate budget
Q_p    = bounded prefill quantum
D_thr  = decode pressure threshold
U_thr  = KV pressure threshold
```

The candidate changes the budget only when prefill and decode overlap and one
pressure condition is active:

```text
mixed = (D > 0) and (P > 0)

pressure =
    D >= D_thr
    or
    U_kv >= U_thr
```

Then:

```text
B_t = min(
    B,
    max(
        B_min,
        D + Q_p
    )
)
```

Otherwise:

```text
B_t = B
```

The intuition is:

```text
reserve enough step capacity for active decode progress
+
allow bounded prefill progress
```

rather than using one static reduced budget for every step.

### Important limitation

Changing the total budget does not mathematically guarantee that every decode
request is scheduled before every prefill request. The actual vLLM request
ordering still matters.

Therefore M4 traces must verify the real scheduled prompt/output mix. If the
candidate does not actually protect decode work in the measured runtime, the
hypothesis is rejected or the mechanism must be redesigned.

This limitation is intentional and must not be hidden in the report.

## 6. Why the candidate subclasses the M4 scheduler

The M7 class is:

```text
AdaptiveBudgetScheduler
        ↓
TracingScheduler (M4)
        ↓
vLLM Scheduler
```

Therefore candidate runs preserve the M4 runtime trace path.

Each scheduling step also writes an M7 policy-decision artifact:

```text
step_id
base_budget
applied_budget
reason
running_prefill
running_decode
waiting_prefill
waiting_decode
kv_usage_ratio
```

This creates a direct evidence path:

```text
policy decision
   ↓
actual M4 schedule
   ↓
client M3 latency
```

## 7. Speculative decoding scope

The reference adaptive-budget implementation fails fast when speculative
decoding is enabled.

That is deliberate.

The current Qwen3-8B baseline does not use speculative decoding, and silently
pretending that "one active decode request = one scheduled token" remains true
under speculative execution would make the algorithm contract incorrect.

Support can be added later if the project expands the experiment scope.

## 8. Freeze the experiment before running

Create the real experiment JSON from:

```text
configs/optimization/m7_adaptive_budget_experiment.example.json
```

Then:

```bash
python scripts/optimization/freeze_experiment.py   --experiment configs/optimization/m7_adaptive_budget_experiment.json   --output-dir artifacts/optimization/adaptive_budget_v1
```

This records fingerprints for both arms.

The following must remain identical between baseline and candidate unless the
candidate mechanism explicitly requires otherwise:

```text
model
model revision
dtype
workload
prompt/output distribution
concurrency/arrival pattern
prefix-cache policy
chunked-prefill baseline state
GPU selection
measurement boundary
warmup/repeat policy
```

## 9. Baseline controls

For an adaptive-budget claim, also run static controls such as:

```text
max_num_batched_tokens = 256
512
1024
2048
```

The purpose is to answer:

> Is adaptation useful, or did we merely discover that a smaller static token
> budget fits this workload better?

If a static configuration provides the same benefit with equal or better
trade-offs, the adaptive mechanism is not justified.

## 10. Correctness gate

Scheduler/runtime optimizations are expected to preserve model semantics.

Run the deterministic correctness workload once per arm:

```bash
python scripts/optimization/run_correctness_probe.py   --base-url http://127.0.0.1:8000   --model qwen3-8b   --output artifacts/optimization/<experiment>/baseline_correctness.jsonl
```

and candidate:

```bash
python scripts/optimization/run_correctness_probe.py   --base-url http://127.0.0.1:8000   --model qwen3-8b   --output artifacts/optimization/<experiment>/candidate_correctness.jsonl
```

Compare:

```bash
python scripts/optimization/compare_correctness.py   --baseline artifacts/optimization/<experiment>/baseline_correctness.jsonl   --candidate artifacts/optimization/<experiment>/candidate_correctness.jsonl   --output artifacts/optimization/<experiment>/correctness_comparison.json
```

The strict default compares:

```text
case IDs
prompt hash
success
prompt tokens
completion tokens
finish reason
response hash
```

If batching-level floating-point nondeterminism causes text differences, do not
blindly disable the check. Investigate first, then document why a relaxed
correctness criterion is valid.

## 11. Repeated performance experiment

M7 requires repeated data, not a single best run.

The provided example uses:

```text
5 baseline repetitions
5 candidate repetitions
```

with at least three required by code.

Keep all raw M3 result JSON files.

Never replace the raw data with only one table or screenshot.

## 12. Objective and guardrails

An optimization experiment has one primary objective.

Example:

```text
objective:
    minimize TPOT P99

guardrails:
    output throughput regression <= 5%
    TTFT P99 regression <= 10%
    error rate regression <= 0%
```

The exact thresholds are part of the experiment contract and must be frozen
before looking at candidate results.

Do not change a guardrail after seeing a regression just to make the result
pass.

## 13. Full trade-off reporting

Suppose a candidate yields:

```text
TPOT P99        -18%
throughput       -3%
TTFT P99         +4%
```

All three values belong in the report.

The M7 report never prints only the objective metric.

The final `claim_ready` condition requires:

```text
selection gate passed
AND
correctness passed
AND
objective metric exists
AND
all frozen guardrails passed
```

Even when `claim_ready=true`, the report still shows every measured regression.

## 14. Negative results

A rejected candidate is not wasted work.

Keep:

```text
Observation Sheet
Hypothesis
Profile Evidence
Change
Before/After
Trade-off
Conclusion
```

Examples of useful negative conclusions:

```text
dynamic budget did not change the actual scheduled prompt/decode mix
static budget matched the adaptive policy
tail latency moved but profiler evidence did not support the proposed cause
throughput regression exceeded the frozen guardrail
KV pressure was not the true bottleneck
```

These conclusions are valuable engineering evidence.

## 15. Running an experiment arm

After freezing an experiment, print the exact command:

```bash
python scripts/optimization/print_experiment_arm_command.py   --experiment artifacts/optimization/adaptive_budget_v1/experiment.json   --arm baseline   --gpus 3   --hf-home "$HOME/workspace/.cache/huggingface"
```

Then repeat with:

```text
--arm candidate
```

Use the M6 managed-run wrapper for real executions so every server/benchmark run
keeps its command, git state, log, and completion status.

## 16. Decision-trace analysis

Candidate server runs emit a decision trace.

Analyze it with:

```bash
python scripts/optimization/analyze_budget_decisions.py   artifacts/optimization/adaptive_budget_<pid>.jsonl   --output artifacts/optimization/<experiment>/decision_summary.json
```

A result such as:

```text
changed_fraction = 0.00
```

means the candidate was effectively inactive for that workload. Do not claim an
algorithmic improvement from such a run.

## 17. Final analysis

Prepare:

```text
configs/optimization/m7_result_manifest.json
```

from the example, then run:

```bash
python scripts/optimization/analyze_experiment.py   --manifest configs/optimization/m7_result_manifest.json   --output artifacts/optimization/<experiment>/analysis.json
```

Render:

```bash
python scripts/optimization/render_optimization_report.py   artifacts/optimization/<experiment>/analysis.json   --output reports/optimization_report.md
```

## 18. M7 exit criteria

M7 becomes `EXPERIMENT COMPLETE` only when:

```text
[ ] one candidate passes all eight selection conditions
[ ] upstream-gap audit is current
[ ] baseline configuration is frozen
[ ] correctness passes
[ ] >=3 repeated runs exist for both arms
[ ] raw artifacts are preserved
[ ] profiler/runtime evidence explains the mechanism
[ ] objective is evaluated
[ ] every guardrail is evaluated
[ ] trade-offs are fully reported
[ ] negative/control results are retained
[ ] report links the exact workload/environment/commit
```

Until then:

```text
CODE READY          yes
GPU VALIDATED       pending
EXPERIMENT COMPLETE pending
```

is the correct project status.

## Interview checks

You should eventually be able to answer without notes:

1. Why is changing `max_num_batched_tokens` not automatically an original
   optimization?
2. Why can a lower scheduling budget improve TPOT but hurt TTFT/throughput?
3. Why must adaptive-budget claims include static-budget controls?
4. What evidence proves the policy actually changed scheduler behavior?
5. Why are client TTFT/TPOT and internal scheduler metrics not interchangeable?
6. Why is correctness a hard gate for a scheduler optimization?
7. What would make you reject your own candidate?
8. If the bottleneck turns out to be NCCL or CPU launch overhead, what happens
   to the adaptive-budget candidate?
