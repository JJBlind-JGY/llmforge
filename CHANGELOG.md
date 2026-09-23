# Changelog

All notable changes to LLMForge Infra are documented in this file.

The project uses semantic release milestones for the public repository while
keeping performance claims tied to their individual experiment reports.

## [Unreleased]

### Planned

- Complete the M3 serving benchmark matrix.
- Validate M4 runtime tracing on the pinned vLLM environment.
- Complete M5 distributed-inference experiments.
- Validate M6 production observability in live serving runs.
- Select and evaluate the final M7 optimization from measured evidence.
- Complete M8 vLLM/SGLang cross-engine validation.
- Record a real upstream issue, patch, or pull-request attempt.

## [0.1.0-alpha.1] - 2026-09-23

### Added

- Reproducible benchmark and environment foundation.
- Native CUDA and Triton performance exercises and reports.
- Transformer resource accounting, KV-cache demonstrations, and
  Prefill/Decode profiling.
- OpenAI-compatible serving benchmark client and workload generator.
- vLLM runtime source discovery, scheduler/KV instrumentation, and trace
  analysis infrastructure.
- NCCL/topology/distributed-inference experiment infrastructure.
- Production-oriented configuration, logging, metrics, tracing, health, and
  reproducible run tooling.
- Evidence-gated optimization framework with correctness and regression gates.
- vLLM/SGLang cross-engine adapter and analysis framework.
- Public learning path, hardware tiers, compatibility guide, reproduction
  documentation, and CPU-only synthetic examples.
- Community health files and release/portfolio readiness audit.

### Changed

- Public project branding and Python distribution use `LLMForge Infra` /
  `llmforge` while the import namespace remains `llmforge`.
- Documentation is organized by systems concepts rather than only by M0–M8
  construction history.
- Measured reports, pending report templates, raw local artifacts, and public
  examples have separate repository semantics.
- CI validates the full CPU-capable suite and the built wheel.

### Validation Status

This is an **alpha engineering release**.

Validated project evidence currently includes the completed M1/M2 GPU and
Transformer-runtime experiments.

The later serving/runtime/distributed/optimization/cross-engine engineering
paths are present, but their complete live experiment campaigns are not implied
by this release tag.

The README project-status table is the public source of truth for that
distinction.
