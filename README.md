# LLMForge

LLMForge is a production-oriented LLM inference infrastructure and performance engineering project.

The project is built incrementally to study and engineer the core layers of modern LLM inference systems, including:

- GPU execution and performance profiling
- Transformer prefill/decode execution
- KV-cache management
- Continuous batching and request scheduling
- vLLM runtime internals
- Multi-GPU and distributed inference
- Observability and SLO-oriented benchmarking
- Profiler-driven runtime optimization

## Development Model

LLMForge uses a split development workflow:

- **Windows workstation:** source development, tests, documentation, and Git workflow
- **Linux GPU server:** CUDA builds, LLM serving, profiling, and multi-GPU benchmarks

## Project Status

Current milestone:

**M0 — Reproducible AI Infra Engineering Foundation**

The project is under active development.