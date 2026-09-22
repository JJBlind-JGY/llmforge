# M6 optional dependency merge

The M6 core path uses only the Python standard library plus code already present
in LLMForge.

OpenTelemetry is optional. If the repository already has
`[project.optional-dependencies]`, merge the following entry into that section:

```toml
observability = [
    "opentelemetry-api>=1.30,<2",
    "opentelemetry-sdk>=1.30,<2",
    "opentelemetry-exporter-otlp-proto-http>=1.30,<2",
]
```

Do not create a second `[project.optional-dependencies]` table.

The default JSONL tracing path does not require these dependencies.

Install the optional bridge only when testing OTLP export:

```bash
uv pip install -e ".[observability]"
```
