# LLMForge Configuration

The files in this directory are reproducible experiment defaults and examples.

They are not assumptions about every user's machine.

## Principles

Public configurations should avoid machine-specific identifiers such as:

- hostnames;
- usernames;
- absolute home directories;
- GPU UUIDs;
- private network addresses.

Hardware-dependent values such as GPU indices and topology should be verified
on the target machine before running an experiment.

## GPU Selection

Most single-GPU examples use:

```text
CUDA_VISIBLE_DEVICES=0
```

This is only a portable default.

Use another physical GPU when appropriate:

```text
CUDA_VISIBLE_DEVICES=3 ... 
```
## Distributed Profiles

Before using multi-GPU profiles, inspect the target system:

```text
python scripts/distributed/capture_topology.py
```

Topology-sensitive benchmark results are not portable across machines.

## Local Overrides

Do not commit credentials or private machine information.

Prefer shell environment variables or ignored local configuration files for
machine-specific settings.
