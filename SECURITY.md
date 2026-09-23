# Security Policy

LLMForge Infra is a research and performance-engineering project. It is not a
hosted service, but security issues can still exist in its source code,
configuration handling, dependency usage, Docker tooling, or documentation.

## Supported Versions

The actively maintained line is the latest version on the `main` branch and the
latest tagged alpha/stable release.

Older development snapshots are not guaranteed to receive security fixes.

## Reporting a Vulnerability

Please **do not open a public GitHub issue** for a vulnerability that could
expose credentials, enable unintended code execution, or create a meaningful
security risk for users.

Preferred reporting path:

1. Open the repository's **Security** tab.
2. Use **Report a vulnerability / Private vulnerability reporting** when it is
   available.
3. Include a minimal reproduction, affected commit/version, impact, and any
   suggested mitigation.

If private vulnerability reporting is not enabled, contact the repository
maintainer through the GitHub profile and request a private reporting channel
before sharing sensitive details.

## What Counts as a Security Issue?

Examples include:

```text
credential or token exposure
unsafe handling of user-supplied paths or commands
unexpected arbitrary code execution
dependency/supply-chain issues with direct project impact
container configuration that meaningfully weakens isolation
private machine data unintentionally committed by project tooling
```

The following are normally **not** security reports:

```text
a benchmark result that differs from another machine
a performance regression
an unsupported CUDA/runtime combination
a documentation typo
a normal runtime crash without a security impact
```

Use a regular issue for those cases.

## Secrets and Experiment Artifacts

Do not commit:

```text
API keys
tokens
passwords
private IP addresses
private hostnames
user-specific absolute home paths
GPU UUIDs when they are not required for a public report
```

Local raw experiment data belongs under the gitignored `artifacts/` tree unless
it has been intentionally sanitized for public release.

## Disclosure

Please allow reasonable time for triage and remediation before public
disclosure.
