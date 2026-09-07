"""Utilities for native benchmark executables."""

from __future__ import annotations


def parse_key_value_output(output: str) -> dict[str, str | int | float | bool]:
    """Parse newline-separated key=value benchmark output."""

    result: dict[str, str | int | float | bool] = {}

    for line in output.splitlines():
        if "=" not in line:
            continue

        key, raw_value = line.split("=", maxsplit=1)

        key = key.strip()
        value = raw_value.strip()

        if value == "true":
            result[key] = True
            continue

        if value == "false":
            result[key] = False
            continue

        try:
            result[key] = int(value)
            continue
        except ValueError:
            pass

        try:
            result[key] = float(value)
            continue
        except ValueError:
            pass

        result[key] = value
    return result
