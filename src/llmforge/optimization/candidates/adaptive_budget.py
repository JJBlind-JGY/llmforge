"""Pure policy for the adaptive mixed prefill/decode budget candidate."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class AdaptiveBudgetConfig:
    min_budget: int = 256
    prefill_quantum: int = 256
    decode_pressure: int = 4
    kv_pressure: float = 0.90

    def validate(self) -> None:
        if self.min_budget <= 0:
            raise ValueError(
                "min_budget must be positive."
            )

        if self.prefill_quantum < 0:
            raise ValueError(
                "prefill_quantum must be "
                "non-negative."
            )

        if self.decode_pressure <= 0:
            raise ValueError(
                "decode_pressure must be "
                "positive."
            )

        if not 0.0 <= self.kv_pressure <= 1.0:
            raise ValueError(
                "kv_pressure must be "
                "in [0, 1]."
            )

    @classmethod
    def from_environment(
        cls,
    ) -> "AdaptiveBudgetConfig":
        config = cls(
            min_budget=int(
                os.environ.get(
                    "LLMFORGE_M7_MIN_BUDGET",
                    "256",
                )
            ),
            prefill_quantum=int(
                os.environ.get(
                    "LLMFORGE_M7_PREFILL_QUANTUM",
                    "256",
                )
            ),
            decode_pressure=int(
                os.environ.get(
                    "LLMFORGE_M7_DECODE_PRESSURE",
                    "4",
                )
            ),
            kv_pressure=float(
                os.environ.get(
                    "LLMFORGE_M7_KV_PRESSURE",
                    "0.90",
                )
            ),
        )

        config.validate()
        return config


@dataclass(frozen=True)
class SchedulerPressure:
    running_prefill: int
    running_decode: int
    waiting_prefill: int
    waiting_decode: int
    kv_usage_ratio: float

    @property
    def has_mixed_work(
        self,
    ) -> bool:
        return (
            self.running_decode > 0
            and (
                self.running_prefill
                + self.waiting_prefill
                > 0
            )
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BudgetDecision:
    base_budget: int
    applied_budget: int
    reason: str
    pressure: SchedulerPressure

    @property
    def changed(
        self,
    ) -> bool:
        return (
            self.applied_budget
            != self.base_budget
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["changed"] = (
            self.changed
        )
        return data


def is_prefill_request(
    request: object,
) -> bool:
    prompt_tokens = getattr(
        request,
        "num_prompt_tokens",
        None,
    )

    computed_tokens = getattr(
        request,
        "num_computed_tokens",
        None,
    )

    if (
        prompt_tokens is None
        or computed_tokens is None
    ):
        return False

    return int(
        computed_tokens
    ) < int(
        prompt_tokens
    )


def count_request_phases(
    requests: Iterable[
        object
    ],
) -> tuple[int, int]:
    prefill = 0
    decode = 0

    for request in requests:
        if is_prefill_request(
            request
        ):
            prefill += 1
        else:
            decode += 1

    return prefill, decode


def choose_budget(
    *,
    config: AdaptiveBudgetConfig,
    base_budget: int,
    pressure: SchedulerPressure,
) -> BudgetDecision:
    config.validate()

    if base_budget <= 0:
        raise ValueError(
            "base_budget must be positive."
        )

    if not pressure.has_mixed_work:
        return BudgetDecision(
            base_budget=base_budget,
            applied_budget=base_budget,
            reason="no_mixed_prefill_decode",
            pressure=pressure,
        )

    decode_pressure = (
        pressure.running_decode
        >= config.decode_pressure
    )

    kv_pressure = (
        pressure.kv_usage_ratio
        >= config.kv_pressure
    )

    if not (
        decode_pressure
        or kv_pressure
    ):
        return BudgetDecision(
            base_budget=base_budget,
            applied_budget=base_budget,
            reason="below_pressure_threshold",
            pressure=pressure,
        )

    # The candidate reserves room for active decode requests and allows a
    # bounded amount of prefill progress in the same step.
    target = max(
        config.min_budget,
        pressure.running_decode
        + config.prefill_quantum,
    )

    applied = min(
        base_budget,
        target,
    )

    reasons = []

    if decode_pressure:
        reasons.append(
            "decode_pressure"
        )

    if kv_pressure:
        reasons.append(
            "kv_pressure"
        )

    return BudgetDecision(
        base_budget=base_budget,
        applied_budget=applied,
        reason="+".join(
            reasons
        ),
        pressure=pressure,
    )
