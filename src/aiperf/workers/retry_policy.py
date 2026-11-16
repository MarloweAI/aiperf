# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Supported retry decisions shared by native and installed HTTP workers."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any, Literal

import orjson
from pydantic import ConfigDict, Field, model_validator

from aiperf.config.base import BaseConfig


class RetryPolicy(BaseConfig):
    """Fully resolved policy; legacy unbounded behavior is explicit."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    version: Literal[1] = Field(default=1, description="Retry policy contract version.")
    strategy: Literal["off", "fixed", "exponential"] = Field(
        default="off", description="Local retry schedule."
    )
    on_429: Literal["follow_retry_policy", "leave_provider"] = Field(
        default="follow_retry_policy", description="Profiling refusal action."
    )
    wait_seconds: float = Field(
        default=10, ge=0, description="Fixed or initial exponential wait, seconds."
    )
    backoff_factor: float = Field(
        default=2, ge=1, description="Exponential multiplier."
    )
    max_wait_seconds: float = Field(
        default=30, ge=0, description="Local wait cap, including jitter, seconds."
    )
    jitter_fraction: float = Field(
        default=0, ge=0, le=1, description="Additional fraction of base wait."
    )
    server_hints: Literal["ignore", "prefer", "minimum"] = Field(
        default="ignore", description="Server hint selection."
    )
    max_attempts: int | None = Field(
        default=6, ge=1, description="Total HTTP attempts including the first."
    )
    max_elapsed_seconds: float | None = Field(
        default=120, gt=0, description="Whole logical request budget, seconds."
    )
    phases: Literal["all", "warmup"] = Field(
        default="all", description="Phases eligible for retry."
    )
    partial_stream: Literal["retry", "stop"] = Field(
        default="stop", description="Whether replaying partial output is permitted."
    )
    jitter_seed: int = Field(
        default=42, ge=0, description="Seed for the independent retry stream."
    )
    legacy_unbounded: bool = Field(
        default=False,
        description="Explicit legacy policy with optional unbounded limits.",
    )

    @model_validator(mode="after")
    def _bounds(self) -> RetryPolicy:
        if not self.legacy_unbounded and (
            self.max_attempts is None or self.max_elapsed_seconds is None
        ):
            raise ValueError(
                "new retry policies require finite attempt and elapsed bounds"
            )
        return self


@dataclass(slots=True, frozen=True)
class RetryHint:
    """Only the allowed retry header information; no arbitrary headers."""

    seconds: float | None = None
    """Selected nonnegative server delay in seconds."""
    source: str | None = None
    """Header interpretation that supplied the selected delay."""
    unusable_reason: str | None = None
    """Why no header could be used, or why milliseconds fell back."""


@dataclass(slots=True, frozen=True)
class RetryDecision:
    """A pure decision; waiting and HTTP transport remain with the worker."""

    action: Literal["retry", "stop", "leave_provider"]
    """Next action for this logical turn."""
    wait_seconds: float | None
    """Effective wait for retry, otherwise unavailable."""
    reason: str
    """Stable decision or terminal reason."""


def retry_jitter_draw(
    policy: RetryPolicy, *, conversation_id: str, turn_index: int, attempt_number: int
) -> float:
    """A reproducible trace-turn/attempt draw, independent of wire UUID and timing."""
    key = orjson.dumps(
        [
            policy.version,
            policy.jitter_seed,
            conversation_id,
            turn_index,
            attempt_number,
        ]
    )
    bits = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") >> 11
    return bits / (1 << 53)


def parse_retry_hint(
    headers: Mapping[str, str] | None, received_at_epoch_seconds: float
) -> RetryHint:
    """Parse millisecond, integer-second or HTTP-date hints with fixed precedence."""
    allowed = {
        str(k).lower(): str(v).strip()
        for k, v in (headers or {}).items()
        if str(k).lower() in {"retry-after", "retry-after-ms"}
    }
    rejected: list[str] = []
    if "retry-after-ms" in allowed:
        try:
            seconds = float(allowed["retry-after-ms"]) / 1000
            if math.isfinite(seconds) and seconds >= 0:
                return RetryHint(seconds, "retry-after-ms")
        except ValueError:
            pass
        rejected.append("invalid_retry_after_ms")
    if "retry-after" in allowed:
        parsed = _standard_retry_hint(
            allowed["retry-after"], received_at_epoch_seconds, rejected
        )
        if parsed is not None:
            return parsed
        rejected.append("invalid_retry_after")
    return RetryHint(unusable_reason=",".join(rejected) or "missing")


def _standard_retry_hint(
    value: str, received_at_epoch_seconds: float, rejected: list[str]
) -> RetryHint | None:
    if value.isascii() and value.isdigit():
        try:
            seconds = float(value)
            if math.isfinite(seconds):
                return RetryHint(seconds, "retry-after", ",".join(rejected) or None)
        except (ValueError, OverflowError):
            pass
    try:
        date = parsedate_to_datetime(value)
        if date.tzinfo is not None and math.isfinite(received_at_epoch_seconds):
            seconds = max(0, date.timestamp() - received_at_epoch_seconds)
            if math.isfinite(seconds):
                return RetryHint(
                    seconds, "retry-after-date", ",".join(rejected) or None
                )
    except (ValueError, TypeError, OverflowError):
        pass
    return None


def retry_hint_from_record(record: Any) -> RetryHint:
    """Use transport receipt timing rather than the time a decision happens."""
    trace = record.trace_data
    headers = getattr(trace, "response_headers", None) if trace is not None else None
    perf_ns = (
        getattr(trace, "response_headers_received_perf_ns", None)
        if trace is not None
        else None
    )
    received = record.timestamp_ns / 1e9
    if isinstance(perf_ns, int):
        received += (perf_ns - record.start_perf_ns) / 1e9
    return parse_retry_hint(headers, received)


def resolve_retry_policy(endpoint: Any, phase: str) -> RetryPolicy:
    """Resolve endpoint flags once, preserving explicitly selected legacy behavior."""
    strategy = getattr(endpoint, "retry_policy", None)
    phases = getattr(endpoint, "retry_phases", "all")
    warmup = str(phase) == "warmup"
    legacy = strategy is None
    enabled = getattr(endpoint, "retry_failed_requests", False) or (
        warmup and getattr(endpoint, "retry_failed_warmup_requests", False)
    )
    if legacy:
        attempts = getattr(endpoint, "retry_max_attempts", 0)
        return RetryPolicy(
            strategy="fixed" if enabled else "off",
            wait_seconds=getattr(endpoint, "retry_wait_seconds", 10),
            max_wait_seconds=getattr(endpoint, "retry_wait_seconds", 10),
            max_attempts=attempts or None,
            max_elapsed_seconds=None,
            partial_stream="retry",
            legacy_unbounded=True,
            phases="all"
            if getattr(endpoint, "retry_failed_requests", False)
            or getattr(endpoint, "retry_on_429", "follow_retry_policy")
            == "leave_provider"
            else "warmup",
            on_429=getattr(endpoint, "retry_on_429", "follow_retry_policy")
            if not warmup
            else "follow_retry_policy",
        )
    values = {
        name: getattr(endpoint, "retry_" + name)
        for name in RetryPolicy.model_fields
        if hasattr(endpoint, "retry_" + name)
        and getattr(endpoint, "retry_" + name) is not None
    }
    values["strategy"] = "off" if phases == "warmup" and not warmup else strategy
    if not values.get("max_attempts"):
        values["max_attempts"] = 6
    if warmup or phases == "warmup":
        values["on_429"] = "follow_retry_policy"
    return RetryPolicy(**values)


def remaining_attempt_seconds(
    policy: RetryPolicy, elapsed_seconds: float, request_timeout_seconds: float
) -> float:
    """Bound an active HTTP call by both its timeout and logical budget."""
    if policy.max_elapsed_seconds is None:
        return request_timeout_seconds
    return max(
        0, min(request_timeout_seconds, policy.max_elapsed_seconds - elapsed_seconds)
    )


def decide_retry(
    policy: RetryPolicy,
    *,
    attempt_number: int,
    elapsed_seconds: float,
    status_code: int | None,
    retryable: bool,
    partial_output: bool = False,
    jitter_draw: float = 0,
    hint: RetryHint | None = None,
    admission_remaining_seconds: float | None = None,
) -> RetryDecision:
    """No clocks, RNG, waiting or I/O: every bound is checked before dispatch."""
    if status_code == 429 and policy.on_429 == "leave_provider":
        return RetryDecision("leave_provider", None, "http_429_rerouted_away")
    if admission_remaining_seconds is not None and admission_remaining_seconds <= 0:
        return RetryDecision("stop", None, "phase_cutoff")
    if not retryable:
        return RetryDecision("stop", None, "not_retryable")
    if partial_output and policy.partial_stream == "stop":
        return RetryDecision("stop", None, "partial_stream")
    if policy.strategy == "off":
        return RetryDecision("stop", None, "retries_off")
    if policy.max_attempts is not None and attempt_number >= policy.max_attempts:
        return RetryDecision("stop", None, "max_attempts")
    if (
        not 0 <= jitter_draw < 1
        or not math.isfinite(elapsed_seconds)
        or elapsed_seconds < 0
    ):
        raise ValueError("invalid retry decision input")
    wait = _effective_wait(policy, attempt_number, jitter_draw, hint)
    if (
        policy.max_elapsed_seconds is not None
        and elapsed_seconds + wait >= policy.max_elapsed_seconds
    ):
        return RetryDecision("stop", None, "logical_deadline")
    if admission_remaining_seconds is not None and wait >= admission_remaining_seconds:
        return RetryDecision("stop", None, "phase_cutoff")
    return RetryDecision("retry", wait, "retryable_error")


def _effective_wait(
    policy: RetryPolicy, attempt_number: int, jitter_draw: float, hint: RetryHint | None
) -> float:
    try:
        base = policy.wait_seconds * (
            policy.backoff_factor ** (attempt_number - 1)
            if policy.strategy == "exponential"
            else 1
        )
    except OverflowError:
        base = math.inf
    wait = min(
        base * (1 + policy.jitter_fraction * jitter_draw), policy.max_wait_seconds
    )
    if hint is not None and hint.seconds is not None:
        if policy.server_hints == "prefer":
            wait = hint.seconds
        elif policy.server_hints == "minimum":
            wait = max(wait, hint.seconds)
    return wait
