# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy conformance: bounds, hints, phases, partial output and departure."""

from types import SimpleNamespace

import pytest

from aiperf.workers.retry_policy import (
    RetryPolicy,
    decide_retry,
    parse_retry_hint,
    remaining_attempt_seconds,
    resolve_retry_policy,
)


def decision(policy: RetryPolicy, **kwargs):
    return decide_retry(
        policy,
        **{
            "attempt_number": 1,
            "elapsed_seconds": 0,
            "status_code": 429,
            "retryable": True,
            **kwargs,
        },
    )


@pytest.mark.parametrize("attempt,expected", [(1, 5), (2, 5), (240, 5)])
def test_strict_zero_jitter_exact_wait(attempt: int, expected: float) -> None:
    p = RetryPolicy(
        strategy="fixed",
        wait_seconds=5,
        max_wait_seconds=5,
        max_attempts=241,
        max_elapsed_seconds=1200,
    )
    d = decision(
        p,
        attempt_number=attempt,
        jitter_draw=0.99,
        hint=parse_retry_hint({"Retry-After": "100"}, 0),
    )
    assert (d.action, d.wait_seconds) == ("retry", expected)
    assert decision(p, attempt_number=241).reason == "max_attempts"
    assert decision(p, elapsed_seconds=1196).reason == "logical_deadline"


def test_exponential_jitter_and_local_cap() -> None:
    p = RetryPolicy(
        strategy="exponential",
        wait_seconds=2,
        backoff_factor=2,
        jitter_fraction=0.25,
        max_wait_seconds=9,
    )
    assert [
        decision(p, attempt_number=k, jitter_draw=0.5).wait_seconds for k in range(1, 5)
    ] == [2.25, 4.5, 9, 9]


@pytest.mark.parametrize(
    "mode,expected", [("ignore", 5), ("prefer", 2), ("minimum", 5)]
)
def test_hint_selection(mode: str, expected: float) -> None:
    p = RetryPolicy(strategy="fixed", wait_seconds=5, server_hints=mode)
    assert (
        decision(p, hint=parse_retry_hint({"Retry-After": "2"}, 0)).wait_seconds
        == expected
    )


@pytest.mark.parametrize(
    "headers,seconds,source",
    [
        ({"retry-after-ms": "12.5", "Retry-After": "99"}, 0.0125, "retry-after-ms"),
        ({"retry-after-ms": "NaN", "Retry-After": "2"}, 2, "retry-after"),
        ({"RETRY-AFTER": "0"}, 0, "retry-after"),
        ({"Retry-After": "Thu, 01 Jan 1970 00:00:10 GMT"}, 10, "retry-after-date"),
        ({"Retry-After": "Wed, 31 Dec 1969 23:59:59 GMT"}, 0, "retry-after-date"),
        ({"Retry-After": "-1"}, None, None),
        ({"Retry-After": "1.5"}, None, None),
        ({"retry-after-ms": "inf"}, None, None),
        ({"Retry-After": "garbage"}, None, None),
    ],
)
def test_hint_parsing(headers: dict, seconds: float | None, source: str | None) -> None:
    h = parse_retry_hint(headers, 0)
    assert h.seconds == seconds
    assert h.source == source
    if seconds is None:
        assert h.unusable_reason


def test_server_wait_not_shortened_to_local_cap_or_budget() -> None:
    p = RetryPolicy(
        strategy="fixed",
        wait_seconds=1,
        max_wait_seconds=2,
        server_hints="prefer",
        max_elapsed_seconds=120,
    )
    h = parse_retry_hint({"Retry-After": "100"}, 0)
    assert decision(p, hint=h).wait_seconds == 100
    assert decision(p, hint=h, elapsed_seconds=21).reason == "logical_deadline"
    assert decision(p, hint=h, admission_remaining_seconds=50).reason == "phase_cutoff"


def test_departure_only_on_profiling_429() -> None:
    endpoint = SimpleNamespace(retry_policy="fixed", retry_on_429="leave_provider")
    profile = resolve_retry_policy(endpoint, "profiling")
    warmup = resolve_retry_policy(endpoint, "warmup")
    assert decision(profile).action == "leave_provider"
    assert decision(warmup).action == "retry"
    assert decision(profile, status_code=500).action == "retry"
    assert decision(profile, status_code=400, retryable=False).action == "stop"


def test_partial_and_cutoff_are_explicit() -> None:
    p = RetryPolicy(strategy="fixed", partial_stream="stop", max_elapsed_seconds=1200)
    assert decision(p, partial_output=True).reason == "partial_stream"
    assert decision(p, admission_remaining_seconds=0).reason == "phase_cutoff"
    assert remaining_attempt_seconds(p, 1190, 30) == 10
    assert remaining_attempt_seconds(p, 1200, 30) == 0


def test_resolved_policy_legacy_and_finite_defaults() -> None:
    p = resolve_retry_policy(SimpleNamespace(retry_failed_requests=True), "profiling")
    assert p.legacy_unbounded and p.strategy == "fixed"
    assert p.max_attempts is None and p.max_elapsed_seconds is None
    p = resolve_retry_policy(
        SimpleNamespace(retry_policy="fixed", retry_phases="warmup"), "profiling"
    )
    assert p.strategy == "off"
    with pytest.raises(ValueError):
        RetryPolicy(strategy="fixed", max_attempts=0)
    with pytest.raises(ValueError):
        RetryPolicy(strategy="fixed", wait_seconds=float("nan"))
