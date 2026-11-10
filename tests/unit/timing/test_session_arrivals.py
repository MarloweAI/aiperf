# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Open-loop session arrival schedules and their journal.

Failure modes covered: a schedule that changes under the same seed, a gamma
shape that does not give the requested CV, a mean rate off by the shape
factor, arrivals past the phase end, and a summary that reports missing
observations as zero or hides sessions the client held back.
"""

from __future__ import annotations

import statistics

import pytest
from pytest import param

from aiperf.common import random_generator as rng
from aiperf.config.phases import SessionArrivalsConfig
from aiperf.timing.session_arrivals import (
    SessionArrival,
    build_arrival_offsets,
    session_arrival_journal,
)


def _offsets(seed: int, config: SessionArrivalsConfig, duration: float) -> list[float]:
    rng.reset()
    rng.init(seed)
    return build_arrival_offsets(config, duration)


def _intervals(offsets: list[float]) -> list[float]:
    return [b - a for a, b in zip([0.0, *offsets], offsets, strict=False)]


def _cv(values: list[float]) -> float:
    return statistics.pstdev(values) / statistics.fmean(values)


def test_build_arrival_offsets_same_seed_same_schedule() -> None:
    config = SessionArrivalsConfig(rate=0.5, pattern="gamma", cv=2.0, ceiling=64)
    assert _offsets(7, config, 600.0) == _offsets(7, config, 600.0)
    assert _offsets(7, config, 600.0) != _offsets(8, config, 600.0)


@pytest.mark.parametrize(
    "pattern,cv",
    [
        param("poisson", 1.0, id="poisson"),
        param("gamma", 1.0, id="gamma-cv1"),
        param("gamma", 2.0, id="gamma-cv2"),
        param("gamma", 4.0, id="gamma-cv4"),
    ],
)  # fmt: skip
def test_build_arrival_offsets_mean_rate_and_cv(pattern: str, cv: float) -> None:
    config = SessionArrivalsConfig(rate=2.0, pattern=pattern, cv=cv, ceiling=64)
    intervals = _intervals(_offsets(42, config, 100_000.0))
    assert len(intervals) > 150_000
    assert statistics.fmean(intervals) == pytest.approx(0.5, rel=0.05)
    assert _cv(intervals) == pytest.approx(cv, rel=0.05)


def test_build_arrival_offsets_stay_inside_the_duration() -> None:
    config = SessionArrivalsConfig(rate=10.0, pattern="poisson", cv=1.0, ceiling=64)
    offsets = _offsets(42, config, 30.0)
    assert offsets == sorted(offsets)
    assert offsets[0] > 0.0
    assert offsets[-1] < 30.0


@pytest.mark.parametrize(
    "kwargs,message",
    [
        param({"pattern": "poisson", "cv": 2.0}, "Poisson", id="poisson-cv"),
        param({"pattern": "constant", "cv": 1.0}, "'poisson' or 'gamma'", id="constant"),
        param({"pattern": "gamma", "cv": 0.0}, "greater than 0", id="zero-cv"),
        param({"pattern": "gamma"}, "need cv", id="gamma-without-cv"),
    ],
)  # fmt: skip
def test_session_arrival_config_rejects_invalid_processes(
    kwargs: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SessionArrivalsConfig(rate=1.0, ceiling=64, **kwargs)


def test_session_arrival_journal_reports_lag_ceiling_waits_and_unstarted() -> None:
    start_ns = 1_000_000_000_000
    second = 1_000_000_000
    config = SessionArrivalsConfig(rate=0.5, pattern="gamma", cv=2.0, ceiling=4096)
    arrivals = [
        SessionArrival(
            index=0,
            scheduled_offset_sec=1.0,
            fired_ns=start_ns + 1 * second + 2_000_000,
            ceiling_full=False,
            issued_ns=start_ns + 1 * second + 3_000_000,
            started=True,
            conversation_id="trace_a",
            x_correlation_id="corr_a",
        ),
        SessionArrival(
            index=1,
            scheduled_offset_sec=2.0,
            fired_ns=start_ns + 2 * second,
            ceiling_full=True,
            issued_ns=start_ns + 7 * second,
            started=True,
            conversation_id="trace_b",
            x_correlation_id="corr_b",
        ),
        SessionArrival(index=2, scheduled_offset_sec=9.5),
    ]
    journal = session_arrival_journal(
        config,
        arrivals,
        initial_sessions=8,
        start_ns=start_ns,
        duration_sec=10.0,
    )

    assert journal["schema_version"] == 1
    assert journal["config"] == {
        "rate_per_sec": 0.5,
        "pattern": "gamma",
        "cv": 2.0,
        "gamma_shape": 0.25,
        "ceiling": 4096,
    }
    assert journal["initial_sessions"] == 8
    summary = journal["summary"]
    assert summary["scheduled"] == 3
    assert summary["scheduled_rate_per_sec"] == pytest.approx(0.3)
    assert summary["fired"] == 2
    assert summary["started"] == 2
    assert summary["not_started"] == 1
    assert summary["ceiling_waits"] == 1
    assert summary["held_back"] is True
    assert summary["start_lag_ms"]["max"] == pytest.approx(5000.0)
    assert summary["fire_lag_ms"]["max"] == pytest.approx(2.0)
    assert journal["arrivals"][2] == {
        "index": 2,
        "scheduled_offset_sec": 9.5,
        "scheduled_ns": start_ns + 9_500_000_000,
        "fired_ns": None,
        "ceiling_full": None,
        "issued_ns": None,
        "started": False,
        "conversation_id": None,
        "x_correlation_id": None,
    }


def test_session_arrival_journal_without_observations_is_unavailable_not_zero() -> None:
    config = SessionArrivalsConfig(rate=0.5, pattern="poisson", cv=1.0, ceiling=64)
    journal = session_arrival_journal(
        config,
        [SessionArrival(index=0, scheduled_offset_sec=1.0)],
        initial_sessions=4,
        start_ns=None,
        duration_sec=10.0,
    )
    summary = journal["summary"]
    assert summary["fired"] == 0
    assert summary["held_back"] is False
    assert summary["fire_lag_ms"] == {"p50": None, "p99": None, "max": None}
    assert summary["start_lag_ms"] == {"p50": None, "p99": None, "max": None}
    assert journal["arrivals"][0]["scheduled_ns"] is None
