# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Open-loop session arrivals for AGENTIC_REPLAY profiling.

With a profiling phase's ``session_arrivals`` (``--session-arrival-rate``
and friends on the CLI), new sessions start on an arrival schedule
instead of when a lane's session tree drains. The schedule comes from the same
interval generators ``--request-rate`` uses (Poisson, or Gamma with shape
1/CV^2), seeded from ``--random-seed``, and is fixed before the phase starts.
Inside a session nothing changes: each turn waits for the previous reply and
then the recorded, capped gap.

The journal records each scheduled arrival against when the client actually
started it, so a report can tell whether the intended offered load was
delivered or held back by the client's own session ceiling.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import orjson

from aiperf.common import random_generator as rng
from aiperf.common.constants import NANOS_PER_MILLIS, NANOS_PER_SECOND

if TYPE_CHECKING:
    from aiperf.config.phases import SessionArrivalsConfig

SESSION_ARRIVALS_FILE = "session_arrivals.json"
SESSION_ARRIVALS_SCHEMA_VERSION = 2


def build_arrival_offsets(
    config: SessionArrivalsConfig, duration_sec: float
) -> list[float]:
    """Scheduled arrival offsets in seconds from the phase start, all below
    ``duration_sec``. The first arrival comes one inter-arrival time after the
    start; the warmed lanes are the population at time zero."""
    if not math.isfinite(duration_sec) or duration_sec <= 0:
        raise ValueError("session arrival duration must be positive and finite")
    if any(step.at_seconds >= duration_sec for step in config.rate_schedule):
        raise ValueError("arrival rate changes must precede phase end")
    stream = (
        random.Random(config.seed)
        if config.seed is not None
        else rng.derive("timing.session.arrivals")
    )
    boundaries = [
        (0.0, config.rate),
        *[(step.at_seconds, step.rate) for step in config.rate_schedule],
        (duration_sec, 0.0),
    ]
    offsets: list[float] = []
    segment = 0
    now = 0.0
    while now < duration_sec:
        hazard = (
            stream.expovariate(1.0)
            if config.pattern == "poisson"
            else stream.gammavariate(config.gamma_shape, 1.0 / config.gamma_shape)
        )
        while segment < len(boundaries) - 1:
            end = boundaries[segment + 1][0]
            rate = boundaries[segment][1]
            available = (end - now) * rate
            if hazard < available:
                now += hazard / rate
                offsets.append(now)
                break
            hazard -= available
            now = end
            segment += 1
        else:
            break
    return offsets


@dataclass(slots=True)
class SessionArrival:
    """One scheduled session arrival and what the client did with it."""

    index: int
    """Position in the schedule."""
    scheduled_offset_sec: float
    """Scheduled time, seconds after the phase start."""
    fired_ns: int | None = None
    """Wall clock (time.time_ns) when the arrival loop released it; None if never."""
    ceiling_full: bool | None = None
    """Whether every session slot was taken when its task went to start it, so
    it waited for one; None if never released."""
    issued_ns: int | None = None
    """Wall clock when its first credit was issued or refused; None if never."""
    started: bool = False
    """Whether its first credit was issued."""
    conversation_id: str | None = None
    """Trace of the session it started."""
    x_correlation_id: str | None = None
    """X-Correlation-ID of the session it started, as the request records carry it."""


def _scheduled_ns(arrival: SessionArrival, start_ns: int | None) -> int | None:
    if start_ns is None:
        return None
    return start_ns + round(arrival.scheduled_offset_sec * NANOS_PER_SECOND)


def _lag_statistics(lags_ns: list[int]) -> dict[str, float | None]:
    """Nearest-rank p50 and p99 and the maximum, in milliseconds."""
    if not lags_ns:
        return {"p50": None, "p99": None, "max": None}
    ordered = sorted(lags_ns)

    def rank(quantile: float) -> float:
        return (
            ordered[max(0, math.ceil(quantile * len(ordered)) - 1)] / NANOS_PER_MILLIS
        )

    return {"p50": rank(0.50), "p99": rank(0.99), "max": ordered[-1] / NANOS_PER_MILLIS}


def session_arrival_journal(
    config: SessionArrivalsConfig,
    arrivals: list[SessionArrival],
    *,
    initial_sessions: int,
    start_ns: int | None,
    duration_sec: float,
    initial_assignments: list[str] | None = None,
    root_events: list[dict[str, Any]] | None = None,
    complete: bool = False,
    maintained_baseline: bool = False,
) -> dict[str, Any]:
    """The ``session_arrivals.json`` document for a finished profiling phase.

    Lags are against each arrival's scheduled time. Without a phase start the
    scheduled times and lags are unavailable (None), never zero.
    """
    fire_lags: list[int] = []
    start_lags: list[int] = []
    rows: list[dict[str, Any]] = []
    for arrival in arrivals:
        scheduled_ns = _scheduled_ns(arrival, start_ns)
        if scheduled_ns is not None and arrival.fired_ns is not None:
            fire_lags.append(arrival.fired_ns - scheduled_ns)
        if (
            scheduled_ns is not None
            and arrival.started
            and arrival.issued_ns is not None
        ):
            start_lags.append(arrival.issued_ns - scheduled_ns)
        rows.append(
            {
                "index": arrival.index,
                "scheduled_offset_sec": arrival.scheduled_offset_sec,
                "scheduled_offset_ns": round(
                    arrival.scheduled_offset_sec * NANOS_PER_SECOND
                ),
                "scheduled_ns": scheduled_ns,
                "fired_ns": arrival.fired_ns,
                "ceiling_full": arrival.ceiling_full,
                "issued_ns": arrival.issued_ns,
                "started": arrival.started,
                "conversation_id": arrival.conversation_id,
                "x_correlation_id": arrival.x_correlation_id,
            }
        )
    started = sum(1 for arrival in arrivals if arrival.started)
    ceiling_waits = sum(1 for arrival in arrivals if arrival.ceiling_full is True)
    document = {
        "schema_version": 3 if maintained_baseline else SESSION_ARRIVALS_SCHEMA_VERSION,
        "complete": complete,
        "config": {
            "rate_per_sec": config.rate,
            "pattern": config.pattern,
            "cv": config.effective_cv,
            "gamma_shape": config.gamma_shape,
            "ceiling": config.ceiling,
            "seed": config.seed,
            "rate_schedule": [step.model_dump() for step in config.rate_schedule],
        },
        "initial_sessions": initial_sessions,
        "initial_assignments": initial_assignments or [],
        "root_events": root_events or [],
        "schedule_sha256": hashlib.sha256(
            orjson.dumps(
                [
                    {
                        "index": a.index,
                        "scheduled_offset_ns": round(
                            a.scheduled_offset_sec * NANOS_PER_SECOND
                        ),
                    }
                    for a in arrivals
                ],
                option=orjson.OPT_SORT_KEYS,
            )
        ).hexdigest(),
        "assignments_sha256": hashlib.sha256(
            orjson.dumps(
                {
                    "initial": initial_assignments or [],
                    "arrivals": [a.conversation_id for a in arrivals],
                },
                option=orjson.OPT_SORT_KEYS,
            )
        ).hexdigest(),
        "start_ns": start_ns,
        "duration_sec": duration_sec,
        "summary": {
            "scheduled": len(arrivals),
            "scheduled_rate_per_sec": len(arrivals) / duration_sec,
            "fired": sum(1 for arrival in arrivals if arrival.fired_ns is not None),
            "started": started,
            "not_started": len(arrivals) - started,
            "ceiling_waits": ceiling_waits,
            "held_back": ceiling_waits > 0,
            "fire_lag_ms": _lag_statistics(fire_lags),
            "start_lag_ms": _lag_statistics(start_lags),
        },
        "arrivals": rows,
    }
    if maintained_baseline:
        document["maintained_baseline"] = True
    return document
