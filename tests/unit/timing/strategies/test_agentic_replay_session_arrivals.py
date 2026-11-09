# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Open-loop session arrivals in AgenticReplayStrategy PROFILING.

Failure modes covered: a drained lane still recycling (closed loop leaking
into open loop), an arrival starting mid-trace instead of at turn 0, a
session's next turn sent before its reply, one held arrival delaying the ones
after it, the ceiling not replacing the lane count as the session limit, and a
journal missing from the artifact directory.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest

from aiperf.common.enums import CacheBustTarget, CreditPhase
from aiperf.common.loop_scheduler import LoopScheduler
from aiperf.config.phases import SessionArrivalsConfig
from aiperf.credit.structs import TurnToSend
from aiperf.timing.session_arrivals import SESSION_ARRIVALS_FILE, SessionArrival
from aiperf.timing.trajectory_source import Trajectory
from tests.unit.conftest import _REAL_SLEEP
from tests.unit.timing.strategies.test_agentic_replay import (
    _make_credit,
    _make_strategy,
)


def _run(**phase_fields: object) -> MagicMock:
    phase = SimpleNamespace(
        **{
            "burst_phase_starts": False,
            "system_idle_gap_cap_seconds": None,
            "duration": 60.0,
            "session_arrivals": SessionArrivalsConfig(rate=0.2, ceiling=512),
            **phase_fields,
        }
    )
    run = MagicMock()
    run.cfg.get_profiling_phases.return_value = [phase]
    run.cfg.get_cache_bust_target.return_value = CacheBustTarget.NONE
    run.benchmark_id = "bench"
    return run


def _issuer(capture) -> AsyncMock:
    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture
    issuer.set_session_limit = MagicMock()
    issuer.can_acquire_and_start_new_session = MagicMock(return_value=True)
    return issuer


async def _drain() -> None:
    for _ in range(50):
        await _REAL_SLEEP(0)


def _open_loop_strategy(
    issuer: AsyncMock, *, artifact_dir=None, progress=None, **phase_fields
):
    strategy, _, scheduler, src = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=[
            Trajectory(conversation_id="trace_0", start_turn_index=0),
            Trajectory(conversation_id="trace_1", start_turn_index=1),
        ],
        num_traces=5,
        turns_per_trace=4,
        issuer=issuer,
        scheduler=LoopScheduler(),
        run=_run(**phase_fields),
        progress=progress,
    )
    strategy.config.artifact_dir = artifact_dir
    strategy.stop_checker.can_start_new_session.return_value = True
    strategy.lifecycle.started_at_perf_ns = time.perf_counter_ns()
    strategy.lifecycle.started_at_ns = time.time_ns()
    return strategy, scheduler, src


@pytest.mark.asyncio
async def test_open_loop_setup_raises_session_limit_to_ceiling() -> None:
    issuer = _issuer(AsyncMock(return_value=True))
    strategy, _, _ = _open_loop_strategy(issuer)
    await strategy.setup_phase()
    issuer.set_session_limit.assert_called_once_with(512)


@pytest.mark.asyncio
async def test_closed_loop_setup_keeps_lane_session_limit() -> None:
    issuer = _issuer(AsyncMock(return_value=True))
    strategy, _, _ = _open_loop_strategy(
        issuer,
    )
    strategy._session_arrival_config = None
    await strategy.setup_phase()
    issuer.set_session_limit.assert_not_called()


@pytest.mark.asyncio
async def test_open_loop_drained_lane_does_not_recycle() -> None:
    issuer = _issuer(AsyncMock(return_value=True))
    strategy, _, _ = _open_loop_strategy(issuer)
    await strategy.setup_phase()
    await strategy._dispatch_recycled_on_lane(0)
    issuer.issue_credit.assert_not_called()


@pytest.mark.asyncio
async def test_open_loop_arrivals_start_turn_zero_sessions_on_new_lanes() -> None:
    issued: list[TurnToSend] = []

    async def capture(turn: TurnToSend) -> bool:
        issued.append(turn)
        return True

    issuer = _issuer(capture)
    strategy, _, _ = _open_loop_strategy(issuer)
    scheduled = len(strategy._session_arrivals)
    assert scheduled > 0
    await strategy.setup_phase()
    await strategy.execute_phase()
    await strategy._session_arrival_task
    await _drain()

    warm, arrivals = issued[:2], issued[2:]
    assert sorted((t.conversation_id, t.turn_index) for t in warm) == [
        ("trace_0", 1),
        ("trace_1", 2),
    ]
    assert len(arrivals) == scheduled
    assert all(t.turn_index == 0 for t in arrivals)
    assert len({t.x_correlation_id for t in arrivals}) == scheduled
    # Arrival roots come from the dataset sampler in its order.
    assert [t.conversation_id for t in arrivals][:5] == [f"trace_{i}" for i in range(5)]
    lanes = [strategy._correlation_to_lane[t.x_correlation_id] for t in arrivals]
    assert lanes == list(range(2, 2 + scheduled))
    assert all(a.started and a.fired_ns is not None for a in strategy._session_arrivals)
    assert [a.x_correlation_id for a in strategy._session_arrivals] == [
        t.x_correlation_id for t in arrivals
    ]


@pytest.mark.asyncio
async def test_open_loop_session_turn_waits_for_previous_reply() -> None:
    issued: list[TurnToSend] = []

    async def capture(turn: TurnToSend) -> bool:
        issued.append(turn)
        return True

    issuer = _issuer(capture)
    strategy, scheduler, _ = _open_loop_strategy(issuer)
    strategy._session_arrivals = strategy._session_arrivals[:1]
    await strategy.setup_phase()
    await strategy.execute_phase()
    await strategy._session_arrival_task
    await _drain()

    first = issued[-1]
    assert first.turn_index == 0
    assert [t for t in issued if t.x_correlation_id == first.x_correlation_id] == [
        first
    ]

    await strategy.handle_credit_return(
        _make_credit(
            conversation_id=first.conversation_id,
            x_correlation_id=first.x_correlation_id,
            turn_index=0,
            num_turns=4,
        )
    )
    await _drain()
    follow_up = [t for t in issued if t.x_correlation_id == first.x_correlation_id]
    assert [t.turn_index for t in follow_up] == [0, 1]


@pytest.mark.asyncio
async def test_open_loop_held_arrival_does_not_delay_later_arrivals() -> None:
    issued: list[TurnToSend] = []
    slot = asyncio.Event()

    async def capture(turn: TurnToSend) -> bool:
        issued.append(turn)
        if turn.turn_index == 0 and sum(t.turn_index == 0 for t in issued) == 1:
            await slot.wait()
        return True

    issuer = _issuer(capture)
    issuer.can_acquire_and_start_new_session.side_effect = [False] + [True] * 1000
    strategy, _, _ = _open_loop_strategy(issuer)
    arrivals = strategy._session_arrivals
    assert len(arrivals) >= 3
    await strategy.setup_phase()
    await strategy.execute_phase()
    await strategy._session_arrival_task
    await _drain()

    assert arrivals[0].ceiling_full is True
    assert arrivals[0].started is False
    assert all(a.started for a in arrivals[1:])
    assert all(a.fired_ns is not None for a in arrivals)

    slot.set()
    await _drain()
    assert arrivals[0].started is True
    assert arrivals[0].issued_ns is not None


@pytest.mark.asyncio
async def test_open_loop_finalize_writes_the_journal(tmp_path) -> None:
    issuer = _issuer(AsyncMock(return_value=True))
    strategy, _, _ = _open_loop_strategy(issuer, artifact_dir=tmp_path)
    await strategy.setup_phase()
    await strategy.execute_phase()
    await strategy._session_arrival_task
    await _drain()
    await strategy.finalize_phase()

    journal = orjson.loads((tmp_path / SESSION_ARRIVALS_FILE).read_bytes())
    assert journal["config"]["rate_per_sec"] == 0.2
    assert journal["config"]["ceiling"] == 512
    assert journal["initial_sessions"] == 2
    assert journal["duration_sec"] == 60.0
    assert journal["summary"]["scheduled"] == len(strategy._session_arrivals)
    assert journal["summary"]["started"] == journal["summary"]["scheduled"]
    assert journal["summary"]["held_back"] is False


@pytest.mark.asyncio
async def test_open_loop_arrivals_released_together_each_see_the_slots_left() -> None:
    """Arrivals due at once (a gamma burst, or a loop behind schedule) are
    released without a yield between them. Each must be checked against the
    slots the earlier ones took, not the slots free when the burst began."""
    free = {"slots": 1}
    blocked = asyncio.Event()

    async def capture(turn: TurnToSend) -> bool:
        if turn.turn_index == 0:  # arrivals; warmed lanes resume past turn 0
            if free["slots"] == 0:
                await blocked.wait()
            else:
                free["slots"] -= 1
        return True

    issuer = _issuer(capture)
    issuer.can_acquire_and_start_new_session.side_effect = lambda: free["slots"] > 0
    strategy, _, _ = _open_loop_strategy(issuer)
    strategy._session_arrivals = [
        SessionArrival(index=index, scheduled_offset_sec=0.0) for index in range(3)
    ]
    await strategy.setup_phase()
    await strategy.execute_phase()
    await strategy._session_arrival_task
    await _drain()

    assert [a.ceiling_full for a in strategy._session_arrivals] == [False, True, True]
    blocked.set()
    await _drain()


@pytest.mark.asyncio
async def test_open_loop_idle_cap_still_rechecks_when_replay_work_drains(
    time_traveler_no_patch_sleep,
) -> None:
    """The arrival loop waits for the whole phase. If the replay scheduler
    tracked it, the scheduler would never drain, so the system-idle cap would
    skip the recheck after a capped timer's turn is held by a replay barrier,
    and a later in-session turn would stay as far out as its recorded gap."""
    progress = MagicMock()
    progress.in_flight = 0
    issuer = _issuer(AsyncMock(return_value=True))
    strategy, scheduler, _ = _open_loop_strategy(
        issuer, progress=progress, system_idle_gap_cap_seconds=0.05
    )
    strategy._session_arrivals = [SessionArrival(index=0, scheduled_offset_sec=1e3)]
    await strategy.setup_phase()
    await strategy.execute_phase()
    issued_at: list[float] = []

    async def held_by_barrier() -> None:
        return None

    async def far_turn() -> None:
        issued_at.append(time.monotonic())

    scheduler.schedule_later(0.1, held_by_barrier())
    scheduler.schedule_later(2_000.0, far_turn())
    idle_started_at = time.monotonic()
    strategy.enforce_system_idle_cap(in_flight_requests=0)
    await asyncio.sleep(0.15)

    assert len(issued_at) == 1
    assert issued_at[0] - idle_started_at == pytest.approx(0.05, abs=0.02)
    await strategy.finalize_phase()
    assert strategy._session_arrival_task.cancelled()


@pytest.mark.asyncio
async def test_open_loop_schedule_counts_from_the_phase_start() -> None:
    """Arrivals already due when the warmed lanes have dispatched are released
    at once, and their lag counts from the phase start the duration stop uses,
    so the schedule cannot run past the phase end."""
    issuer = _issuer(AsyncMock(return_value=True))
    strategy, _, _ = _open_loop_strategy(issuer)
    strategy.lifecycle.started_at_perf_ns -= 5 * 10**9
    strategy.lifecycle.started_at_ns -= 5 * 10**9
    strategy._session_arrivals = [SessionArrival(index=0, scheduled_offset_sec=1.0)]
    await strategy.setup_phase()
    await strategy.execute_phase()
    await strategy._session_arrival_task
    await _drain()

    arrival = strategy._session_arrivals[0]
    assert strategy._session_arrival_start_ns == strategy.lifecycle.started_at_ns
    scheduled_ns = strategy.lifecycle.started_at_ns + 10**9
    assert arrival.fired_ns - scheduled_ns > 3.9 * 10**9


@pytest.mark.asyncio
async def test_open_loop_phase_stop_leaves_the_rest_unfired_and_unstarted(
    tmp_path,
) -> None:
    """When the phase stops starting sessions, the loop releases nothing more,
    an arrival still waiting to start is cancelled with the phase's other
    replay work, and the journal records both as not started."""
    held = asyncio.Event()

    async def capture(turn: TurnToSend) -> bool:
        if turn.turn_index == 0 and not held.is_set():
            held.set()
            await asyncio.Event().wait()
        return True

    issuer = _issuer(capture)
    strategy, scheduler, _ = _open_loop_strategy(issuer, artifact_dir=tmp_path)
    arrivals = [
        SessionArrival(index=i, scheduled_offset_sec=offset)
        for i, offset in enumerate((0.0, 1e3, 1e3))
    ]
    strategy._session_arrivals = arrivals
    strategy.stop_checker.can_start_new_session.side_effect = lambda: (
        sum(a.fired_ns is not None for a in arrivals) < 2
    )
    await strategy.setup_phase()
    await strategy.execute_phase()
    await strategy._session_arrival_task
    await _drain()

    # Arrival 0 is waiting inside issue_credit; arrival 1 fired as the phase
    # stopped, so it was refused, which is not a ceiling wait; 2 never fired.
    assert held.is_set()
    assert [a.fired_ns is not None for a in arrivals] == [True, True, False]
    assert [a.started for a in arrivals] == [False, False, False]
    assert [a.ceiling_full for a in arrivals] == [False, False, None]
    for task in scheduler.cancel_all():
        with contextlib.suppress(asyncio.CancelledError):
            await task
    await strategy.finalize_phase()

    journal = orjson.loads((tmp_path / SESSION_ARRIVALS_FILE).read_bytes())
    summary = journal["summary"]
    assert (summary["fired"], summary["started"], summary["not_started"]) == (2, 0, 3)
    assert summary["ceiling_waits"] == 0
    assert journal["arrivals"][0]["issued_ns"] is None
    assert journal["arrivals"][1]["issued_ns"] is not None
