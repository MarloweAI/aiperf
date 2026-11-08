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
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest

from aiperf.common.enums import CacheBustTarget, CreditPhase
from aiperf.common.loop_scheduler import LoopScheduler
from aiperf.config.phases import SessionArrivalsConfig
from aiperf.credit.structs import TurnToSend
from aiperf.timing.session_arrivals import SESSION_ARRIVALS_FILE
from aiperf.timing.trajectory_source import Trajectory
from tests.unit.conftest import _REAL_SLEEP
from tests.unit.timing.strategies.test_agentic_replay import (
    _make_credit,
    _make_strategy,
)


def _run(**phase_fields: object) -> MagicMock:
    phase = SimpleNamespace(
        burst_phase_starts=False,
        system_idle_gap_cap_seconds=None,
        duration=60.0,
        session_arrivals=SessionArrivalsConfig(rate=0.2, ceiling=512),
        **phase_fields,
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


def _open_loop_strategy(issuer: AsyncMock, *, artifact_dir=None, **phase_fields):
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
    )
    strategy.config.artifact_dir = artifact_dir
    strategy.stop_checker.can_start_new_session.return_value = True
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
