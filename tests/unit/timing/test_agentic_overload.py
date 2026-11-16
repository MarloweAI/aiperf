# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Virtual-time full phase: independent roots, child departure and cutoff drain."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import orjson
import pytest

from aiperf.common.enums import CacheBustTarget, CreditPhase
from aiperf.common.models import ConversationMetadata, DatasetMetadata, TurnMetadata
from aiperf.config.phases import SessionArrivalsConfig
from aiperf.credit.callback_handler import CreditCallbackHandler
from aiperf.credit.messages import CreditReturn
from aiperf.dataset.dataset_samplers import SequentialSampler
from aiperf.plugin.enums import TimingMode
from aiperf.timing.concurrency import ConcurrencyManager
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.runner import PhaseRunner
from aiperf.timing.session_tree import SessionTreeRegistry
from aiperf.timing.trajectory_source import TrajectorySource
from tests.unit.timing.test_agentic_cutoff_drain import _dataset, _publisher

pytestmark = pytest.mark.looptime


class Worker:
    def __init__(self, departure: bool, delay: float) -> None:
        self.departure = departure
        self.delay = delay
        self.callback = None
        self.sent = []
        self.returned = []
        self.tasks = set()
        self.cancel_all_credits = AsyncMock()
        self.wait_for_workers = AsyncMock()
        self.mark_credits_complete = lambda: None
        self.sticky_router = None

    async def send_credit(self, *, credit) -> None:
        self.sent.append((asyncio.get_running_loop().time(), credit))
        task = asyncio.create_task(self.answer(credit))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def answer(self, credit) -> None:
        loop = asyncio.get_running_loop()
        completed = loop.create_future()
        loop.call_later(self.delay, completed.set_result, None)
        await completed
        self.returned.append(credit)
        leave = self.departure and credit.agent_depth > 0
        await self.callback.on_credit_return(
            "worker",
            CreditReturn(
                credit=credit,
                error="HTTP 429" if leave else None,
                provider_departure=leave,
            ),
        )


async def run_phase(
    tmp_path, *, departure: bool, delay: float, long_root: bool = False
):
    metadata = _dataset("continuation")
    if long_root:
        metadata = DatasetMetadata(
            sampling_strategy="sequential",
            conversations=[
                ConversationMetadata(
                    conversation_id="root",
                    turns=[
                        TurnMetadata(timestamp_ms=0),
                        TurnMetadata(timestamp_ms=60_000, delay_ms=60_000),
                    ],
                )
            ],
        )
    source = TrajectorySource(
        dataset_metadata=metadata,
        dataset_sampler=SequentialSampler(["root"]),
        concurrency=1,
        random_seed=42,
        start_min_ratio=0,
        start_max_ratio=0,
        allow_dataset_wrap=True,
    )
    arrivals = SessionArrivalsConfig(
        rate=0.2,
        ceiling=100,
        seed=42,
        rate_schedule=[
            {"at_seconds": 10, "rate": 0.6},
            {"at_seconds": 20, "rate": 0.2},
        ],
    )
    profile = SimpleNamespace(duration=30, session_arrivals=arrivals)
    run = SimpleNamespace(
        benchmark_id=tmp_path.name,
        random_seed=42,
        cfg=SimpleNamespace(
            get_profiling_phases=lambda: [profile],
            get_cache_bust_target=lambda: CacheBustTarget.FIRST_TURN_PREFIX,
            get_default_dataset=lambda: None,
        ),
    )
    config = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.AGENTIC_REPLAY,
        concurrency=1,
        expected_duration_sec=30,
        grace_period_sec=60,
        artifact_dir=tmp_path,
    )
    manager = ConcurrencyManager()
    registry = SessionTreeRegistry(manager)
    callback = CreditCallbackHandler(manager, session_tree_registry=registry)
    worker = Worker(departure, delay)
    worker.callback = callback
    runner = PhaseRunner(
        config=config,
        conversation_source=source,
        phase_publisher=_publisher(),
        credit_router=worker,
        concurrency_manager=manager,
        callback_handler=callback,
        session_tree_registry=registry,
        run=run,
    )
    stats = await runner.run(is_final_phase=True)
    return (
        stats,
        worker,
        registry,
        orjson.loads((tmp_path / "session_arrivals.json").read_bytes()),
    )


@pytest.mark.parametrize("delay", [0.5, 3.0])
async def test_arrival_manifest_independent_of_child_departure_and_response_speed(
    tmp_path, delay, time_traveler_no_patch_sleep, monkeypatch
) -> None:
    async def inline_io(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", inline_io)
    a = tmp_path / "strict"
    b = tmp_path / "depart"
    a.mkdir()
    b.mkdir()
    strict, strict_worker, _, sj = await run_phase(a, departure=False, delay=0.5)
    departed, worker, registry, dj = await run_phase(b, departure=True, delay=delay)
    assert not strict.grace_period_timeout_triggered
    assert not departed.grace_period_timeout_triggered
    assert sj["complete"] and dj["complete"]
    assert (
        strict_worker.sent[0][1].cache_bust_marker
        != worker.sent[0][1].cache_bust_marker
    )
    assert sj["schedule_sha256"] == dj["schedule_sha256"]
    assert sj["assignments_sha256"] == dj["assignments_sha256"]
    assert sj["summary"]["scheduled"] == dj["summary"]["started"]
    assert registry.open_count() == 0
    events = [e for e in dj["root_events"] if e["event"] == "departed"]
    assert events
    assert len({e["root_correlation_id"] for e in events}) == len(events)
    for event in events:
        root = event["root_correlation_id"]
        attempts = [
            credit
            for _, credit in worker.sent
            if credit.effective_root_correlation_id == root
        ]
        assert sum(c.agent_depth > 0 for c in attempts) == 1
    assert len(strict_worker.sent) >= len(worker.sent)


async def test_long_unfinished_roots_have_complete_cutoff_census_after_http_drain(
    tmp_path, time_traveler_no_patch_sleep, monkeypatch
):
    async def inline_io(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", inline_io)
    stats, worker, registry, journal = await run_phase(
        tmp_path, departure=False, delay=0.5, long_root=True
    )
    assert not stats.grace_period_timeout_triggered
    assert stats.requests_completed == len(worker.sent)
    assert registry.open_count() == 0
    assert journal["complete"]
    started = {
        e["root_correlation_id"]
        for e in journal["root_events"]
        if e["event"] == "started"
    }
    censored = {
        e["root_correlation_id"]
        for e in journal["root_events"]
        if e["event"] == "censored" and e["reason"] == "phase_cutoff"
    }
    assert started == censored
    assert all(credit.turn_index == 0 for _, credit in worker.sent)
