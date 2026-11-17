# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mixed cohorts must retain whole-tree baseline lanes without coupling arrivals
to responses, replacing departed roots, or losing recycle/cutoff census events.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import orjson
import pytest

from aiperf.common.enums import ConversationBranchMode, CreditPhase
from aiperf.common.models import (
    ConversationBranchInfo,
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
)
from aiperf.config import BenchmarkConfig, BenchmarkRun
from aiperf.config.phases import SessionArrivalsConfig
from aiperf.credit.callback_handler import CreditCallbackHandler
from aiperf.credit.messages import CreditReturn
from aiperf.dataset.dataset_samplers import SequentialSampler
from aiperf.plugin.enums import DatasetSamplingStrategy, TimingMode
from aiperf.timing.concurrency import ConcurrencyManager
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.runner import PhaseRunner
from aiperf.timing.session_arrivals import build_arrival_offsets
from aiperf.timing.session_tree import SessionTreeRegistry
from aiperf.timing.trajectory_source import TrajectorySource
from tests.unit.timing.strategies._shared_helpers import (
    _build_real_trajectory_source,
    _make_dataset,
)
from tests.unit.timing.test_agentic_cutoff_drain import _publisher, _Worker

pytestmark = pytest.mark.looptime


def test_precomputed_burst_assignments_do_not_consume_baseline_sampler() -> None:
    dataset = _make_dataset(num_traces=5, turns_per_trace=3)
    source = _build_real_trajectory_source(dataset=dataset, trajectories=[])
    assigned = source.precompute_arrival_conversation_ids(8)
    assert assigned == [f"trace_{i % 5}" for i in range(8)]
    assert [source.next_recycle_conversation_id() for _ in range(11)] == [
        f"trace_{i % 5}" for i in range(11)
    ]
    assert assigned == [f"trace_{i % 5}" for i in range(8)]


@pytest.mark.parametrize("multiplier", [2, 3, 4])
def test_zero_rate_segments_preserve_seeded_burst_only_window(multiplier: int) -> None:
    config = SessionArrivalsConfig(
        rate=0,
        ceiling=4096,
        seed=42,
        rate_schedule=[
            {"at_seconds": 900, "rate": multiplier / 90},
            {"at_seconds": 1200, "rate": 0},
        ],
    )
    offsets = build_arrival_offsets(config, 1800)
    assert offsets
    assert all(900 < offset < 1200 for offset in offsets)
    assert offsets == build_arrival_offsets(config, 1800)


def _dataset() -> DatasetMetadata:
    return DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="root",
                turns=[TurnMetadata(timestamp_ms=0, branch_ids=["bg"]), TurnMetadata()],
                branches=[
                    ConversationBranchInfo(
                        branch_id="bg",
                        child_conversation_ids=["child"],
                        mode=ConversationBranchMode.SPAWN,
                        is_background=True,
                        start_timestamp_ms=1,
                    )
                ],
            ),
            ConversationMetadata(
                conversation_id="child",
                is_root=False,
                agent_depth=1,
                parent_conversation_id="root",
                turns=[TurnMetadata(timestamp_ms=1), TurnMetadata(delay_ms=2000)],
            ),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


class ObservedWorker(_Worker):
    def __init__(
        self, registry: SessionTreeRegistry, *, latency: float, depart: bool
    ) -> None:
        super().__init__()
        self.registry = registry
        self.latency = latency
        self.depart = depart
        self.departed: set[str] = set()
        self.dispatches: list[tuple[object, float, int]] = []
        self.returns: list[tuple[object, float]] = []

    async def send_credit(self, *, credit) -> None:
        self.dispatches.append(
            (
                credit,
                asyncio.get_running_loop().time(),
                self.registry.open_count(CreditPhase.PROFILING),
            )
        )
        await super().send_credit(credit=credit)

    async def _answer(self, credit) -> None:
        await asyncio.sleep(self.latency * (3 if credit.agent_depth else 1))
        root = credit.effective_root_correlation_id
        leaving = self.depart and credit.agent_depth > 0 and root not in self.departed
        if leaving:
            self.departed.add(root)
        self.returns.append((credit, asyncio.get_running_loop().time()))
        await self.callback.on_credit_return(
            "worker",
            CreditReturn(
                credit=credit,
                provider_departure=leaving,
                error="HTTP429" if leaving else None,
            ),
        )


async def _run(
    tmp_path: Path, *, arrivals: bool, latency: float = 0.25, depart: bool = False
) -> tuple[dict, ObservedWorker]:
    arrival_config = (
        {
            "rate": 0,
            "ceiling": 4096,
            "seed": 42,
            "rate_schedule": [
                {"at_seconds": 3, "rate": 4},
                {"at_seconds": 7, "rate": 0},
            ],
        }
        if arrivals
        else None
    )
    cfg = BenchmarkConfig.model_validate(
        {
            "models": ["fixture"],
            "endpoint": {"type": "completions", "urls": ["http://localhost:8000/v1"]},
            "datasets": [{"type": "synthetic", "name": "fixture"}],
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "concurrency": 16,
                    "duration": 12,
                    "timing_mode": "agentic_replay",
                    "session_maintain_baseline": True,
                    "session_arrivals": arrival_config,
                }
            ],
        }
    )
    run = BenchmarkRun(
        benchmark_id="mixed-fixture",
        cfg=cfg,
        artifact_dir=tmp_path,
        random_seed=42,
        variables={},
    )
    source = TrajectorySource(
        dataset_metadata=_dataset(),
        dataset_sampler=SequentialSampler(["root"]),
        concurrency=16,
        random_seed=42,
        start_min_ratio=0,
        start_max_ratio=0,
        allow_dataset_wrap=True,
    )
    config = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.AGENTIC_REPLAY,
        concurrency=16,
        expected_duration_sec=12,
        grace_period_sec=20,
        artifact_dir=tmp_path,
    )
    manager = ConcurrencyManager()
    registry = SessionTreeRegistry(manager)
    worker = ObservedWorker(registry, latency=latency, depart=depart)
    callback = CreditCallbackHandler(manager, session_tree_registry=registry)
    worker.callback = callback
    policy = MagicMock()
    policy.next_cancellation_delay_ns.return_value = None
    runner = PhaseRunner(
        config=config,
        conversation_source=source,
        phase_publisher=_publisher(),
        credit_router=worker,
        concurrency_manager=manager,
        cancellation_policy=policy,
        callback_handler=callback,
        session_tree_registry=registry,
        run=run,
    )
    stats = await runner.run(is_final_phase=True)
    assert not stats.grace_period_timeout_triggered
    assert stats.in_flight_requests == 0
    return orjson.loads((tmp_path / "session_arrivals.json").read_bytes()), worker


@pytest.mark.parametrize("arrivals", [False, True])
async def test_baseline_recycles_only_after_slow_descendants_and_census_reconciles(
    tmp_path: Path, arrivals: bool, time_traveler_no_patch_sleep
) -> None:
    journal, worker = await _run(tmp_path, arrivals=arrivals)
    assert journal["schema_version"] == 3
    assert journal["maintained_baseline"] is True
    assert journal["complete"] is True
    events = journal["root_events"]
    starts = {e["root_correlation_id"]: e for e in events if e["event"] == "started"}
    terminal = {
        e["root_correlation_id"]: e
        for e in events
        if e["event"] in {"drained", "departed", "censored"}
    }
    assert starts.keys() == terminal.keys()
    recycled = [e for e in starts.values() if e["origin"] == "recycle"]
    assert recycled
    assert len([e for e in starts.values() if e["origin"] == "initial"]) == 16
    assert (
        len([e for e in starts.values() if e["origin"] == "arrival"])
        == journal["summary"]["started"]
    )
    for event in recycled:
        previous = event["predecessor_root_id"]
        assert terminal[previous]["event"] == "drained"
        assert terminal[previous]["at_ns"] <= event["at_ns"]
        assert starts[previous]["lane"] == event["lane"]
        assert event["cohort"] == "baseline"
    active = set()
    peak_baseline = 0
    for event in events:
        if event["cohort"] != "baseline":
            continue
        root = event["root_correlation_id"]
        if event["event"] == "started":
            active.add(root)
        else:
            active.discard(root)
        peak_baseline = max(peak_baseline, len(active))
        assert len(active) <= 16
    assert peak_baseline == 16
    for event in recycled:
        previous = event["predecessor_root_id"]
        root_end = max(
            t
            for c, t in worker.returns
            if c.effective_root_correlation_id == previous and c.agent_depth == 0
        )
        child_end = max(
            t
            for c, t in worker.returns
            if c.effective_root_correlation_id == previous and c.agent_depth > 0
        )
        next_start = min(
            t
            for c, t, _ in worker.dispatches
            if c.effective_root_correlation_id == event["root_correlation_id"]
        )
        assert root_end < child_end <= next_start


async def test_slow_responses_do_not_change_extra_offsets_or_assignments(
    tmp_path: Path, time_traveler_no_patch_sleep
) -> None:
    fast_dir, slow_dir = tmp_path / "fast", tmp_path / "slow"
    fast_dir.mkdir()
    slow_dir.mkdir()
    fast, _ = await _run(fast_dir, arrivals=True, latency=0.25)
    slow, _ = await _run(slow_dir, arrivals=True, latency=0.75)
    assert fast["schedule_sha256"] == slow["schedule_sha256"]
    assert fast["assignments_sha256"] == slow["assignments_sha256"]
    assert fast["summary"]["started"] == slow["summary"]["started"] > 0
    assert all(3 < row["scheduled_offset_sec"] < 7 for row in fast["arrivals"])


@pytest.mark.parametrize("arrivals", [False, True])
async def test_child_429_retires_baseline_and_burst_without_replacement(
    tmp_path: Path, arrivals: bool, time_traveler_no_patch_sleep
) -> None:
    journal, worker = await _run(tmp_path, arrivals=arrivals, depart=True)
    events = journal["root_events"]
    assert worker.departed
    assert not [e for e in events if e["origin"] == "recycle"]
    assert {
        e["root_correlation_id"] for e in events if e["event"] == "departed"
    } == worker.departed
    assert all(not c.agent_depth or c.turn_index == 0 for c, _, _ in worker.dispatches)
    assert (
        len(
            [e for e in events if e["event"] == "started" and e["cohort"] == "baseline"]
        )
        == 16
    )
