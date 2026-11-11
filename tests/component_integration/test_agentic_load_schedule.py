# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scheduled AgentX admission preserves baseline trees and drains retired HTTP work."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from aiperf.common.enums import ConversationBranchMode, CreditPhase
from aiperf.common.loop_scheduler import LoopScheduler
from aiperf.common.models import (
    ConversationBranchInfo,
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
)
from aiperf.config.flags import CLIConfig
from aiperf.config.flags.resolver import resolve_config
from aiperf.config.phases import ConcurrencyPhase
from aiperf.credit.callback_handler import CreditCallbackHandler
from aiperf.credit.issuer import CreditIssuer
from aiperf.credit.messages import CreditReturn
from aiperf.credit.structs import Credit
from aiperf.dataset.dataset_samplers import SequentialSampler
from aiperf.plugin.enums import DatasetSamplingStrategy, TimingMode
from aiperf.timing.branch_orchestrator import BranchOrchestrator
from aiperf.timing.concurrency import ConcurrencyManager
from aiperf.timing.config import CreditPhaseConfig, _build_profiling_config
from aiperf.timing.phase.lifecycle import PhaseLifecycle
from aiperf.timing.phase.progress_tracker import PhaseProgressTracker
from aiperf.timing.phase.stop_conditions import StopConditionChecker
from aiperf.timing.request_cancellation import RequestCancellationConfig
from aiperf.timing.session_tree import SessionTreeRegistry
from aiperf.timing.strategies.agentic_replay import AgenticReplayStrategy
from aiperf.timing.trajectory_source import TrajectorySource

pytestmark = pytest.mark.component_integration


def test_schedule_validates_order_duration_and_baseline() -> None:
    phase = ConcurrencyPhase(
        name="profiling",
        type="concurrency",
        concurrency=24,
        duration=1800,
        timing_mode=TimingMode.AGENTIC_REPLAY,
        agentic_concurrency_schedule="900:96,1200:24",
    )
    assert [
        (step.at_seconds, step.concurrency)
        for step in phase.agentic_concurrency_schedule
    ] == [(900, 96), (1200, 24)]
    for schedule in ("1200:96,900:24", "900:12", "1800:96", "nan:96", "900:0"):
        with pytest.raises(ValidationError):
            ConcurrencyPhase(
                name="profiling",
                type="concurrency",
                concurrency=24,
                duration=1800,
                timing_mode=TimingMode.AGENTIC_REPLAY,
                agentic_concurrency_schedule=schedule,
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("root_pending", [True, False])
async def test_retired_tree_releases_only_after_dispatched_requests_return(
    root_pending: bool,
) -> None:
    manager = ConcurrencyManager()
    manager.configure_for_phase(CreditPhase.PROFILING, 2, None)
    registry = SessionTreeRegistry(manager)
    assert await manager.acquire_session_slot(CreditPhase.PROFILING, lambda: True)
    registry.open_tree("extra", CreditPhase.PROFILING, root_pending=root_pending)
    registry.register_descendants("extra", 3)
    registry.on_request_issued("extra")
    registry.on_request_issued("extra")
    registry.retire_tree("extra")
    assert registry.open_count() == 1
    registry.on_request_returned("extra")
    assert registry.open_count() == 1
    registry.on_request_returned("extra")
    assert registry.open_count() == 0
    assert manager.get_session_stats().release_count == 1


@pytest.mark.asyncio
async def test_schedule_keeps_baseline_identity_and_capacity_during_extra_drain() -> (
    None
):
    metadata = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id=f"trace-{index}", turns=[TurnMetadata(), TurnMetadata()]
            )
            for index in range(100)
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    source = TrajectorySource(
        dataset_metadata=metadata,
        dataset_sampler=SequentialSampler(
            [c.conversation_id for c in metadata.conversations]
        ),
        concurrency=24,
        random_seed=42,
    )
    manager = ConcurrencyManager()
    manager.configure_for_phase(CreditPhase.PROFILING, 24, None)
    registry = SessionTreeRegistry(manager)
    issuer = MagicMock()
    issuer.issue_credit = AsyncMock(return_value=True)
    strategy = AgenticReplayStrategy(
        config=CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.AGENTIC_REPLAY,
            concurrency=24,
            expected_duration_sec=1800,
            agentic_concurrency_schedule=[
                {"at_seconds": 900, "concurrency": 96},
                {"at_seconds": 1200, "concurrency": 24},
            ],
        ),
        conversation_source=source,
        scheduler=MagicMock(),
        stop_checker=MagicMock(),
        credit_issuer=issuer,
        lifecycle=MagicMock(),
        session_tree_registry=registry,
        concurrency_manager=manager,
    )
    identities = [(t.conversation_id, t.x_correlation_id) for t in source.trajectories]
    await strategy.setup_phase()
    await strategy._apply_concurrency_target(96)
    assert issuer.issue_credit.await_count == 72
    extras = [call.args[0] for call in issuer.issue_credit.await_args_list]
    for turn in extras:
        assert await manager.acquire_session_slot(CreditPhase.PROFILING, lambda: True)
        registry.open_tree(
            turn.effective_root_correlation_id, CreditPhase.PROFILING, root_pending=True
        )
        registry.on_request_issued(turn.effective_root_correlation_id)
    await strategy._apply_concurrency_target(24)
    assert [
        (t.conversation_id, t.x_correlation_id) for t in source.trajectories
    ] == identities
    assert all(
        strategy.is_retired_root(turn.effective_root_correlation_id) for turn in extras
    )
    assert manager.session_slot_available(CreditPhase.PROFILING), (
        "baseline replacements must not wait for extra HTTP drain"
    )
    assert registry.open_count() == 72
    for turn in extras:
        registry.on_request_returned(turn.effective_root_correlation_id)
    assert registry.open_count() == 0
    assert issuer.issue_credit.await_count == 72, "retired lanes must not recycle"


@pytest.mark.asyncio
@pytest.mark.parametrize("delayed_child", [False, True])
async def test_real_credit_path_stops_extra_branches_and_keeps_baseline_turns(
    delayed_child: bool,
) -> None:
    """Use real admission, callback, branch and slot owners; the router captures credits."""
    roots = [
        ConversationMetadata(
            conversation_id=f"root-{index}",
            turns=[
                TurnMetadata(branch_ids=["background"]),
                TurnMetadata(),
                TurnMetadata(),
            ],
            branches=[
                ConversationBranchInfo(
                    branch_id="background",
                    child_conversation_ids=["child"],
                    mode=ConversationBranchMode.SPAWN,
                    is_background=True,
                    start_timestamp_ms=0,
                )
            ],
        )
        for index in range(4)
    ]
    child = ConversationMetadata(
        conversation_id="child",
        turns=[
            TurnMetadata(timestamp_ms=60000 if delayed_child else None),
            TurnMetadata(),
        ],
        is_root=False,
        agent_depth=1,
        parent_conversation_id="root-0",
    )
    metadata = DatasetMetadata(
        conversations=[*roots, child],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    source = TrajectorySource(
        dataset_metadata=metadata,
        dataset_sampler=SequentialSampler([r.conversation_id for r in roots]),
        concurrency=1,
        random_seed=42,
    )
    config = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.AGENTIC_REPLAY,
        concurrency=1,
        expected_duration_sec=1800,
        agentic_concurrency_schedule="900:3,1200:1",
    )
    manager = ConcurrencyManager()
    manager.configure_for_phase(CreditPhase.PROFILING, 1, None)
    registry = SessionTreeRegistry(manager)
    lifecycle = PhaseLifecycle(config)
    lifecycle.start()
    progress = PhaseProgressTracker(config)
    stop = StopConditionChecker(config, lifecycle, progress.counter)
    credits: list[Credit] = []
    router = MagicMock()
    router.send_credit = AsyncMock(side_effect=lambda *, credit: credits.append(credit))
    policy = MagicMock()
    policy.next_cancellation_delay_ns.return_value = None
    issuer = CreditIssuer(
        phase=CreditPhase.PROFILING,
        stop_checker=stop,
        progress=progress,
        concurrency_manager=manager,
        credit_router=router,
        cancellation_policy=policy,
        lifecycle=lifecycle,
        session_tree_registry=registry,
    )
    scheduler = LoopScheduler()
    branches = BranchOrchestrator(
        source, issuer, session_tree_registry=registry, scheduler=scheduler
    )
    strategy = AgenticReplayStrategy(
        config=config,
        conversation_source=source,
        scheduler=scheduler,
        stop_checker=stop,
        credit_issuer=issuer,
        lifecycle=lifecycle,
        concurrency_manager=manager,
        session_tree_registry=registry,
        branch_orchestrator=branches,
        progress=progress,
    )
    callback = CreditCallbackHandler(
        manager, branch_orchestrator=branches, session_tree_registry=registry
    )
    callback.register_phase(
        phase=CreditPhase.PROFILING,
        progress=progress,
        lifecycle=lifecycle,
        stop_checker=stop,
        strategy=strategy,
    )
    await strategy.setup_phase()
    await strategy._dispatch_recycled_on_lane(0)
    baseline = credits[0]
    await strategy._apply_concurrency_target(3)
    extra = credits[1]
    await callback.on_credit_return(
        "worker", CreditReturn(credit=extra, cancelled=False, first_token_sent=False)
    )
    if delayed_child:
        assert scheduler.pending_count == 1
    else:
        assert any(
            c.agent_depth == 1
            and c.effective_root_correlation_id == extra.effective_root_correlation_id
            for c in credits
        )
    await strategy._apply_concurrency_target(1)
    retired_credits = [
        c
        for c in credits
        if strategy.is_retired_root(c.effective_root_correlation_id)
        and c.id != extra.id
    ]
    count_at_cutoff = len(credits)
    for credit in retired_credits:
        await callback.on_credit_return(
            "worker",
            CreditReturn(credit=credit, cancelled=False, first_token_sent=False),
        )
    assert len(credits) == count_at_cutoff
    assert registry.open_count() == 1, "only the continuous baseline tree remains"
    assert not branches.has_pending_branch_work()
    await callback.on_credit_return(
        "worker", CreditReturn(credit=baseline, cancelled=False, first_token_sent=False)
    )
    assert any(
        c.x_correlation_id == baseline.x_correlation_id and c.turn_index == 1
        for c in credits
    )
    scheduler.cancel_all()
    branches.cleanup()
    registry.release_all(CreditPhase.PROFILING)


def test_schedule_control_timers_follow_wall_clock_and_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = AgenticReplayStrategy.__new__(AgenticReplayStrategy)
    strategy._load_schedule_enabled = True
    strategy._load_schedule = ConcurrencyPhase(
        name="profiling",
        type="concurrency",
        concurrency=24,
        duration=1800,
        agentic_concurrency_schedule="900:96,1200:24",
    ).agentic_concurrency_schedule
    strategy._load_timers = []
    strategy._load_schedule_closed = False
    strategy.lifecycle = MagicMock(started_at_perf_sec=100.0)
    loop = MagicMock()
    monkeypatch.setattr(
        "aiperf.timing.strategies.agentic_replay.asyncio.get_running_loop", lambda: loop
    )
    monkeypatch.setattr(
        "aiperf.timing.strategies.agentic_replay.time.perf_counter", lambda: 150.0
    )
    strategy._start_load_schedule()
    assert [call.args[0] for call in loop.call_later.call_args_list] == [850.0, 1150.0]
    strategy.stop_load_schedule()
    assert loop.call_later.return_value.cancel.call_count == 2
    assert strategy._load_timers == []


def test_cli_schedule_reaches_native_profiling_config() -> None:
    resolved = resolve_config(
        CLIConfig(
            model_names=["example"],
            concurrency=24,
            benchmark_duration=1800,
            agentic_concurrency_schedule="900:96,1200:24",
        )
    )
    phase = resolved.benchmark.get_profiling_phases()[0]
    phase.timing_mode = TimingMode.AGENTIC_REPLAY
    config = _build_profiling_config(
        phase,
        default_cancellation=RequestCancellationConfig(),
        phase_index=0,
        profiling_index=0,
    )
    assert [
        (step.at_seconds, step.concurrency)
        for step in config.agentic_concurrency_schedule
    ] == [(900, 96), (1200, 24)]


def test_peak_schedule_concurrency_obeys_dataset_wrap_policy() -> None:
    metadata = DatasetMetadata(
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        conversations=[
            ConversationMetadata(
                conversation_id="only-root", turns=[TurnMetadata(), TurnMetadata()]
            )
        ],
    )
    with pytest.raises(ValueError, match="concurrency 96 exceeds"):
        TrajectorySource(
            dataset_metadata=metadata,
            dataset_sampler=SequentialSampler(["only-root"]),
            concurrency=24,
            max_concurrency=96,
            random_seed=42,
            expected_duration_sec=1800,
        )
