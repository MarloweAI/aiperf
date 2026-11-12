# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A subagent turn still waiting on its replay timer at the duration cutoff
must not hold the AgentX profiling phase open for the grace period."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import ConversationBranchMode, CreditPhase
from aiperf.common.models import (
    ConversationBranchInfo,
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
)
from aiperf.credit.callback_handler import CreditCallbackHandler
from aiperf.credit.messages import CreditReturn
from aiperf.dataset.dataset_samplers import SequentialSampler
from aiperf.plugin.enums import DatasetSamplingStrategy, TimingMode
from aiperf.timing.concurrency import ConcurrencyManager
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.runner import PhaseRunner
from aiperf.timing.session_tree import SessionTreeRegistry
from aiperf.timing.trajectory_source import TrajectorySource

pytestmark = pytest.mark.looptime

DURATION_SEC = 30.0
GRACE_SEC = 600.0
REQUEST_SEC = 0.5
TIMER_PAST_CUTOFF_MS = 60_000.0
# After the sampled start (t* = 0), so the child is spawned by the root's
# first return rather than resumed as a live stream.
SPAWN_MS = 5_000.0


def _dataset(pending_turn: str) -> DatasetMetadata:
    """One root that spawns one background subagent whose next request is
    recorded after the cutoff: its first request (``first``) or its second
    (``continuation``)."""
    if pending_turn == "first":
        child_turns = [
            TurnMetadata(timestamp_ms=SPAWN_MS + TIMER_PAST_CUTOFF_MS),
            TurnMetadata(),
        ]
    else:
        child_turns = [
            TurnMetadata(timestamp_ms=SPAWN_MS),
            TurnMetadata(delay_ms=TIMER_PAST_CUTOFF_MS),
        ]
    root = ConversationMetadata(
        conversation_id="root",
        turns=[TurnMetadata(timestamp_ms=0.0, branch_ids=["bg"]), TurnMetadata()],
        branches=[
            ConversationBranchInfo(
                branch_id="bg",
                child_conversation_ids=["child"],
                mode=ConversationBranchMode.SPAWN,
                is_background=True,
                start_timestamp_ms=SPAWN_MS,
            )
        ],
    )
    child = ConversationMetadata(
        conversation_id="child",
        turns=child_turns,
        is_root=False,
        agent_depth=1,
        parent_conversation_id="root",
    )
    return DatasetMetadata(
        conversations=[root, child],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


class _Worker:
    """Credit router whose worker answers every request after REQUEST_SEC."""

    def __init__(self) -> None:
        self.callback: CreditCallbackHandler | None = None
        self.sent: list = []
        self.returned_at: list[float] = []
        self._tasks: set[asyncio.Task] = set()
        self.cancel_all_credits = AsyncMock()
        self.wait_for_workers = AsyncMock()
        self.mark_credits_complete = MagicMock()
        self.sticky_router = None

    async def send_credit(self, *, credit) -> None:
        self.sent.append(credit)
        task = asyncio.get_running_loop().create_task(self._answer(credit))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _answer(self, credit) -> None:
        await asyncio.sleep(REQUEST_SEC)
        self.returned_at.append(asyncio.get_running_loop().time())
        await self.callback.on_credit_return(
            "worker",
            CreditReturn(credit=credit, cancelled=False, first_token_sent=False),
        )


def _publisher() -> MagicMock:
    publisher = MagicMock()
    for name in (
        "publish_phase_start",
        "publish_phase_baseline_request",
        "publish_phase_sending_complete",
        "publish_phase_complete",
        "publish_progress",
        "publish_credits_complete",
    ):
        setattr(publisher, name, AsyncMock())
    return publisher


@pytest.mark.parametrize("pending_turn", ["first", "continuation"])
async def test_subagent_timer_cancelled_at_cutoff_does_not_wait_out_grace(
    pending_turn: str, time_traveler_no_patch_sleep, caplog: pytest.LogCaptureFixture
) -> None:
    metadata = _dataset(pending_turn)
    source = TrajectorySource(
        dataset_metadata=metadata,
        dataset_sampler=SequentialSampler(["root"]),
        concurrency=1,
        random_seed=42,
        start_min_ratio=0.0,
        start_max_ratio=0.0,
        allow_dataset_wrap=True,
    )
    config = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.AGENTIC_REPLAY,
        concurrency=1,
        expected_duration_sec=DURATION_SEC,
        grace_period_sec=GRACE_SEC,
    )
    manager = ConcurrencyManager()
    registry = SessionTreeRegistry(manager)
    worker = _Worker()
    callback = CreditCallbackHandler(manager, session_tree_registry=registry)
    worker.callback = callback
    cancel_policy = MagicMock()
    cancel_policy.next_cancellation_delay_ns.return_value = None
    runner = PhaseRunner(
        config=config,
        conversation_source=source,
        phase_publisher=_publisher(),
        credit_router=worker,
        concurrency_manager=manager,
        cancellation_policy=cancel_policy,
        callback_handler=callback,
        session_tree_registry=registry,
    )
    loop = asyncio.get_running_loop()
    caplog.set_level(logging.WARNING, logger="aiperf.timing.branch_orchestrator")

    started = loop.time()
    stats = await runner.run(is_final_phase=True)
    finished = loop.time()

    child_requests = [c for c in worker.sent if c.agent_depth > 0]
    expected_child_requests = 0 if pending_turn == "first" else 1
    assert len(child_requests) == expected_child_requests
    assert stats.timeout_triggered
    assert not stats.grace_period_timeout_triggered
    assert worker.returned_at, "the replay must have sent requests"
    assert finished - started < DURATION_SEC + REQUEST_SEC + 1.0, (
        "the phase must end at the cutoff, not after the grace period"
    )
    last_return = max(worker.returned_at) - started
    assert (stats.requests_end_ns - stats.start_ns) / 1e9 == pytest.approx(
        max(DURATION_SEC, last_return), abs=0.01
    ), "the reported end is the cutoff or the last return, never the grace end"
    assert not any("leaked state" in r.getMessage() for r in caplog.records)
