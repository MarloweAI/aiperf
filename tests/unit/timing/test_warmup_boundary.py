# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the warmup boundary hook and where PhaseOrchestrator runs it."""

from __future__ import annotations

import asyncio
import contextlib
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import param

from aiperf.common.enums import CreditPhase, LifecycleState, ProfileCancelReason
from aiperf.common.messages import ProfileCancelCommand
from aiperf.config.warmup_boundary import (
    BOUNDARY_DONE_MARKER,
    WARMUP_DONE_MARKER,
    WarmupBoundaryHookConfig,
)
from aiperf.plugin.enums import ArrivalPattern, TimingMode
from aiperf.timing.config import TimingConfig
from aiperf.timing.phase.publisher import PhasePublisher
from aiperf.timing.phase_orchestrator import PhaseOrchestrator
from aiperf.timing.warmup_boundary import WarmupBoundaryHook, WarmupBoundaryHookError
from tests.unit.timing.conftest import (
    MockCreditRouter,
    make_dataset,
    make_phase_config,
)

_WARMUP_REQUESTS = 2
_PROFILING_REQUESTS = 3


def _write_script(path: Path, body: str) -> Path:
    path.write_text(f"set -e\n{body}\n")
    return path


def _hook_config(tmp_path: Path, body: str) -> WarmupBoundaryHookConfig:
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    return WarmupBoundaryHookConfig(
        script=_write_script(tmp_path / "hook.sh", body), marker_dir=marker_dir
    )


def _sleeping_script_body(pid_file: Path, child_file: Path) -> str:
    """A script that starts a child command and waits for it."""
    return f'echo $$ > "{pid_file}"\nsleep 60 &\necho $! > "{child_file}"\nwait'


async def _wait_for_pids(task: asyncio.Future, *files: Path) -> list[int]:
    for _ in range(500):
        if all(f.exists() and f.read_text().strip() for f in files):
            return [int(f.read_text()) for f in files]
        await _pause()
    raise AssertionError(f"the script did not write {files}")


async def _is_gone(pid: int) -> bool:
    """True when ``pid`` exits (a zombie waiting for init counts as exited)."""
    for _ in range(200):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        stat = Path(f"/proc/{pid}/stat")
        with contextlib.suppress(FileNotFoundError):
            if stat.read_text().split(") ", 1)[1].startswith("Z"):
                return True
        await _pause()
    return False


async def _pause() -> None:
    """Wait 10 ms of loop time. ``asyncio.sleep`` is a no-op in unit tests."""
    loop = asyncio.get_running_loop()
    done = loop.create_future()
    loop.call_later(0.01, done.set_result, None)
    await done


def _read_marker_ns(path: Path) -> int:
    text = path.read_text()
    assert text.endswith("\n")
    return int(text)


def _phase(phase: CreditPhase, requests: int, *, seamless: bool = False):
    return make_phase_config(
        phase=phase,
        timing_mode=TimingMode.REQUEST_RATE,
        request_count=requests,
        concurrency=1,
        arrival_pattern=ArrivalPattern.CONCURRENCY_BURST,
        seamless=seamless,
    )


def _orchestrator(
    hook: WarmupBoundaryHookConfig | None, router: MockCreditRouter
) -> PhaseOrchestrator:
    pub = MagicMock()
    pub.publish = AsyncMock()
    config = TimingConfig(
        phase_configs=[
            _phase(CreditPhase.WARMUP, _WARMUP_REQUESTS),
            _phase(CreditPhase.PROFILING, _PROFILING_REQUESTS),
        ],
        warmup_boundary_hook=hook,
    )
    return PhaseOrchestrator(
        config=config,
        phase_publisher=PhasePublisher(pub_client=pub, service_id="test"),
        credit_router=router,
        dataset_metadata=make_dataset(["c0", "c1", "c2", "c3"]),
        service_id="test-orchestrator",
    )


@pytest.mark.asyncio
class TestWarmupBoundaryHook:
    """The hook component alone, with real bash scripts."""

    async def test_run_writes_warmup_marker_before_script_and_boundary_after(
        self, tmp_path: Path
    ) -> None:
        seen = tmp_path / "seen"
        config = _hook_config(tmp_path, f'ls "{tmp_path / "markers"}" > "{seen}"')

        await WarmupBoundaryHook(config).run()

        assert seen.read_text().split() == [WARMUP_DONE_MARKER]
        warmup_ns = _read_marker_ns(config.marker_dir / WARMUP_DONE_MARKER)
        boundary_ns = _read_marker_ns(config.marker_dir / BOUNDARY_DONE_MARKER)
        assert 0 < warmup_ns <= boundary_ns

    @pytest.mark.parametrize("status", [1, 3, 127])  # fmt: skip
    async def test_run_nonzero_exit_raises_without_boundary_marker(
        self, tmp_path: Path, status: int
    ) -> None:
        config = _hook_config(tmp_path, f"exit {status}")

        with pytest.raises(WarmupBoundaryHookError, match=f"status {status}"):
            await WarmupBoundaryHook(config).run()

        assert (config.marker_dir / WARMUP_DONE_MARKER).exists()
        assert not (config.marker_dir / BOUNDARY_DONE_MARKER).exists()

    async def test_run_missing_script_raises_without_boundary_marker(
        self, tmp_path: Path
    ) -> None:
        marker_dir = tmp_path / "markers"
        marker_dir.mkdir()
        config = WarmupBoundaryHookConfig(
            script=tmp_path / "missing.sh", marker_dir=marker_dir
        )

        with pytest.raises(WarmupBoundaryHookError, match="status 127"):
            await WarmupBoundaryHook(config).run()

        assert not (marker_dir / BOUNDARY_DONE_MARKER).exists()

    async def test_run_cancelled_kills_script_and_its_children(
        self, tmp_path: Path
    ) -> None:
        config = _hook_config(
            tmp_path, _sleeping_script_body(tmp_path / "pid", tmp_path / "child")
        )

        task = asyncio.create_task(WarmupBoundaryHook(config).run())
        pids = await _wait_for_pids(task, tmp_path / "pid", tmp_path / "child")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        for pid in pids:
            assert await _is_gone(pid), f"process {pid} still runs"
        assert not (config.marker_dir / BOUNDARY_DONE_MARKER).exists()

    async def test_prepare_removes_markers_of_an_earlier_run(
        self, tmp_path: Path
    ) -> None:
        config = _hook_config(tmp_path, "true")
        for name in (WARMUP_DONE_MARKER, BOUNDARY_DONE_MARKER):
            (config.marker_dir / name).write_text("1\n")

        await WarmupBoundaryHook(config).prepare()

        assert list(config.marker_dir.iterdir()) == []

    async def test_prepare_missing_marker_dir_raises(self, tmp_path: Path) -> None:
        config = WarmupBoundaryHookConfig(
            script=_write_script(tmp_path / "hook.sh", "true"),
            marker_dir=tmp_path / "missing",
        )

        with pytest.raises(WarmupBoundaryHookError, match="does not exist"):
            await WarmupBoundaryHook(config).prepare()


@pytest.mark.asyncio
class TestPhaseOrchestratorWarmupBoundary:
    """PhaseOrchestrator runs the hook between warmup and profiling."""

    async def test_hook_runs_after_warmup_and_before_profiling(
        self, tmp_path: Path
    ) -> None:
        config = _hook_config(tmp_path, "true")
        router = MockCreditRouter(auto_return=True)
        orchestrator = _orchestrator(config, router)
        hook = orchestrator._warmup_boundary_hook
        run = hook.run
        phases_at_hook: list[CreditPhase] = []

        async def _record_then_run() -> None:
            phases_at_hook.extend(c.phase for c in router.sent_credits)
            await run()

        hook.run = _record_then_run
        await orchestrator.initialize()
        await orchestrator.start()

        assert phases_at_hook == [CreditPhase.WARMUP] * _WARMUP_REQUESTS
        assert [c.phase for c in router.sent_credits] == (
            [CreditPhase.WARMUP] * _WARMUP_REQUESTS
            + [CreditPhase.PROFILING] * _PROFILING_REQUESTS
        )
        assert (config.marker_dir / WARMUP_DONE_MARKER).exists()
        assert (config.marker_dir / BOUNDARY_DONE_MARKER).exists()

    async def test_hook_failure_stops_run_before_profiling(
        self, tmp_path: Path
    ) -> None:
        config = _hook_config(tmp_path, "exit 2")
        router = MockCreditRouter(auto_return=True)
        router.cancel_all_credits = AsyncMock()
        orchestrator = _orchestrator(config, router)
        await orchestrator.initialize()

        # The lifecycle mixin turns the error into a FAILED state, which
        # TimingManager reports to the SystemController.
        with pytest.raises(asyncio.CancelledError, match="exited with status 2"):
            await orchestrator.start()

        assert orchestrator.state == LifecycleState.FAILED
        cancels = [
            call.args[0]
            for call in orchestrator._phase_publisher._pub_client.publish.await_args_list
            if isinstance(call.args[0], ProfileCancelCommand)
        ]
        assert [c.reason for c in cancels] == [
            ProfileCancelReason.WARMUP_BOUNDARY_HOOK_FAILURE
        ]
        assert "exited with status 2" in cancels[0].reason_detail
        assert [c.phase for c in router.sent_credits] == (
            [CreditPhase.WARMUP] * _WARMUP_REQUESTS
        )
        router.cancel_all_credits.assert_awaited()
        assert (config.marker_dir / WARMUP_DONE_MARKER).exists()
        assert not (config.marker_dir / BOUNDARY_DONE_MARKER).exists()

    async def test_missing_marker_dir_stops_run_before_profiling(
        self, tmp_path: Path
    ) -> None:
        config = WarmupBoundaryHookConfig(
            script=_write_script(tmp_path / "hook.sh", "true"),
            marker_dir=tmp_path / "missing",
        )
        router = MockCreditRouter(auto_return=True)
        orchestrator = _orchestrator(config, router)
        await orchestrator.initialize()

        with pytest.raises(asyncio.CancelledError, match="does not exist"):
            await orchestrator.start()

        assert router.sent_credits == []

    async def test_stale_boundary_marker_is_removed_before_a_failing_hook(
        self, tmp_path: Path
    ) -> None:
        config = _hook_config(tmp_path, "exit 1")
        (config.marker_dir / BOUNDARY_DONE_MARKER).write_text("1\n")
        orchestrator = _orchestrator(config, MockCreditRouter(auto_return=True))
        await orchestrator.initialize()

        with pytest.raises(asyncio.CancelledError, match="exited with status 1"):
            await orchestrator.start()

        assert not (config.marker_dir / BOUNDARY_DONE_MARKER).exists()

    async def test_cancel_during_hook_kills_it_and_skips_profiling(
        self, tmp_path: Path
    ) -> None:
        config = _hook_config(
            tmp_path, _sleeping_script_body(tmp_path / "pid", tmp_path / "child")
        )
        router = MockCreditRouter(auto_return=True)
        orchestrator = _orchestrator(config, router)
        await orchestrator.initialize()

        start = asyncio.ensure_future(orchestrator.start())
        pids = await _wait_for_pids(start, tmp_path / "pid", tmp_path / "child")
        await orchestrator.cancel()
        await asyncio.wait_for(start, timeout=10)

        for pid in pids:
            assert await _is_gone(pid), f"process {pid} still runs"
        assert {c.phase for c in router.sent_credits} == {CreditPhase.WARMUP}
        assert (config.marker_dir / WARMUP_DONE_MARKER).exists()
        assert not (config.marker_dir / BOUNDARY_DONE_MARKER).exists()

    async def test_without_hook_writes_no_markers(self, tmp_path: Path) -> None:
        router = MockCreditRouter(auto_return=True)
        orchestrator = _orchestrator(None, router)
        await orchestrator.initialize()
        await orchestrator.start()

        assert orchestrator._warmup_boundary_hook is None
        assert len(router.sent_credits) == _WARMUP_REQUESTS + _PROFILING_REQUESTS
        assert list(tmp_path.iterdir()) == []

    async def test_cancelled_warmup_skips_hook(self, tmp_path: Path) -> None:
        orchestrator = _orchestrator(
            _hook_config(tmp_path, "true"), MockCreditRouter(auto_return=True)
        )
        orchestrator._warmup_boundary_hook.run = AsyncMock()
        runner = MagicMock(was_cancelled=True)

        await orchestrator._run_warmup_boundary_hook(runner)

        orchestrator._warmup_boundary_hook.run.assert_not_awaited()


class TestTimingConfigWarmupBoundary:
    """TimingConfig refuses a hook with no single, non-seamless warmup phase."""

    @pytest.mark.parametrize(
        "phases",
        [
            param([CreditPhase.PROFILING], id="no-warmup"),
            param([CreditPhase.WARMUP, CreditPhase.WARMUP, CreditPhase.PROFILING], id="two-warmups"),
        ],
    )  # fmt: skip
    def test_hook_without_exactly_one_warmup_raises(
        self, tmp_path: Path, phases: list[CreditPhase]
    ) -> None:
        with pytest.raises(ValueError, match="exactly one warmup phase"):
            TimingConfig(
                phase_configs=[_phase(p, 1) for p in phases],
                warmup_boundary_hook=_hook_config(tmp_path, "true"),
            )

    def test_hook_after_seamless_warmup_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="seamless"):
            TimingConfig(
                phase_configs=[
                    _phase(CreditPhase.WARMUP, 1, seamless=True),
                    _phase(CreditPhase.PROFILING, 1),
                ],
                warmup_boundary_hook=_hook_config(tmp_path, "true"),
            )

    def test_no_hook_allows_any_phases(self) -> None:
        config = TimingConfig(phase_configs=[_phase(CreditPhase.PROFILING, 1)])
        assert config.warmup_boundary_hook is None
