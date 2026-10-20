# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``--benchmark-id`` and the warmup boundary hook on the AgentX request path.

Uses the offline request path of ``test_agentx_request_bodies_golden``: CLI
flags, ``run_benchmark``, the weka trace loader, ``PhaseOrchestrator`` and a
real ``Worker``.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import orjson
import pytest

from aiperf.config.warmup_boundary import BOUNDARY_DONE_MARKER, WARMUP_DONE_MARKER
from tests.unit.regression.test_agentx_request_bodies_golden import (
    _FIXED_UUID,
    _GOLDEN,
    agentx_requests,
)

_MARKER = re.compile(r"\[rid:[0-9a-f]{12}\]")


def _golden_entries() -> list[dict]:
    return orjson.loads(_GOLDEN.read_bytes())["requests"]


def _without_markers(entries: list[dict]) -> list[str]:
    return [_MARKER.sub("[rid]", e["body"]) for e in entries]


@pytest.mark.asyncio
async def test_fixed_benchmark_id_repeats_bodies_with_new_correlation_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_tokenizer,
    skip_service_registration,
) -> None:
    extra = {"benchmark_id": "agentx-warm-01"}
    first_run, first_credits, first = await agentx_requests(
        tmp_path / "first", monkeypatch, extra
    )
    second_run, second_credits, second = await agentx_requests(
        tmp_path / "second", monkeypatch, extra
    )

    assert first_run.benchmark_id == second_run.benchmark_id == "agentx-warm-01"
    assert first == second
    first_ids = {c.x_correlation_id for c in first_credits}
    assert first_ids.isdisjoint(c.x_correlation_id for c in second_credits)

    golden = _golden_entries()
    assert [e["body"] for e in first] != [e["body"] for e in golden]
    assert _without_markers(first) == _without_markers(golden)


@pytest.mark.asyncio
async def test_boundary_hook_runs_and_leaves_bodies_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_tokenizer,
    skip_service_registration,
) -> None:
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    seen = tmp_path / "seen"
    script = tmp_path / "boundary.sh"
    script.write_text(f'ls "{marker_dir}" > "{seen}"\n')

    run, _, entries = await agentx_requests(
        tmp_path,
        monkeypatch,
        {"warmup_boundary_hook": script, "warmup_boundary_dir": marker_dir},
    )

    assert run.benchmark_id == _FIXED_UUID.hex[:12]
    assert seen.read_text().split() == [WARMUP_DONE_MARKER]
    assert (marker_dir / BOUNDARY_DONE_MARKER).exists()
    assert entries == _golden_entries()


@pytest.mark.asyncio
async def test_failing_boundary_hook_stops_before_profiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_tokenizer,
    skip_service_registration,
) -> None:
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    script = tmp_path / "boundary.sh"
    script.write_text("exit 4\n")

    # PhaseOrchestrator's lifecycle turns the hook error into CancelledError
    # and the FAILED state.
    with pytest.raises(asyncio.CancelledError, match="exited with status 4"):
        await agentx_requests(
            tmp_path,
            monkeypatch,
            {"warmup_boundary_hook": script, "warmup_boundary_dir": marker_dir},
        )

    assert (marker_dir / WARMUP_DONE_MARKER).exists()
    assert not (marker_dir / BOUNDARY_DONE_MARKER).exists()
