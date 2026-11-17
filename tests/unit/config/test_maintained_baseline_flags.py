# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The opt-in flag reaches profiling and preserves zero-rate arrival boundaries."""

from pathlib import Path

import pytest

from aiperf.config.flags import CLIConfig
from aiperf.config.flags.resolver import resolve_config
from tests.unit.cli_runner.test_agentic_replay_phase_override import (
    _AGENTIC_PROFILING_YAML,
)


def test_maintained_baseline_cli_overlays_agentic_profiling(tmp_path: Path) -> None:
    path = tmp_path / "input.yaml"
    path.write_text(_AGENTIC_PROFILING_YAML.replace("requests: 100", "duration: 1800"))
    config = resolve_config(
        CLIConfig(
            session_maintain_baseline=True,
            session_arrival_rate=0,
            session_concurrency_ceiling=4096,
            session_arrival_schedule="900:0.03333333333333333,1200:0",
        ),
        path,
    )
    phase = config.benchmark.get_profiling_phases()[0]
    assert phase.session_maintain_baseline is True
    assert phase.session_arrivals.rate == 0
    assert phase.session_arrivals.rate_schedule[-1].rate == 0


def test_zero_arrival_segment_requires_cohort_journal(tmp_path: Path) -> None:
    path = tmp_path / "input.yaml"
    path.write_text(_AGENTIC_PROFILING_YAML.replace("requests: 100", "duration: 1800"))
    with pytest.raises(ValueError, match="session_maintain_baseline"):
        resolve_config(
            CLIConfig(session_arrival_rate=0, session_concurrency_ceiling=4096), path
        )
