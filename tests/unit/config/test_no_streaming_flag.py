# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``--no-streaming``: an explicit non-streaming request mode, including under AgentX.

The AgentX scenario turns streaming on when ``--streaming`` is unset and
honors an explicit ``--no-streaming`` (or ``--streaming=false``) without a
violation, so the run stays ``submission_valid=true``.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.common.scenario import apply_scenario
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.converter import convert_cli_to_aiperf
from aiperf.config.resolution.plan import BenchmarkRun

_AGENTX = (
    "--url",
    "http://localhost:8000",
    "--model",
    "test-model",
    "--endpoint-type",
    "chat",
    "--scenario",
    "inferencex-agentx-mvp",
    "--public-dataset",
    "semianalysis_cc_traces_weka_062126",
    "--extra-inputs",
    "ignore_eos:true",
    "--benchmark-duration",
    "900",
    "--concurrency",
    "2",
)


# Same cyclopts capture pattern as tests/unit/config/test_auto_plot_fields.py.
def _parse_cli_args(argv: list[str]) -> CLIConfig:
    from cyclopts import App

    captured: dict[str, CLIConfig] = {}
    app = App(name="test_profile")

    @app.default
    def _runner(*, cli_config: CLIConfig) -> None:  # pragma: no cover - capture only
        captured["uc"] = cli_config

    try:
        app(argv, exit_on_error=False)
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise
    return captured["uc"]


def _outcome(*extra: str):
    cli = _parse_cli_args([*_AGENTX, *extra])
    config = convert_cli_to_aiperf(cli)
    run = BenchmarkRun(
        benchmark_id="test-run",
        cfg=config.benchmark,
        artifact_dir=config.benchmark.artifacts.dir,
    )
    return run, apply_scenario(run)


@pytest.mark.parametrize(
    ("extra", "streaming"),
    [
        param((), True, id="unset-streams"),
        param(("--streaming",), True, id="explicit-on"),
        param(("--no-streaming",), False, id="no-streaming"),
        param(("--streaming=false",), False, id="streaming-equals-false"),
    ],
)
def test_agentx_streaming_request_mode(extra: tuple[str, ...], streaming: bool) -> None:
    run, outcome = _outcome(*extra)
    assert run.cfg.endpoint.streaming is streaming
    assert outcome.violations == []
    assert outcome.submission_valid is True
    assert ("streaming" in outcome.applied_locks) is streaming


def test_no_streaming_is_explicit() -> None:
    cli = _parse_cli_args([*_AGENTX, "--no-streaming"])
    assert cli.streaming is False
    assert "streaming" in cli.model_fields_set
