# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``--screening``: a short, labeled run of a scenario.

``--screening`` relaxes only the scenario's profiling-duration floor. Every
other lock stays enforced. The run is always stamped ``submission_valid=False``
with the ``screening`` reason.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.common.scenario import ScenarioLockError, apply_scenario
from aiperf.common.scenario.base import SCREENING_REASON, UNSAFE_OVERRIDE_REASON
from aiperf.config.config import BenchmarkConfig
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.converter import convert_cli_to_aiperf
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.config.resolution.resolvers import ScenarioResolver
from aiperf.exporters.aggregate.aggregate_base_exporter import (
    _build_run_metadata_dict,
    compute_submission_outcome,
)

_SCENARIO = "inferencex-agentx-mvp"
_FLOOR = 900


def _build_run(
    *,
    duration: float = 120,
    screening: bool = True,
    unsafe_override: bool = False,
    streaming: bool = True,
    scenario: str | None = _SCENARIO,
) -> BenchmarkRun:
    body: dict[str, Any] = {
        "models": ["my-model"],
        "endpoint": {
            "urls": ["http://localhost:8000/v1/chat/completions"],
            "type": "chat",
            "streaming": streaming,
            "extra": {"ignore_eos": True},
        },
        "datasets": [
            {
                "name": "main",
                "type": "public",
                "dataset": "semianalysis_cc_traces_weka_with_subagents",
            }
        ],
        "phases": [
            {
                "name": "profiling",
                "type": "concurrency",
                "concurrency": 8,
                "duration": duration,
            }
        ],
        "scenario": scenario,
        "screening": screening,
        "unsafe_override": unsafe_override,
    }
    cfg = BenchmarkConfig.model_validate(body)
    return BenchmarkRun(
        benchmark_id="test-run",
        cfg=cfg,
        artifact_dir=Path("/tmp/aiperf-screening-test"),
    )


@pytest.mark.parametrize(
    "duration",
    [
        param(30, id="far-below-floor"),
        param(_FLOOR - 1, id="just-below-floor"),
        param(_FLOOR, id="at-floor"),
        param(3600, id="above-floor"),
    ],
)  # fmt: skip
def test_screening_accepts_any_duration_and_is_always_stamped(
    duration: float,
) -> None:
    run = _build_run(duration=duration)

    outcome = apply_scenario(run)

    assert outcome.violations == []
    assert outcome.submission_valid is False
    assert outcome.submission_invalid_reasons == [SCREENING_REASON]
    assert run.cfg.get_profiling_phases()[0].duration == duration


def test_without_screening_short_duration_still_raises() -> None:
    run = _build_run(duration=120, screening=False)

    with pytest.raises(ScenarioLockError, match="--benchmark-duration"):
        apply_scenario(run)


def test_without_screening_valid_run_is_unchanged() -> None:
    outcome = apply_scenario(_build_run(duration=1800, screening=False))

    assert outcome.submission_valid is True
    assert outcome.submission_invalid_reasons == []
    assert "min_benchmark_duration" in outcome.applied_locks


def test_screening_keeps_other_locks_enforced() -> None:
    run = _build_run(duration=120, streaming=False)

    with pytest.raises(ScenarioLockError) as exc:
        apply_scenario(run)

    flags = [v.flag for v in exc.value.violations]
    assert "--benchmark-duration" not in flags
    assert flags


def test_screening_with_unsafe_override_stamps_both_reasons() -> None:
    run = _build_run(duration=120, streaming=False, unsafe_override=True)

    outcome = apply_scenario(run)

    assert outcome.submission_valid is False
    assert outcome.submission_invalid_reasons == [
        SCREENING_REASON,
        UNSAFE_OVERRIDE_REASON,
    ]
    assert all(v.flag != "--benchmark-duration" for v in outcome.violations)


def test_screening_with_unsafe_override_and_no_violation_is_screening_only() -> None:
    outcome = apply_scenario(_build_run(duration=120, unsafe_override=True))

    assert outcome.violations == []
    assert outcome.submission_invalid_reasons == [SCREENING_REASON]


def test_screening_warning_is_logged_once_at_run_start(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="aiperf.common.scenario.validator")
    run = _build_run(duration=120)

    ScenarioResolver().resolve(run)
    apply_scenario(run)  # The aggregate export re-resolves. It must not log.

    lines = [
        r.getMessage() for r in caplog.records if "SCREENING RUN" in r.getMessage()
    ]
    assert len(lines) == 1
    assert f"{_FLOOR}s profiling-duration floor is relaxed" in lines[0]


def test_no_screening_warning_without_the_flag(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="aiperf.common.scenario.validator")

    ScenarioResolver().resolve(_build_run(duration=1800, screening=False))

    assert not [r for r in caplog.records if "SCREENING RUN" in r.getMessage()]


def test_screening_with_unset_duration_keeps_scenario_default() -> None:
    run = _build_run(duration=120)
    run.cfg.get_profiling_phases()[0].duration = None

    outcome = apply_scenario(run)

    assert run.cfg.get_profiling_phases()[0].duration == 1800.0
    assert outcome.submission_invalid_reasons == [SCREENING_REASON]


def test_screening_relaxes_every_profiling_phase() -> None:
    run = _build_run(duration=120)
    second = run.cfg.get_profiling_phases()[0].model_copy(update={"name": "p2"})
    run.cfg.phases.append(second)

    outcome = apply_scenario(run)

    assert len(run.cfg.get_profiling_phases()) == 2
    assert outcome.violations == []
    assert outcome.submission_invalid_reasons == [SCREENING_REASON]


def test_screening_without_scenario_is_refused() -> None:
    with pytest.raises(ValidationError, match="--screening marks a short run"):
        _build_run(scenario=None)


def test_screening_flag_reaches_benchmark_config() -> None:
    cli = CLIConfig.model_validate(
        {
            "model_names": ["my-model"],
            "urls": ["http://localhost:8000"],
            "endpoint_type": "chat",
            "scenario": _SCENARIO,
            "screening": True,
            "benchmark_duration": 120,
        }
    )

    assert convert_cli_to_aiperf(cli).benchmark.screening is True


def test_screening_flag_unset_leaves_default() -> None:
    cli = CLIConfig.model_validate(
        {
            "model_names": ["my-model"],
            "urls": ["http://localhost:8000"],
            "endpoint_type": "chat",
            "request_count": 5,
        }
    )

    assert convert_cli_to_aiperf(cli).benchmark.screening is False


def test_export_metadata_marks_screening_run() -> None:
    valid, reasons = compute_submission_outcome(
        scenario_name=_SCENARIO,
        validator_submission_valid=False,
        validator_reasons=[SCREENING_REASON],
        total_responses=10,
    )

    metadata = _build_run_metadata_dict(
        scenario_name=_SCENARIO,
        submission_valid=valid,
        submission_invalid_reasons=reasons,
    )

    assert metadata == {
        "scenario": _SCENARIO,
        "submission_valid": False,
        "submission_invalid_reasons": [SCREENING_REASON],
        "screening": True,
    }


def test_export_metadata_without_screening_has_no_screening_key() -> None:
    metadata = _build_run_metadata_dict(
        scenario_name=_SCENARIO,
        submission_valid=False,
        submission_invalid_reasons=[UNSAFE_OVERRIDE_REASON],
    )

    assert "screening" not in metadata


def test_screening_in_yaml_config(tmp_path: Path) -> None:
    from aiperf.config.flags.resolver import resolve_config

    config_file = tmp_path / "aiperf.yaml"
    config_file.write_text(
        "benchmark:\n"
        "  models: [my-model]\n"
        "  endpoint: {type: chat, urls: ['http://localhost:8000']}\n"
        "  datasets: [{name: main, type: public, dataset: semianalysis_cc_traces_weka_with_subagents}]\n"
        "  phases: [{name: profiling, type: concurrency, concurrency: 8, duration: 120}]\n"
        f"  scenario: {_SCENARIO}\n"
        "  screening: true\n"
    )

    assert resolve_config(CLIConfig(), config_file).benchmark.screening is True


def test_screening_flag_with_yaml_config_is_applied(tmp_path: Path) -> None:
    from aiperf.config.flags.resolver import resolve_config

    config_file = tmp_path / "aiperf.yaml"
    config_file.write_text(_yaml_body(scenario=True))

    config = resolve_config(CLIConfig(screening=True), config_file)

    assert config.benchmark.screening is True


def test_screening_flag_with_yaml_config_without_scenario_is_refused(
    tmp_path: Path,
) -> None:
    from aiperf.config.flags.resolver import resolve_config

    config_file = tmp_path / "aiperf.yaml"
    config_file.write_text(_yaml_body(scenario=False))

    with pytest.raises(ValidationError, match="--screening marks a short run"):
        resolve_config(CLIConfig(screening=True), config_file)


def _yaml_body(*, scenario: bool) -> str:
    body = (
        "benchmark:\n"
        "  models: [my-model]\n"
        "  endpoint: {type: chat, urls: ['http://localhost:8000']}\n"
        "  datasets: [{name: main, type: public, dataset: "
        "semianalysis_cc_traces_weka_with_subagents}]\n"
        "  phases: [{name: profiling, type: concurrency, concurrency: 8, "
        "duration: 1800}]\n"
    )
    if scenario:
        body += f"  scenario: {_SCENARIO}\n"
    return body


def test_cancelled_screening_run_keeps_the_stamp() -> None:
    valid, reasons = compute_submission_outcome(
        scenario_name=_SCENARIO,
        validator_submission_valid=False,
        validator_reasons=[SCREENING_REASON],
        was_cancelled=True,
    )

    metadata = _build_run_metadata_dict(
        scenario_name=_SCENARIO,
        submission_valid=valid,
        submission_invalid_reasons=reasons,
    )

    assert metadata["screening"] is True
    assert metadata["submission_invalid_reasons"] == [SCREENING_REASON, "run_cancelled"]


async def _aggregate_export(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, is_sweep: bool
) -> dict:
    """Stamp a real screening plan and export it with the real aggregate exporter."""
    from types import SimpleNamespace

    import orjson

    from aiperf.cli_runner import _aggregate
    from aiperf.exporters.aggregate import (
        AggregateConfidenceJsonExporter,
        AggregateExporterConfig,
    )
    from aiperf.orchestrator.aggregation.base import AggregateResult

    monkeypatch.setattr(_aggregate, "_sum_runtime_response_counts", lambda _r: (10, 0))
    plan = SimpleNamespace(configs=[_build_run(duration=120).cfg], is_sweep=is_sweep)
    aggregate = AggregateResult(
        aggregation_type="confidence",
        num_runs=2,
        num_successful_runs=2,
        failed_runs=[],
        metrics={},
        metadata={},
    )
    _aggregate._stamp_scenario_submission_metadata(aggregate, [], plan)
    exporter = AggregateConfidenceJsonExporter(
        AggregateExporterConfig(result=aggregate, output_dir=tmp_path)
    )
    out_path = await exporter.export()
    return orjson.loads(Path(out_path).read_bytes())


def _metadata(exported: dict) -> dict:
    return exported.get("metadata", exported)


@pytest.mark.asyncio
async def test_aggregate_export_keeps_screening(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    metadata = _metadata(await _aggregate_export(monkeypatch, tmp_path, is_sweep=False))

    assert metadata["screening"] is True
    assert metadata["submission_valid"] is False
    assert metadata["submission_invalid_reasons"] == [SCREENING_REASON]


@pytest.mark.asyncio
async def test_aggregate_export_of_screening_sweep_keeps_screening(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    metadata = _metadata(await _aggregate_export(monkeypatch, tmp_path, is_sweep=True))

    assert metadata["screening"] is True
    assert metadata["submission_invalid_reasons"] == [
        SCREENING_REASON,
        "scenario_with_sweep",
    ]


@pytest.mark.asyncio
async def test_aggregate_export_keeps_screening_when_reresolution_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _fail(_run) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("aiperf.common.scenario.apply_scenario", _fail)

    metadata = _metadata(await _aggregate_export(monkeypatch, tmp_path, is_sweep=False))

    assert metadata["screening"] is True
    assert metadata["submission_valid"] is False
    assert metadata["submission_invalid_reasons"] == [
        SCREENING_REASON,
        "scenario_reresolve_failed",
    ]
