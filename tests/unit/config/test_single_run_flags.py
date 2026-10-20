# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``--benchmark-id`` and ``--warmup-boundary-hook/-dir``.

Covers flag parsing into ``AIPerfConfig``, the plan and the ``BenchmarkRun``
(``run_benchmark`` and ``aiperf service``), the YAML+CLI path, and each
combination the options refuse.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf import cli_runner
from aiperf.config import AIPerfConfig, BenchmarkConfig
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.resolver import resolve_config
from aiperf.config.loader import build_benchmark_plan
from aiperf.config.resolution.plan import BenchmarkPlan
from aiperf.config.warmup_boundary import WarmupBoundaryHookConfig

_BASE_ARGV = [
    "--model",
    "test-model",
    "--url",
    "http://localhost:8000",
    "--endpoint-type",
    "chat",
    "--request-count",
    "5",
]
_HOOK_ARGV = [
    "--warmup-boundary-hook",
    "/opt/hooks/boundary.sh",
    "--warmup-boundary-dir",
    "/tmp/markers",
]
_AGENTX_ARGV_WITHOUT_INPUT = [
    "--model",
    "test-model",
    "--url",
    "http://localhost:8000",
    "--endpoint-type",
    "chat",
    "--streaming",
    "--scenario",
    "inferencex-agentx-mvp",
    "--custom-dataset-type",
    "weka_trace",
    "--concurrency",
    "1",
    "--benchmark-duration",
    "900",
]


# Same cyclopts capture pattern as tests/unit/config/test_baseten_replay_flags.py
# (not importable from there: tests/unit/config is not a package).
def _parse_cli_args(argv: list[str]) -> CLIConfig:
    """Parse ``argv`` through cyclopts into a ``CLIConfig`` (no execution)."""
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


def _config(argv: list[str]) -> AIPerfConfig:
    return resolve_config(_parse_cli_args(argv), None)


def _run_benchmark_capture(plan: BenchmarkPlan):
    """Run ``run_benchmark`` up to the launch and return the ``BenchmarkRun``."""
    captured = {}

    def _capture(run, on_complete=None) -> None:
        captured["run"] = run

    with (
        patch.object(cli_runner, "_preflight_artifact_dir"),
        patch.object(cli_runner, "_preflight_accuracy_deps"),
        patch.object(cli_runner, "_preflight_fd_limit"),
        patch.object(cli_runner, "_preflight_endpoint_ready"),
        patch.object(cli_runner, "_run_single_benchmark", side_effect=_capture),
        patch("aiperf.common.bootstrap.register_sigusr1_faulthandler"),
    ):
        cli_runner.run_benchmark(plan)
    return captured["run"]


def _benchmark_dict(**extra) -> dict:
    return {
        "models": ["test-model"],
        "endpoint": {"type": "chat", "urls": ["http://localhost:8000"]},
        "datasets": [{"name": "default", "type": "synthetic"}],
        **extra,
    }


_HOOK = {"script": "/opt/hooks/boundary.sh", "marker_dir": "/tmp/markers"}


class TestBenchmarkIdFlag:
    def test_unset_flag_leaves_benchmark_id_none(self) -> None:
        config = _config(_BASE_ARGV)

        assert config.benchmark_id is None
        assert build_benchmark_plan(config).benchmark_id is None

    def test_flag_reaches_config_and_plan(self) -> None:
        config = _config([*_BASE_ARGV, "--benchmark-id", "agentx_warm-01"])

        assert config.benchmark_id == "agentx_warm-01"
        assert build_benchmark_plan(config).benchmark_id == "agentx_warm-01"

    def test_run_benchmark_uses_fixed_id_without_calling_uuid4(self) -> None:
        plan = build_benchmark_plan(
            _config([*_BASE_ARGV, "--benchmark-id", "fixed-id"])
        )

        with patch.object(cli_runner, "uuid4", side_effect=AssertionError):
            run = _run_benchmark_capture(plan)

        assert run.benchmark_id == "fixed-id"

    def test_run_benchmark_without_flag_uses_random_id(self) -> None:
        plan = build_benchmark_plan(_config(_BASE_ARGV))

        first = _run_benchmark_capture(plan)
        second = _run_benchmark_capture(plan)

        assert first.benchmark_id != second.benchmark_id
        assert len(first.benchmark_id) == 12

    def test_fixed_id_leaves_other_uuid4_calls_random(self) -> None:
        plan = build_benchmark_plan(
            _config([*_BASE_ARGV, "--benchmark-id", "fixed-id"])
        )
        run = _run_benchmark_capture(plan)

        assert run.benchmark_id == "fixed-id"
        assert uuid.uuid4() != uuid.uuid4()
        assert plan.sweep_id != build_benchmark_plan(_config(_BASE_ARGV)).sweep_id

    def test_service_command_uses_fixed_id(self) -> None:
        from aiperf.cli_commands.service import service

        cli = _parse_cli_args([*_BASE_ARGV, "--benchmark-id", "svc-id"])
        with patch("aiperf.common.bootstrap.bootstrap_and_run_service") as bootstrap:
            service(service_type=MagicMock(), cli_config=cli)

        assert bootstrap.call_args.kwargs["run"].benchmark_id == "svc-id"

    @pytest.mark.parametrize(
        "benchmark_id",
        [
            param("", id="empty"),
            param("a" * 65, id="too-long"),
            param("../escape", id="path-traversal"),
            param("a/b", id="slash"),
            param("with space", id="space"),
            param("dot.id", id="dot"),
        ],
    )  # fmt: skip
    def test_path_unsafe_id_is_refused(self, benchmark_id: str) -> None:
        with pytest.raises(ValidationError, match="benchmark_id"):
            _config([*_BASE_ARGV, "--benchmark-id", benchmark_id])

    def test_longest_allowed_id_is_accepted(self) -> None:
        assert _config([*_BASE_ARGV, "--benchmark-id", "A" * 64]).benchmark_id

    @pytest.mark.parametrize(
        "extra_argv",
        [
            param(["--num-profile-runs", "2"], id="multi-run"),
            param(["--concurrency", "1,2"], id="sweep"),
        ],
    )  # fmt: skip
    def test_multiple_runs_are_refused(self, extra_argv: list[str]) -> None:
        with pytest.raises(ValidationError, match="--benchmark-id fixes the id"):
            _config([*_BASE_ARGV, *extra_argv, "--benchmark-id", "fixed-id"])

    def test_plan_with_trials_refuses_benchmark_id(self) -> None:
        plan = build_benchmark_plan(_config(_BASE_ARGV))

        with pytest.raises(ValidationError, match="more than one benchmark"):
            BenchmarkPlan(
                configs=plan.configs,
                variations=plan.variations,
                trials=2,
                benchmark_id="fixed-id",
            )

    def test_variant_with_benchmark_id_is_refused(self) -> None:
        with pytest.raises(TypeError, match="benchmark-id fixes the id"):
            _config(
                [
                    *_BASE_ARGV,
                    "--variant",
                    "a:concurrency=1",
                    "--variant",
                    "b:benchmark-id=fixed-id",
                ]
            )

    def test_yaml_envelope_and_cli_override(self, tmp_path: Path) -> None:
        config_file = tmp_path / "aiperf.yaml"
        config_file.write_text(
            "benchmark_id: from-yaml\n"
            "benchmark:\n"
            "  models: [test-model]\n"
            "  endpoint: {type: chat, urls: ['http://localhost:8000']}\n"
            "  datasets: [{name: default, type: synthetic}]\n"
            "  phases: [{name: profiling, type: concurrency, requests: 5}]\n"
        )

        from_yaml = resolve_config(CLIConfig(), config_file)
        from_cli = resolve_config(CLIConfig(benchmark_id="from-cli"), config_file)

        assert from_yaml.benchmark_id == "from-yaml"
        assert from_cli.benchmark_id == "from-cli"


class TestWarmupBoundaryHookFlags:
    def test_unset_flags_leave_hook_none(self) -> None:
        assert _config(_BASE_ARGV).benchmark.warmup_boundary_hook is None

    def test_flags_build_one_hook_object(self) -> None:
        config = _config([*_BASE_ARGV, "--warmup-request-count", "3", *_HOOK_ARGV])

        assert config.benchmark.warmup_boundary_hook == WarmupBoundaryHookConfig(
            script=Path("/opt/hooks/boundary.sh"), marker_dir=Path("/tmp/markers")
        )

    @pytest.mark.parametrize(
        "flag_pair",
        [
            param(_HOOK_ARGV[:2], id="hook-only"),
            param(_HOOK_ARGV[2:], id="dir-only"),
        ],
    )  # fmt: skip
    def test_one_flag_without_the_other_is_refused(self, flag_pair: list[str]) -> None:
        with pytest.raises(ValueError, match="go together"):
            _config([*_BASE_ARGV, "--warmup-request-count", "3", *flag_pair])

    def test_hook_without_warmup_phase_is_refused(self) -> None:
        with pytest.raises(
            ValidationError, match="exactly one warmup phase. The run has 0"
        ):
            _config([*_BASE_ARGV, *_HOOK_ARGV])

    def test_agentic_scenario_needs_no_declared_warmup(self, tmp_path: Path) -> None:
        config = _config(
            [*_AGENTX_ARGV_WITHOUT_INPUT, "--input-file", str(tmp_path), *_HOOK_ARGV]
        )

        assert config.benchmark.warmup_boundary_hook is not None

    @pytest.mark.parametrize(
        "extra_argv",
        [
            param(["--num-profile-runs", "2"], id="multi-run"),
            param(["--concurrency", "1,2"], id="sweep"),
        ],
    )  # fmt: skip
    def test_multiple_runs_are_refused(self, extra_argv: list[str]) -> None:
        with pytest.raises(ValidationError, match="--warmup-boundary-hook runs"):
            _config(
                [
                    *_BASE_ARGV,
                    "--warmup-request-count",
                    "3",
                    *extra_argv,
                    *_HOOK_ARGV,
                ]
            )

    def test_variant_that_adds_the_hook_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="--warmup-boundary-hook runs"):
            build_benchmark_plan(
                _config(
                    [
                        *_BASE_ARGV,
                        "--warmup-request-count",
                        "3",
                        "--variant",
                        "a:concurrency=1",
                        "--variant",
                        "b:warmup-boundary-hook=/opt/h.sh,warmup-boundary-dir=/tmp",
                    ]
                )
            )

    def test_scenario_sweep_run_that_adds_the_hook_is_refused(
        self, tmp_path: Path
    ) -> None:
        config_file = tmp_path / "aiperf.yaml"
        config_file.write_text(
            "benchmark:\n"
            "  models: [test-model]\n"
            "  endpoint: {type: chat, urls: ['http://localhost:8000']}\n"
            "  datasets: [{name: default, type: synthetic}]\n"
            "  phases:\n"
            "    - {name: warmup, kind: warmup, type: concurrency, requests: 2}\n"
            "    - {name: profiling, type: concurrency, requests: 5}\n"
            "sweep:\n"
            "  type: scenarios\n"
            "  runs:\n"
            "    - {name: plain}\n"
            "    - name: hooked\n"
            "      benchmark:\n"
            "        warmup_boundary_hook: {script: /opt/h.sh, marker_dir: /tmp}\n"
        )

        with pytest.raises(ValidationError, match="--warmup-boundary-hook runs"):
            build_benchmark_plan(resolve_config(CLIConfig(), config_file))

    def test_yaml_with_two_profiling_phases_accepts_hook_flags(
        self, tmp_path: Path
    ) -> None:
        config_file = tmp_path / "aiperf.yaml"
        config_file.write_text(
            "benchmark:\n"
            "  models: [test-model]\n"
            "  endpoint: {type: chat, urls: ['http://localhost:8000']}\n"
            "  datasets: [{name: default, type: synthetic}]\n"
            "  phases:\n"
            "    - {name: warmup, kind: warmup, type: concurrency, requests: 2}\n"
            "    - {name: p1, kind: profiling, type: concurrency, requests: 5}\n"
            "    - {name: p2, kind: profiling, type: concurrency, requests: 5}\n"
        )
        cli = CLIConfig(
            warmup_boundary_hook=Path("/opt/h.sh"), warmup_boundary_dir=Path("/tmp")
        )

        config = resolve_config(cli, config_file)

        assert config.benchmark.warmup_boundary_hook == WarmupBoundaryHookConfig(
            script=Path("/opt/h.sh"), marker_dir=Path("/tmp")
        )

    def test_hook_reaches_timing_config(self, tmp_path: Path) -> None:
        from aiperf.timing.config import TimingConfig

        plan = build_benchmark_plan(
            _config(
                [
                    *_BASE_ARGV,
                    "--warmup-request-count",
                    "3",
                    *_HOOK_ARGV,
                    "--output-artifact-dir",
                    str(tmp_path),
                ]
            )
        )
        run = _run_benchmark_capture(plan)

        timing = TimingConfig.from_run(run)

        assert timing.warmup_boundary_hook == plan.configs[0].warmup_boundary_hook
        assert [p.phase for p in timing.phase_configs] == ["warmup", "profiling"]
        assert timing.phase_configs[0].seamless is False


class TestWarmupBoundaryHookPhaseShapes:
    """YAML-shaped configs: the warmup phase count and the seamless decision."""

    def test_one_warmup_then_profiling_is_accepted(self) -> None:
        config = BenchmarkConfig.model_validate(
            _benchmark_dict(
                phases=[
                    {
                        "name": "warmup",
                        "kind": "warmup",
                        "type": "concurrency",
                        "requests": 2,
                    },
                    {"name": "profiling", "type": "concurrency", "requests": 5},
                ],
                warmup_boundary_hook=_HOOK,
            )
        )

        assert config.warmup_boundary_hook.script == Path(_HOOK["script"])

    def test_two_warmups_are_refused(self) -> None:
        with pytest.raises(
            ValidationError, match="exactly one warmup phase. The run has 2"
        ):
            BenchmarkConfig.model_validate(
                _benchmark_dict(
                    phases=[
                        {
                            "name": "w1",
                            "kind": "warmup",
                            "type": "concurrency",
                            "requests": 2,
                        },
                        {
                            "name": "w2",
                            "kind": "warmup",
                            "type": "concurrency",
                            "requests": 2,
                        },
                        {"name": "profiling", "type": "concurrency", "requests": 5},
                    ],
                    warmup_boundary_hook=_HOOK,
                )
            )

    def test_seamless_profiling_after_warmup_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="seamless phase 'profiling'"):
            BenchmarkConfig.model_validate(
                _benchmark_dict(
                    phases=[
                        {
                            "name": "warmup",
                            "kind": "warmup",
                            "type": "concurrency",
                            "requests": 2,
                        },
                        {
                            "name": "profiling",
                            "type": "concurrency",
                            "requests": 5,
                            "seamless": True,
                        },
                    ],
                    warmup_boundary_hook=_HOOK,
                )
            )

    def test_warmup_as_last_phase_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="profiling phase directly after"):
            BenchmarkConfig.model_validate(
                _benchmark_dict(
                    phases=[
                        {"name": "profiling", "type": "concurrency", "requests": 5},
                        {
                            "name": "warmup",
                            "kind": "warmup",
                            "type": "concurrency",
                            "requests": 2,
                        },
                    ],
                    warmup_boundary_hook=_HOOK,
                )
            )

    def test_hook_needs_both_fields(self) -> None:
        with pytest.raises(ValidationError, match="markerDir"):
            WarmupBoundaryHookConfig.model_validate({"script": "/opt/hooks/b.sh"})
