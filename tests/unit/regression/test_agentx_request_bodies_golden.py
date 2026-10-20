# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AgentX-style request bodies stay byte-identical to aiperf 49634d1c.

The golden fixture ``fixtures/agentx_request_bodies_49634d1c.json`` comes from
aiperf commit 49634d1c6c2e6dcc18979d704a65e7c7064efc49 (tag agentx-v1.0.4). We
generated it with this same test module at that commit: copy this file and
``fixtures/agentx_traces/`` into a checkout of 49634d1c and run the test with
``REGENERATE_AGENTX_GOLDEN=1``.

The test runs the request path offline, without the benchmark-id and boundary
hook options:

1. ``run_benchmark`` builds the run from CLI flags for the
   ``inferencex-agentx-mvp`` scenario (agentic replay, first-turn-prefix
   cache bust). ``uuid4`` in ``aiperf.cli_runner`` is fixed, so the default
   benchmark id is deterministic without ``--benchmark-id``.
2. The weka trace loader reads the local trace fixture. Prompt text synthesis
   is stubbed, as in the loader's own tests.
3. ``TimingConfig.from_run`` and ``PhaseOrchestrator`` run the warmup and
   profiling phases with a router that records each credit and returns it.
   Concurrency is 1 and the profiling phase stops at a request count, so the
   credit sequence does not depend on wall-clock timing.
4. A real ``Worker`` turns each credit into a request: marker injection, the
   chat endpoint's payload formatting and ``orjson`` encoding. The test keeps
   the bytes that reach the transport.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import UUID

import orjson
import pytest

from aiperf import cli_runner
from aiperf.common.environment import Environment
from aiperf.common.models import DatasetMetadata
from aiperf.config.flags import CLIConfig
from aiperf.config.flags.resolver import resolve_config
from aiperf.config.loader import build_benchmark_plan
from aiperf.config.resolution.resolvers import build_default_resolver_chain
from aiperf.credit.messages import CreditReturn
from aiperf.credit.structs import Credit, CreditContext
from aiperf.dataset.loader.weka_trace import WekaTraceLoader
from aiperf.timing.config import TimingConfig
from aiperf.timing.phase.publisher import PhasePublisher
from aiperf.timing.phase_orchestrator import PhaseOrchestrator
from aiperf.workers.worker import Worker

_FIXTURES = Path(__file__).parent / "fixtures"
_TRACES = _FIXTURES / "agentx_traces"
_GOLDEN = _FIXTURES / "agentx_request_bodies_49634d1c.json"
_BASE_COMMIT = "49634d1c6c2e6dcc18979d704a65e7c7064efc49"
_FIXED_UUID = UUID("a6e47b0d-5c1f-4d2e-9a3b-7c8d9e0f1a2b")
_BLOCK_SIZE = 16


class _RecordingRouter:
    """Credit router that records each credit and returns it at once."""

    def __init__(self) -> None:
        self.sent_credits: list[Credit] = []
        self._return_cb = None
        self._tasks: set[asyncio.Task] = set()

    async def send_credit(self, credit: Credit) -> None:
        self.sent_credits.append(credit)
        task = asyncio.get_running_loop().create_task(self._return(credit))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _return(self, credit: Credit) -> None:
        await self._return_cb(
            "worker-1",
            CreditReturn(credit=credit, cancelled=False, first_token_sent=True),
        )

    async def wait_for_workers(self, timeout: float) -> None:
        return None

    async def cancel_all_credits(self) -> None:
        return None

    def mark_credits_complete(self) -> None:
        return None

    def set_return_callback(self, cb) -> None:
        self._return_cb = cb

    def set_first_token_callback(self, cb) -> None:
        return None


class _DatasetClient:
    """Serves a fresh copy of a loaded conversation by id, as the mmap client does."""

    def __init__(self, conversations: list) -> None:
        self._by_id = {c.session_id: c for c in conversations}

    async def get_conversation(self, conversation_id: str):
        return self._by_id[conversation_id].model_copy(deep=True)

    async def stop(self) -> None:
        return None


def _cli_config(artifact_dir: Path, extra: dict | None = None) -> CLIConfig:
    """AgentX-style flags. ``extra`` adds or replaces flags by field name."""
    return CLIConfig.model_validate(
        {
            "scenario": "inferencex-agentx-mvp",
            "unsafe_override": True,
            "model_names": ["claude-opus-4-5-20251101"],
            "urls": ["http://localhost:8000"],
            "endpoint_type": "chat",
            "streaming": True,
            "custom_dataset_type": "weka_trace",
            "input_file": str(_TRACES),
            "tokenizer_name": "test-tok",
            "concurrency": 1,
            "random_seed": 42,
            "warmup_requests_per_lane": 2,
            "benchmark_duration": 60,
            "request_count": 20,
            "extra_inputs": ["ignore_eos:true"],
            "artifact_directory": str(artifact_dir),
            **(extra or {}),
        }
    )


def _build_run(artifact_dir: Path, extra: dict | None = None):
    """Build the run through ``run_benchmark`` and capture it before launch.

    ``uuid4`` in ``aiperf.cli_runner`` returns ``_FIXED_UUID``, so a run
    without a fixed benchmark id gets ``_FIXED_UUID.hex[:12]``.
    """
    plan = build_benchmark_plan(resolve_config(_cli_config(artifact_dir, extra), None))
    captured = {}

    def _capture(run, on_complete=None) -> None:
        captured["run"] = run

    with (
        patch.object(cli_runner, "uuid4", return_value=_FIXED_UUID),
        patch.object(cli_runner, "_preflight_artifact_dir"),
        patch.object(cli_runner, "_preflight_accuracy_deps"),
        patch.object(cli_runner, "_preflight_fd_limit"),
        patch.object(cli_runner, "_preflight_endpoint_ready"),
        patch.object(cli_runner, "_run_single_benchmark", side_effect=_capture),
        patch("aiperf.common.bootstrap.register_sigusr1_faulthandler"),
    ):
        cli_runner.run_benchmark(plan)
    run = captured["run"]
    build_default_resolver_chain().resolve_all(run)
    return run


def _load_conversations(run, monkeypatch: pytest.MonkeyPatch) -> list:
    loader = WekaTraceLoader(
        filename=str(_TRACES), run=run, default_block_size=_BLOCK_SIZE
    )
    monkeypatch.setattr(Environment.DATASET, "WEKA_PARALLEL_WORKERS", 1)
    monkeypatch.setattr(
        loader,
        "synthesize_prompts_from_hash_ids",
        lambda reqs: {r.key: f"prompt-{r.key}" for r in reqs},
    )
    monkeypatch.setattr(
        loader,
        "sample_partial_tail_tokens",
        lambda n_tokens, seed: [0] * max(n_tokens, 0),
    )
    monkeypatch.setattr(
        loader, "sample_partial_tail", lambda n_tokens, seed: "x" * max(n_tokens, 0)
    )
    monkeypatch.setattr(
        loader,
        "_decode_block_tokens",
        lambda hash_ids: [0] * (len(hash_ids) * _BLOCK_SIZE),
    )
    monkeypatch.setattr(
        loader, "_decode_tokens_to_text", lambda tokens: "x" * len(tokens)
    )
    loader.prompt_generator = MagicMock()
    loader.prompt_generator._cache = {}
    loader._tokenizer_name = "test-tok"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = _BLOCK_SIZE
    return loader.convert_to_conversations(loader.load_dataset())


async def _issue_credits(run, conversations: list) -> list[Credit]:
    publisher_client = MagicMock()

    async def _publish(*args, **kwargs) -> None:
        return None

    publisher_client.publish = _publish
    router = _RecordingRouter()
    orchestrator = PhaseOrchestrator(
        config=TimingConfig.from_run(run),
        phase_publisher=PhasePublisher(pub_client=publisher_client, service_id="t"),
        credit_router=router,
        dataset_metadata=DatasetMetadata(
            conversations=[c.metadata() for c in conversations],
            sampling_strategy=run.cfg.get_default_dataset().sampling,
        ),
        run=run,
        service_id="golden-orchestrator",
    )
    await orchestrator.initialize()
    await asyncio.wait_for(orchestrator.start(), timeout=60)
    await orchestrator.stop()
    return router.sent_credits


async def _request_bodies(run, conversations: list, credits: list[Credit]) -> list:
    worker = Worker(run=run, service_id="golden-worker")
    await worker.initialize()
    await worker.start()
    worker._dataset_client = _DatasetClient(conversations)
    transport = worker.inference_client.transport
    send_request = transport.send_request
    bodies: list[bytes] = []

    async def _capture(request_info, payload, **kwargs):
        bodies.append(bytes(payload))
        return await send_request(request_info, payload=payload, **kwargs)

    transport.send_request = _capture
    entries = []
    try:
        for credit in credits:
            before = len(bodies)
            await worker._process_credit(CreditContext(credit=credit, drop_perf_ns=0))
            assert len(bodies) == before + 1, f"no request body for {credit!r}"
            entries.append(
                {
                    "phase": str(credit.phase),
                    "conversation_id": credit.conversation_id,
                    "turn_index": credit.turn_index,
                    "sha256": hashlib.sha256(bodies[-1]).hexdigest(),
                    "body": bodies[-1].decode(),
                }
            )
    finally:
        await worker.stop()
    return entries


async def agentx_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: dict | None = None
) -> tuple[object, list[Credit], list[dict]]:
    """Return the run, its credits and its request bodies for AgentX-style flags."""
    run = _build_run(tmp_path / "artifacts", extra)
    conversations = _load_conversations(run, monkeypatch)
    credits = await _issue_credits(run, conversations)
    return run, credits, await _request_bodies(run, conversations, credits)


@pytest.mark.asyncio
async def test_agentx_request_bodies_match_base_commit_without_new_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_tokenizer,
    skip_service_registration,
) -> None:
    run, _, entries = await agentx_requests(tmp_path, monkeypatch)
    assert run.benchmark_id == _FIXED_UUID.hex[:12]

    if os.environ.get("REGENERATE_AGENTX_GOLDEN") == "1":
        _GOLDEN.write_bytes(
            orjson.dumps(
                {"base_commit": _BASE_COMMIT, "requests": entries},
                option=orjson.OPT_INDENT_2,
            )
            + b"\n"
        )

    golden = orjson.loads(_GOLDEN.read_bytes())
    assert golden["base_commit"] == _BASE_COMMIT
    assert entries == golden["requests"]
