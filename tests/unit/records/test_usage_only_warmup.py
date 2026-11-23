# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``--warmup-accept-usage-only``: a complete one-token warmup reply without text.

The streams copy what SGLang and Rill send for GLM-5.3 when the one warmup
token is ``<think>`` and the reasoning parser removes it; the non-streamed
replies copy SGLang's JSON completion for the same request. Every failure case
must stay invalid: profiling requests, gateway errors, cancelled or incomplete
replies, replies of another form than the body asked for, and replies longer
than one token.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import orjson
import pytest

from aiperf.common.enums import CreditPhase, ModelSelectionStrategy
from aiperf.common.models import ErrorDetails, RequestInfo, Text, Turn
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import (
    RecordContext,
    RequestRecord,
    SSEMessage,
    TextResponse,
)
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.plugin.enums import EndpointType
from aiperf.records.inference_result_parser import InferenceResultParser
from aiperf.records.usage_only_warmup import (
    is_usage_only_one_token_warmup,
    one_token_body_streams,
)
from aiperf.workers.inference_client import InferenceClient

_ID = "chatcmpl-1"
_CHUNK = "chat.completion.chunk"
_USAGE = {"prompt_tokens": 100, "completion_tokens": 1, "total_tokens": 101}


def _sglang_chunks(usage: dict | None = None) -> list[dict | str]:
    """SGLang: role chunk with empty content, finish chunk, usage chunk."""
    return [
        {
            "id": _ID,
            "object": _CHUNK,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}],
        },
        {
            "id": _ID,
            "object": _CHUNK,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
        },
        {"id": _ID, "object": _CHUNK, "choices": [], "usage": usage or dict(_USAGE)},
        "[DONE]",
    ]


def _rill_chunks() -> list[dict | str]:
    """Rill: finish chunk with a null reasoning delta, SGLang usage layout."""
    usage = dict(_USAGE, reasoning_tokens=0, prompt_tokens_details={"cached_tokens": 0})
    return [
        {
            "id": _ID,
            "object": _CHUNK,
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning_content": None},
                    "logprobs": None,
                    "finish_reason": "length",
                    "matched_stop": None,
                }
            ],
        },
        {"id": _ID, "object": _CHUNK, "choices": [], "usage": usage},
        "[DONE]",
    ]


def _messages(chunks: list[dict | str], start: int = 2) -> list[SSEMessage]:
    return [
        SSEMessage.parse(
            "data: " + (c if isinstance(c, str) else orjson.dumps(c).decode()),
            perf_ns=start + i,
        )
        for i, c in enumerate(chunks)
    ]


def _record(
    chunks: list[dict | str] | None = None,
    *,
    phase: CreditPhase = CreditPhase.WARMUP,
    max_tokens: int = 1,
    body_streams: bool | None = True,
) -> RequestRecord:
    """A record as the worker sends it: body checked, then stripped (AgentX)."""
    chunks = _sglang_chunks() if chunks is None else chunks
    context = RecordContext(
        credit_phase=phase,
        credit_num=0,
        conversation_id="trace",
        turn_index=0,
        max_tokens=max_tokens,
        x_request_id="sai-req",
        x_correlation_id="session",
        one_token_body_streams=body_streams,
    )
    return RequestRecord(
        request_info=context,
        model_name="test-model",
        status=200,
        start_perf_ns=1,
        end_perf_ns=len(chunks) + 10,
        responses=_messages(chunks),
    )


@pytest.mark.parametrize(
    "chunks", [_sglang_chunks(), _rill_chunks()], ids=["sglang", "rill"]
)
def test_complete_one_token_reply_without_text_is_accepted(chunks) -> None:
    assert is_usage_only_one_token_warmup(_record(chunks))


def _variant(name: str) -> list[dict | str]:
    chunks = _sglang_chunks()
    if name == "finish_stop":
        chunks[1]["choices"][0]["finish_reason"] = "stop"
    if name == "usage_on_finish_chunk":
        chunks[1]["usage"] = chunks.pop(2)["usage"]
    if name == "null_delta":
        chunks[1]["choices"][0]["delta"] = None
    if name == "cumulative_usage":
        chunks[0]["usage"] = dict(_USAGE, completion_tokens=0, total_tokens=100)
    if name == "null_error":
        chunks[2]["error"] = None
    return chunks


@pytest.mark.parametrize(
    "name",
    [
        "finish_stop",
        "usage_on_finish_chunk",
        "null_delta",
        "cumulative_usage",
        "null_error",
    ],
)
def test_other_complete_shapes_are_accepted(name) -> None:
    assert is_usage_only_one_token_warmup(_record(_variant(name)))


@pytest.mark.parametrize(
    "body, expected",
    [
        ({"max_tokens": 1, "stream": True}, True),
        ({"max_completion_tokens": 1, "stream": True, "n": 1}, True),
        ({"max_tokens": 1, "stream": False}, False),
        ({"max_tokens": 1}, False),
        ({"max_tokens": 2, "stream": True}, None),
        ({"max_tokens": 2, "stream": False}, None),
        ({"max_tokens": 1, "max_completion_tokens": 8, "stream": True}, None),
        ({"max_tokens": True, "stream": True}, None),
        ({"stream": True}, None),
        ({"stream": False}, None),
        ({"max_tokens": 1, "stream": True, "n": 2}, None),
        ({"max_tokens": 1, "stream": "true"}, None),
        ([1], None),
    ],
)
def test_body_check(body, expected) -> None:
    assert one_token_body_streams(orjson.dumps(body)) is expected


def test_body_check_needs_a_body() -> None:
    assert one_token_body_streams(None) is None
    assert one_token_body_streams(b"{not json") is None


def _finalized(
    *,
    accept: bool = True,
    phase: CreditPhase = CreditPhase.WARMUP,
    extra: list[tuple[str, object]] | None = None,
    streaming: bool = True,
) -> RecordContext:
    """Format the body with the real chat endpoint, then finalize and strip it."""
    endpoint = ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name="test-model")],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        ),
        endpoint=EndpointInfo(
            type=EndpointType.CHAT,
            base_url="http://localhost:8000",
            streaming=streaming,
            use_server_token_count=True,
            extra=extra or [],
        ),
    )
    turn = Turn(role="user", texts=[Text(contents=["hello"])], max_tokens=1)
    info = RequestInfo(
        model_endpoint=endpoint,
        turns=[turn],
        turn_index=0,
        credit_num=0,
        credit_phase=phase,
        x_request_id="sai-req",
        x_correlation_id="session",
        conversation_id="trace",
    )
    info.payload_bytes = orjson.dumps(
        ChatEndpoint(model_endpoint=endpoint).format_payload(info)
    )
    client = SimpleNamespace(
        model_endpoint=endpoint,
        strip_record_payload_bytes=True,
        accept_usage_only_warmup=accept,
        _enrich_request_record=InferenceClient._enrich_request_record,
    )
    record = InferenceClient._finalize_request_record(
        client, record=RequestRecord(status=200), request_info=info
    )
    assert record.request_info.payload_bytes is None
    return record.request_info


def test_worker_checks_the_formatted_body_before_stripping_it() -> None:
    assert _finalized().one_token_body_streams is True
    assert _finalized(streaming=False).one_token_body_streams is False


def test_the_check_survives_the_trip_to_the_record_processor() -> None:
    """Worker and record processor are separate processes: the record crosses
    ZMQ as JSON, after the body was stripped."""
    record = RequestRecord(
        request_info=_finalized(),
        model_name="test-model",
        status=200,
        start_perf_ns=1,
        end_perf_ns=20,
        responses=_messages(_sglang_chunks()),
    )
    received = RequestRecord.model_validate_json(record.model_dump_json())
    assert received.request_info.payload_bytes is None
    assert received.request_info.one_token_body_streams is True
    assert is_usage_only_one_token_warmup(received)


@pytest.mark.parametrize("streaming", [True, False], ids=["stream", "json"])
@pytest.mark.parametrize(
    "case",
    ["flag_off", "profiling", "extra_cap", "extra_legacy_cap", "extra_n"],
)
def test_worker_marks_only_one_token_warmup_bodies(case, streaming) -> None:
    kwargs: dict = {"streaming": streaming}
    if case == "flag_off":
        kwargs["accept"] = False
    if case == "profiling":
        kwargs["phase"] = CreditPhase.PROFILING
    if case == "extra_cap":
        kwargs["extra"] = [("max_completion_tokens", 8)]
    if case == "extra_legacy_cap":
        kwargs["extra"] = [("max_tokens", 8)]
    if case == "extra_n":
        kwargs["extra"] = [("n", 2)]
    context = _finalized(**kwargs)
    assert context.one_token_body_streams is None
    # The record processor then refuses the exemption, for either reply form.
    assert not is_usage_only_one_token_warmup(_record(body_streams=None))
    assert not is_usage_only_one_token_warmup(_json_record(body_streams=None))


@pytest.mark.parametrize(
    "usage",
    [
        dict(_USAGE, reasoning_tokens=1),
        dict(_USAGE, completion_tokens_details={"reasoning_tokens": 1}),
        dict(_USAGE, completion_tokens_details=None),
    ],
)
def test_one_reasoning_token_is_accepted(usage) -> None:
    assert is_usage_only_one_token_warmup(_record(_sglang_chunks(usage)))


def test_keep_alive_comments_are_ignored() -> None:
    record = _record()
    record.responses.insert(1, SSEMessage.parse(": keep-alive", perf_ns=2))
    assert is_usage_only_one_token_warmup(record)


def _break(failure: str) -> RequestRecord:
    chunks = _sglang_chunks()
    body_streams: bool | None = True
    max_tokens = 1
    phase = CreditPhase.WARMUP
    if failure == "missing_done":
        chunks.pop()
    if failure == "two_done":
        chunks.insert(2, "[DONE]")
    if failure == "missing_finish":
        chunks[1]["choices"][0]["finish_reason"] = None
    if failure == "two_finishes":
        chunks.insert(
            2,
            {
                "id": _ID,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        )
    if failure == "abort_finish":
        chunks[1]["choices"][0]["finish_reason"] = "abort"
    if failure == "inconsistent_id":
        chunks[2]["id"] = "other"
    if failure == "missing_id":
        del chunks[2]["id"]
    if failure == "total_mismatch":
        chunks[2]["usage"]["total_tokens"] = 102
    if failure == "two_tokens":
        chunks[2]["usage"].update(completion_tokens=2, total_tokens=102)
    if failure == "no_usage":
        chunks[2]["usage"] = None
    if failure == "zero_prompt":
        chunks[2]["usage"].update(prompt_tokens=0, total_tokens=1)
    if failure == "two_reasoning":
        chunks[2]["usage"]["reasoning_tokens"] = 2
    if failure == "two_choices":
        chunks[1]["choices"].append(
            {"index": 1, "delta": {}, "finish_reason": "length"}
        )
    if failure == "choice_index":
        chunks[1]["choices"][0]["index"] = 1
    if failure == "content":
        chunks[0]["choices"][0]["delta"]["content"] = "hello"
    if failure == "reasoning_content":
        chunks[1]["choices"][0]["delta"]["reasoning_content"] = "hm"
    if failure == "tool_call":
        chunks[1]["choices"][0]["delta"]["tool_calls"] = [{"index": 0}]
    if failure == "other_role":
        chunks[0]["choices"][0]["delta"]["role"] = "tool"
    if failure == "error_chunk":
        chunks.insert(2, {"id": _ID, "error": {"message": "upstream connect error"}})
    if failure == "bad_json":
        chunks.insert(2, "{not json")
    if failure == "choice_after_finish":
        chunks.insert(2, {"id": _ID, "choices": [{"index": 0, "delta": {}}]})
    if failure == "profiling":
        phase = CreditPhase.PROFILING
    if failure == "record_max_tokens":
        max_tokens = 2
    if failure == "body_not_one_token":
        body_streams = None
    if failure == "empty_error_object":
        chunks.insert(2, {"id": _ID, "choices": [], "error": {}})
    if failure == "usage_shrinks":
        chunks.insert(
            1,
            {
                "id": _ID,
                "object": _CHUNK,
                "choices": [],
                "usage": dict(_USAGE, completion_tokens=2, total_tokens=102),
            },
        )
    if failure == "usage_goes_back":
        # 1, then 0, then 1: the upper bound alone does not catch it.
        for count in (0, 1):
            chunks.insert(
                1,
                {
                    "id": _ID,
                    "object": _CHUNK,
                    "choices": [],
                    "usage": dict(
                        _USAGE, completion_tokens=count, total_tokens=100 + count
                    ),
                },
            )
    if failure == "usage_without_count":
        chunks.insert(
            1,
            {
                "id": _ID,
                "object": _CHUNK,
                "choices": [],
                "usage": {"prompt_tokens": 100},
            },
        )
    record = _record(
        chunks, phase=phase, max_tokens=max_tokens, body_streams=body_streams
    )
    if failure == "http504":
        record.status = 504
    if failure == "error":
        record.error = ErrorDetails(type="TimeoutError", message="stream timeout")
    if failure == "cancelled":
        record.cancellation_perf_ns = 5
    if failure == "missing_end":
        record.end_perf_ns = None
    if failure == "response_after_end":
        record.responses[-1].perf_ns = record.end_perf_ns + 1
    if failure == "responses_out_of_order":
        record.responses[1].perf_ns = 1_000
        record.end_perf_ns = 2_000
    if failure == "event_error":
        record.responses.insert(
            2, SSEMessage.parse('event: error\ndata: {"id": "chatcmpl-1"}', perf_ns=4)
        )
    if failure == "text_response":
        record.responses = [
            TextResponse(perf_ns=3, text=orjson.dumps(chunks[1]).decode())
        ]
    if failure == "no_responses":
        record.responses = []
    return record


_FAILURES = [
    "missing_done",
    "two_done",
    "missing_finish",
    "two_finishes",
    "abort_finish",
    "inconsistent_id",
    "missing_id",
    "total_mismatch",
    "two_tokens",
    "no_usage",
    "zero_prompt",
    "two_reasoning",
    "two_choices",
    "choice_index",
    "content",
    "reasoning_content",
    "tool_call",
    "other_role",
    "error_chunk",
    "bad_json",
    "choice_after_finish",
    "profiling",
    "record_max_tokens",
    "body_not_one_token",
    "empty_error_object",
    "usage_shrinks",
    "usage_without_count",
    "usage_goes_back",
    "http504",
    "error",
    "cancelled",
    "missing_end",
    "response_after_end",
    "responses_out_of_order",
    "event_error",
    "text_response",
    "no_responses",
]


@pytest.mark.parametrize("failure", _FAILURES)
def test_rule_requires_a_complete_successful_one_token_warmup(failure) -> None:
    assert not is_usage_only_one_token_warmup(_break(failure))


def _sglang_completion(usage: dict | None = None) -> dict:
    """SGLang's non-streamed reply when the reasoning parser removes the one token."""
    return {
        "id": _ID,
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": None,
                    "tool_calls": None,
                },
                "logprobs": None,
                "finish_reason": "length",
                "matched_stop": None,
            }
        ],
        "usage": usage
        or dict(_USAGE, reasoning_tokens=0, prompt_tokens_details={"cached_tokens": 0}),
    }


def _json_record(
    completion: dict | str | None = None,
    *,
    phase: CreditPhase = CreditPhase.WARMUP,
    max_tokens: int = 1,
    body_streams: bool | None = False,
) -> RequestRecord:
    """A non-streamed warmup record: one text response, as the transport keeps it."""
    completion = _sglang_completion() if completion is None else completion
    text = (
        completion if isinstance(completion, str) else orjson.dumps(completion).decode()
    )
    record = _record([], phase=phase, max_tokens=max_tokens, body_streams=body_streams)
    record.end_perf_ns = 20
    record.responses = [
        TextResponse(perf_ns=20, text=text, content_type="application/json")
    ]
    return record


def _json_variant(name: str) -> dict:
    completion = _sglang_completion()
    choice = completion["choices"][0]
    if name == "finish_stop":
        choice["finish_reason"] = "stop"
    if name == "empty_content":
        choice["message"] = {"role": "assistant", "content": ""}
    if name == "null_error":
        completion["error"] = None
    if name == "one_reasoning_token":
        completion["usage"]["reasoning_tokens"] = 1
    return completion


@pytest.mark.parametrize(
    "name",
    ["sglang", "finish_stop", "empty_content", "null_error", "one_reasoning_token"],
)
def test_complete_non_streamed_one_token_reply_is_accepted(name) -> None:
    assert is_usage_only_one_token_warmup(_json_record(_json_variant(name)))


def _json_break(failure: str) -> RequestRecord:
    completion: dict | str = _sglang_completion()
    choice = completion["choices"][0]
    kwargs: dict = {}
    if failure == "content":
        choice["message"]["content"] = "hello"
    if failure == "reasoning_content":
        choice["message"]["reasoning_content"] = "hm"
    if failure == "tool_call":
        choice["message"]["tool_calls"] = [{"id": "call"}]
    if failure == "other_role":
        choice["message"]["role"] = "tool"
    if failure == "no_message":
        del choice["message"]
    if failure == "missing_finish":
        choice["finish_reason"] = None
    if failure == "abort_finish":
        choice["finish_reason"] = "abort"
    if failure == "choice_index":
        choice["index"] = 1
    if failure == "two_choices":
        completion["choices"].append(dict(choice, index=1))
    if failure == "no_choices":
        completion["choices"] = []
    if failure == "missing_id":
        del completion["id"]
    if failure == "error_object":
        completion["error"] = {"message": "upstream connect error"}
    if failure == "two_tokens":
        completion["usage"].update(completion_tokens=2, total_tokens=102)
    if failure == "no_usage":
        del completion["usage"]
    if failure == "total_mismatch":
        completion["usage"]["total_tokens"] = 102
    if failure == "two_reasoning":
        completion["usage"]["reasoning_tokens"] = 2
    if failure == "bad_json":
        completion = "{not json"
    if failure == "not_an_object":
        completion = "[1]"
    if failure == "profiling":
        kwargs["phase"] = CreditPhase.PROFILING
    if failure == "record_max_tokens":
        kwargs["max_tokens"] = 2
    if failure == "body_not_one_token":
        kwargs["body_streams"] = None
    if failure == "body_streamed":
        kwargs["body_streams"] = True
    record = _json_record(completion, **kwargs)
    if failure == "http504":
        record.status = 504
    if failure == "error":
        record.error = ErrorDetails(type="TimeoutError", message="request timeout")
    if failure == "cancelled":
        record.cancellation_perf_ns = 5
    if failure == "missing_end":
        record.end_perf_ns = None
    if failure == "response_after_end":
        record.responses[0].perf_ns = record.end_perf_ns + 1
    if failure == "two_responses":
        record.responses.append(record.responses[0])
    if failure == "sse_reply":
        record.responses = _messages(_sglang_chunks())
    if failure == "no_responses":
        record.responses = []
    return record


_JSON_FAILURES = [
    "content",
    "reasoning_content",
    "tool_call",
    "other_role",
    "no_message",
    "missing_finish",
    "abort_finish",
    "choice_index",
    "two_choices",
    "no_choices",
    "missing_id",
    "error_object",
    "two_tokens",
    "no_usage",
    "total_mismatch",
    "two_reasoning",
    "bad_json",
    "not_an_object",
    "profiling",
    "record_max_tokens",
    "body_not_one_token",
    "body_streamed",
    "http504",
    "error",
    "cancelled",
    "missing_end",
    "response_after_end",
    "two_responses",
    "sse_reply",
    "no_responses",
]


@pytest.mark.parametrize("failure", _JSON_FAILURES)
def test_non_streamed_rule_requires_a_complete_successful_one_token_warmup(
    failure,
) -> None:
    assert not is_usage_only_one_token_warmup(_json_break(failure))


def _parser(accept: bool, streaming: bool = True) -> InferenceResultParser:
    from tests.unit.conftest import make_run_from_cli

    def communication_init(self, run, **kwargs):
        from aiperf.common.mixins.aiperf_lifecycle_mixin import AIPerfLifecycleMixin

        AIPerfLifecycleMixin.__init__(self, **kwargs)
        self.run = run
        for method in [
            "trace_or_debug",
            "debug",
            "info",
            "warning",
            "error",
            "exception",
        ]:
            setattr(self, method, MagicMock())

    flags = {"warmup_accept_usage_only": True} if accept else {}
    cli = CLIConfig(
        model_names=["test-model"],
        streaming=streaming,
        use_server_token_count=True,
        **flags,
    )
    with patch("aiperf.common.mixins.CommunicationMixin.__init__", communication_init):
        return InferenceResultParser(run=make_run_from_cli(cli))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chunks", [_sglang_chunks(), _rill_chunks()], ids=["sglang", "rill"]
)
async def test_default_keeps_the_content_check(chunks) -> None:
    parsed = await _parser(accept=False).parse_request_record(_record(chunks))
    assert parsed.has_error
    assert parsed.request.error.type == "InvalidInferenceResultError"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chunks", [_sglang_chunks(), _rill_chunks()], ids=["sglang", "rill"]
)
async def test_flag_accepts_the_reply_without_inventing_content(chunks) -> None:
    parsed = await _parser(accept=True).parse_request_record(_record(chunks))
    assert parsed.valid and not parsed.has_error
    assert parsed.usage_only_warmup
    assert parsed.content_responses == []
    assert parsed.token_counts.input == 100
    assert parsed.token_counts.output + (parsed.token_counts.reasoning or 0) == 1
    assert parsed.request.responses is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["profiling", "http504", "error_chunk", "cancelled", "two_tokens"]
)
async def test_flag_never_exempts_profiling_or_failed_requests(failure) -> None:
    parsed = await _parser(accept=True).parse_request_record(_break(failure))
    assert not parsed.valid
    assert not parsed.usage_only_warmup


@pytest.mark.asyncio
async def test_flag_does_not_change_a_reply_with_text() -> None:
    chunks = _sglang_chunks()
    chunks[0]["choices"][0]["delta"]["content"] = "Hi"
    parsed = await _parser(accept=True).parse_request_record(_record(chunks))
    assert parsed.valid and not parsed.usage_only_warmup
    assert len(parsed.content_responses) == 1


def test_cli_flag_reaches_the_benchmark_config() -> None:
    from tests.unit.conftest import make_run_from_cli

    off = make_run_from_cli(CLIConfig(model_names=["m"]))
    on = make_run_from_cli(CLIConfig(model_names=["m"], warmup_accept_usage_only=True))
    assert off.cfg.warmup_accept_usage_only is False
    assert on.cfg.warmup_accept_usage_only is True


@pytest.mark.asyncio
async def test_non_streamed_default_keeps_the_content_check() -> None:
    parsed = await _parser(accept=False, streaming=False).parse_request_record(
        _json_record()
    )
    assert parsed.has_error
    assert parsed.request.error.type == "InvalidInferenceResultError"


@pytest.mark.asyncio
async def test_flag_accepts_the_non_streamed_reply_without_inventing_content() -> None:
    parsed = await _parser(accept=True, streaming=False).parse_request_record(
        _json_record()
    )
    assert parsed.valid and not parsed.has_error
    assert parsed.usage_only_warmup
    assert parsed.content_responses == []
    assert parsed.token_counts.input == 100
    assert parsed.token_counts.output + (parsed.token_counts.reasoning or 0) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["profiling", "http504", "two_tokens", "error_object"]
)
async def test_flag_never_exempts_non_streamed_profiling_or_failed_requests(
    failure,
) -> None:
    parsed = await _parser(accept=True, streaming=False).parse_request_record(
        _json_break(failure)
    )
    assert not parsed.valid
    assert not parsed.usage_only_warmup


@pytest.mark.asyncio
async def test_flag_does_not_change_a_non_streamed_reply_with_text() -> None:
    completion = _sglang_completion()
    completion["choices"][0]["message"]["content"] = "Hi"
    parsed = await _parser(accept=True, streaming=False).parse_request_record(
        _json_record(completion)
    )
    assert parsed.valid and not parsed.usage_only_warmup
    assert len(parsed.content_responses) == 1
