# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recognize a complete one-token warmup reply that has no text.

Agentic replay sends its warmup requests with ``max_tokens=1``. A reasoning
model can spend that one token on a marker such as ``</think>``. The server's
reasoning parser removes the marker, so the stream has no text: only a finish
chunk, a usage chunk and ``[DONE]``. ``ParsedResponseRecord.valid`` needs text,
so aiperf counts this complete reply as a failed request.

``--warmup-accept-usage-only`` accepts such a reply as valid. This module holds
the rule in two parts:

- ``body_streams_one_token`` reads the request body as sent. The worker calls it
  before it may strip ``payload_bytes`` and keeps the result in
  ``RecordContext.streams_one_token``.
- ``is_usage_only_one_token_warmup`` reads the raw SSE stream, so the record
  processor calls it before it frees ``RequestRecord.responses``.
"""

from __future__ import annotations

import sys
from typing import Any

import orjson

from aiperf.common.enums import CreditPhase
from aiperf.common.models.record_models import RequestRecord, SSEMessage

_DONE = "[DONE]"
_FINISH_REASONS = frozenset({"length", "stop"})
_WIRE_TOKEN_CAPS = ("max_tokens", "max_completion_tokens")


def is_usage_only_one_token_warmup(record: RequestRecord) -> bool:
    """Return True only for a complete warmup reply of one token without text.

    All of these must be true:

    - The request is in the warmup phase, its ``max_tokens`` is 1, and the
      worker found that the body as sent streams one choice capped at 1 token
      (``RecordContext.streams_one_token``).
    - HTTP status 200, no error, not cancelled, valid timestamps.
    - Every response is an SSE message. The stream ends with exactly one
      ``[DONE]``. All other data are JSON chunks with one ``id`` and no
      ``error`` key with a value other than null.
    - At most one choice per chunk, with index 0. No delta has text: every
      delta value other than ``role: "assistant"`` is empty.
    - Exactly one ``finish_reason``, ``length`` or ``stop``. No choice follows it.
    - No usage report shows more than 1 completion token or a smaller count than
      an earlier report. The last usage reports ``completion_tokens`` 1,
      ``prompt_tokens`` > 0, ``total_tokens`` = prompt + completion, and 0 or 1
      reasoning tokens.

    A profiling request is never accepted. A reply with text is never accepted
    here, because the normal check accepts it.
    """
    info = record.request_info
    if (
        info is None
        or info.credit_phase != CreditPhase.WARMUP
        or info.max_tokens != 1
        or not info.streams_one_token
        or record.status != 200
        or record.error is not None
        or record.was_cancelled
        or record.end_perf_ns is None
        or not 0 <= record.start_perf_ns < record.end_perf_ns < sys.maxsize
    ):
        return False
    data = _stream_data(record)
    if data is None or not data or data[-1] != _DONE or _DONE in data[:-1]:
        return False
    try:
        chunks = [orjson.loads(item) for item in data[:-1]]
    except orjson.JSONDecodeError:
        return False
    return _chunks_are_one_finished_token(chunks)


def body_streams_one_token(payload_bytes: bytes | None) -> bool:
    """Return True when the request body as sent streams one choice capped at 1 token.

    The body is the final one: endpoint extras and ``extra_body`` can replace
    the turn's ``max_tokens``, so ``RecordContext.max_tokens`` alone is not
    enough. Every token cap in the body must be 1.
    """
    if payload_bytes is None:
        return False
    try:
        payload = orjson.loads(payload_bytes)
    except orjson.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    caps = [payload[key] for key in _WIRE_TOKEN_CAPS if key in payload]
    n = payload.get("n", 1)
    return (
        bool(caps)
        and all(type(cap) is int and cap == 1 for cap in caps)
        and payload.get("stream") is True
        and type(n) is int
        and n == 1
    )


def _stream_data(record: RequestRecord) -> list[str] | None:
    """Return the non-empty SSE data in order, or None if the stream is wrong.

    Messages without data (for example keep-alive comments) are skipped.
    """
    data: list[str] = []
    previous = record.start_perf_ns
    for response in record.responses or []:
        if not isinstance(response, SSEMessage):
            return None
        if not previous <= response.perf_ns <= record.end_perf_ns:
            return None
        previous = response.perf_ns
        if any(
            packet.name == "event" and packet.value == "error"
            for packet in response.packets
        ):
            return None
        if content := response.extract_data_content():
            data.append(content)
    return data


def _chunks_are_one_finished_token(chunks: list[Any]) -> bool:
    identifier: str | None = None
    finished = False
    usage: dict[str, Any] | None = None
    completion = 0
    for chunk in chunks:
        if not isinstance(chunk, dict) or chunk.get("error") is not None:
            return False
        chunk_id = chunk.get("id")
        if not isinstance(chunk_id, str) or not chunk_id:
            return False
        if identifier is None:
            identifier = chunk_id
        elif chunk_id != identifier:
            return False
        choices = chunk.get("choices")
        if not isinstance(choices, list) or len(choices) > 1:
            return False
        for choice in choices:
            if (
                not isinstance(choice, dict)
                or finished
                or choice.get("index") != 0
                or not _delta_has_no_text(choice.get("delta"))
            ):
                return False
            reason = choice.get("finish_reason")
            if reason is not None:
                if reason not in _FINISH_REASONS:
                    return False
                finished = True
        if chunk.get("usage"):
            usage = chunk["usage"]
            # Cumulative reports may grow (0, then 1) but never pass 1 or shrink.
            count = usage.get("completion_tokens") if isinstance(usage, dict) else None
            if type(count) is not int or not completion <= count <= 1:
                return False
            completion = count
    return finished and _usage_is_one_token(usage)


def _delta_has_no_text(delta: Any) -> bool:
    if delta is None:
        return True
    if not isinstance(delta, dict):
        return False
    return all(
        (key == "role" and value == "assistant") or value in (None, "", [], {})
        for key, value in delta.items()
    )


def _usage_is_one_token(usage: Any) -> bool:
    if not isinstance(usage, dict):
        return False
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    total = usage.get("total_tokens")
    if not all(type(value) is int for value in (prompt, completion, total)):
        return False
    if prompt <= 0 or completion != 1 or total != prompt + completion:
        return False
    # OpenAI reports reasoning tokens in completion_tokens_details. SGLang and
    # Rill report them at the top level.
    details = usage.get("completion_tokens_details") or {}
    if not isinstance(details, dict):
        return False
    for reasoning in (usage.get("reasoning_tokens"), details.get("reasoning_tokens")):
        if reasoning is not None and (
            type(reasoning) is not int or reasoning not in (0, 1)
        ):
            return False
    return True
