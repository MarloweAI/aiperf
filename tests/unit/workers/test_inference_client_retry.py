# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retrying failed requests for reliability experiments.

Failure modes covered: a replica lost before the first token (gateway 503),
a replica lost mid-stream (stream ends without [DONE]), a non-retryable client
error, a client-side exception without an HTTP status, the attempt cap, retries
left off by default, endpoints that never send [DONE], a second first-token
report for one request, and the phase cancelling a request during its wait.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.enums import ModelSelectionStrategy
from aiperf.common.models import ErrorDetails, SSEMessage
from aiperf.common.models.dataset_models import Text, Turn
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import RequestInfo, RequestRecord, SSEField
from aiperf.plugin.enums import EndpointType
from aiperf.workers.inference_client import InferenceClient


def make_client(
    endpoint_type=EndpointType.CHAT, **endpoint_options
) -> tuple[InferenceClient, MagicMock]:
    model_endpoint = ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name="test-model")],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        ),
        endpoint=EndpointInfo(
            type=endpoint_type,
            base_url="http://localhost:8000/v1/test",
            streaming=True,
            **endpoint_options,
        ),
    )
    transport = MagicMock()
    endpoint = MagicMock()
    endpoint.get_endpoint_headers.return_value = {}
    endpoint.get_endpoint_params.return_value = {}
    endpoint.format_payload.return_value = {}
    entry = MagicMock()
    entry.name = "http"
    entry.metadata = {"url_schemes": ["http", "https"]}

    def get_class(protocol, name):
        return (
            (lambda **kwargs: endpoint)
            if protocol == "endpoint"
            else (lambda **kwargs: transport)
        )

    with (
        patch(
            "aiperf.workers.inference_client.plugins.get_class", side_effect=get_class
        ),
        patch(
            "aiperf.workers.inference_client.plugins.list_entries", return_value=[entry]
        ),
    ):
        client = InferenceClient(model_endpoint=model_endpoint, service_id="svc")
    return client, transport


def request_info(client: InferenceClient) -> RequestInfo:
    return RequestInfo(
        model_endpoint=client.model_endpoint,
        turns=[Turn(texts=[Text(contents=["hi"])])],
        turn_index=0,
        credit_num=0,
        credit_phase="profiling",
        x_request_id="sai-00000000-0000-0000-0000-000000000001",
        x_correlation_id="c",
        conversation_id="conv",
    )


def failed(start: int, code: int | None, kind: str) -> RequestRecord:
    return RequestRecord(
        start_perf_ns=start, timestamp_ns=start, end_perf_ns=start + 1,
        error=ErrorDetails(code=code, type=kind, message=kind),
    )  # fmt: skip


def streamed(start: int, *, done: bool) -> RequestRecord:
    packets = [
        SSEMessage(perf_ns=start + 5, packets=[SSEField(name="data", value='{"x":1}')])
    ]
    if done:
        packets.append(
            SSEMessage(
                perf_ns=start + 6, packets=[SSEField(name="data", value="[DONE]")]
            )
        )
    return RequestRecord(start_perf_ns=start, timestamp_ns=start, end_perf_ns=start + 7, responses=packets)  # fmt: skip


@pytest.fixture
def no_wait():
    with patch(
        "aiperf.workers.inference_client.asyncio.sleep", new=AsyncMock()
    ) as sleep:
        yield sleep


@pytest.mark.asyncio
async def test_lost_replica_is_retried_and_measured_from_the_first_attempt(no_wait):
    client, transport = make_client(retry_failed_requests=True, retry_wait_seconds=10.0)
    transport.send_request = AsyncMock(
        side_effect=[
            failed(100, 503, "Service Unavailable"),
            streamed(200, done=False),
            streamed(300, done=True),
        ]
    )
    record = await client.send_request(request_info(client))
    assert record.error is None
    assert record.start_perf_ns == 100
    assert [error.type for error in record.retried_errors] == [
        "Service Unavailable",
        "IncompleteStreamError",
    ]
    assert transport.send_request.await_count == 3
    assert [call.args for call in no_wait.await_args_list] == [(10.0,), (10.0,)]
    sent_ids = {
        call.args[0].x_request_id for call in transport.send_request.await_args_list
    }
    assert sent_ids == {"sai-00000000-0000-0000-0000-000000000001"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options,responses,attempts",
    [
        pytest.param({"retry_failed_requests": True}, [failed(1, 400, "Bad Request")], 1, id="client_error"),
        pytest.param({"retry_failed_requests": True}, [failed(1, 499, "RequestCancellationError")], 1, id="cancelled"),
        pytest.param({"retry_failed_requests": True, "retry_max_attempts": 2}, [failed(1, 503, "x"), failed(2, None, "ClientPayloadError")], 2, id="cap"),
        pytest.param({}, [failed(1, 503, "x")], 1, id="off_by_default"),
        pytest.param({}, [streamed(1, done=False)], 1, id="off_keeps_short_streams"),
        pytest.param({"retry_failed_requests": True}, [failed(1, None, "ValueError")], 1, id="client_bug_without_status"),
    ],
)  # fmt: skip
async def test_requests_that_must_not_be_retried(no_wait, options, responses, attempts):
    client, transport = make_client(**options)
    transport.send_request = AsyncMock(side_effect=responses)
    record = await client.send_request(request_info(client))
    assert transport.send_request.await_count == attempts
    assert len(record.retried_errors) == attempts - 1


@pytest.mark.asyncio
async def test_endpoints_without_done_marker_keep_their_complete_streams(no_wait):
    client, transport = make_client(EndpointType.RESPONSES, retry_failed_requests=True)
    transport.send_request = AsyncMock(side_effect=[streamed(1, done=False)])
    record = await client.send_request(request_info(client))
    assert record.error is None and transport.send_request.await_count == 1


@pytest.mark.asyncio
async def test_first_token_is_reported_once_across_attempts(no_wait):
    client, transport = make_client(retry_failed_requests=True)
    reported = AsyncMock(return_value=True)

    async def attempt(info, payload, first_token_callback=None):
        message = SSEMessage(
            perf_ns=1, packets=[SSEField(name="data", value='{"x":1}')]
        )
        if first_token_callback is not None:
            await first_token_callback(1, message)
        return streamed(1, done=transport.send_request.await_count > 1)

    transport.send_request = AsyncMock(side_effect=attempt)
    record = await client.send_request(
        request_info(client), first_token_callback=reported
    )
    assert record.error is None and transport.send_request.await_count == 2
    assert reported.await_count == 1


@pytest.mark.asyncio
async def test_phase_cancellation_during_the_wait_ends_the_request():
    client, transport = make_client(retry_failed_requests=True)
    transport.send_request = AsyncMock(
        side_effect=[failed(1, 503, "Service Unavailable")]
    )
    with (
        patch(
            "aiperf.workers.inference_client.asyncio.sleep",
            new=AsyncMock(side_effect=asyncio.CancelledError),
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        await client.send_request(request_info(client))
    assert transport.send_request.await_count == 1
