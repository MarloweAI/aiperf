# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The existing worker notification closes sibling retry gates, preserving drain."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiperf.common.messages import SessionProviderDepartureNotification
from aiperf.workers.worker import Worker
from tests.unit.workers.test_inference_client_retry import (
    failed,
    make_client,
    request_info,
    streamed,
)


@pytest.mark.asyncio
async def test_child_refusal_propagates_to_waiting_sibling_without_a_second_http_attempt():
    sibling, sibling_transport = make_client(retry_policy="fixed", retry_wait_seconds=5)
    refusing, refusing_transport = make_client(
        retry_policy="fixed", retry_on_429="leave_provider"
    )
    sibling_transport.send_request = AsyncMock(
        return_value=failed(100, 500, "retryable")
    )
    refusing_transport.send_request = AsyncMock(
        return_value=failed(100, 429, "refused")
    )
    receiving_worker = SimpleNamespace(inference_client=sibling)
    notifications = []

    async def publish(message):
        notifications.append(message)
        await Worker._on_provider_departure(receiving_worker, message)

    sending_worker = SimpleNamespace(
        inference_client=refusing, service_id="worker-b", publish=publish
    )
    refusing.depart_root = lambda root, observed: Worker.depart_provider_root(
        sending_worker, root, observed
    )
    waiting = asyncio.Event()
    event = sibling.provider_departure_event("c")
    original_wait = event.wait

    async def observe_wait():
        waiting.set()
        return await original_wait()

    event.wait = observe_wait
    pending = asyncio.create_task(sibling.send_request(request_info(sibling)))
    await waiting.wait()
    refusal = await refusing.send_request(request_info(refusing))
    drained = await pending
    assert refusal.retry_terminal_reason == "http_429_rerouted_away"
    assert drained.retry_terminal_reason == "root_departed"
    assert drained.error.code == 500
    assert sibling_transport.send_request.await_count == 1
    assert refusing_transport.send_request.await_count == 1
    assert len(notifications) == 1
    assert isinstance(notifications[0], SessionProviderDepartureNotification)
    assert (
        sibling.provider_departure_times["c"]["observed_ns"]
        == notifications[0].observed_ns
    )


@pytest.mark.asyncio
async def test_departure_does_not_cancel_already_started_sibling_transport():
    client, transport = make_client(retry_policy="fixed")
    started = asyncio.Event()
    finish = asyncio.Event()

    async def active(*args, **kwargs):
        started.set()
        await finish.wait()
        return streamed(100, done=True)

    transport.send_request = AsyncMock(side_effect=active)
    pending = asyncio.create_task(client.send_request(request_info(client)))
    await started.wait()
    client.apply_provider_departure("c", 1)
    finish.set()
    result = await pending
    assert result.error is None
    assert transport.send_request.await_count == 1
