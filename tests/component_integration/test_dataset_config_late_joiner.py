# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for record-path services that miss DatasetConfiguredNotification.

DatasetManager publishes DatasetConfiguredNotification once, on PUB/SUB, with no
replay. RecordProcessors are spawned with the workers, after the controller's
registration barrier, so when the dataset configures quickly (a dataset mmap
cache hit configures in well under a second) a RecordProcessor can subscribe
after the broadcast. It then waited DATASET.CONFIGURATION_TIMEOUT on its first
record and aborted the run with no records processed.
"""

from __future__ import annotations

import pytest

from aiperf.common.environment import Environment
from aiperf.common.messages import DatasetConfiguredNotification
from aiperf.plugin.enums import ServiceType
from aiperf.records.records_manager import RecordsManager
from tests.harness import (
    FakeCommunication,  # noqa: F401
    FakeServiceManager,
    FakeTransport,  # noqa: F401
)
from tests.harness.fake_communication import FakePubClient
from tests.harness.utils import AIPerfCLI

pytestmark = pytest.mark.component_integration

_REQUEST_COUNT = 8

_PROFILE_CMD = f"""
    aiperf profile \
        --model gpt2 \
        --endpoint-type chat \
        --request-count {_REQUEST_COUNT} \
        --concurrency 2 \
        --osl 2 \
        --isl 2 \
        --workers-max 2 \
        --random-seed 42 \
        --ui simple
"""

# A missed notification used to cost the whole configuration timeout. Keep it
# short so a regression fails in seconds; a configured run never waits on it.
_CONFIGURATION_TIMEOUT = 5.0


@pytest.fixture
def short_dataset_configuration_timeout(monkeypatch) -> None:
    monkeypatch.setattr(
        Environment.DATASET, "CONFIGURATION_TIMEOUT", _CONFIGURATION_TIMEOUT
    )


def _assert_every_request_recorded(result) -> None:
    assert result.exit_code == 0, result.stderr[-4000:]
    assert result.request_count == _REQUEST_COUNT
    assert result.jsonl is not None
    assert len(result.jsonl) == _REQUEST_COUNT


@pytest.mark.component_integration
@pytest.mark.usefixtures("short_dataset_configuration_timeout")
class TestDatasetConfigLateJoiner:
    def test_record_processor_started_after_notification_processes_records(
        self, cli: AIPerfCLI, monkeypatch
    ):
        """The observed failure: the RecordProcessor registers only after
        DatasetManager has published, so the broadcast never reaches it."""
        original_run_service = FakeServiceManager.run_service
        original_publish = FakePubClient.publish
        deferred: list[tuple[FakeServiceManager, int]] = []
        started_late: list[int] = []
        published = False

        async def run_service(self, service_type, num_replicas=1):
            if service_type == ServiceType.RECORD_PROCESSOR and not published:
                deferred.append((self, num_replicas))
                return
            await original_run_service(self, service_type, num_replicas)

        async def publish(self, message):
            nonlocal published
            await original_publish(self, message)
            if isinstance(message, DatasetConfiguredNotification) and not published:
                published = True
                # Start the deferred replicas before DatasetManager answers
                # PROFILE_CONFIGURE, so they are up before any record exists.
                for manager, num_replicas in deferred:
                    await original_run_service(
                        manager, ServiceType.RECORD_PROCESSOR, num_replicas
                    )
                    started_late.append(num_replicas)

        monkeypatch.setattr(FakeServiceManager, "run_service", run_service)
        monkeypatch.setattr(FakePubClient, "publish", publish)

        result = cli.run_sync(_PROFILE_CMD, timeout=60.0, assert_success=False)

        assert started_late, "no RecordProcessor was started after the notification"
        _assert_every_request_recorded(result)

    def test_records_manager_that_missed_notification_processes_records(
        self, cli: AIPerfCLI, monkeypatch
    ):
        """RecordsManager is gated the same way; withhold the broadcast from
        its subscription as if it had not subscribed yet."""
        original_publish = FakePubClient.publish
        withheld: list[RecordsManager] = []

        async def publish(self, message):
            if not isinstance(message, DatasetConfiguredNotification) or withheld:
                await original_publish(self, message)
                return
            topic = self._determine_topic(message)
            for comm in self.bus.communications:
                for sub_client in comm.sub_clients:
                    callbacks = sub_client.subscriptions.get(topic, [])
                    for callback in list(callbacks):
                        owner = getattr(callback, "__self__", None)
                        if isinstance(owner, RecordsManager):
                            callbacks.remove(callback)
                            withheld.append(owner)
            await original_publish(self, message)

        monkeypatch.setattr(FakePubClient, "publish", publish)

        result = cli.run_sync(_PROFILE_CMD, timeout=60.0, assert_success=False)

        assert withheld, "the notification was not withheld from RecordsManager"
        _assert_every_request_recorded(result)
