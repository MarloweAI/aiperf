# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from dataclasses import asdict
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import orjson

from aiperf.common.enums import CreditPhase, RequestContentType
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import (
    ErrorDetails,
    ModelEndpointInfo,
    RecordContext,
    RequestInfo,
    RequestRecord,
    SSEMessage,
)
from aiperf.common.redact import redact_headers
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, TransportType
from aiperf.records.usage_only_warmup import body_streams_one_token
from aiperf.workers.retry_policy import (
    decide_retry,
    remaining_attempt_seconds,
    resolve_retry_policy,
    retry_hint_from_record,
    retry_jitter_draw,
)

if TYPE_CHECKING:
    from aiperf.transports.base_transports import FirstTokenCallback


# Failures without an HTTP status that a lost or restarting replica causes. Any
# other exception without a status (a payload or client bug) would fail the same
# way on every attempt, so it is not retried.
TRANSPORT_ERROR_TYPES: frozenset[str] = frozenset(
    {
        "ClientConnectionError",
        "ClientConnectorError",
        "ClientOSError",
        "ClientPayloadError",
        "ConnectionRefusedError",
        "ConnectionResetError",
        "ContentLengthError",
        "IncompleteStreamError",
        "ServerDisconnectedError",
        "ServerTimeoutError",
        "TimeoutError",
        "TransferEncodingError",
    }
)

# Endpoints whose SSE streams end with `data: [DONE]` (OpenAI chat and
# completions); a stream from these that stops before it was cut short.
DONE_TERMINATED_ENDPOINTS: frozenset[str] = frozenset({"chat", "completions"})


def is_retryable_error(error: ErrorDetails | None) -> bool:
    """Whether a user would retry this failure.

    Server faults (5xx), rate limits (429) and transport failures are retried.
    Other 4xx responses, including context overflow, would fail the same way
    again; a cancellation (499) is the benchmark ending the request.
    """
    if error is None:
        return False
    code = error.code
    if code is None or code == 0:
        return error.type in TRANSPORT_ERROR_TYPES
    return code == 429 or code >= 500


def mark_incomplete_stream(record: RequestRecord) -> None:
    """Fail an SSE response that ended without `data: [DONE]`.

    A replica lost mid-answer can end the stream at a chunk boundary without a
    transport error; the record would otherwise count as a short success.
    """
    if record.error is not None or not record.responses:
        return
    if not all(isinstance(message, SSEMessage) for message in record.responses):
        return
    for message in reversed(record.responses):
        if any(
            packet.name == "data" and packet.value == "[DONE]"
            for packet in message.packets
        ):
            return
    record.error = ErrorDetails(
        type="IncompleteStreamError",
        message="SSE stream ended before data: [DONE]",
    )


def _once(
    callback: FirstTokenCallback | None, first: RequestRecord
) -> FirstTokenCallback | None:
    """Report a request's first token once across its attempts.

    The worker's callback tells the router a prefill finished. A stream cut after
    its first token and then retried must not report a second one, so later
    attempts skip the callback once any attempt, including the first, reported it.
    """
    if callback is None:
        return None
    reported = any(isinstance(message, SSEMessage) for message in first.responses)

    async def wrapper(ttft_ns: int, message: SSEMessage) -> bool:
        nonlocal reported
        if reported:
            return True
        reported = await callback(ttft_ns, message)
        return reported

    return wrapper


def detect_transport_from_url(url: str) -> str:
    """Detect transport type from URL scheme.

    Looks up registered transports and matches their url_schemes metadata
    against the URL's scheme.

    Args:
        url: URL to detect transport for.

    Returns:
        Transport plugin name (e.g., 'http').

    Raises:
        ValueError: If no transport supports the URL scheme.
    """
    parsed = urlparse(url)
    # urlparse mishandles URLs without schemes (e.g., 'localhost:8765')
    if parsed.scheme and not parsed.netloc:
        parsed = urlparse(f"http://{url}")
    scheme = parsed.scheme.lower() if parsed.scheme else "http"

    for entry in plugins.list_entries(PluginType.TRANSPORT):
        if scheme in entry.metadata.get("url_schemes", []):
            return entry.name

    raise ValueError(f"No transport found for URL scheme '{scheme}' in: {url}")


class InferenceClient(AIPerfLifecycleMixin):
    """Inference client for the worker."""

    def __init__(
        self,
        model_endpoint: ModelEndpointInfo,
        service_id: str,
        *,
        strip_record_payload_bytes: bool = False,
        accept_usage_only_warmup: bool = False,
        **kwargs,
    ):
        super().__init__(model_endpoint=model_endpoint, service_id=service_id, **kwargs)
        self.model_endpoint = model_endpoint
        self.service_id = service_id
        self._provider_departures: dict[str, asyncio.Event] = {}
        self.provider_departure_times: dict[str, dict[str, int]] = {}
        # When True, omit canonical request payload bytes from the slim
        # RecordContext after dispatch (memory optimization for large prompts).
        # Resolved by the worker via record payload-retention auto-detection.
        self.strip_record_payload_bytes = strip_record_payload_bytes
        # ``--warmup-accept-usage-only``: check one-token warmup bodies before
        # stripping them (``aiperf.records.usage_only_warmup``).
        self.accept_usage_only_warmup = accept_usage_only_warmup

        # Detect and set transport type if not explicitly set
        if not model_endpoint.transport:
            model_endpoint.transport = TransportType(
                detect_transport_from_url(model_endpoint.endpoint.base_url)
            )

        # Create endpoint and transport instances
        EndpointClass = plugins.get_class(
            PluginType.ENDPOINT, self.model_endpoint.endpoint.type
        )
        self.endpoint = EndpointClass(model_endpoint=self.model_endpoint)
        TransportClass = plugins.get_class(
            PluginType.TRANSPORT, str(self.model_endpoint.transport)
        )
        self.transport = TransportClass(model_endpoint=self.model_endpoint)
        self.attach_child_lifecycle(self.transport)

    async def _send_request_to_transport(
        self,
        request_info: RequestInfo,
        first_token_callback: FirstTokenCallback | None = None,
    ) -> RequestRecord:
        """Send request via transport.

        Handles the complete request lifecycle:
        1. Populates endpoint headers and params on request_info
        2. Formats the payload using the endpoint
        3. Sends the request via the transport

        Note: Cancellation is handled by the transport layer, which ensures the
        request is always sent before being cancelled (simulating real client behavior).

        Args:
            request_info: The request information (includes cancel_after_ns).
            first_token_callback: Optional callback fired on first SSE message with ttft_ns

        Returns:
            RequestRecord containing the response data and metadata.
        """
        request_info.endpoint_headers = self.endpoint.get_endpoint_headers(request_info)
        request_info.endpoint_params = self.endpoint.get_endpoint_params(request_info)
        if request_info.payload_bytes is not None:
            formatted_payload: dict[str, Any] | bytes = request_info.payload_bytes
        else:
            current_turn = request_info.turns[-1] if request_info.turns else None
            if current_turn and current_turn.raw_payload is not None:
                formatted_payload = current_turn.raw_payload
            else:
                formatted_payload = self.endpoint.format_payload(request_info)
        # Canonicalise to bytes and stash on request_info. Two wins: (1) the
        # transport skips its own orjson.dumps on the dict path, (2) the
        # record processor can read the exact wire payload for raw-export.
        # Multipart endpoints (image_edit / image_generation / video) are the
        # exception: the transport must receive the structured dict so it can
        # build the FormData. Pre-dumped bytes would be sent verbatim as JSON
        # and the server's form parser would reject them (422, prompt=null).
        wire_payload: dict[str, Any] | bytes = formatted_payload
        if isinstance(formatted_payload, dict):
            encoded = orjson.dumps(formatted_payload)
            request_info.payload_bytes = encoded
            is_multipart = (
                self.model_endpoint.endpoint.request_content_type
                == RequestContentType.MULTIPART_FORM_DATA
            )
            wire_payload = formatted_payload if is_multipart else encoded
        else:
            request_info.payload_bytes = formatted_payload
        return await self.transport.send_request(
            request_info,
            payload=wire_payload,
            first_token_callback=first_token_callback,
        )

    async def _send_request_internal(
        self,
        request_info: RequestInfo,
        first_token_callback: FirstTokenCallback | None = None,
    ) -> RequestRecord:
        """Send request to transport and handle exceptions.

        Cancellation is now handled at the transport layer, which ensures the
        request is always sent before being cancelled.
        """
        pre_send_perf_ns, pre_send_timestamp_ns = None, None
        try:
            # Save the current perf_ns before sending the request so it can be used to calculate
            # the start_perf_ns of the request in case of an exception.
            pre_send_perf_ns, pre_send_timestamp_ns = (
                time.perf_counter_ns(),
                time.time_ns(),
            )

            # Transport handles cancellation internally (cancel_after_ns is in request_info)
            result = await self._send_request_to_transport(
                request_info=request_info, first_token_callback=first_token_callback
            )

            if self.is_debug_enabled:
                self.debug(
                    f"pre_send_perf_ns to start_perf_ns latency: {result.start_perf_ns - pre_send_perf_ns} ns"
                )
            return result
        except Exception as e:
            self.error(
                f"Error calling inference server API at {self.model_endpoint.endpoint.base_url}: {e!r}"
            )
            return RequestRecord(
                request_info=request_info,
                timestamp_ns=pre_send_timestamp_ns or time.time_ns(),
                # Try and use the pre_send_perf_ns if it is available, otherwise use the current time.
                start_perf_ns=pre_send_perf_ns or time.perf_counter_ns(),
                end_perf_ns=time.perf_counter_ns(),
                error=ErrorDetails.from_exception(e),
            )

    def provider_departure_event(self, root_correlation_id: str) -> asyncio.Event:
        """Shared per-worker gate for every concurrent turn/retry of one root."""
        return self._provider_departures.setdefault(
            root_correlation_id, asyncio.Event()
        )

    def root_has_departed(self, root_correlation_id: str) -> bool:
        """True once a local or remote measured refusal closed this root."""
        return self.provider_departure_event(root_correlation_id).is_set()

    def apply_provider_departure(
        self,
        root_correlation_id: str,
        observed_ns: int,
        *,
        applied_ns: int | None = None,
    ) -> None:
        """Close retries without cancelling any already-active HTTP operation."""
        self.provider_departure_event(root_correlation_id).set()
        previous = self.provider_departure_times.get(root_correlation_id)
        if previous is None or observed_ns < previous["observed_ns"]:
            self.provider_departure_times[root_correlation_id] = {
                "observed_ns": observed_ns,
                "applied_ns": applied_ns if applied_ns is not None else time.time_ns(),
            }

    async def send_request(
        self,
        request_info: RequestInfo,
        first_token_callback: FirstTokenCallback | None = None,
    ) -> RequestRecord:
        """Send a request to the inference API. Will return an error record if the call fails.

        Args:
            request_info: The request information.
            first_token_callback: Optional callback fired on first SSE message with ttft_ns

        Returns:
            RequestRecord containing the response data and metadata.
        """
        if not request_info.turns and not request_info.payload_bytes:
            raise ValueError(
                f"RequestInfo has no turns (credit_num={request_info.credit_num}, "
                f"conversation_id={request_info.conversation_id})"
            )
        if self.is_trace_enabled and request_info.turns:
            self.trace(f"Calling inference API for turn: {request_info.turns[-1]}")
        endpoint = self.model_endpoint.endpoint
        if (
            endpoint.retry_policy is not None
            or endpoint.retry_on_429 == "leave_provider"
        ):
            record = await self._send_with_retry_policy(
                request_info, first_token_callback
            )
        else:
            record = await self._send_request_internal(
                request_info, first_token_callback
            )
        if endpoint.retry_policy is None and (
            endpoint.retry_failed_requests
            or (
                endpoint.retry_failed_warmup_requests
                and request_info.credit_phase == CreditPhase.WARMUP
            )
        ):
            record = await self._retry_until_final(
                record, request_info, first_token_callback
            )
        # Redact sensitive headers on the request_info now that the transport has
        # consumed them.  This prevents raw credentials from flowing back through
        # ZMQ messages (which are TRACE-logged as serialised JSON / repr).
        request_info.endpoint_headers = (
            redact_headers(request_info.endpoint_headers) or {}
        )
        final_info = (
            record.request_info
            if endpoint.retry_policy is not None
            and isinstance(record.request_info, RequestInfo)
            else request_info
        )
        return self._finalize_request_record(record=record, request_info=final_info)

    async def _send_with_retry_policy(
        self, request_info: RequestInfo, first_token_callback: FirstTokenCallback | None
    ) -> RequestRecord:
        """Run one resolved policy, retaining ordinary native rows for retries."""
        endpoint = self.model_endpoint.endpoint
        policy = resolve_retry_policy(endpoint, str(request_info.credit_phase))
        root = request_info.root_correlation_id or request_info.x_correlation_id
        departed = self.provider_departure_event(root)
        logical_id = request_info.x_request_id
        started = time.perf_counter()
        attempt = 1
        while True:
            elapsed = time.perf_counter() - started
            timeout = remaining_attempt_seconds(policy, elapsed, endpoint.timeout)
            info = (
                request_info
                if attempt == 1
                else request_info.model_copy(update={"x_request_id": str(uuid.uuid4())})
            )
            if departed.is_set():
                raise asyncio.CancelledError("root departed before transport dispatch")
            if timeout <= 0:
                record = RequestRecord(
                    request_info=info,
                    error=ErrorDetails(
                        type="LogicalRequestStopped",
                        message="root_departed"
                        if departed.is_set()
                        else "logical_deadline",
                    ),
                )
                record.retry_terminal_reason = (
                    "root_departed" if departed.is_set() else "logical_deadline"
                )
                return record
            record = await self._send_policy_attempt(
                info, first_token_callback, timeout
            )
            hint = retry_hint_from_record(record)
            decision = decide_retry(
                policy,
                attempt_number=attempt,
                elapsed_seconds=time.perf_counter() - started,
                status_code=record.error.code
                if record.error is not None
                else record.status,
                retryable=is_retryable_error(record.error),
                partial_output=bool(record.responses) and record.error is not None,
                jitter_draw=retry_jitter_draw(
                    policy,
                    conversation_id=request_info.conversation_id,
                    turn_index=request_info.turn_index,
                    attempt_number=attempt,
                ),
                hint=hint,
            )
            record.retry_metadata = {
                "logical_request_id": logical_id,
                "attempt_number": attempt,
                "action": decision.action,
                "wait_seconds": decision.wait_seconds,
                "reason": decision.reason,
                "hint": asdict(hint),
                "policy_version": policy.version,
                "resolved_policy": policy.model_dump(),
            }
            record.retry_terminal_reason = (
                decision.reason if decision.action != "retry" else None
            )
            if decision.action == "leave_provider":
                observed_ns = time.time_ns()
                self.apply_provider_departure(root, observed_ns)
                notify = getattr(self, "depart_root", None)
                if notify is not None:
                    await notify(root, observed_ns)
                record.retry_metadata["departure"] = self.provider_departure_times[root]
                return record
            if decision.action != "retry":
                return record
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(departed.wait(), decision.wait_seconds)
            if departed.is_set():
                record.retry_terminal_reason = "root_departed"
                record.retry_metadata.update(action="stop", reason="root_departed")
                record.retry_metadata["departure"] = self.provider_departure_times[root]
                return record
            emit = getattr(self, "attempt_callback", None)
            if emit is not None:
                await emit(
                    self._finalize_request_record(record=record, request_info=info)
                )
            attempt += 1

    async def _send_policy_attempt(
        self,
        info: RequestInfo,
        first_token_callback: FirstTokenCallback | None,
        timeout: float,
    ) -> RequestRecord:
        """One HTTP operation, with a remaining logical budget applied."""
        endpoint = self.model_endpoint.endpoint
        try:
            async with asyncio.timeout(timeout):
                record = await self._send_request_internal(info, first_token_callback)
        except TimeoutError:
            record = RequestRecord(
                request_info=info,
                error=ErrorDetails(
                    type="TimeoutError",
                    message="logical request or HTTP deadline expired",
                ),
            )
        record.request_info = info
        if endpoint.streaming and str(endpoint.type) in DONE_TERMINATED_ENDPOINTS:
            mark_incomplete_stream(record)
        return record

    async def _retry_until_final(
        self,
        record: RequestRecord,
        request_info: RequestInfo,
        first_token_callback: FirstTokenCallback | None,
    ) -> RequestRecord:
        """Resend a failed request, as a user would, until it succeeds or stops.

        The request keeps its X-Request-ID across attempts, so every attempt's
        traces and dispatch records belong to one logical request. The returned
        record starts at the first attempt, so latency and TTFT include failed
        attempts and waits. Phase cancellation ends the loop by cancelling the
        task, including during the wait.
        """
        endpoint = self.model_endpoint.endpoint
        first_token_callback = _once(first_token_callback, record)
        first_start_perf_ns = record.start_perf_ns
        first_timestamp_ns = record.timestamp_ns
        retried: list[ErrorDetails] = []
        while True:
            if endpoint.streaming and str(endpoint.type) in DONE_TERMINATED_ENDPOINTS:
                mark_incomplete_stream(record)
            if not is_retryable_error(record.error):
                break
            if endpoint.retry_max_attempts and (
                len(retried) + 1 >= endpoint.retry_max_attempts
            ):
                break
            retried.append(record.error)
            self.warning(
                f"Retrying request {request_info.x_request_id} after attempt "
                f"{len(retried)} failed: {record.error.type} {record.error.code}"
            )
            await asyncio.sleep(endpoint.retry_wait_seconds)
            record = await self._send_request_internal(
                request_info, first_token_callback
            )
        record.retried_errors = retried
        record.start_perf_ns = first_start_perf_ns
        record.timestamp_ns = first_timestamp_ns
        return record

    @staticmethod
    def _enrich_request_record(
        record: RequestRecord, request_info: RequestInfo
    ) -> RequestRecord:
        """Attach a slim ``RecordContext`` (downcast from ``RequestInfo``) to
        the record before the ZMQ hop to the record processor.

        The full ``RequestInfo`` carries transport-only extras
        (``model_endpoint``, ``turns``, ``system_message``,
        ``user_context_message``, ``endpoint_headers``, ``endpoint_params``,
        ``drop_perf_ns``, ``cancel_after_ns``, ...) that the record-processor
        pipeline never reads; downcasting saves ~500-900 bytes per record at
        high throughput. The full ``turns`` list never travels — live records
        drive off the canonical ``payload_bytes``.
        """
        ctx_field_names = set(RecordContext.model_fields.keys())
        ri_dump = request_info.model_dump(include=ctx_field_names)
        record.request_info = RecordContext.model_validate(ri_dump)
        return record

    def _finalize_request_record(
        self,
        *,
        record: RequestRecord,
        request_info: RequestInfo,
    ) -> RequestRecord:
        """Enrich a RequestRecord with the original request info.

        The hoisted metric inputs ``max_tokens``, ``audio_duration_seconds``,
        and ``scheduled_send_ms`` live only on the originating turn — they are
        NOT ``RecordContext`` fields on ``request_info`` and so are not copied
        by the downcast in ``_enrich_request_record``. Populate them explicitly
        so the record processor (``osl_mismatch`` / ``audio_duration`` /
        ``replay_send_schedule_offset`` metrics) reads them directly off the
        slim record without the full ``turns`` list on the wire:
        ``max_tokens`` and ``scheduled_send_ms`` from the dispatch (last) turn,
        ``audio_duration_seconds`` from the first turn (ASR requests are
        single-turn; mirrors the pre-hoist ``turns[0]`` read).
        """
        last_turn = request_info.turns[-1] if request_info.turns else None
        first_turn = request_info.turns[0] if request_info.turns else None
        turn_model = last_turn.model if last_turn else None
        record.model_name = turn_model or self.model_endpoint.primary_model_name
        self._enrich_request_record(record, request_info)

        if record.request_info is not None:
            record.request_info.max_tokens = last_turn.max_tokens if last_turn else None
            record.request_info.audio_duration_seconds = (
                first_turn.audio_duration_seconds if first_turn else None
            )
            record.request_info.scheduled_send_ms = (
                float(last_turn.timestamp)
                if last_turn is not None and last_turn.timestamp is not None
                else None
            )

            # The usage-only warmup rule needs the body as sent, so it is
            # checked here, before the bytes may be stripped.
            if (
                self.accept_usage_only_warmup
                and record.request_info.credit_phase == CreditPhase.WARMUP
                and record.request_info.max_tokens == 1
            ):
                record.request_info.streams_one_token = body_streams_one_token(
                    record.request_info.payload_bytes
                )

            # When stripping is enabled (large-prompt memory optimization,
            # resolved by the worker's payload-retention auto-detection), drop
            # the canonical request payload bytes from the slim record context
            # after dispatch.
            if self.strip_record_payload_bytes:
                record.request_info.payload_bytes = None

        # If this is the first turn, calculate the credit drop latency
        if request_info.turn_index == 0 and request_info.drop_perf_ns is not None:
            record.credit_drop_latency = (
                record.start_perf_ns - request_info.drop_perf_ns
            )

        # Always redact at this boundary to guarantee no raw headers leak downstream,
        # even if a transport pre-populates record.request_headers.
        source_headers = (
            record.request_headers
            if record.request_headers is not None
            else request_info.endpoint_headers
        )
        record.request_headers = redact_headers(source_headers)
        return record
