# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Flat endpoint retry controls shared by authored and runtime endpoint models."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from aiperf.common.finite import FiniteFloat
from aiperf.config.base import BaseConfig


class RetryDefaults:
    RETRY_FAILED_REQUESTS = False
    RETRY_FAILED_WARMUP_REQUESTS = False
    RETRY_WAIT_SECONDS = 10.0
    RETRY_MAX_ATTEMPTS = 0


class RetryOptionsMixin(BaseConfig):
    """One schema for new retry controls; consumers retain their legacy fields."""

    retry_policy: Literal["off", "fixed", "exponential"] | None = Field(
        default=None,
        description="Local retry schedule; None preserves explicit legacy retry flags.",
    )

    retry_server_hints: Literal["ignore", "prefer", "minimum"] = Field(
        default="ignore", description="Selection of allowed server retry hints."
    )

    retry_on_429: Literal["follow_retry_policy", "leave_provider"] = Field(
        default="follow_retry_policy",
        description="Profiling action on HTTP 429; warmup always follows retry policy.",
    )

    retry_max_elapsed_seconds: FiniteFloat | None = Field(
        default=None,
        description="Whole logical-request deadline in seconds; new policies default to 120.",
    )

    retry_backoff_factor: FiniteFloat = Field(
        default=2, description="Exponential retry multiplier."
    )

    retry_max_wait_seconds: FiniteFloat = Field(
        default=30, description="Local wait cap including jitter, seconds."
    )

    retry_jitter_fraction: FiniteFloat = Field(
        default=0, description="Additional seeded jitter fraction."
    )

    retry_phases: Literal["all", "warmup"] = Field(
        default="all", description="Phases eligible for retry."
    )

    retry_partial_stream: Literal["retry", "stop"] = Field(
        default="stop", description="Explicit replay of partial output."
    )

    retry_jitter_seed: int = Field(
        default=42, ge=0, description="Independent retry random seed."
    )

    retry_failed_requests: Annotated[
        bool,
        Field(
            default=RetryDefaults.RETRY_FAILED_REQUESTS,
            description="Retry a request that fails with HTTP 5xx or 429, a connection "
            "error, or a stream that ends before `data: [DONE]`, the way a user retries. "
            "The conversation waits for its retried turn. Latency and TTFT are measured "
            "from the first attempt. For reliability experiments; leave off for "
            "performance measurements.",
        ),
    ]

    retry_failed_warmup_requests: Annotated[
        bool,
        Field(
            default=RetryDefaults.RETRY_FAILED_WARMUP_REQUESTS,
            description="Retry only warmup requests, with the same rules, wait and attempt cap as --retry-failed-requests. A router that refuses the warmup burst (429 or 503) then delays the warmup instead of failing it, while profiling requests stay first attempts. Has no effect when --retry-failed-requests is set.",
        ),
    ]

    retry_wait_seconds: Annotated[
        float,
        Field(
            gt=0.0,
            default=RetryDefaults.RETRY_WAIT_SECONDS,
            description="Seconds to wait before each retry when retries are enabled.",
        ),
    ]

    retry_max_attempts: Annotated[
        int,
        Field(
            ge=0,
            default=RetryDefaults.RETRY_MAX_ATTEMPTS,
            description="Maximum attempts per request including the first when retries "
            "are enabled (0 = keep retrying until the request succeeds or the benchmark "
            "phase cancels it).",
        ),
    ]

    @model_validator(mode="after")
    def _validate_retry_policy(self) -> Self:
        from aiperf.workers.retry_policy import resolve_retry_policy

        if self.retry_policy is not None:
            if self.retry_failed_requests or self.retry_failed_warmup_requests:
                raise ValueError(
                    "--retry-policy cannot combine with legacy --retry-failed-* flags"
                )
            if (
                "retry_max_attempts" in self.model_fields_set
                and self.retry_max_attempts == 0
            ):
                raise ValueError(
                    "new retry policies require a positive --retry-max-attempts"
                )
            resolved = resolve_retry_policy(self, "profiling")
            self.retry_max_attempts = resolved.max_attempts
            self.retry_max_elapsed_seconds = resolved.max_elapsed_seconds
        elif any(
            getattr(self, name) != self.__class__.model_fields[name].default
            for name in (
                "retry_on_429",
                "retry_server_hints",
                "retry_max_elapsed_seconds",
                "retry_backoff_factor",
                "retry_max_wait_seconds",
                "retry_jitter_fraction",
                "retry_phases",
                "retry_partial_stream",
                "retry_jitter_seed",
            )
        ):
            if self.retry_on_429 != "leave_provider":
                raise ValueError("new retry options require --retry-policy")
        return self
