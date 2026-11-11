# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wall-clock session targets for one continuous AgentX profiling phase."""

from __future__ import annotations

from typing import Annotated

from pydantic import BeforeValidator, ConfigDict, Field

from aiperf.config.base import BaseConfig


class ConcurrencyStep(BaseConfig):
    """Change admitted tree count without replacing baseline sessions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    at_seconds: float = Field(
        gt=0, allow_inf_nan=False, description="Seconds since profiling began."
    )
    concurrency: int = Field(
        gt=0, description="Target admitted session-tree lanes after this boundary."
    )


def parse_concurrency_schedule(value: object) -> object:
    """Accept native YAML points or the CLI form ``900:96,1200:24``."""
    if not isinstance(value, str):
        return value
    points: list[dict[str, str]] = []
    for item in value.split(","):
        parts = item.split(":")
        if len(parts) != 2:
            raise ValueError(
                "agentic concurrency schedule requires seconds:concurrency pairs"
            )
        points.append({"at_seconds": parts[0], "concurrency": parts[1]})
    return points


ConcurrencySchedule = Annotated[
    tuple[ConcurrencyStep, ...], BeforeValidator(parse_concurrency_schedule)
]
