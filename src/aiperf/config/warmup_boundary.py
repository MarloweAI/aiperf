# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warmup/profiling boundary hook settings.

``WarmupBoundaryHookConfig`` holds the script and the marker directory of the
hook that runs between the warmup phase and the profiling phase. One object
holds both values, so a configuration with only one of them cannot exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import ConfigDict, Field

from aiperf.config.base import BaseConfig

WARMUP_DONE_MARKER = "warmup.done"
"""Marker file written when the warmup phase returns, before the hook runs."""

BOUNDARY_DONE_MARKER = "boundary.done"
"""Marker file written when the hook exits with status 0."""


class WarmupBoundaryHookConfig(BaseConfig):
    """A bash script that runs after the warmup phase and before profiling.

    Before the warmup phase, aiperf removes old markers from ``marker_dir``.
    It writes ``<marker_dir>/warmup.done`` when the warmup phase returns,
    runs ``bash <script>`` and waits for it. A non-zero exit status stops the
    run, so profiling never starts. On exit status 0, aiperf writes
    ``<marker_dir>/boundary.done`` and the profiling phase starts. Each marker
    holds ``time.time_ns()`` and a newline.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    script: Annotated[
        Path,
        Field(
            description="Bash script that aiperf runs with `bash <script>` after "
            "the warmup phase completes and before the profiling phase starts. "
            "The profiling phase waits for the script. A non-zero exit status "
            "stops the run before profiling.",
        ),
    ]

    marker_dir: Annotated[
        Path,
        Field(
            description="Existing directory for the marker files. Before the "
            "warmup phase, aiperf removes old markers there. It writes "
            "`warmup.done` when the warmup phase returns and `boundary.done` "
            "when the script exits with status 0. Each file holds "
            "`time.time_ns()` and a newline.",
        ),
    ]
