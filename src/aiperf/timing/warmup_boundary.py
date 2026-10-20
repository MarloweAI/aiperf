# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the warmup boundary hook between the warmup and profiling phases."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import time
from pathlib import Path

import aiofiles
import aiofiles.os

from aiperf.common.constants import IS_WINDOWS
from aiperf.common.exceptions import AIPerfError
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.config.warmup_boundary import (
    BOUNDARY_DONE_MARKER,
    WARMUP_DONE_MARKER,
    WarmupBoundaryHookConfig,
)


class WarmupBoundaryHookError(AIPerfError):
    """The warmup boundary hook cannot run or exited with a non-zero status."""


class WarmupBoundaryHook(AIPerfLoggerMixin):
    """Runs one ``bash <script>`` at the warmup/profiling boundary.

    ``prepare()`` runs before the warmup phase. It checks the marker directory
    and removes markers that an earlier run left. ``run()`` writes
    ``warmup.done``, runs the script and waits for it. It raises
    ``WarmupBoundaryHookError`` on a non-zero exit status, so the caller stops
    before profiling. On exit status 0 it writes ``boundary.done``.
    """

    def __init__(self, config: WarmupBoundaryHookConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self._config = config

    async def prepare(self) -> None:
        marker_dir = self._config.marker_dir
        if not await aiofiles.os.path.isdir(marker_dir):
            raise WarmupBoundaryHookError(
                f"The warmup boundary marker directory {marker_dir} does not exist."
            )
        for name in (WARMUP_DONE_MARKER, BOUNDARY_DONE_MARKER):
            with contextlib.suppress(FileNotFoundError):
                await aiofiles.os.remove(marker_dir / name)

    async def run(self) -> None:
        script = self._config.script
        await _write_marker(self._config.marker_dir / WARMUP_DONE_MARKER)
        self.notice(f"Warmup complete. Running the warmup boundary hook {script}")
        status = await _run_bash(script)
        if status != 0:
            self.error(f"The warmup boundary hook {script} exited with status {status}")
            raise WarmupBoundaryHookError(
                f"The warmup boundary hook {script} exited with status {status}. "
                "Profiling does not start."
            )
        self.info(f"The warmup boundary hook {script} exited with status 0")
        await _write_marker(self._config.marker_dir / BOUNDARY_DONE_MARKER)
        self.notice("The warmup boundary hook succeeded. Profiling starts.")


async def _run_bash(script: Path) -> int:
    """Run ``bash <script>`` and return its exit status.

    The script runs in its own process group. When the waiting task is
    cancelled, the whole group gets SIGKILL, so a stopped benchmark does not
    leave the script or the commands it started running.
    """
    session = {} if IS_WINDOWS else {"start_new_session": True}
    process = await asyncio.create_subprocess_exec("bash", str(script), **session)
    try:
        return await process.wait()
    except asyncio.CancelledError:
        if process.returncode is None:
            _kill_process_group(process)
            await process.wait()
        raise


def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    if IS_WINDOWS:
        process.kill()
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)


async def _write_marker(path: Path) -> None:
    async with aiofiles.open(path, "w") as marker:
        await marker.write(f"{time.time_ns()}\n")
