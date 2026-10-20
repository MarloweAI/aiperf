# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Characters allowed in a fixed benchmark id (``--benchmark-id``)."""

from __future__ import annotations

BENCHMARK_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
"""Pattern of a fixed benchmark id.

aiperf puts the benchmark id into directory names (``aiperf_mmap_{id}``), so a
fixed id is limited to path-safe characters.
"""
