---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Warmup Phase Configuration
---

# Warmup Phase Configuration

The warmup phase runs before your actual benchmark to prepare the system for steady-state measurement. This guide explains when and how to configure warmup for accurate benchmarking results.

## Why Use Warmup?

When benchmarking starts, several "cold-start" effects can pollute your measurements:

```
Without warmup:                          With warmup:

Latency                                  Latency
   ▲                                        ▲
   │ ▓▓                                     │              Profiling starts
   │ ▓▓▓    ← Cold-start spikes             │              after system is warm
   │ ▓▓▓▓▓    pollute results               │                    │
   │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓                  │  Warmup    ▼  ▓▓▓▓▓▓▓▓▓▓▓▓▓
   │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓               │  ▓▓▓▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓▓▓▓▓▓▓▓
   └─────────────────────────────▶          └──────────────────────────────▶
                              Time                                       Time
```

**Cold-start effects include:**

| Effect | Cause | Impact |
|--------|-------|--------|
| **JIT compilation** | Python/PyTorch compiling code paths | Higher initial latency |
| **KV cache allocation** | Server allocating GPU memory | Memory pressure, timeouts |
| **Connection establishment** | New TCP/TLS handshakes for HTTP connections | Network latency spikes |
| **CUDA kernel compilation** | First-run kernel JIT | GPU stalls |
| **Model loading** | Lazy weight loading on first inference | Extreme latency outliers |

## Quick Start

Add warmup with a simple request count:

```bash
aiperf profile \
    --model your-model \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --request-rate 10 \
    --warmup-request-count 50 \
    --request-count 500
```

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     Using Request_Rate strategy
INFO     AIPerf System is WARMING UP

Warming Up: 50/50 |████████████████████████| 100% [00:05<00:00]

INFO     Warmup completed, starting profiling phase
INFO     AIPerf System is PROFILING

Profiling: 500/500 |████████████████████████| 100% [00:50<00:00]

INFO     Benchmark completed successfully
INFO     Results saved to: artifacts/your-model-chat-rate10/

JSON Export: artifacts/your-model-chat-rate10/profile_export_aiperf.json
```

This sends 50 warmup requests before the 500 profiling requests begin. Warmup metrics are discarded.

## Warmup Trigger Options

You can trigger warmup with **count-based** or **duration-based** stopping:

### Count-Based Warmup

```bash
# Stop after 100 warmup requests
--warmup-request-count 100

# OR stop after 20 sessions complete (for multi-turn)
--num-warmup-sessions 20
```

### Duration-Based Warmup

```bash
# Run warmup for 30 seconds
--warmup-duration 30
```

### Combined (First One Wins)

```bash
# Warmup stops when EITHER condition is met
--warmup-duration 60 \
--warmup-request-count 200
```

## Warmup-Specific Load Settings

By default, warmup inherits your profiling settings. Override them for different warmup behavior:

### Different Concurrency

```bash
aiperf profile \
    --model your-model \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --concurrency 100 \
    --warmup-concurrency 20 \
    --warmup-request-count 50 \
    --request-count 500
```

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     AIPerf System is WARMING UP
INFO     Warmup concurrency: 20 (profiling will use: 100)

Warming Up: 50/50 |████████████████████████| 100% [00:12<00:00]

INFO     Warmup completed, starting profiling phase
INFO     AIPerf System is PROFILING

Profiling: 500/500 |████████████████████████| 100% [01:15<00:00]

INFO     Benchmark completed successfully
INFO     Results saved to: artifacts/your-model-chat-concurrency100/

JSON Export: artifacts/your-model-chat-concurrency100/profile_export_aiperf.json
```

Warmup runs at 20 concurrent requests, then profiling runs at 100.

### Different Request Rate

```bash
aiperf profile \
    --model your-model \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --request-rate 50 \
    --warmup-request-rate 10 \
    --warmup-duration 30 \
    --benchmark-duration 120
```

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     AIPerf System is WARMING UP
INFO     Warmup rate: 10.0 req/s (profiling will use: 50.0 req/s)

Warming Up: [00:30] - Running for 30 seconds...

INFO     Warmup completed, starting profiling phase
INFO     AIPerf System is PROFILING

Profiling: [02:00] - Running for 120 seconds...

INFO     Benchmark completed successfully
INFO     Results saved to: artifacts/your-model-chat-rate50/

JSON Export: artifacts/your-model-chat-rate50/profile_export_aiperf.json
```

Warmup sends at 10 QPS, then profiling runs at 50 QPS.

### Different Arrival Pattern

```bash
aiperf profile \
    --model your-model \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --request-rate 20 \
    --arrival-pattern gamma \
    --arrival-smoothness 2.0 \
    --warmup-arrival-pattern constant \
    --warmup-duration 30 \
    --benchmark-duration 120
```

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     AIPerf System is WARMING UP
INFO     Warmup pattern: constant (profiling will use: gamma with smoothness 2.0)

Warming Up: [00:30] - Running for 30 seconds...

INFO     Warmup completed, starting profiling phase
INFO     AIPerf System is PROFILING

Profiling: [02:00] - Running for 120 seconds...

INFO     Benchmark completed successfully
INFO     Results saved to: artifacts/your-model-chat-rate20/

JSON Export: artifacts/your-model-chat-rate20/profile_export_aiperf.json
```

Warmup uses predictable constant arrivals; profiling uses gamma arrivals with reduced variance (smoothness > 1 = smoother than Poisson).

## Warmup with Ramping

Warmup can include its own gradual ramp-up:

```bash
aiperf profile \
    --model your-model \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --concurrency 100 \
    --concurrency-ramp-duration 30 \
    --warmup-concurrency 50 \
    --warmup-concurrency-ramp-duration 10 \
    --warmup-request-count 200 \
    --benchmark-duration 120
```

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     AIPerf System is WARMING UP
INFO     Warmup ramping from 1 to 50 over 10 seconds

Warming Up: 200/200 |████████████████████████| 100% [00:15<00:00]

INFO     Warmup completed, starting profiling phase
INFO     AIPerf System is PROFILING
INFO     Profiling ramping from 1 to 100 over 30 seconds

Profiling: [02:00] - Running for 120 seconds...

INFO     Benchmark completed successfully
INFO     Results saved to: artifacts/your-model-chat-concurrency100/

JSON Export: artifacts/your-model-chat-concurrency100/profile_export_aiperf.json
```

**Timeline:**

```
    Warmup Phase (ramps 1→50 over 10s)     Profiling Phase (ramps 1→100 over 30s)
    ─────────────────────────────────────  ──────────────────────────────────────────
                   ●━━━━━━ 50                                              ●━━━━━━ 100
              ●────┘                                                  ●────┘
         ●────┘                                                  ●────┘
    ●────┘                                                  ●────┘
    └──────────┬────────────────────────┬──────────────────────────────────────────▶
              10s                    Warmup ends              30s                Time
                                   (wait for responses)
```

## Grace Period

By default, AIPerf waits indefinitely for all warmup responses before starting profiling. When using duration-based warmup (`--warmup-duration`), you can limit this wait:

```bash
# Wait max 10 seconds for stragglers after warmup requests sent
--warmup-grace-period 10
```

This prevents slow warmup responses from delaying the profiling phase indefinitely.

## Warmup Boundary Hook

Use `--warmup-boundary-hook SCRIPT` with `--warmup-boundary-dir DIR` to run a
bash script between the warmup phase and the profiling phase. For example, the
script can record server counters or save the server's cache.

```bash
--warmup-request-count 100 \
--warmup-boundary-hook /opt/hooks/boundary.sh \
--warmup-boundary-dir /tmp/boundary
```

The timing manager runs these steps when the warmup phase completes:

1. It writes `DIR/warmup.done`.
2. It runs `bash SCRIPT` and waits for it. The profiling phase does not start
   until the script exits.
3. A non-zero exit status stops the run. Profiling does not start,
   `DIR/boundary.done` does not exist and aiperf exits with a non-zero status.
4. On exit status 0, it writes `DIR/boundary.done` and the profiling phase
   starts.

Each marker file holds `time.time_ns()` and a newline. Before the warmup
phase starts, aiperf checks that `DIR` exists and removes the `warmup.done` and
`boundary.done` files of an earlier run. If the warmup phase is cancelled, the
script does not run and profiling does not start. If the run is cancelled while
the script runs, aiperf kills the script's process group.

Rules:

- The two flags go together. One flag without the other is an error.
- The run needs exactly one warmup phase, and a profiling phase must follow it.
  An agentic replay scenario makes its own warmup phase.
- The profiling phase after the warmup cannot be seamless, because a seamless
  phase does not wait for the warmup requests.
- A parameter sweep or `--num-profile-runs` > 1 cannot use the hook.

## Multi-Turn Warmup

For multi-turn benchmarks, warmup by session count ensures complete conversations:

```bash
aiperf profile \
    --model your-model \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --session-turns-mean 5 \
    --num-warmup-sessions 10 \
    --request-count 500
```

This completes 10 full conversations (each ~5 turns) before profiling begins.

## Prefill Concurrency Warmup

When using [prefill concurrency](./prefill-concurrency.md) to limit simultaneous prefill operations, you can configure warmup separately:

```bash
aiperf profile \
    --model your-model \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --concurrency 50 \
    --prefill-concurrency 5 \
    --warmup-concurrency 20 \
    --warmup-prefill-concurrency 2 \
    --warmup-request-count 50 \
    --benchmark-duration 120
```

Warmup runs with lower limits (20 concurrent, 2 prefill), then profiling uses full limits.

## Examples

### Minimal Warmup

Just warm up connections and caches:

```bash
aiperf profile \
    --model Qwen/Qwen2.5-7B-Instruct \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --concurrency 50 \
    --warmup-request-count 20 \
    --request-count 500
```

### Production-Like Warmup

Simulate gradual traffic increase:

```bash
aiperf profile \
    --model Qwen/Qwen2.5-7B-Instruct \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --request-rate 100 \
    --concurrency 200 \
    --concurrency-ramp-duration 60 \
    --warmup-request-rate 20 \
    --warmup-concurrency 50 \
    --warmup-concurrency-ramp-duration 15 \
    --warmup-duration 30 \
    --benchmark-duration 300
```

### Long-Context Warmup

For long prompts, use lower warmup concurrency to avoid OOM:

```bash
aiperf profile \
    --model Qwen/Qwen2.5-7B-Instruct \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --synthetic-input-tokens-mean 32000 \
    --output-tokens-mean 500 \
    --concurrency 20 \
    --prefill-concurrency 3 \
    --warmup-concurrency 5 \
    --warmup-prefill-concurrency 1 \
    --warmup-request-count 10 \
    --benchmark-duration 120
```

## CLI Reference

### Stop Conditions (at least one required for warmup)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--warmup-request-count` | int | None | Stop warmup after this many requests |
| `--num-warmup-sessions` | int | None | Stop warmup after this many sessions complete |
| `--warmup-duration` | float | None | Stop warmup after this many seconds |

### Load Settings (inherit from profiling if not set)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--warmup-concurrency` | int | `--concurrency` | Session concurrency during warmup |
| `--warmup-prefill-concurrency` | int | `--prefill-concurrency` | Prefill concurrency during warmup |
| `--warmup-request-rate` | float | `--request-rate` | Request rate during warmup |
| `--warmup-arrival-pattern` | str | `--arrival-pattern` | Arrival pattern during warmup |

### Ramping (inherit from profiling if not set)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--warmup-concurrency-ramp-duration` | float | `--concurrency-ramp-duration` | Ramp duration for warmup concurrency |
| `--warmup-prefill-concurrency-ramp-duration` | float | `--prefill-concurrency-ramp-duration` | Ramp duration for warmup prefill |
| `--warmup-request-rate-ramp-duration` | float | `--request-rate-ramp-duration` | Ramp duration for warmup rate |

### Other

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--warmup-grace-period` | float | ∞ | Max seconds to wait for warmup responses after stop condition. Requires `--warmup-duration`. |
| `--warmup-boundary-hook` | path | - | Bash script to run after warmup and before profiling. Requires `--warmup-boundary-dir`. See [Warmup Boundary Hook](#warmup-boundary-hook). |
| `--warmup-boundary-dir` | path | - | Existing directory for `warmup.done` and `boundary.done`. Requires `--warmup-boundary-hook`. |

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Warmup takes too long | Grace period waiting for slow responses | Set `--warmup-grace-period` |
| Cold-start still visible | Insufficient warmup | Increase `--warmup-request-count` or `--warmup-duration` |
| OOM during warmup | Warmup concurrency too high | Lower `--warmup-concurrency` and `--warmup-prefill-concurrency` |
| Warmup not running | No warmup trigger set | Add `--warmup-request-count`, `--num-warmup-sessions`, or `--warmup-duration` |

## Related Documentation

- [Gradual Ramping](./ramping.md) — Smooth ramp-up for concurrency and rate
- [Prefill Concurrency](./prefill-concurrency.md) — Memory-safe long-context benchmarking
- [Timing Modes Reference](../benchmark-modes/timing-modes-reference.md) — Complete CLI compatibility matrix
