# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Score a ThunderAgent rollout on the metrics that subagents actually move.

An agent that dispatches subagents turns one trajectory into a *tree* of
programs. Three things go wrong when the scheduler cannot see that tree, and all
three are visible in the router's own metrics:

``leaked_programs``
    The router drops a program only when a request marks it final. Nothing
    marked a subagent final before this recipe released them, so every subagent
    left a resident ``Program`` for the life of the run, inflating the working
    set that pause/resume decisions are computed from. Should be 0.
``unpinned_turn_rate``
    Fraction of program turns that reached a replica without a pin — each one
    re-prefills a context the scheduler meant to keep resident. This is the
    metric ThunderAgent exists to lower.
``prefix_hit_rate``
    vLLM's own view of the same thing, as a cross-check that does not depend on
    the router's bookkeeping being right.

Usage::

    python -m recipe.dynamo.benchmarks.thunderagent_subagent_bench \\
        --before before.json --after after.json --label ta

where the JSON files are captures of the router's ``metrics`` endpoint plus an
optional ``vllm`` block of Prometheus counters. ``--before`` may be omitted when
the router started with the run.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchResult:
    """One variant's score on a subagent rollout.

    Attributes:
        label: Variant name, e.g. ``ta`` or ``dynamo``.
        programs_created: Programs the router admitted over the window.
        programs_ended: Programs released over the window.
        leaked_programs: Admitted minus released. Non-zero means subagent
            programs stayed resident after their trajectory finished.
        program_requests: Turns that carried a program id.
        unpinned_turns: Program turns that reached a replica without a pin.
        unpinned_turn_rate: ``unpinned_turns / program_requests``.
        pauses: Tool-boundary pauses the scheduler performed.
        forced_resumes: Resumes that fired on the starvation timer rather than
            on headroom — a high count means the pause threshold is too tight.
        prefix_hit_rate: vLLM prefix-cache hits over queries, when supplied.
        wall_clock_s: Rollout wall-clock, when supplied.
    """

    label: str
    programs_created: int
    programs_ended: int
    leaked_programs: int
    program_requests: int
    unpinned_turns: int
    unpinned_turn_rate: float | None
    pauses: int
    forced_resumes: int
    prefix_hit_rate: float | None
    wall_clock_s: float | None


def _counter(snapshot: dict[str, Any], name: str) -> int:
    """Read one counter, treating a missing snapshot or key as zero."""
    return int((snapshot.get("counters") or {}).get(name, 0))


def _ratio(numerator: float, denominator: float) -> float | None:
    """Return ``numerator / denominator``, or ``None`` when undefined."""
    return numerator / denominator if denominator else None


def summarize(
    after: dict[str, Any],
    before: dict[str, Any] | None = None,
    *,
    label: str = "ta",
    wall_clock_s: float | None = None,
) -> BenchResult:
    """Reduce a pair of router metric snapshots to one comparable row.

    Counters are differenced so a router shared across variants still yields
    per-run numbers.

    :param after: Router ``metrics`` payload captured after the rollout.
    :type after: dict[str, Any]
    :param before: The same payload captured before it; omit if the router
        started with the run.
    :type before: dict[str, Any] | None
    :param label: Variant name carried into the result.
    :type label: str
    :param wall_clock_s: Measured rollout duration, when available.
    :type wall_clock_s: float | None
    :return: The scored row.
    :rtype: BenchResult
    :raises ValueError: If a counter went backwards, which means the two
        snapshots came from different router processes.
    """
    before = before or {}

    def delta(name: str) -> int:
        value = _counter(after, name) - _counter(before, name)
        if value < 0:
            raise ValueError(
                f"counter {name!r} decreased across the window (snapshots are from different router processes)"
            )
        return value

    created = delta("programs_created_total")
    ended = delta("programs_ended_total")
    program_requests = delta("program_requests_total")
    unpinned = delta("unpinned_turns_total")

    vllm_after = after.get("vllm") or {}
    vllm_before = before.get("vllm") or {}
    queries = float(vllm_after.get("prefix_cache_queries_total", 0)) - float(
        vllm_before.get("prefix_cache_queries_total", 0)
    )
    hits = float(vllm_after.get("prefix_cache_hits_total", 0)) - float(vllm_before.get("prefix_cache_hits_total", 0))

    return BenchResult(
        label=label,
        programs_created=created,
        programs_ended=ended,
        leaked_programs=created - ended,
        program_requests=program_requests,
        unpinned_turns=unpinned,
        unpinned_turn_rate=_ratio(unpinned, program_requests),
        pauses=delta("program_pauses_total"),
        forced_resumes=delta("forced_resumes_total"),
        prefix_hit_rate=_ratio(hits, queries),
        wall_clock_s=wall_clock_s,
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_table(results: list[BenchResult]) -> str:
    """Render scored variants as a fixed-width table, one row per variant."""
    if not results:
        return "(no results)"
    columns = list(asdict(results[0]))
    rows = [[_fmt(getattr(result, column)) for column in columns] for result in results]
    widths = [max(len(column), *(len(row[index]) for row in rows)) for index, column in enumerate(columns)]
    lines = [
        "  ".join(column.ljust(widths[index]) for index, column in enumerate(columns)),
        "  ".join("-" * width for width in widths),
    ]
    lines += ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after", required=True, type=Path, help="metrics captured after the rollout")
    parser.add_argument("--before", type=Path, help="metrics captured before the rollout")
    parser.add_argument("--label", default="ta", help="variant name for this row")
    parser.add_argument("--wall-clock-s", type=float, help="measured rollout duration")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = parser.parse_args(argv)

    result = summarize(
        json.loads(args.after.read_text()),
        json.loads(args.before.read_text()) if args.before else None,
        label=args.label,
        wall_clock_s=args.wall_clock_s,
    )
    print(json.dumps(asdict(result), indent=2) if args.json else render_table([result]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
