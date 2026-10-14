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
"""Score a subagent rollout from the prototype ThunderAgent router's ``/metrics``.

Reduces a before/after pair of metrics captures (plus an optional ``vllm`` block
of Prometheus counters) to leaked programs, unpinned-turn rate, pause counts and
prefix-cache hit rate::

    python -m recipe.dynamo.benchmarks.thunderagent_subagent_bench \\
        --before before.json --after after.json --label ta
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchResult:
    """One variant's score: ``leaked_programs`` is admitted minus released programs,
    ``unpinned_turn_rate`` the share of program turns that reached a replica without
    a pin, ``forced_resumes`` the resumes fired by the starvation timer."""

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
    """Difference the ``before`` and ``after`` metrics captures into one row.

    Raises ValueError when a counter went backwards, i.e. the two captures come
    from different router processes.
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
