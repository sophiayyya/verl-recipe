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
"""Scoring rules for the subagent ThunderAgent benchmark."""

import pytest
from recipe.dynamo.benchmarks.thunderagent_subagent_bench import (
    BenchResult,
    render_table,
    summarize,
)


def _metrics(**counters):
    return {"counters": counters}


def test_a_balanced_run_leaks_nothing():
    result = summarize(_metrics(programs_created_total=9, programs_ended_total=9))
    assert result.leaked_programs == 0


def test_unreleased_subagents_surface_as_leaked_programs():
    """One root plus two subagents per trajectory, with only roots released."""
    result = summarize(_metrics(programs_created_total=9, programs_ended_total=3))
    assert result.leaked_programs == 6


def test_unpinned_turn_rate_is_per_program_request():
    result = summarize(_metrics(program_requests_total=200, unpinned_turns_total=10))
    assert result.unpinned_turn_rate == pytest.approx(0.05)


def test_rates_are_none_rather_than_zero_when_undefined():
    """A run with no program traffic must not report a flattering 0.0."""
    result = summarize(_metrics())
    assert result.unpinned_turn_rate is None
    assert result.prefix_hit_rate is None


def test_counters_are_differenced_against_the_before_snapshot():
    result = summarize(
        _metrics(programs_created_total=30, programs_ended_total=28),
        _metrics(programs_created_total=20, programs_ended_total=20),
    )
    assert (result.programs_created, result.programs_ended, result.leaked_programs) == (10, 8, 2)


def test_a_counter_going_backwards_is_rejected():
    """Two snapshots from different router processes would silently produce
    negative deltas that read as impossibly good results."""
    with pytest.raises(ValueError, match="decreased across the window"):
        summarize(_metrics(programs_created_total=1), _metrics(programs_created_total=5))


def test_prefix_hit_rate_is_differenced_from_the_vllm_block():
    # 640 hits over 800 queries in the window, not 900/1000 over all time.
    after = {**_metrics(), "vllm": {"prefix_cache_queries_total": 1000, "prefix_cache_hits_total": 740}}
    before = {**_metrics(), "vllm": {"prefix_cache_queries_total": 200, "prefix_cache_hits_total": 100}}
    assert summarize(after, before).prefix_hit_rate == pytest.approx(0.8)


def test_missing_counters_are_treated_as_zero():
    assert summarize({}).programs_created == 0


def test_table_renders_one_row_per_variant_with_aligned_columns():
    rows = [
        summarize(_metrics(programs_created_total=9, programs_ended_total=9), label="ta"),
        summarize(_metrics(programs_created_total=9, programs_ended_total=3), label="dynamo"),
    ]
    table = render_table(rows)
    lines = table.splitlines()
    assert lines[0].split()[0] == "label"
    assert len(lines) == 4  # header, rule, two rows
    assert all(isinstance(row, BenchResult) for row in rows)


def test_empty_result_set_renders_a_placeholder():
    assert render_table([]) == "(no results)"
