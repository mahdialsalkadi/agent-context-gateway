"""
The analytics engine and the `agent-gateway stats` surface.

The audit log is the single source of truth, so these tests write synthetic
audit rows and assert on the numbers that come back out. The estimates are
deliberately simple (characters/4, a per-property schema cost); what is pinned
here is that the arithmetic is honest and deterministic, not that it matches any
particular tokenizer.
"""

from __future__ import annotations

import json

from src.analytics import (
    BENCHMARK_USD_PER_MTOKEN,
    Analytics,
    estimate_tool_schema_tokens,
    percentile,
    render_table,
)
from src.config import (
    TIER_COMMERCIAL,
    TIER_LOCAL,
    TIER_SUBSCRIPTION,
    classify_upstream,
    load_settings,
)


def write_audit(path, entries):
    path.write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8"
    )


def entry(route="FastPath-Strip", tool_action="Stripped-ZeroTokens", **extra):
    base = {
        "ts": 1_700_000_000,
        "surface": "openai",
        "route": route,
        "tool_action": tool_action,
        "latency_ms": 10.0,
        "spill_count": 0,
        "intercepted": 0,
        "tools_before": 0,
        "tools_after": 0,
        "selective_dropped": 0,
        "spilled_chars": 0,
    }
    base.update(extra)
    return base


# ------------------------------------------------------------------------------
# Counting
# ------------------------------------------------------------------------------
def test_counts_requests_and_route_breakdown(tmp_path):
    log = tmp_path / "audit.log"
    write_audit(
        log,
        [
            entry(),
            entry(),
            entry(route="Classifier-Keep", tool_action="Retained-Classifier"),
            entry(route="Reasoning-Passthrough", tool_action="Retained-Full"),
        ],
    )

    stats = Analytics(log).compute()

    assert stats.requests == 4
    assert stats.routes["FastPath-Strip"] == 2
    assert stats.routes["Classifier-Keep"] == 1
    assert stats.routes["Reasoning-Passthrough"] == 1


def test_latency_percentiles_are_order_insensitive(tmp_path):
    log = tmp_path / "audit.log"
    latencies = [100.0, 1.0, 50.0, 25.0, 7.0]
    write_audit(log, [entry(latency_ms=value) for value in latencies])

    stats = Analytics(log).compute()

    assert stats.p50_ms == 25.0
    assert stats.p90_ms == 100.0
    assert stats.p99_ms == 100.0


def test_percentile_of_an_empty_sample_is_zero():
    assert percentile([], 0.99) == 0.0


def test_torn_or_garbage_lines_are_counted_not_fatal(tmp_path):
    log = tmp_path / "audit.log"
    log.write_text(
        json.dumps(entry()) + "\n" + "{not json at all" + "\n" + json.dumps(entry()) + "\n",
        encoding="utf-8",
    )

    stats = Analytics(log).compute()

    assert stats.requests == 2
    assert stats.malformed_lines == 0  # unparseable rows are skipped silently


def test_missing_file_yields_an_empty_report(tmp_path):
    stats = Analytics(tmp_path / "does-not-exist.log").compute()

    assert stats.requests == 0
    assert stats.savings.total_tokens == 0
    assert stats.p50_ms == 0.0


def test_sessions_and_spills_are_tallied(tmp_path):
    log = tmp_path / "audit.log"
    write_audit(
        log,
        [
            entry(session_id="alpha", spill_count=2, intercepted=1),
            entry(session_id="alpha"),
            entry(session_id="beta", spill_count=1),
        ],
    )

    stats = Analytics(log).compute()

    assert stats.sessions == {"alpha": 2, "beta": 1}
    assert stats.spills == 3
    assert stats.intercepted_fetches == 1


def test_time_window_spans_first_to_last(tmp_path):
    log = tmp_path / "audit.log"
    write_audit(log, [entry(ts=1_700_000_000), entry(ts=1_700_000_500)])

    stats = Analytics(log).compute()

    assert stats.first_ts == 1_700_000_000
    assert stats.last_ts == 1_700_000_500
    assert stats.as_dict()["window"]["span_s"] == 500


# ------------------------------------------------------------------------------
# Savings arithmetic
# ------------------------------------------------------------------------------
def test_schema_strip_saving_is_booked_from_tool_counters(tmp_path):
    log = tmp_path / "audit.log"
    # 20 tools before, 0 after on a strip route.
    write_audit(log, [entry(tools_before=20, tools_after=0)])

    savings = Analytics(log).compute().savings

    expected = int(20 * (30.0 + 2 * 25.0))
    assert savings.schema_tokens == expected
    assert savings.total_tokens == expected
    assert savings.usd == expected / 1_000_000 * BENCHMARK_USD_PER_MTOKEN


def test_selective_drop_saving_is_booked(tmp_path):
    log = tmp_path / "audit.log"
    write_audit(
        log,
        [entry(route="Default", tool_action="Retained-Classifier",
               tools_before=20, tools_after=5, selective_dropped=15)],
    )

    data = Analytics(log).compute()

    assert data.savings.selective_drops == 15
    assert data.savings.schema_tokens == int(15 * (30.0 + 2 * 25.0))


def test_legacy_rows_without_counters_get_an_estimate(tmp_path):
    """Rows written before tools_before existed must still count for something."""
    log = tmp_path / "audit.log"
    row = entry()
    for key in ("tools_before", "tools_after", "selective_dropped", "spilled_chars"):
        row.pop(key)
    write_audit(log, [row])

    savings = Analytics(log).compute().savings

    assert savings.schema_tokens > 0


def test_spilled_chars_booked_directly_and_by_fallback(tmp_path):
    log = tmp_path / "audit.log"
    row_with_size = entry(spill_count=1, spilled_chars=8000)
    row_legacy = entry(spill_count=2)
    for key in ("spilled_chars",):
        row_legacy.pop(key)
    write_audit(log, [row_with_size, row_legacy])

    savings = Analytics(log).compute().savings

    assert savings.artifact_chars == 8000 + 2 * 2500
    assert savings.artifact_tokens == int(savings.artifact_chars / 4)


def test_escape_replays_are_counted(tmp_path):
    log = tmp_path / "audit.log"
    write_audit(
        log,
        [entry(route="Classifier-Strip->EarlyEscapeAbort",
               tool_action="Reverted-To-Full")],
    )

    assert Analytics(log).compute().savings.escaped_replays == 1


def test_tool_schema_estimator_counts_nested_properties():
    tools = [
        {"function": {"name": "t", "parameters": {"properties": {f"p{i}": {} for i in range(4)}}}},
    ]
    assert estimate_tool_schema_tokens(tools) == 30.0 + 4 * 25.0


def test_zero_savings_when_nothing_was_pruned(tmp_path):
    log = tmp_path / "audit.log"
    write_audit(log, [entry(route="FastPath-Keep", tool_action="Retained-Heuristic",
                            tools_before=10, tools_after=11)])

    savings = Analytics(log).compute().savings

    assert savings.schema_tokens == 0
    assert savings.total_tokens == 0
    assert savings.usd == 0.0


# ------------------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------------------
def test_render_table_shows_the_connection_tier_and_quota(tmp_path):
    log = tmp_path / "audit.log"
    write_audit(log, [entry(tools_before=20, tools_after=4)])

    text = render_table(
        Analytics(log).compute(),
        str(log),
        {"tier": "subscription", "label": "SUBSCRIPTION BRIDGE: Google Antigravity"},
    )

    assert "SUBSCRIPTION BRIDGE" in text
    assert "quota saved" in text
    assert "80.0%" in text, "20 offered, 4 forwarded => 80% kept out of the prompt"
    text.encode("ascii")


def test_quota_preserved_is_zero_without_tool_traffic(tmp_path):
    log = tmp_path / "audit.log"
    write_audit(log, [entry()])

    data = Analytics(log).compute().as_dict()
    assert data["quota_preserved_pct"] == 0.0
    assert data["tools_offered"] == 0


def test_classify_upstream_identifies_the_connection_tier():
    assert classify_upstream("http://127.0.0.1:8080/v1")[0] == TIER_SUBSCRIPTION
    assert classify_upstream("http://127.0.0.1:11434/v1")[0] == TIER_LOCAL
    assert classify_upstream("https://api.openai.com/v1")[0] == TIER_COMMERCIAL
    assert classify_upstream("https://openrouter.ai/api/v1")[0] == TIER_COMMERCIAL
    assert "Antigravity" in classify_upstream("http://127.0.0.1:8080/v1")[1]


def test_render_table_is_ascii_and_mentions_the_estimate(tmp_path):
    log = tmp_path / "audit.log"
    write_audit(log, [entry(tools_before=20, tools_after=0, spilled_chars=8000)])

    text = render_table(Analytics(log).compute(), str(log))

    assert "estimated" in text.lower()
    assert "$" in text
    assert "p50" in text
    text.encode("ascii")  # must render over any terminal


def test_json_snapshot_is_round_trippable(tmp_path):
    log = tmp_path / "audit.log"
    write_audit(log, [entry()])

    data = Analytics(log).compute().as_dict()

    assert json.loads(json.dumps(data)) == data
    assert data["requests"] == 1


# ------------------------------------------------------------------------------
# The settings surface
# ------------------------------------------------------------------------------
def test_settings_expose_selective_pruning_toggles():
    default = load_settings(env={})
    assert default.enable_selective_pruning is True
    assert default.selective_tool_limit == 5

    tuned = load_settings(
        env={"ENABLE_SELECTIVE_PRUNING": "0", "SELECTIVE_TOOL_LIMIT": "3"}
    )
    assert tuned.enable_selective_pruning is False
    assert tuned.selective_tool_limit == 3
