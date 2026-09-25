"""
Token and cost analytics over the gateway's own audit log.

The audit log is written per request by `gateway.audit()` and is the single
source of truth here: the analytics process reads the file, it never talks to
the gateway. That keeps the hot request path free of any accounting work and
means the numbers can be recomputed at any time from a plain text file.

Estimates, not measurements. Exact token counts would need the upstream's
tokenizer, which no OpenAI-compatible API exposes. The estimators here are the
standard characters/4 for prose and the widely used ~25 tokens per property for
a tool schema, and the money figure is derived from them -- so every number
shown to the user is labelled as an estimate and derived by code that can be
read in one sitting.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# A mid-range frontier price: several prominent models cluster at $2.50-3.75 per
# million prompt tokens, so the middle of that band is the honest benchmark.
BENCHMARK_USD_PER_MTOKEN = 3.00

# Characters per token for prose. Code tends lower, prose higher; 4 is the
# conventional midpoint and is what the gateway's own token estimates use.
CHARS_PER_TOKEN = 4.0

# Tokens per property in a JSON-schema tool definition. Derived from real
# payloads: name, description, type scaffolding and each property object
# average out near this. Used for the schema-pruning saving.
TOKENS_PER_TOOL_PROPERTY = 25.0
TOOL_SCHEMA_OVERHEAD_TOKENS = 30.0

PRUNED_ROUTES = ("FastPath-Strip", "Classifier-Strip")


def estimate_tokens(text: str) -> int:
    """Cheap character-based token estimate (matches src.bridge)."""
    return int(len(text or "") / CHARS_PER_TOKEN)


def estimate_tool_schema_tokens(tools: Iterable[Any]) -> int:
    """Token estimate for a list of tool schemas.

    Counts nested properties, since a schema's cost is dominated by its
    parameters rather than its name and description.
    """
    total = 0
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        properties = (
            function.get("parameters", {}).get("properties", {})
            if isinstance(function, dict)
            else {}
        )
        if not isinstance(properties, dict):
            properties = {}
        total += TOOL_SCHEMA_OVERHEAD_TOKENS + len(properties) * TOKENS_PER_TOOL_PROPERTY
    return int(total)


def percentile(values: List[float], fraction: float) -> float:
    """Nearest-rank percentile; 0.0 for an empty sample."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


@dataclass
class Savings:
    """Where the saved tokens came from.

    `schema_tokens` counts tool schemas the upstream never saw: whole-schema
    strips (tools_before - tools_after == 0 on a stripped route) and selective
    sub-tool drops alike. `artifact_chars` counts oversized tool output replaced
    by a handle. They are estimated separately and added.
    """

    schema_tokens: int = 0
    artifact_chars: int = 0
    selective_drops: int = 0
    escaped_replays: int = 0
    # Raw tool counts, so the dashboard can say what fraction of the schema the
    # upstream never had to read. This is the number that matters to a flat-rate
    # plan: it is quota *not spent*, not money not paid.
    tools_before_total: int = 0
    tools_after_total: int = 0

    @property
    def artifact_tokens(self) -> int:
        return int(self.artifact_chars / CHARS_PER_TOKEN)

    @property
    def total_tokens(self) -> int:
        return self.schema_tokens + self.artifact_tokens

    @property
    def usd(self) -> float:
        return self.total_tokens / 1_000_000 * BENCHMARK_USD_PER_MTOKEN

    @property
    def tool_reduction_pct(self) -> float:
        """Share of the offered tool schema the upstream never had to read."""
        if self.tools_before_total <= 0:
            return 0.0
        dropped = self.tools_before_total - self.tools_after_total
        return max(0.0, min(100.0, dropped / self.tools_before_total * 100.0))


@dataclass
class Stats:
    requests: int = 0
    routes: Dict[str, int] = field(default_factory=dict)
    tool_actions: Dict[str, int] = field(default_factory=dict)
    spills: int = 0
    intercepted_fetches: int = 0
    sessions: Dict[str, int] = field(default_factory=dict)
    latencies_ms: List[float] = field(default_factory=list)
    savings: Savings = field(default_factory=Savings)
    first_ts: Optional[int] = None
    last_ts: Optional[int] = None
    malformed_lines: int = 0
    benchmark_usd_per_mtoken: float = BENCHMARK_USD_PER_MTOKEN

    @property
    def p50_ms(self) -> float:
        return percentile(self.latencies_ms, 0.50)

    @property
    def p90_ms(self) -> float:
        return percentile(self.latencies_ms, 0.90)

    @property
    def p99_ms(self) -> float:
        return percentile(self.latencies_ms, 0.99)

    def as_dict(self) -> Dict[str, Any]:
        """JSON-safe snapshot, used by `--json` and the tests."""
        span = None
        if self.first_ts is not None and self.last_ts is not None:
            span = self.last_ts - self.first_ts
        return {
            "requests": self.requests,
            "routes": dict(sorted(self.routes.items(), key=lambda kv: -kv[1])),
            "tool_actions": dict(sorted(self.tool_actions.items(), key=lambda kv: -kv[1])),
            "spills": self.spills,
            "intercepted_fetches": self.intercepted_fetches,
            "sessions": len(self.sessions),
            "pruned_schemas_tokens": self.savings.schema_tokens,
            "spilled_chars": self.savings.artifact_chars,
            "selective_drops": self.savings.selective_drops,
            "escape_replays": self.savings.escaped_replays,
            "tools_offered": self.savings.tools_before_total,
            "tools_forwarded": self.savings.tools_after_total,
            "quota_preserved_pct": round(self.savings.tool_reduction_pct, 1),
            "estimated_tokens_saved": self.savings.total_tokens,
            "estimated_usd_saved": round(self.savings.usd, 4),
            "benchmark_usd_per_mtoken": BENCHMARK_USD_PER_MTOKEN,
            "latency_ms": {
                "p50": self.p50_ms,
                "p90": self.p90_ms,
                "p99": self.p99_ms,
            },
            "window": {"first_ts": self.first_ts, "last_ts": self.last_ts, "span_s": span},
            "malformed_lines": self.malformed_lines,
        }


class Analytics:
    """Aggregate the gateway audit log into savings and latency statistics.

    Reads are defensive line by line: a torn write from a crash must cost one
    malformed row, never the whole report.
    """

    def __init__(
        self,
        audit_path: Path | str,
        benchmark_usd_per_mtoken: float = BENCHMARK_USD_PER_MTOKEN,
    ) -> None:
        self.audit_path = Path(audit_path)
        self.benchmark_usd_per_mtoken = benchmark_usd_per_mtoken

    # --- ingestion ---------------------------------------------------------
    def read_entries(self) -> Iterable[Dict[str, Any]]:
        try:
            with open(self.audit_path, "r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(entry, dict):
                        yield entry
        except OSError:
            return

    def compute(self) -> Stats:
        stats = Stats(benchmark_usd_per_mtoken=self.benchmark_usd_per_mtoken)
        for entry in self.read_entries():
            stats.requests += 1

            route = str(entry.get("route") or "unknown")
            stats.routes[route] = stats.routes.get(route, 0) + 1
            action = str(entry.get("tool_action") or "unknown")
            stats.tool_actions[action] = stats.tool_actions.get(action, 0) + 1

            stats.spills += int(entry.get("spill_count") or 0)
            stats.intercepted_fetches += int(entry.get("intercepted") or 0)
            session = entry.get("session_id")
            if session:
                stats.sessions[str(session)] = stats.sessions.get(str(session), 0) + 1

            latency = entry.get("latency_ms")
            if isinstance(latency, (int, float)) and latency >= 0:
                stats.latencies_ms.append(float(latency))

            timestamp = entry.get("ts")
            if isinstance(timestamp, (int, float)):
                timestamp = int(timestamp)
                stats.first_ts = (
                    timestamp if stats.first_ts is None else min(stats.first_ts, timestamp)
                )
                stats.last_ts = (
                    timestamp if stats.last_ts is None else max(stats.last_ts, timestamp)
                )

            self._account_savings(entry, stats.savings)

        return stats

    def _account_savings(self, entry: Dict[str, Any], savings: Savings) -> None:
        route = str(entry.get("route") or "")

        if "EarlyEscapeAbort" in route:
            savings.escaped_replays += 1

        # Tool schemas the upstream never saw. A stripped route sends none at
        # all (the saving is everything the request carried); a selective route
        # sends the kept subset (the saving is the dropped count). `tools_before`
        # is recorded by the gateway after the synthetic fetch_log was appended,
        # which is exactly what would have been sent without the gateway.
        tools_before = entry.get("tools_before")
        tools_after = entry.get("tools_after")
        if tools_before is None and route in PRUNED_ROUTES:
            # Audit rows from before the tool counters existed. Estimate from
            # the action alone, using the observed-average schema size.
            tools_before, tools_after = 10, 0
        if isinstance(tools_before, int) and isinstance(tools_after, int):
            dropped = max(0, int(tools_before) - int(tools_after))
            if dropped:
                savings.schema_tokens += int(
                    dropped * (TOOL_SCHEMA_OVERHEAD_TOKENS + 2 * TOKENS_PER_TOOL_PROPERTY)
                )
            savings.selective_drops += int(entry.get("selective_dropped") or 0)
            savings.tools_before_total += max(0, int(tools_before))
            savings.tools_after_total += max(0, int(tools_after))

        # Spilled artifact bodies: the gateway records the characters it did
        # not send in `spilled_chars`; older rows fall back to the observed
        # mean size of a spill notice.
        spilled_chars = entry.get("spilled_chars")
        if isinstance(spilled_chars, (int, float)):
            savings.artifact_chars += int(spilled_chars)
        else:
            savings.artifact_chars += int(entry.get("spill_count") or 0) * 2500


# ------------------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------------------
def _bar(value: float, maximum: float, width: int = 24) -> str:
    if maximum <= 0:
        return " " * width
    filled = min(width, max(0, round(value / maximum * width)))
    return "#" * filled + "." * (width - filled)


def render_table(
    stats: Stats, source: str, connection: Optional[Dict[str, str]] = None
) -> str:
    """Plain-text dashboard. ASCII only, so it renders over any SSH session."""
    data = stats.as_dict()
    savings = stats.savings
    routes = data["routes"]
    top_routes = list(routes.items())[:6]
    max_route = max(routes.values()) if routes else 0

    lines: List[str] = []
    add = lines.append
    add("agent-context-gateway -- savings dashboard")
    add("=" * 62)
    if connection:
        add(f"  connection       [{connection.get('label', connection.get('tier', '?'))}]")
    add(f"  audit source     {source}")
    if data["window"]["first_ts"]:
        add(
            f"  window           {data['window']['span_s'] or 0}s "
            f"({data['window']['first_ts']} .. {data['window']['last_ts']})"
        )
    add("")
    add(f"  requests         {data['requests']:<8} sessions        {data['sessions']}")
    add(f"  spills           {data['spills']:<8} fetch_log hits  {data['intercepted_fetches']}")
    add("")
    add("  routes")
    for route, count in top_routes:
        add(f"    {route:<34} {count:>6}  {_bar(count, max_route)}")
    if len(routes) > len(top_routes):
        add(f"    ... and {len(routes) - len(top_routes)} more")
    add("")
    add("  quota preserved (the point of the gateway)")
    add(
        f"    rate-limit quota saved   {data['quota_preserved_pct']:>6.1f}%"
        f"   ({savings.tools_after_total:,} of {savings.tools_before_total:,} tools forwarded)"
    )
    add("")
    add("  estimated savings (see note)")
    add(
        f"    tool schemas pruned   {savings.schema_tokens:>10,} tokens"
        f"   ({data['selective_drops']} selective drops)"
    )
    add(f"    spilled tool output   {savings.artifact_tokens:>10,} tokens   ({savings.artifact_chars:,} chars)")
    add(f"    escape replays        {savings.escaped_replays:>10,}")
    add(f"    {'-' * 46}")
    add(f"    TOTAL                 {savings.total_tokens:>10,} tokens")
    add(f"    ~ ${savings.usd:,.4f} at ${stats.benchmark_usd_per_mtoken:.2f}/M prompt tokens")
    add("")
    add("  latency (gateway overhead, ms)")
    add(f"    p50 {stats.p50_ms:>8.2f}   p90 {stats.p90_ms:>8.2f}   p99 {stats.p99_ms:>8.2f}")
    if data["malformed_lines"]:
        add(f"  note: {data['malformed_lines']} malformed audit lines skipped")
    add("")
    add(
        "  note: token and dollar figures are estimates; exact counts require the"
    )
    add("  upstream's tokenizer, which no OpenAI-compatible API exposes.")
    return "\n".join(lines)


def render_live(stats_source, interval: float = 2.0) -> None:
    """Re-render the dashboard in place until Ctrl-C.

    `stats_source` is a zero-argument callable returning (Stats, source_label),
    optionally with a third connection-tier mapping, so live mode re-reads the
    log each frame and shows fresh numbers.
    """
    try:
        while True:
            frame_source = stats_source()
            if len(frame_source) == 3:
                stats, source, connection = frame_source
            else:
                stats, source = frame_source
                connection = None
            frame = render_table(stats, source, connection)
            print("\033[2J\033[H" + frame, flush=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[dash] stopped.")


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    from .config import load_settings

    parser = argparse.ArgumentParser(
        prog="agent-gateway-stats",
        description="Token and cost savings from the gateway's audit log.",
    )
    parser.add_argument("--audit-log", help="override the audit log path")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--live", action="store_true", help="refresh in place until Ctrl-C")
    parser.add_argument("--interval", type=float, default=2.0, help="live refresh seconds")
    args = parser.parse_args(argv)

    settings = load_settings()
    path = Path(args.audit_log) if args.audit_log else settings.audit_log_path
    analytics = Analytics(path)

    if args.live:
        render_live(
            lambda: (analytics.compute(), str(path), settings.connection), args.interval
        )
        return 0

    stats = analytics.compute()
    if args.json:
        payload = stats.as_dict()
        payload["connection"] = settings.connection
        print(json.dumps(payload, indent=2))
    else:
        print(render_table(stats, str(path), settings.connection))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
