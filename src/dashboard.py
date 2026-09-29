"""Embedded web dashboard: one HTML page, three JSON endpoints, zero build step.

The page is a Python string served by FastAPI at GET /ui. Vanilla JS only --
the only remote reference is the Tailwind CDN script, which the page degrades
gracefully without. All dynamic data comes from /ui/api/* endpoints on this
same origin, so the dashboard works on a loopback port with no CORS setup.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from fastapi.responses import HTMLResponse, JSONResponse


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>agent-context-gateway</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  body { background:#0b1120; color:#e2e8f0; font-family:ui-sans-serif,system-ui,sans-serif; }
  .card { background:#111a2e; border:1px solid #1e293b; border-radius:12px; }
  .badge { font-size:11px; padding:2px 8px; border-radius:9999px; font-weight:600; }
  .strip { background:#7f1d1d; color:#fecaca; }
  .keep  { background:#14532d; color:#bbf7d0; }
  .pass  { background:#1e3a8a; color:#bfdbfe; }
  .sel   { background:#78350f; color:#fde68a; }
  .num   { font-family:ui-monospace,monospace; }
  .refreshing { opacity:.55; }
</style>
</head>
<body class="min-h-screen">
<div id="root" class="max-w-5xl mx-auto px-4 py-8 space-y-6">

  <header class="flex items-center justify-between">
    <div>
      <h1 class="text-xl font-bold">agent-context-gateway</h1>
      <p id="sub" class="text-sm text-slate-400">Context Telemetry & Semantic Pruning</p>
    </div>
    <div class="flex items-center gap-2">
      <div id="mode" class="badge pass">Transparent Proxy</div>
      <div id="conn" class="badge keep">connecting…</div>
    </div>
  </header>

  <section class="grid grid-cols-2 md:grid-cols-4 gap-4">
    <div class="card p-4">
      <div class="text-xs text-slate-400">rate-limit quota saved</div>
      <div id="quota" class="num text-2xl font-bold mt-1 text-amber-300">–</div>
      <div id="usd" class="num text-xs text-emerald-400 mt-1">–</div>
    </div>
    <div class="card p-4">
      <div class="text-xs text-slate-400">pruned tools count</div>
      <div id="pruned-tools" class="num text-2xl font-bold mt-1 text-emerald-400">–</div>
      <div id="pruned-sub" class="num text-xs text-slate-400 mt-1">–</div>
    </div>
    <div class="card p-4">
      <div class="text-xs text-slate-400">requests</div>
      <div id="requests" class="num text-2xl font-bold mt-1">–</div>
    </div>
    <div class="card p-4">
      <div class="text-xs text-slate-400">p50 / p90 latency</div>
      <div id="latency" class="num text-2xl font-bold mt-1">–</div>
    </div>
  </section>

  <section class="card p-4">
    <div class="flex items-center justify-between mb-3">
      <h2 class="font-semibold">recent requests</h2>
      <span class="text-xs text-slate-500">last 20 · from the audit log</span>
    </div>
    <div class="overflow-x-auto">
    <table class="w-full text-sm">
      <thead><tr class="text-left text-xs text-slate-500 border-b border-slate-800">
        <th class="py-1 pr-3">timestamp</th>
        <th class="py-1 pr-3">agent</th>
        <th class="py-1 pr-3">route</th>
        <th class="py-1 pr-3">selected tools</th>
        <th class="py-1 pr-3">latency</th></tr></thead>
      <tbody id="feed"><tr><td colspan="5" class="py-3 text-slate-500">no requests yet</td></tr></tbody>
    </table>
    </div>
  </section>

  <section class="card p-4">
    <div class="flex items-center justify-between mb-3">
      <h2 class="font-semibold">knowledge graph</h2>
      <span class="text-xs text-slate-500">SQLite · WAL · facts already earned</span>
    </div>
    <table class="w-full text-sm">
      <thead><tr class="text-left text-xs text-slate-500 border-b border-slate-800">
        <th class="py-1 pr-3">source</th><th class="py-1 pr-3">relation</th>
        <th class="py-1 pr-3">target</th><th class="py-1 pr-3">status</th><th></th></tr></thead>
      <tbody id="graph"><tr><td colspan="5" class="py-3 text-slate-500">loading…</td></tr></tbody>
    </table>
  </section>

  <footer class="text-xs text-slate-500 space-y-1">
    <p>quota saved = share of the offered tool schemas the upstream never had to read (the hourly limit you did not touch).</p>
    <p>* dollars are an estimate at the benchmark price; exact token counts need the upstream's tokenizer.</p>
    <p>auto-refreshes every 3s · <span id="err" class="text-rose-400"></span></p>
  </footer>
</div>

<script>
const $ = (id) => document.getElementById(id);
const badgeClass = (route) =>
  route.includes("Strip") ? "badge strip"
  : route.includes("Passthrough") ? "badge pass"
  : (route.includes("Selective") || route.includes("Jev-Routed") || route.includes("Skill")) ? "badge sel"
  : "badge keep";

function badgeFor(route) {
  const span = document.createElement("span");
  span.className = badgeClass(route);
  span.textContent = route;
  return span;
}

async function jget(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(url + " -> " + r.status);
  return r.json();
}

async function refreshStats() {
  const d = await jget("/ui/api/stats");
  $("quota").textContent = (d.quota_preserved_pct || 0).toFixed(1) + "%";
  $("usd").textContent = "~$" + (d.estimated_usd_saved || 0).toFixed(4) + " saved on metered plan";
  $("pruned-tools").textContent = (d.pruned_tools_count || 0).toLocaleString();
  $("pruned-sub").textContent = (d.estimated_tokens_saved || 0).toLocaleString() + " tokens saved";
  $("requests").textContent = (d.requests || 0).toLocaleString();
  $("latency").textContent = ((d.latency_ms && d.latency_ms.p50) ? d.latency_ms.p50.toFixed(0) : "0") + " / " +
    ((d.latency_ms && d.latency_ms.p90) ? d.latency_ms.p90.toFixed(0) : "0") + "ms";

  const mode = d.mode || (d.recent && d.recent.some(r => r.route.includes("Skill")) ? "Native Skill" : "Transparent Proxy");
  $("mode").textContent = mode;
  $("mode").className = "badge " + (mode.includes("Skill") ? "sel" : "pass");

  const feed = $("feed");
  feed.textContent = "";
  if (!d.recent || !d.recent.length) {
    feed.innerHTML = '<tr><td colspan="5" class="py-3 text-slate-500">no requests yet</td></tr>';
  } else {
    for (const row of d.recent) {
      const tr = document.createElement("tr");
      tr.className = "border-b border-slate-900";

      const time = document.createElement("td");
      time.className = "py-1 pr-3 text-slate-400 num";
      time.textContent = row.time;

      const agent = document.createElement("td");
      agent.className = "py-1 pr-3 font-medium text-slate-300";
      agent.textContent = row.agent || "–";

      const route = document.createElement("td");
      route.className = "py-1 pr-3";
      route.appendChild(badgeFor(row.route));

      const tools = document.createElement("td");
      tools.className = "py-1 pr-3 num text-slate-300";
      tools.textContent = row.tools;

      const lat = document.createElement("td");
      lat.className = "py-1 pr-3 num text-slate-400";
      lat.textContent = row.latency;

      tr.append(time, agent, route, tools, lat);
      feed.appendChild(tr);
    }
  }
}

async function refreshGraph() {
  const d = await jget("/ui/api/memory");
  const tbody = $("graph");
  tbody.textContent = "";
  if (!d.relations.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="py-3 text-slate-500">no facts stored yet</td></tr>';
    return;
  }
  for (const rel of d.relations) {
    const tr = document.createElement("tr");
    tr.className = "border-b border-slate-900";
    tr.innerHTML =
      '<td class="py-1 pr-3 num">' + rel.source + "</td>" +
      '<td class="py-1 pr-3 text-slate-400">' + rel.predicate + "</td>" +
      '<td class="py-1 pr-3 num">' + rel.target + "</td>" +
      '<td class="py-1 pr-3"><span class="badge ' + (rel.status === "permanent" ? "keep" : "pass") + '">' + rel.status + "</span></td>";
    const td = document.createElement("td");
    td.className = "py-1 text-right";
    const btn = document.createElement("button");
    btn.className = "text-xs text-rose-400 hover:underline";
    btn.textContent = "forget";
    btn.onclick = () => forget(rel.id, tr);
    td.appendChild(btn);
    tr.appendChild(td);
    tbody.appendChild(tr);
  }
}

async function forget(id, row) {
  row.style.opacity = ".4";
  try {
    const r = await fetch("/ui/api/memory", {
      method: "DELETE",
      headers: {"content-type": "application/json"},
      body: JSON.stringify({id}),
    });
    if (!r.ok) throw new Error(r.status);
    row.remove();
  } catch (e) {
    row.style.opacity = "1";
    $("err").textContent = "delete failed: " + e.message;
  }
}

let failures = 0;
async function tick() {
  document.body.classList.add("refreshing");
  try {
    await Promise.all([refreshStats(), refreshGraph()]);
    failures = 0;
    $("conn").textContent = "live";
    $("conn").className = "badge keep";
  } catch (e) {
    failures += 1;
    $("err").textContent = e.message;
    if (failures > 1) {
      $("conn").textContent = "offline";
      $("conn").className = "badge strip";
    }
  } finally {
    document.body.classList.remove("refreshing");
  }
}
tick();
setInterval(tick, 3000);
</script>
</body>
</html>"""


def dashboard_response() -> HTMLResponse:
    return HTMLResponse(DASHBOARD_HTML)


def stats_payload(analytics, recent_limit: int = 20, connection=None) -> Dict[str, Any]:
    """Analytics numbers plus the recent-request feed, UI-shaped."""
    data = analytics.compute().as_dict()
    recent = recent_requests(analytics, recent_limit)
    data["recent"] = recent
    data["connection"] = connection or {}

    tools_offered = data.get("tools_offered") or 0
    tools_forwarded = data.get("tools_forwarded") or 0
    data["pruned_tools_count"] = max(0, tools_offered - tools_forwarded)

    # Determine mode: Native Skill vs Transparent Proxy
    if any("Skill" in str(r.get("route", "")) for r in recent):
        data["mode"] = "Native Skill"
    else:
        tier = (connection or {}).get("tier")
        if tier == "subscription":
            data["mode"] = "Native Skill"
        else:
            data["mode"] = "Transparent Proxy"

    return data


def recent_requests(analytics, limit: int = 20) -> list:
    """The last `limit` request rows, newest first, formatted for the feed."""
    import time as _time

    rows = []
    for entry in analytics.read_entries():
        if entry.get("surface") == "memory":
            continue
        rows.append(entry)
    rows = rows[-limit:][::-1]

    feed = []
    for entry in rows:
        ts = entry.get("ts") or 0
        raw_ts = entry.get("timestamp")
        time_display = "–"
        if ts:
            try:
                time_display = _time.strftime("%H:%M:%S", _time.localtime(float(ts)))
            except Exception:
                pass
        elif raw_ts:
            try:
                from datetime import datetime as _dt
                dt = _dt.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
                time_display = dt.astimezone().strftime("%H:%M:%S")
            except Exception:
                time_display = raw_ts.split("T")[-1][:8] if "T" in str(raw_ts) else str(raw_ts)

        agent = str(entry.get("agent") or entry.get("surface") or "–")
        route = str(entry.get("route") or "unknown")

        before = entry.get("tools_in") if "tools_in" in entry else entry.get("tools_before")
        after = entry.get("tools_out") if "tools_out" in entry else entry.get("tools_after")
        selected_tools = entry.get("selected_tools") or []

        if selected_tools and isinstance(selected_tools, list):
            tools_str = f"[{', '.join(selected_tools)}]"
            if isinstance(before, int) and isinstance(after, int):
                tools_str = f"{before} → {after} {tools_str}"
        elif isinstance(before, int) and isinstance(after, int):
            tools_str = f"{before} → {after}"
        elif isinstance(before, int):
            tools_str = f"{before} → 0"
        else:
            tools_str = "–"

        latency = entry.get("latency_ms")
        feed.append(
            {
                "time": time_display,
                "agent": agent,
                "route": route,
                "tools": tools_str,
                "selected_tools": selected_tools if isinstance(selected_tools, list) else [],
                "latency": f"{latency:.0f}ms" if isinstance(latency, (int, float)) else "–",
            }
        )
    return feed


def memory_relations(graph_memory, limit: int = 50) -> list:
    """Current relations, oldest-last, with ids so the UI can delete them."""
    try:
        connection = graph_memory.connect()
        try:
            rows = connection.execute(
                """
                SELECT id, source, predicate, target, status, confidence
                FROM relations
                ORDER BY last_observed DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        finally:
            connection.close()
    except Exception:
        return []
    return [dict(row) for row in rows]


def forget_relation(graph_memory, relation_id: int) -> bool:
    """Delete one relation by id; returns True when a row was removed."""
    if not isinstance(relation_id, int) or relation_id <= 0:
        return False
    try:
        connection = graph_memory.connect()
        try:
            with connection:
                cursor = connection.execute(
                    "DELETE FROM relations WHERE id = ?", (relation_id,)
                )
            return cursor.rowcount > 0
        finally:
            connection.close()
    except Exception:
        return False


def profile_payload(available: list, active: str) -> Dict[str, Any]:
    return {"available": sorted(available), "active": active}
