"""
Embedded web dashboard: one HTML page, three JSON endpoints, zero build step.

The page is a Python string served by FastAPI at GET /ui. Vanilla JS only --
the only remote reference is the Tailwind CDN script, which the page degrades
gracefully without. All dynamic data comes from /ui/api/* endpoints on this
same origin, so the dashboard works on a loopback port with no CORS setup.

The analytics figures are the same estimates `agent-gateway stats` shows; the
UI exists to make them visible, not to pretend they are exact.
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
      <p id="sub" class="text-sm text-slate-400">loading…</p>
    </div>
    <div id="conn" class="badge keep">connecting…</div>
  </header>

  <section class="grid grid-cols-2 md:grid-cols-4 gap-4">
    <div class="card p-4"><div class="text-xs text-slate-400">tokens saved</div>
      <div id="tokens" class="num text-2xl font-bold mt-1">–</div></div>
    <div class="card p-4"><div class="text-xs text-slate-400">dollars saved*</div>
      <div id="usd" class="num text-2xl font-bold mt-1 text-emerald-400">–</div></div>
    <div class="card p-4"><div class="text-xs text-slate-400">requests</div>
      <div id="requests" class="num text-2xl font-bold mt-1">–</div></div>
    <div class="card p-4"><div class="text-xs text-slate-400">p50 / p90 latency</div>
      <div id="latency" class="num text-2xl font-bold mt-1">–</div></div>
  </section>

  <section class="card p-4">
    <div class="flex items-center justify-between mb-3">
      <h2 class="font-semibold">profile</h2>
      <span id="profile-note" class="text-xs text-slate-500">switching restarts nothing; state is shared</span>
    </div>
    <div id="profiles" class="flex flex-wrap gap-2">
      <span class="text-sm text-slate-500">loading…</span>
    </div>
    <p id="profile-msg" class="text-xs mt-2 text-slate-400"></p>
  </section>

  <section class="card p-4">
    <div class="flex items-center justify-between mb-3">
      <h2 class="font-semibold">recent requests</h2>
      <span class="text-xs text-slate-500">last 20 · from the audit log</span>
    </div>
    <div class="overflow-x-auto">
    <table class="w-full text-sm">
      <thead><tr class="text-left text-xs text-slate-500 border-b border-slate-800">
        <th class="py-1 pr-3">time</th><th class="py-1 pr-3">route</th>
        <th class="py-1 pr-3">tools</th><th class="py-1 pr-3">latency</th></tr></thead>
      <tbody id="feed"><tr><td colspan="4" class="py-3 text-slate-500">no requests yet</td></tr></tbody>
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
    <p>* dollars are an estimate at the benchmark price; exact token counts need the upstream's tokenizer.</p>
    <p>auto-refreshes every 3s · <span id="err" class="text-rose-400"></span></p>
  </footer>
</div>

<script>
const $ = (id) => document.getElementById(id);
const badgeClass = (route) =>
  route.includes("Strip") ? "badge strip"
  : route.includes("Passthrough") ? "badge pass"
  : route.includes("Selective") ? "badge sel"
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
  $("tokens").textContent = d.estimated_tokens_saved.toLocaleString();
  $("usd").textContent = "$" + d.estimated_usd_saved.toFixed(4);
  $("requests").textContent = d.requests.toLocaleString();
  $("latency").textContent = d.latency_ms.p50.toFixed(0) + " / " + d.latency_ms.p90.toFixed(0) + "ms";
  $("sub").textContent = "benchmark $" + d.benchmark_usd_per_mtoken.toFixed(2) + "/M prompt tokens";

  const feed = $("feed");
  feed.textContent = "";
  if (!d.recent.length) {
    feed.innerHTML = '<tr><td colspan="4" class="py-3 text-slate-500">no requests yet</td></tr>';
  }
  for (const row of d.recent) {
    const tr = document.createElement("tr");
    tr.className = "border-b border-slate-900";
    const time = document.createElement("td");
    time.className = "py-1 pr-3 text-slate-400 num";
    time.textContent = row.time;
    const route = document.createElement("td");
    route.className = "py-1 pr-3";
    route.appendChild(badgeFor(row.route));
    const tools = document.createElement("td");
    tools.className = "py-1 pr-3 num";
    tools.textContent = row.tools;
    const lat = document.createElement("td");
    lat.className = "py-1 pr-3 num";
    lat.textContent = row.latency;
    tr.append(time, route, tools, lat);
    feed.appendChild(tr);
  }
}

async function refreshProfile() {
  const d = await jget("/ui/api/profile");
  const wrap = $("profiles");
  wrap.textContent = "";
  const current = document.createElement("span");
  current.className = "badge keep";
  current.textContent = "active: " + d.active;
  wrap.appendChild(current);
  for (const p of d.available) {
    if (p === d.active) continue;
    const btn = document.createElement("button");
    btn.className = "badge pass hover:opacity-80";
    btn.textContent = "switch → " + p;
    btn.onclick = () => switchProfile(p);
    wrap.appendChild(btn);
  }
}

async function switchProfile(name) {
  $("profile-msg").textContent = "switching to " + name + " …";
  try {
    const r = await fetch("/ui/api/profile", {
      method: "POST",
      headers: {"content-type": "application/json"},
      body: JSON.stringify({profile: name}),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || r.status);
    $("profile-msg").textContent = d.message || ("now on " + name);
    refreshProfile();
  } catch (e) {
    $("profile-msg").textContent = "switch failed: " + e.message;
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
    await Promise.all([refreshStats(), refreshProfile(), refreshGraph()]);
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


def stats_payload(analytics, recent_limit: int = 20) -> Dict[str, Any]:
    """Analytics numbers plus the recent-request feed, UI-shaped."""
    data = analytics.compute().as_dict()
    data["recent"] = recent_requests(analytics, recent_limit)
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
        before = entry.get("tools_before")
        after = entry.get("tools_after")
        if isinstance(before, int) and isinstance(after, int) and after:
            tools = f"{before} → {after}"
        elif isinstance(before, int):
            tools = f"{before} → 0"
        else:
            tools = "–"
        latency = entry.get("latency_ms")
        feed.append(
            {
                "time": _time.strftime("%H:%M:%S", _time.localtime(ts)) if ts else "–",
                "route": str(entry.get("route") or "unknown"),
                "tools": tools,
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
