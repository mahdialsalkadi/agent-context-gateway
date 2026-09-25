"""
The embedded web dashboard.

`GET /ui` must serve real, self-contained HTML with no build step; the JSON
endpoints behind it must return UI-shaped data; and the graph mutation must
actually delete the row it says it deletes. Everything runs against the real
ASGI app with the offline mock upstream.
"""

from __future__ import annotations

import json

import pytest

from src.dashboard import DASHBOARD_HTML, memory_relations, recent_requests, stats_payload


@pytest.fixture(autouse=True)
def restore_environ():
    """The live profile switch deliberately updates os.environ so the rest of
    the process (analytics CLI, memory audit, respawn) sees the new profile.
    That is correct behaviour in production and poisonous between tests, so
    this file snapshots and restores the environment around every test."""
    saved = dict(__import__("os").environ)
    yield
    __import__("os").environ.clear()
    __import__("os").environ.update(saved)


# ------------------------------------------------------------------------------
# GET /ui
# ------------------------------------------------------------------------------
async def test_ui_serves_html(api):
    response = await api.get("/ui")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    body = response.text
    assert body.lstrip().lower().startswith("<!doctype html")
    assert "</html>" in body


async def test_ui_html_is_self_contained(api):
    """No build step, no framework bootstrapping, no script src except the CDN."""
    response = await api.get("/ui")
    body = response.text

    assert "text/html" in response.headers["content-type"]
    # The interactive parts are vanilla JS over fetch() to same-origin JSON.
    assert "fetch(" in body
    assert "setInterval" in body
    assert "npm " not in body and "webpack" not in body


async def test_ui_is_hidden_from_the_openapi_schema(api):
    schema = (await api.get("/openapi.json")).json()
    assert "/ui" not in schema.get("paths", {})
    assert "/v1/chat/completions" in schema.get("paths", {})


# ------------------------------------------------------------------------------
# /ui/api/stats
# ------------------------------------------------------------------------------
async def test_stats_endpoint_reflects_the_audit_log(api, tmp_path):
    # A request first, so there is something in the audit log.
    await api.post(
        "/v1/chat/completions",
        json={"model": "mock-model", "stream": True, "messages": [{"role": "user", "content": "hello"}]},
    )

    response = await api.get("/ui/api/stats")

    assert response.status_code == 200
    data = response.json()
    assert data["requests"] >= 1
    assert "estimated_tokens_saved" in data
    assert "estimated_usd_saved" in data
    assert "benchmark_usd_per_mtoken" in data
    assert isinstance(data["recent"], list) and data["recent"]
    first = data["recent"][0]
    assert {"time", "route", "tools", "latency"} <= set(first)


def test_recent_requests_hides_memory_rows_and_orders_newest_first(tmp_path, monkeypatch):
    from src.analytics import Analytics

    log = tmp_path / "audit.log"
    log.write_text(
        "\n".join(
            [
                json.dumps({"ts": 1000, "surface": "openai", "route": "FastPath-Strip", "latency_ms": 5.0, "tools_before": 10, "tools_after": 0}),
                json.dumps({"ts": 2000, "surface": "memory", "triples_injected": 2}),
                json.dumps({"ts": 3000, "surface": "openai", "route": "Default", "latency_ms": 7.0, "tools_before": 2, "tools_after": 2}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = recent_requests(Analytics(log), limit=10)

    assert [row["route"] for row in rows] == ["Default", "FastPath-Strip"]
    assert rows[0]["tools"] == "2 → 2"
    assert rows[1]["tools"] == "10 → 0"


def test_stats_payload_shapes_analytics_for_the_ui(tmp_path):
    from src.analytics import Analytics

    log = tmp_path / "audit.log"
    log.write_text("", encoding="utf-8")

    data = stats_payload(Analytics(log))

    assert data["requests"] == 0
    assert data["recent"] == []
    assert "latency_ms" in data


# ------------------------------------------------------------------------------
# /ui/api/profile
# ------------------------------------------------------------------------------
async def test_profile_endpoint_lists_shipped_profiles(api):
    response = await api.get("/ui/api/profile")

    assert response.status_code == 200
    data = response.json()
    assert {"antigravity", "claude", "codex", "hermes", "openrouter"} <= set(data["available"])
    assert data["active"]


async def test_profile_switch_rejects_unknown_names(api):
    response = await api.post(
        "/ui/api/profile", json={"profile": "not-a-profile"}
    )

    assert response.status_code == 404
    assert "does not exist" in response.json()["error"]


async def test_profile_switch_requires_a_name(api):
    response = await api.post("/ui/api/profile", json={})

    assert response.status_code == 400


async def test_profile_switch_applies_settings_live(api):
    """Switching must change the app's settings without a restart."""
    response = await api.post("/ui/api/profile", json={"profile": "antigravity"})

    assert response.status_code == 200
    message = response.json()["message"]
    assert "antigravity" in message
    assert "http://127.0.0.1:8080/v1" in message

    app = api._transport.app
    assert app.state.settings.upstream_base_url == "http://127.0.0.1:8080/v1"
    assert app.state.settings.effective_classifier_mode == "upstream_reused"


# ------------------------------------------------------------------------------
# /ui/api/memory
# ------------------------------------------------------------------------------
async def test_memory_endpoint_lists_relations(api, memory):
    memory.ingest_sync("s1", "The billing service uses PostgreSQL 16.", "")

    response = await api.get("/ui/api/memory")

    assert response.status_code == 200
    relations = response.json()["relations"]
    assert any(rel["target"] == "PostgreSQL" for rel in relations)
    row = relations[0]
    assert {"id", "source", "predicate", "target", "status"} <= set(row)


async def test_forget_deletes_exactly_one_relation(api, memory):
    memory.ingest_sync("s1", "The billing service uses PostgreSQL 16.", "")
    memory.ingest_sync("s2", "The cache requires Memcached.", "")

    relations = memory_relations(memory)
    victim = relations[0]
    survivors = [rel["id"] for rel in relations if rel["id"] != victim["id"]]

    response = await api.request(
        "DELETE", "/ui/api/memory", json={"id": victim["id"]}
    )

    assert response.status_code == 200
    assert response.json()["deleted"] == victim["id"]

    remaining = memory_relations(memory)
    assert len(remaining) == len(relations) - 1
    assert [rel["id"] for rel in remaining] == survivors


async def test_forget_rejects_bad_ids(api):
    for bad in (0, -1, "abc", None):
        response = await api.request("DELETE", "/ui/api/memory", json={"id": bad})
        assert response.status_code == 404


async def test_forget_on_an_empty_graph_is_a_clean_404(api):
    response = await api.request("DELETE", "/ui/api/memory", json={"id": 99999})
    assert response.status_code == 404


# ------------------------------------------------------------------------------
# The page degrades without the CDN
# ------------------------------------------------------------------------------
def test_html_contains_static_fallbacks():
    """With JS disabled or the CDN blocked, the page still shows its sections."""
    assert "recent requests" in DASHBOARD_HTML
    assert "knowledge graph" in DASHBOARD_HTML
    assert "estimated" in DASHBOARD_HTML.lower() or "dollars" in DASHBOARD_HTML.lower()
