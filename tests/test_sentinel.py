"""
Sentinel watchdog behaviour.

The load-bearing assertion is `test_probe_rejects_a_foreign_service`. A status
check that only looks at the HTTP code reports healthy for anything listening on
the port, which makes the watchdog silently useless: the gateway can be dead
while a stale process holds 8090. These tests pin the identity check.
"""

from __future__ import annotations

import httpx
import pytest

from src.sentinel import Sentinel


class FakeResponse:
    """Minimal stand-in for `httpx.Response` as used by the probe."""

    def __init__(self, status_code: int = 200, payload=None, raises=None) -> None:
        self.status_code = status_code
        self._payload = payload
        self._raises = raises

    def json(self):
        if isinstance(self._raises, Exception):
            raise self._raises
        return self._payload


def patch_get(monkeypatch, response=None, error=None):
    """Replace `httpx.get` with a stub, recording the call."""
    calls = []

    def fake_get(url, timeout=None):
        calls.append({"url": url, "timeout": timeout})
        if error is not None:
            raise error
        return response

    monkeypatch.setattr(httpx, "get", fake_get)
    return calls


def healthy_payload(data_dir: str) -> dict:
    """The shape `create_app` actually serves from /health."""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "listen": "127.0.0.1:8090",
        "upstream": "http://127.0.0.1:9099/v1",
        "upstream_key": True,
        "gateway_auth": False,
        "classifier_enabled": False,
        "data_dir": data_dir,
        "log_dir": "/tmp/logs",
        "shm_cache_dir": "/tmp/shm",
        "memory_injection": True,
        "artifacts": {"dir": "/tmp/shm", "files": 0, "bytes": 0},
        "memory": {"relations": 0, "entities": 0, "journal_mode": "wal", "by_status": {}},
        "timestamp": 1790348184.0,
    }


class StubStore:
    def __init__(self, on_prune=None) -> None:
        self.on_prune = on_prune or (lambda *a, **k: {})
        self.calls = 0

    def prune(self, *_args, **_kwargs):
        self.calls += 1
        return self.on_prune()


class StubMemory:
    def __init__(self, on_compact=None) -> None:
        self.on_compact = on_compact or (lambda *a, **k: {})
        self.calls = 0

    def compact(self, *_args, **_kwargs):
        self.calls += 1
        return self.on_compact()


def build_sentinel(settings, store=None, memory=None) -> Sentinel:
    """Sentinel with in-memory collaborators, so a cycle touches nothing on disk."""
    return Sentinel(
        settings,
        memory=memory or StubMemory(),
        store=store or StubStore(),
        health_timeout=1.0,
    )


@pytest.fixture
def sentinel(settings) -> Sentinel:
    return build_sentinel(settings)


# ------------------------------------------------------------------------------
# Identity
# ------------------------------------------------------------------------------
def test_probe_identifies_our_gateway(settings, sentinel, monkeypatch):
    patch_get(monkeypatch, FakeResponse(200, healthy_payload(str(settings.data_dir))))

    info = sentinel.probe()

    assert info["reachable"] is True
    assert info["identified"] is True
    assert info["version"] == "0.1.0"
    assert sentinel.check_health() is True


def test_probe_rejects_a_foreign_service(settings, sentinel, monkeypatch):
    """The exact failure mode: something else answers 200 on our port."""
    patch_get(monkeypatch, FakeResponse(200, {"hello": "i am not a gateway"}))

    info = sentinel.probe()

    assert info["reachable"] is True, "it did answer"
    assert info["identified"] is False, "but it is not ours"
    assert sentinel.check_health() is False


def test_probe_rejects_a_gateway_pointed_at_a_different_data_dir(settings, sentinel, monkeypatch):
    """Same routes, different instance: not the process we are babysitting."""
    patch_get(monkeypatch, FakeResponse(200, healthy_payload("/somewhere/else/data")))

    info = sentinel.probe()

    assert info["identified"] is False
    assert "data_dir mismatch" in info["detail"]


def test_probe_rejects_non_json_and_non_object_payloads(settings, sentinel, monkeypatch):
    patch_get(monkeypatch, FakeResponse(200, raises=ValueError("not json")))
    assert sentinel.probe()["identified"] is False

    patch_get(monkeypatch, FakeResponse(200, ["a", "list"]))
    assert sentinel.probe()["identified"] is False


def test_probe_handles_a_non_200_answer(settings, sentinel, monkeypatch):
    patch_get(monkeypatch, FakeResponse(503, healthy_payload(str(settings.data_dir))))

    info = sentinel.probe()

    assert info["reachable"] is True
    assert info["identified"] is False
    assert "unexpected status 503" in info["detail"]


def test_probe_reports_unreachable_on_a_connection_error(settings, sentinel, monkeypatch):
    patch_get(monkeypatch, error=httpx.ConnectError("connection refused"))

    info = sentinel.probe()

    assert info["reachable"] is False
    assert info["identified"] is False
    assert "unreachable" in info["detail"]


# ------------------------------------------------------------------------------
# Cycle consequences
# ------------------------------------------------------------------------------
def test_cycle_does_not_respawn_over_a_foreign_service(settings, monkeypatch):
    """Respawning cannot win the bind, so the cycle must report instead of churn."""
    patch_get(monkeypatch, FakeResponse(200, {"hello": "i am not a gateway"}))
    sentinel = build_sentinel(settings)
    spawned = []
    monkeypatch.setattr(sentinel, "spawn_gateway", lambda: spawned.append(True) or True)

    report = sentinel.run_cycle()

    assert report["gateway_healthy"] is False
    assert report["gateway_reachable"] is True
    assert report["foreign_service_on_port"] is True
    assert report["respawned"] is False
    assert spawned == [], "must not spawn into a port it cannot bind"


def test_cycle_still_prunes_when_a_foreign_service_holds_the_port(settings, monkeypatch):
    """Maintenance must not be skipped just because the watchdog is confused."""
    patch_get(monkeypatch, FakeResponse(200, {"hello": "i am not a gateway"}))
    store = StubStore()
    memory = StubMemory()
    sentinel = build_sentinel(settings, store=store, memory=memory)
    monkeypatch.setattr(sentinel, "spawn_gateway", lambda: True)

    report = sentinel.run_cycle()

    assert store.calls == 1 and memory.calls == 1
    assert "artifacts" in report and "memory" in report


def test_cycle_respawns_when_nothing_is_listening(settings, monkeypatch):
    patch_get(monkeypatch, error=httpx.ConnectError("connection refused"))
    sentinel = build_sentinel(settings)
    spawned = []
    monkeypatch.setattr(sentinel, "spawn_gateway", lambda: spawned.append(True) or True)

    report = sentinel.run_cycle()

    assert report["gateway_healthy"] is False
    assert report["respawned"] is True
    assert spawned == [True]


def test_status_only_mutates_nothing(settings, monkeypatch):
    patch_get(monkeypatch, FakeResponse(200, {"hello": "i am not a gateway"}))

    def explode(*_args, **_kwargs):
        raise AssertionError("--status must not mutate anything")

    store = StubStore(on_prune=explode)
    memory = StubMemory(on_compact=explode)
    sentinel = build_sentinel(settings, store=store, memory=memory)
    monkeypatch.setattr(sentinel, "spawn_gateway", explode)

    report = sentinel.run_cycle(status_only=True)

    assert report["gateway_healthy"] is False
    assert "health_detail" in report
    assert "respawned" not in report
    assert report["shm_dir"] == str(settings.shm_cache_dir)
    assert store.calls == 0 and memory.calls == 0
