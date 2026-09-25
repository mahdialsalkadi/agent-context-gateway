"""
Shared fixtures.

The whole suite runs offline: the gateway under test talks to a real (but local
and keyless) mock upstream, so the true HTTP and streaming paths are exercised
without spending a cent or needing credentials.
"""

from __future__ import annotations

import httpx
import pytest

from src.artifacts import ArtifactStore
from src.classifier import Classifier
from src.config import Settings
from src.gateway import create_app
from src.memory import GraphMemory
from tests.mock_upstream import MockState, start_mock_upstream


@pytest.fixture(scope="session")
def mock_upstream():
    """A single mock upstream for the whole session."""
    server, state, base_url = start_mock_upstream()
    yield state, base_url
    server.shutdown()
    server.server_close()


@pytest.fixture
def mock_state(mock_upstream) -> MockState:
    """Request log, cleared before each test that asks for it."""
    state, _ = mock_upstream
    state.reset()
    return state


@pytest.fixture
def settings(tmp_path, mock_upstream) -> Settings:
    """Isolated settings: temp dirs, mock upstream, no classifier."""
    _, base_url = mock_upstream
    return Settings(
        host="127.0.0.1",
        port=8090,
        data_dir=tmp_path / "data",
        log_dir=tmp_path / "logs",
        shm_cache_dir=tmp_path / "shm",
        upstream_base_url=base_url,
        upstream_api_key="test-upstream-key",
        classifier_api_url="",
        classifier_api_key="",
        truncate_threshold_chars=200,
        memory_injection=False,
    )


@pytest.fixture
def store(settings) -> ArtifactStore:
    artifact_store = ArtifactStore(settings.shm_cache_dir, settings.truncate_threshold_chars)
    artifact_store.ensure_dir()
    return artifact_store


@pytest.fixture
def memory(settings) -> GraphMemory:
    graph = GraphMemory(settings.db_path)
    graph.init_schema()
    return graph


@pytest.fixture
def classifier(settings) -> Classifier:
    return Classifier(settings)


@pytest.fixture
def app(settings, classifier, store, memory):
    return create_app(
        settings=settings, classifier=classifier, store=store, memory=memory
    )


@pytest.fixture
async def api(app):
    """Client that drives the real ASGI app."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://gateway", timeout=30.0
    ) as client:
        yield client


@pytest.fixture
def app_with_transport(settings, classifier, store, memory):
    """Factory for apps backed by an in-process transport instead of a server."""

    def _build(transport: httpx.AsyncBaseTransport):
        return create_app(
            settings=settings,
            classifier=classifier,
            store=store,
            memory=memory,
            upstream_transport=transport,
        )

    return _build
