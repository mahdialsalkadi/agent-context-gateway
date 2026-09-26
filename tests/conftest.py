"""
Shared fixtures.

The whole suite runs offline: the gateway under test talks to a real (but local
and keyless) mock upstream, so the true HTTP and streaming paths are exercised
without spending a cent or needing credentials.
"""

from __future__ import annotations

import os
import shutil
import tempfile

import httpx
import pytest

# --- hermetic HOME + hermetic environment ------------------------------------
# `src.gateway` resolves settings at import time, and `load_env_file` only fills
# in defaults via os.environ.setdefault. Two real-world leaks would therefore
# reach every test: a developer's `~/.agent-gateway/.env` (via HOME) and the
# repo's own `.env` — which lives at the *project root*, unaffected by HOME, and
# is picked up when `src.gateway`'s import-time `create_app()` loads the default
# env file. Values already in os.environ outrank every profile a test loads, so
# both are purged here: point HOME at a scratch directory AND strip anything a
# real `.env` may have injected, before importing any project module.
_TEST_HOME = tempfile.mkdtemp(prefix="agent-gateway-test-home-")
_ORIGINAL_HOME = os.environ.get("HOME")
os.environ["HOME"] = _TEST_HOME
os.environ["USERPROFILE"] = _TEST_HOME

# Keys a real `.env` can inject and that a test's settings resolution must never
# inherit. Purged before the imports (they would otherwise be seen by the
# import-time create_app), after the imports (its load_env_file re-injects them),
# and around every test (code under test can re-load the file mid-session).
_REAL_ENV_KEYS = (
    "GATEWAY_HOST",
    "GATEWAY_PORT",
    "CLASSIFIER_MODE",
    "CLASSIFIER_MODEL",
    "CLASSIFIER_API_URL",
    "CLASSIFIER_API_KEY",
    "UPSTREAM_BASE_URL",
    "UPSTREAM_API_KEY",
    "LOCAL_JEV_URL",
    "LOCAL_JEV_TIMEOUT_SECONDS",
    "ALLOW_LEGACY_UPSTREAM_PORT",
    "AGENT_GATEWAY_PROFILE",
    "OLLAMA_BASE_URL",
)

for _leaked in _REAL_ENV_KEYS:
    os.environ.pop(_leaked, None)

from src.artifacts import ArtifactStore
from src.classifier import Classifier
from src.config import Settings
from src.gateway import create_app
from src.memory import GraphMemory
from tests.mock_upstream import MockState, start_mock_upstream

# `src.gateway`'s import-time create_app() ran load_env_file() AFTER the strip
# above, re-injecting the real .env's values. Purge again now imports are done.
for _leaked in _REAL_ENV_KEYS:
    os.environ.pop(_leaked, None)


@pytest.fixture(autouse=True)
def _purge_real_env():
    """Keep the developer's real .env out of every test's settings resolution."""
    for key in _REAL_ENV_KEYS:
        os.environ.pop(key, None)
    yield
    for key in _REAL_ENV_KEYS:
        os.environ.pop(key, None)


@pytest.fixture(scope="session", autouse=True)
def _isolated_home():
    """Restore the caller's HOME and bin the scratch directory after the run."""
    yield
    if _ORIGINAL_HOME is not None:
        os.environ["HOME"] = _ORIGINAL_HOME
    shutil.rmtree(_TEST_HOME, ignore_errors=True)


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
