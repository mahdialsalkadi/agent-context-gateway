"""
Tests for Google CloudCode / Gemini / Antigravity proxy interception and Jev pruning.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from src.config import Settings, load_settings
from src.gateway import create_app
from src.cli import build_agent_env


def mock_google_stream_bytes(texts: list[str]) -> list[bytes]:
    """Simulate Google Gemini SSE stream response chunks."""
    chunks = []
    for text in texts:
        payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": text}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
        }
        chunks.append(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))
    return chunks


@pytest.fixture
def mock_upstream_transport():
    """Mock httpx transport that records outbound requests and returns responses."""
    recorded_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded_requests.append(request)
        url_str = str(request.url)

        if "streamGenerateContent" in url_str:
            chunks = mock_google_stream_bytes(["Hello from Google Gemini backend!"])
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=b"".join(chunks),
                request=request,
            )
        elif "loadCodeAssist" in url_str:
            return httpx.Response(
                200,
                json={"cloudaicompanionProject": "projects/test-project", "userTier": "PRO"},
                request=request,
            )
        elif "fetchAvailableModels" in url_str:
            return httpx.Response(
                200,
                json={"models": [{"name": "gemini-2.5-flash"}, {"name": "gemini-3.8-flash"}]},
                request=request,
            )
        return httpx.Response(200, json={"status": "ok"}, request=request)

    transport = httpx.MockTransport(handler)
    transport.recorded = recorded_requests
    return transport


@pytest.fixture
def app_and_audit(tmp_path, mock_upstream_transport):
    audit_file = tmp_path / "audit.log"
    settings = load_settings(
        env={
            "UPSTREAM_BASE_URL": "https://daily-cloudcode-pa.googleapis.com",
            "LOG_DIR": str(tmp_path),
            "CLASSIFIER_MODE": "heuristics",
            "GATEWAY_PORT": "8091",
        }
    )
    app = create_app(settings=settings, upstream_transport=mock_upstream_transport)
    return app, audit_file, mock_upstream_transport


def test_build_agent_env_antigravity():
    env = build_agent_env("antigravity", 8091)
    assert env["HTTP_PROXY"] == "http://127.0.0.1:8091"
    assert env["HTTPS_PROXY"] == "http://127.0.0.1:8091"
    assert env["ALL_PROXY"] == "http://127.0.0.1:8091"
    assert env["GOOGLE_API_ENDPOINT"] == "http://127.0.0.1:8091"
    assert env["DAILY_CLOUDCODE_ENDPOINT"] == "http://127.0.0.1:8091"


def test_google_v1internal_load_code_assist_passthrough(app_and_audit):
    app, audit_file, transport = app_and_audit
    client = TestClient(app)

    resp = client.post(
        "/v1internal:loadCodeAssist",
        json={"metadata": {"ide": "antigravity"}},
        headers={"authorization": "Bearer user-token-123"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["cloudaicompanionProject"] == "projects/test-project"

    assert len(transport.recorded) == 1
    req = transport.recorded[0]
    assert "daily-cloudcode-pa.googleapis.com/v1internal:loadCodeAssist" in str(req.url)
    assert req.headers["authorization"] == "Bearer user-token-123"


def test_google_v1internal_fetch_available_models_passthrough(app_and_audit):
    app, audit_file, transport = app_and_audit
    client = TestClient(app)

    resp = client.get(
        "/v1internal:fetchAvailableModels",
        headers={"authorization": "Bearer user-token-123"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["models"]) == 2

    assert len(transport.recorded) == 1
    req = transport.recorded[0]
    assert "daily-cloudcode-pa.googleapis.com/v1internal:fetchAvailableModels" in str(req.url)


@pytest.mark.asyncio
async def test_google_stream_generate_content_pruning(app_and_audit):
    app, audit_file, transport = app_and_audit
    client = TestClient(app)

    sample_tools = [
        {
            "functionDeclarations": [
                {
                    "name": "view_file",
                    "description": "View file content from filesystem",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
                {
                    "name": "run_command",
                    "description": "Run bash shell command",
                    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
                },
                {
                    "name": "send_email",
                    "description": "Send email via SMTP",
                    "parameters": {"type": "object", "properties": {"to": {"type": "string"}}},
                },
            ]
        }
    ]

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "Can you check the git status with run_command?"}],
            }
        ],
        "tools": sample_tools,
    }

    # Mock Jev router to prune to only run_command
    with patch(
        "src.gateway.route_tools_via_jev",
        new=AsyncMock(return_value=[sample_tools[0]["functionDeclarations"][1]]),
    ):
        resp = client.post(
            "/v1internal:streamGenerateContent?alt=sse",
            json=payload,
            headers={
                "authorization": "Bearer test-gemini-session",
                "x-goog-api-client": "gl-go/1.22",
            },
        )

        assert resp.status_code == 200
        assert resp.headers["X-Proxy-Route"] == "Jev-Routed"
        assert resp.headers["X-Proxy-Tools-Before"] == "3"
        assert resp.headers["X-Proxy-Tools-After"] == "1"

        # Check forwarded request body has pruned function declarations
        assert len(transport.recorded) == 1
        req = transport.recorded[0]
        assert "alt=sse" in str(req.url)
        body = json.loads(req.content)
        assert len(body["tools"][0]["functionDeclarations"]) == 1
        assert body["tools"][0]["functionDeclarations"][0]["name"] == "run_command"

        # Verify audit log was recorded
        assert audit_file.exists()
        lines = [json.loads(l) for l in audit_file.read_text().splitlines() if l.strip()]
        assert len(lines) >= 1
        audit_entry = lines[-1]
        assert audit_entry["agent"] == "antigravity"
        assert audit_entry["surface"] == "google"
        assert audit_entry["route"] == "Jev-Routed"
        assert audit_entry["tools_before"] == 3
        assert audit_entry["tools_after"] == 1
        assert audit_entry["selected_tools"] == ["run_command"]


@pytest.mark.asyncio
async def test_google_stream_generate_content_strips_to_zero(app_and_audit):
    app, audit_file, transport = app_and_audit
    client = TestClient(app)

    sample_tools = [
        {
            "functionDeclarations": [
                {"name": "view_file", "description": "View file"},
                {"name": "run_command", "description": "Run bash"},
            ]
        }
    ]

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "Hello, how are you today?"}],
            }
        ],
        "tools": sample_tools,
    }

    # Mock Jev router returning empty list (conversational prompt)
    with patch("src.gateway.route_tools_via_jev", new=AsyncMock(return_value=[])):
        resp = client.post(
            "/v1internal:streamGenerateContent?alt=sse",
            json=payload,
        )

        assert resp.status_code == 200
        assert resp.headers["X-Proxy-Route"] == "Jev-Strip"
        assert resp.headers["X-Proxy-Tools-Before"] == "2"
        assert resp.headers["X-Proxy-Tools-After"] == "0"

        # Forwarded request body should have tools stripped
        req = transport.recorded[0]
        body = json.loads(req.content)
        assert "tools" not in body

        # Verify audit log
        lines = [json.loads(l) for l in audit_file.read_text().splitlines() if l.strip()]
        assert lines[-1]["route"] == "Jev-Strip"
        assert lines[-1]["tools_after"] == 0


def test_google_stream_generate_content_mid_tool_loop_passthrough(app_and_audit):
    app, audit_file, transport = app_and_audit
    client = TestClient(app)

    sample_tools = [
        {
            "functionDeclarations": [
                {"name": "view_file", "description": "View file"},
            ]
        }
    ]

    # Tool loop turn has functionResponse in parts
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": "read file"}]},
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "view_file",
                            "response": {"content": "file contents"},
                        }
                    }
                ],
            },
        ],
        "tools": sample_tools,
    }

    resp = client.post(
        "/v1internal:streamGenerateContent?alt=sse",
        json=payload,
    )
    assert resp.status_code == 200
    assert resp.headers["X-Proxy-Route"] == "ToolLoop"
    assert resp.headers["X-Proxy-Tools-Before"] == "1"
    assert resp.headers["X-Proxy-Tools-After"] == "1"
