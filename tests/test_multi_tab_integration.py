"""Tests for multi-tab environment integration, agent binary wrappers, and Hermes local model selection."""

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

try:
    import yaml
except ImportError:
    yaml = None

from src import cli
from src.config import Settings
from src.gateway import create_app


def test_update_shell_file_idempotency_and_cleanup(tmp_path):
    shell_file = tmp_path / "config.fish"
    initial_content = (
        "# Some user custom config\n"
        "alias foo='bar'\n"
        "export HTTP_PROXY=\"http://127.0.0.1:8091\"\n"
        "alias agy='HTTP_PROXY=\"http://127.0.0.1:8091\" /bin/agy'\n"
        "set -gx PATH /custom/bin $PATH\n"
    )
    shell_file.write_text(initial_content, encoding="utf-8")

    block_lines = [
        'set -gx AGY_GATEWAY_URL "http://127.0.0.1:8091"',
        'set -gx OPENAI_BASE_URL "http://127.0.0.1:8091/v1"',
    ]
    clean_patterns = [
        r"export\s+HTTP_PROXY=.*:809",
        r"alias\s+agy=.*809",
    ]

    res = cli._update_shell_file(shell_file, block_lines, clean_patterns=clean_patterns)
    assert res is True

    content = shell_file.read_text(encoding="utf-8")
    assert "alias foo='bar'" in content
    assert "set -gx PATH /custom/bin $PATH" in content
    # Legacy lines cleaned
    assert "export HTTP_PROXY" not in content
    assert "alias agy" not in content
    # Block added
    assert cli.SHELL_BLOCK_START in content
    assert 'set -gx AGY_GATEWAY_URL "http://127.0.0.1:8091"' in content
    assert cli.SHELL_BLOCK_END in content

    # Run again with updated port to assert idempotency and update
    block_lines_v2 = [
        'set -gx AGY_GATEWAY_URL "http://127.0.0.1:8092"',
        'set -gx OPENAI_BASE_URL "http://127.0.0.1:8092/v1"',
    ]
    res2 = cli._update_shell_file(shell_file, block_lines_v2, clean_patterns=clean_patterns)
    assert res2 is True

    content_v2 = shell_file.read_text(encoding="utf-8")
    assert content_v2.count(cli.SHELL_BLOCK_START) == 1
    assert content_v2.count(cli.SHELL_BLOCK_END) == 1
    assert 'set -gx AGY_GATEWAY_URL "http://127.0.0.1:8092"' in content_v2
    assert "8091" not in content_v2


def test_install_agent_shims_elf_rename_and_wrapper(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)

    # Simulate an authentic ELF binary at bin/agy
    agy_bin = bin_dir / "agy"
    agy_bin.write_bytes(b"\x7fELF\x02\x01\x01\x00dummy_binary_bytes")

    installed = cli.install_agent_shims(port=8091, bin_dir=bin_dir)
    assert any("agy" in p for p in installed)

    # Verify original ELF was safely renamed to agy-real
    agy_real = bin_dir / "agy-real"
    assert agy_real.exists()
    assert agy_real.read_bytes().startswith(b"\x7fELF")

    # Verify agy is now the executable wrapper script
    assert agy_bin.exists()
    wrapper_text = agy_bin.read_text(encoding="utf-8")
    assert 'export AGY_GATEWAY_URL="${AGY_GATEWAY_URL:-http://127.0.0.1:8091}"' in wrapper_text
    assert "unset HTTP_PROXY" in wrapper_text
    assert "agy-real" in wrapper_text
    assert (agy_bin.stat().st_mode & 0o111) != 0


def test_purge_hermes_proxy_cache(tmp_path, monkeypatch):
    hermes_dir = tmp_path / ".hermes"
    hermes_dir.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # 1. provider_models_cache.json with stale gateway openrouter models
    cache_path = hermes_dir / "provider_models_cache.json"
    cache_data = {
        "openrouter": {"models": ["gpt-4o", "claude-3-opus"]},
        "custom:http://127.0.0.1:8091/v1#abc": {"models": ["openrouter/auto", "qwen-2.5-coder"]},
        "custom:http://127.0.0.1:8090/v1": {"models": ["mistral-large"]},
    }
    cache_path.write_text(json.dumps(cache_data), encoding="utf-8")

    # 2. config.yaml with custom_providers if yaml is available
    config_path = hermes_dir / "config.yaml"
    if yaml is not None:
        config_data = {
            "model": {"default": "qwen"},
            "custom_providers": [
                {
                    "name": "hermes-proxy",
                    "base_url": "http://127.0.0.1:8091/v1",
                    "models": {"openrouter/auto": {}},
                    "models_discovered": True,
                }
            ],
        }
        config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    cli.purge_hermes_proxy_cache(port=8091)

    # Assert cache pruned
    with open(cache_path, "r", encoding="utf-8") as f:
        new_cache = json.load(f)
    assert "openrouter" in new_cache
    assert "custom:http://127.0.0.1:8091/v1#abc" not in new_cache
    assert "custom:http://127.0.0.1:8090/v1" not in new_cache

    # Assert config reset
    if yaml is not None:
        with open(config_path, "r", encoding="utf-8") as f:
            new_config = yaml.safe_load(f)
        provider = new_config["custom_providers"][0]
        assert provider["models_discovered"] is False
        assert provider["models"] == {}


def test_gateway_healthy_verifies_upstream_match(monkeypatch):
    import src.config
    from src.config import Settings

    settings = Settings(
        data_dir=Path("/test/data"),
        upstream_base_url="http://127.0.0.1:11435/v1",
    )
    monkeypatch.setattr(src.config, "load_settings", lambda *args, **kwargs: settings)

    # 1. Matching data_dir AND matching upstream -> Healthy
    monkeypatch.setattr(
        cli,
        "_probe_health",
        lambda timeout=1.5: {
            "status": "ok",
            "data_dir": "/test/data",
            "upstream": "http://127.0.0.1:11435/v1",
        },
    )
    assert cli._gateway_healthy() is True

    # 2. Matching data_dir BUT stale upstream (e.g. was openrouter) -> Not Healthy
    monkeypatch.setattr(
        cli,
        "_probe_health",
        lambda timeout=1.5: {
            "status": "ok",
            "data_dir": "/test/data",
            "upstream": "https://openrouter.ai/api/v1",
        },
    )
    assert cli._gateway_healthy() is False


def test_is_local_upstream_classification():
    local_cfg = Settings(upstream_base_url="http://127.0.0.1:11435/v1")
    assert local_cfg.is_local_upstream is True

    ollama_cfg = Settings(upstream_base_url="http://localhost:11434/v1")
    assert ollama_cfg.is_local_upstream is True

    cloud_cfg = Settings(upstream_base_url="https://openrouter.ai/api/v1")
    assert cloud_cfg.is_local_upstream is False


def test_cmd_integrate(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".bashrc").write_text("", encoding="utf-8")

    args = argparse.Namespace(port=8091)
    ret = cli.cmd_integrate(args)
    assert ret == 0

    captured = capsys.readouterr()
    assert "Multi-tab integration complete" in captured.err
