"""Lifecycle management for the local Jev llama-server sidecar.

Ensures that when CLASSIFIER_MODE=local_jev, llama-server is automatically
started alongside agent-gateway and cleanly terminated when agent-gateway stops,
releasing VRAM back to the system.
"""

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx

from .config import CLASSIFIER_MODE_LOCAL_JEV, Settings


def _jev_pid_path(settings: Settings) -> Path:
    return settings.log_dir / "jev.pid"


def _read_jev_pid(settings: Settings) -> Optional[int]:
    try:
        return int(_jev_pid_path(settings).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _write_jev_pid(settings: Settings, pid: int) -> None:
    try:
        path = _jev_pid_path(settings)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(pid) + "\n", encoding="utf-8")
    except OSError:
        pass


def _clear_jev_pid(settings: Settings) -> None:
    try:
        _jev_pid_path(settings).unlink(missing_ok=True)
    except OSError:
        pass


def _pid_alive(pid: Optional[int]) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def probe_local_jev(url: str, timeout: float = 1.0) -> bool:
    """True when llama-server is healthy and answering on the Jev endpoint."""
    if not url:
        return False
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    try:
        response = httpx.get(f"{base}/v1/models", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def _find_jev_model_path() -> Optional[Path]:
    for candidate in (
        Path.home() / ".cache/llama.cpp/Jev-Q8_0.gguf",
        Path.home() / ".cache/llama.cpp/Jev-Q4_K_M.gguf",
    ):
        if candidate.is_file():
            return candidate
    return None


def _safe_stderr(msg: str) -> None:
    try:
        sys.stderr.write(msg)
        sys.stderr.flush()
    except Exception:
        pass


def ensure_local_jev_running(settings: Settings, wait_seconds: float = 8.0) -> bool:
    """Start the local Jev llama-server if it is not already running."""
    if settings.effective_classifier_mode != CLASSIFIER_MODE_LOCAL_JEV:
        return True

    url = settings.local_jev_url
    if probe_local_jev(url, timeout=0.8):
        return True

    _safe_stderr("[jev] starting local Jev decision server on Vulkan GPU...\n")

    # Strategy 1: systemd --user service if available
    has_systemd_service = False
    try:
        chk = subprocess.run(
            ["systemctl", "--user", "status", "agw-llama.service"],
            capture_output=True,
            timeout=2.0,
        )
        has_systemd_service = chk.returncode in (0, 3)  # 0=active, 3=inactive
    except Exception:
        has_systemd_service = False

    if has_systemd_service:
        try:
            subprocess.run(
                ["systemctl", "--user", "start", "agw-llama.service"],
                capture_output=True,
                timeout=5.0,
            )
        except Exception:
            pass

    # Strategy 2: fallback to direct subprocess spawn
    if not probe_local_jev(url, timeout=1.0) and shutil.which("llama-server"):
        model_path = _find_jev_model_path()
        if model_path:
            parsed = urlparse(url)
            port = parsed.port or 11435
            env = os.environ.copy()
            radv_icd = Path.home() / ".local/share/vulkan/icd.d/radeon_icd.json"
            if radv_icd.is_file():
                env["VK_ICD_FILENAMES"] = str(radv_icd)
            log_file = settings.log_dir / "jev.log"
            handle = open(log_file, "ab")
            proc = subprocess.Popen(
                [
                    "llama-server",
                    "-m",
                    str(model_path),
                    "--port",
                    str(port),
                    "-ngl",
                    "99",
                    "-c",
                    "2048",
                    "--keep",
                    "-1",
                    "--threads",
                    "4",
                    "--host",
                    "127.0.0.1",
                ],
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            _write_jev_pid(settings, proc.pid)

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if probe_local_jev(url, timeout=0.5):
            _safe_stderr("[jev] local Jev server ready (Vulkan GPU resident)\n")
            return True
        time.sleep(0.25)

    _safe_stderr("[jev] warning: local Jev server did not report ready in time\n")
    return False


def stop_local_jev(settings: Settings, wait_seconds: float = 6.0) -> bool:
    """Stop the local Jev llama-server and release VRAM."""
    if settings.effective_classifier_mode != CLASSIFIER_MODE_LOCAL_JEV:
        return True

    _safe_stderr("[jev] stopping local Jev decision server (releasing VRAM)...\n")

    # 1. Stop systemd service if available
    try:
        subprocess.run(
            ["systemctl", "--user", "stop", "agw-llama.service"],
            capture_output=True,
            timeout=5.0,
        )
    except Exception:
        pass

    # 2. Stop spawned PID if any
    pid = _read_jev_pid(settings)
    if _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
        deadline = time.time() + 2.0
        while _pid_alive(pid) and time.time() < deadline:
            time.sleep(0.1)
        if _pid_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    _clear_jev_pid(settings)

    # 3. If any process is still listening on the Jev port, terminate it
    url = settings.local_jev_url
    parsed = urlparse(url)
    port = parsed.port or 11435
    try:
        res = subprocess.run(
            ["fuser", f"{port}/tcp"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        pids = [int(p) for p in res.stdout.strip().split() if p.isdigit()]
        for p in pids:
            try:
                os.kill(p, signal.SIGTERM)
            except OSError:
                pass
        if pids:
            time.sleep(0.5)
            for p in pids:
                if _pid_alive(p):
                    try:
                        os.kill(p, signal.SIGKILL)
                    except OSError:
                        pass
    except Exception:
        pass

    # 4. Verify port is released
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if not probe_local_jev(url, timeout=0.3):
            _safe_stderr("[jev] local Jev server stopped (VRAM freed)\n")
            return True
        time.sleep(0.2)

    _safe_stderr("[jev] warning: local Jev port still responsive after stop\n")
    return False
