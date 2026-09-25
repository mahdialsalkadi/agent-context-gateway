"""
Autonomous maintenance daemon.

Runs one cycle and exits (ideal for a systemd timer or cron), or self-schedules
with `--loop` when the host has no scheduler at all.

A cycle does four things:

1. **Watchdog** -- ping `/health` and respawn the gateway if it has died.
2. **Artifact pruning** -- FIFO-evict spilled context at a size ceiling, plus age
   expiry, so a long session cannot fill the RAM disk.
3. **Graph compaction** -- drop expired ephemeral rows, apply the Ebbinghaus
   decay to stale candidates, and `VACUUM`.
4. **Log rotation** -- size-based rollover for the gateway and audit logs.

Usage::

    agent-sentinel                 # one cycle
    agent-sentinel --status        # report only, mutate nothing
    agent-sentinel --loop --interval 3600
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, Optional

import httpx

from .artifacts import ArtifactStore
from .config import Settings, load_settings
from .memory import GraphMemory

DEFAULT_SHM_MAX_BYTES = 180 * 1024 * 1024
DEFAULT_SHM_MAX_AGE = 7200.0
DEFAULT_LOG_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_LOCK_STALE_SECONDS = 3600.0


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, "") or default))
    except (TypeError, ValueError):
        return default


class Sentinel:
    """One maintenance cycle, parameterised for testability."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        memory: Optional[GraphMemory] = None,
        store: Optional[ArtifactStore] = None,
        shm_max_bytes: Optional[int] = None,
        shm_max_age_seconds: Optional[float] = None,
        log_max_bytes: Optional[int] = None,
        health_timeout: float = 3.0,
    ) -> None:
        self.settings = settings or load_settings()
        self._memory = memory
        self._store = store
        self.shm_max_bytes = (
            shm_max_bytes
            if shm_max_bytes is not None
            else _env_int("SHM_MAX_BYTES", DEFAULT_SHM_MAX_BYTES)
        )
        self.shm_max_age_seconds = (
            shm_max_age_seconds
            if shm_max_age_seconds is not None
            else _env_float("SHM_MAX_AGE_SECONDS", DEFAULT_SHM_MAX_AGE)
        )
        self.log_max_bytes = (
            log_max_bytes
            if log_max_bytes is not None
            else _env_int("LOG_MAX_BYTES", DEFAULT_LOG_MAX_BYTES)
        )
        self.health_timeout = health_timeout
        self.lock_path = self.settings.log_dir / ".sentinel.lock"

    # --- collaborators -----------------------------------------------------
    @property
    def memory(self) -> GraphMemory:
        if self._memory is None:
            self._memory = GraphMemory(self.settings.db_path)
        return self._memory

    @property
    def store(self) -> ArtifactStore:
        if self._store is None:
            self._store = ArtifactStore(
                self.settings.shm_cache_dir, self.settings.truncate_threshold_chars
            )
        return self._store

    @property
    def health_url(self) -> str:
        return f"http://{self.settings.host}:{self.settings.port}/health"

    # --- 1. watchdog -------------------------------------------------------
    def check_health(self) -> bool:
        try:
            return httpx.get(self.health_url, timeout=self.health_timeout).status_code == 200
        except Exception:
            return False

    def spawn_gateway(self) -> bool:
        """Respawn the gateway detached, inheriting this process's environment."""
        module = "src.gateway:app"
        try:
            self.settings.log_dir.mkdir(parents=True, exist_ok=True)
            handle = open(self.settings.gateway_log_path, "ab")
            subprocess.Popen(
                [sys.executable, "-m", "uvicorn", module, "--host", self.settings.host,
                 "--port", str(self.settings.port)],
                stdout=handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True
        except Exception:
            return False

    # --- 4. log rotation (defined before use for readability) --------------
    def rotate_logs(self) -> Dict[str, Any]:
        rotated = []
        for path in (
            self.settings.gateway_log_path,
            self.settings.audit_log_path,
            self.settings.log_dir / "sentinel.log",
        ):
            try:
                if path.exists() and path.stat().st_size > self.log_max_bytes:
                    os.replace(path, path.with_suffix(path.suffix + ".old"))
                    rotated.append(path.name)
            except OSError:
                continue
        return {"rotated": rotated}

    # --- locking -----------------------------------------------------------
    def acquire_lock(self) -> bool:
        try:
            if self.lock_path.exists():
                age = time.time() - self.lock_path.stat().st_mtime
                if age > DEFAULT_LOCK_STALE_SECONDS:
                    self.lock_path.unlink()
                else:
                    return False
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            self.lock_path.write_text(str(os.getpid()), encoding="utf-8")
            return True
        except OSError:
            return True  # fail open: a lock problem must not stop maintenance

    def release_lock(self) -> None:
        try:
            self.lock_path.unlink()
        except OSError:
            pass

    # --- cycle -------------------------------------------------------------
    def run_cycle(self, status_only: bool = False) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "ts": int(time.time()),
            "health_url": self.health_url,
            "gateway_healthy": self.check_health(),
        }

        if status_only:
            report["shm_dir"] = str(self.settings.shm_cache_dir)
            report["db_path"] = str(self.settings.db_path)
            return report

        if not report["gateway_healthy"]:
            report["respawned"] = self.spawn_gateway()

        try:
            report["artifacts"] = self.store.prune(
                self.shm_max_bytes, self.shm_max_age_seconds
            )
        except Exception as exc:
            report["artifacts"] = {"error": str(exc)}

        try:
            report["memory"] = self.memory.compact()
        except Exception as exc:
            report["memory"] = {"error": str(exc)}

        try:
            report["logs"] = self.rotate_logs()
        except Exception as exc:
            report["logs"] = {"error": str(exc)}

        return report


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent-sentinel",
        description="Maintenance daemon for agent-context-gateway.",
    )
    parser.add_argument("--loop", action="store_true", help="run forever at --interval")
    parser.add_argument(
        "--interval", type=int, default=86400, help="loop interval in seconds (default: daily)"
    )
    parser.add_argument("--status", action="store_true", help="report only; change nothing")
    parser.add_argument("--json", action="store_true", help="emit the cycle report as JSON")
    args = parser.parse_args(argv)

    settings = load_settings()
    settings.ensure_dirs()
    sentinel = Sentinel(settings)

    if args.status:
        print(json.dumps(sentinel.run_cycle(status_only=True), indent=2, default=str))
        return 0

    if not args.loop:
        if not sentinel.acquire_lock():
            print("[sentinel] another cycle is already running; skipping.", file=sys.stderr)
            return 0
        try:
            report = sentinel.run_cycle()
        finally:
            sentinel.release_lock()
        if args.json:
            print(json.dumps(report, indent=2, default=str))
        else:
            print("[sentinel] maintenance cycle complete.")
            print(json.dumps(report, indent=2, default=str))
        return 0

    interval = max(60, args.interval)
    print(f"[sentinel] loop mode active, interval {interval}s. Ctrl-C to stop.")
    try:
        while True:
            if sentinel.acquire_lock():
                try:
                    report = sentinel.run_cycle()
                    print(f"[sentinel] {json.dumps(report, default=str)}")
                except Exception as exc:
                    print(f"[sentinel] cycle error: {exc}", file=sys.stderr)
                finally:
                    sentinel.release_lock()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("[sentinel] stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
