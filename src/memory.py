"""
Self-maintaining knowledge graph.

Durable facts learned from a conversation survive across sessions, decay when
they are not reinforced, graduate to permanent once corroborated, and are
injected back into later prompts as a small set of triples.

Three details are more important than they look:

* **WAL plus a busy timeout.** The gateway writes after every turn while the
  sentinel compacts on a schedule; without WAL these collide and one of them
  starts raising `database is locked`.
* **No network call inside a transaction.** Classifier calls are resolved before
  the write transaction opens. Holding a write lock across a 1.5s classifier
  round trip is a reliable way to block every other writer.
* **`VACUUM` runs outside any transaction.** SQLite refuses it otherwise.

Every public method is safe to call from a background task and will not raise
into a request path.
"""

from __future__ import annotations

import math
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# ------------------------------------------------------------------------------
# Tunables
# ------------------------------------------------------------------------------
GRADUATION_SESSIONS = 3
STABILITY_SECONDS = 86400.0  # reference window for the forgetting curve
EPHEMERAL_TTL_SECONDS = 86400.0
DECAY_FLOOR = 0.15
MAX_MATCHED_ENTITIES = 32
MAX_INGEST_TRIPLES = 24

# Sentiment and throwaway markers: worth noting, never worth trusting long.
EPHEMERAL_SENTIMENT_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"\b(hate|dislike|love|annoyed|slow|fast|bad|good|awesome|ugly|nice)\b", re.I),
    re.compile(r"\b(temp|temporary|just testing|ignore this|scratch)\b", re.I),
)

# Deterministic extraction. Single-token targets on purpose: allowing spaces
# would capture trailing conjunctions ("postgres and") and poison the graph.
TRIPLE_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(
        r"([A-Za-z0-9_][A-Za-z0-9_-]{1,40})\s+"
        r"(uses|is configured on|requires|runs on|is located at|depends on|connects to"
        r"|is deployed on|is stored in|is written in|is built with|targets)\s+"
        r"([A-Za-z0-9_.:/@-]{2,80})",
        re.I,
    ),
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE,
    type TEXT,
    updated_at REAL
);
CREATE TABLE IF NOT EXISTS relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT,
    predicate TEXT,
    target TEXT,
    context TEXT,
    status TEXT CHECK(status IN ('candidate', 'permanent', 'ephemeral')),
    confidence REAL DEFAULT 0.3,
    session_count INTEGER DEFAULT 1,
    last_session_id TEXT,
    first_seen REAL,
    last_observed REAL,
    UNIQUE(source, predicate, target)
);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_updated ON entities(updated_at);
CREATE INDEX IF NOT EXISTS idx_rel_lookup ON relations(source, target);
CREATE INDEX IF NOT EXISTS idx_rel_predicate ON relations(source, predicate);
CREATE INDEX IF NOT EXISTS idx_rel_status ON relations(status);
"""


# ------------------------------------------------------------------------------
# Pure helpers
# ------------------------------------------------------------------------------
def classify_epistemic_stance(statement: str) -> str:
    """`ephemeral` for fleeting opinion, else `candidate` for a durable fact."""
    for pattern in EPHEMERAL_SENTIMENT_PATTERNS:
        if pattern.search(statement or ""):
            return "ephemeral"
    return "candidate"


def ebbinghaus_decay(
    confidence: float, last_observed_ts: float, stability: float = STABILITY_SECONDS,
    now: Optional[float] = None,
) -> float:
    """R = e^(-delta_t / S) -- retention of an un-reinforced fact."""
    reference = time.time() if now is None else now
    delta_t = max(0.0, reference - float(last_observed_ts or reference))
    retention = math.exp(-delta_t / stability)
    return round(float(confidence) * retention, 4)


def extract_triples(text: str) -> List[Tuple[str, str, str]]:
    """Regex triple extraction: deterministic, offline, zero latency."""
    found: List[Tuple[str, str, str]] = []
    seen: Set[Tuple[str, str, str]] = set()

    for pattern in TRIPLE_PATTERNS:
        for source, predicate, target in pattern.findall(text or ""):
            triple = (
                source.strip().strip(".,;:()[]'\""),
                predicate.lower().strip(),
                target.strip().strip(".,;:()[]'\""),
            )
            if not all(triple):
                continue
            if triple[0].lower() == triple[2].lower():
                continue
            if triple in seen:
                continue
            seen.add(triple)
            found.append(triple)

    return found[:MAX_INGEST_TRIPLES]


# ------------------------------------------------------------------------------
# Store
# ------------------------------------------------------------------------------
class GraphMemory:
    """Thread-safe-by-construction graph store: one connection per operation."""

    def __init__(
        self,
        db_path: Path | str,
        graduation_sessions: int = GRADUATION_SESSIONS,
        ephemeral_ttl_seconds: float = EPHEMERAL_TTL_SECONDS,
        decay_floor: float = DECAY_FLOOR,
    ) -> None:
        self.db_path = Path(db_path)
        self.graduation_sessions = max(2, int(graduation_sessions))
        self.ephemeral_ttl_seconds = float(ephemeral_ttl_seconds)
        self.decay_floor = float(decay_floor)

    # --- connection --------------------------------------------------------
    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def init_schema(self) -> None:
        conn = self.connect()
        try:
            with conn:
                conn.executescript(SCHEMA)
        finally:
            conn.close()

    @staticmethod
    def _retry(fn, attempts: int = 5):
        """Retry briefly on writer contention rather than failing the turn."""
        last: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                return fn()
            except sqlite3.OperationalError as exc:
                last = exc
                if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                    raise
                time.sleep(0.05 * (2 ** attempt))
        raise last  # type: ignore[misc]

    # --- ingest ------------------------------------------------------------
    def ingest(
        self,
        triples: Sequence[Tuple[str, str, str]],
        session_id: str,
        prompt: str = "",
        supersedes: Optional[Dict[Tuple[str, str], bool]] = None,
    ) -> int:
        """Apply triples in a single transaction. Never raises.

        `supersedes` maps (source, predicate) -> "new value wins". It is computed
        by the caller *before* this call precisely so no network I/O happens
        while the write lock is held.
        """
        triples = [t for t in triples if all(t)][:MAX_INGEST_TRIPLES]
        if not triples:
            return 0

        supersedes = supersedes or {}
        now = time.time()

        def _apply() -> int:
            conn = self.connect()
            try:
                with conn:
                    for source, predicate, target in triples:
                        if supersedes.get((source, predicate)):
                            conn.execute(
                                "DELETE FROM relations WHERE source = ? AND predicate = ?",
                                (source, predicate),
                            )

                        status = classify_epistemic_stance(f"{source} {predicate} {target}")
                        row = conn.execute(
                            """
                            SELECT status, confidence, session_count, last_session_id
                            FROM relations
                            WHERE source = ? AND predicate = ? AND target = ?
                            """,
                            (source, predicate, target),
                        ).fetchone()

                        if row is None:
                            conn.execute(
                                """
                                INSERT OR IGNORE INTO relations
                                    (source, predicate, target, context, status, confidence,
                                     session_count, last_session_id, first_seen, last_observed)
                                VALUES (?, ?, ?, ?, ?, 0.3, 1, ?, ?, ?)
                                """,
                                (
                                    source, predicate, target, (prompt or "")[:200],
                                    status, session_id, now, now,
                                ),
                            )
                        else:
                            sessions = row["session_count"] + (
                                1 if session_id != row["last_session_id"] else 0
                            )
                            new_status = row["status"]
                            if sessions >= self.graduation_sessions:
                                new_status = "permanent"
                            elif new_status == "ephemeral" and status == "candidate":
                                new_status = status

                            conn.execute(
                                """
                                UPDATE relations
                                SET confidence = ?, session_count = ?, status = ?,
                                    last_session_id = ?, last_observed = ?
                                WHERE source = ? AND predicate = ? AND target = ?
                                """,
                                (
                                    min(1.0, round(row["confidence"] + 0.2, 4)),
                                    sessions,
                                    new_status,
                                    session_id,
                                    now,
                                    source,
                                    predicate,
                                    target,
                                ),
                            )

                        for name in (source, target):
                            conn.execute(
                                """
                                INSERT INTO entities (name, type, updated_at)
                                VALUES (?, 'concept', ?)
                                ON CONFLICT(name) DO UPDATE SET updated_at = excluded.updated_at
                                """,
                                (name, now),
                            )
                return len(triples)
            finally:
                conn.close()

        try:
            return self._retry(_apply)
        except Exception:
            return 0

    def ingest_sync(self, session_id: str, prompt: str, response: str) -> int:
        """Ingest without consulting any classifier (used by tests and CLIs)."""
        triples = extract_triples(f"{prompt}\n{response}")
        if not triples:
            return 0
        return self.ingest(triples, session_id, prompt)

    async def ingest_async(self, session_id: str, prompt: str, response: str, classifier) -> int:
        """Ingest a completed turn, resolving conflicts before writing.

        Called from a background task, so it must never raise.
        """
        try:
            triples = extract_triples(f"{prompt}\n{response}")
            if not triples:
                return 0

            # Phase 1 -- decide, with no database lock held.
            supersedes: Dict[Tuple[str, str], bool] = {}
            if classifier is not None:
                conn = self.connect()
                try:
                    for source, predicate, target in triples:
                        row = conn.execute(
                            """
                            SELECT target FROM relations
                            WHERE source = ? AND predicate = ? LIMIT 1
                            """,
                            (source, predicate),
                        ).fetchone()
                        if row is None or row["target"] == target:
                            continue
                        try:
                            verdict = await classifier.value_supersedes(
                                source, predicate, row["target"], target
                            )
                        except Exception:
                            # A classifier that raises must cost us the decision,
                            # never the whole write.
                            verdict = None
                        # Fail-open: if it could not be asked, newest wins.
                        supersedes[(source, predicate)] = (
                            True if verdict is None else bool(verdict)
                        )
                finally:
                    conn.close()

            # Phase 2 -- write atomically.
            return self.ingest(triples, session_id, prompt, supersedes)
        except Exception:
            return 0

    # --- retrieval ---------------------------------------------------------
    def match_entities(self, query: str, limit: int = MAX_MATCHED_ENTITIES) -> List[str]:
        """Entity names appearing in `query`, matched on token boundaries.

        Boundary matching matters: the entity `post` must not fire for a query
        about `postgres`.
        """
        try:
            conn = self.connect()
            try:
                rows = conn.execute(
                    """
                    SELECT name FROM entities
                    WHERE name IS NOT NULL
                    ORDER BY updated_at DESC LIMIT 5000
                    """
                ).fetchall()
            finally:
                conn.close()
        except Exception:
            return []

        haystack = (query or "").lower()
        matched: List[str] = []
        for row in rows:
            name = row["name"]
            if not name or len(name) < 3:
                continue
            try:
                if re.search(rf"(?<![a-z0-9_]){re.escape(name.lower())}(?![a-z0-9_])", haystack):
                    matched.append(name)
            except re.error:
                continue
            if len(matched) >= limit:
                break
        return matched

    def retrieve(self, query: str, max_tokens: int = 200, limit: int = 8) -> str:
        """Delimited block of winning triples, or "" when nothing matches."""
        try:
            entities = self.match_entities(query)
            if not entities:
                return ""

            placeholders = ",".join("?" for _ in entities)
            conn = self.connect()
            try:
                rows = conn.execute(
                    f"""
                    SELECT source, predicate, target, confidence, status
                    FROM relations
                    WHERE (source IN ({placeholders}) OR target IN ({placeholders}))
                      AND confidence > 0.2
                      AND status != 'ephemeral'
                    ORDER BY confidence DESC, last_observed DESC
                    LIMIT ?
                    """,
                    (*entities, *entities, int(limit)),
                ).fetchall()
            finally:
                conn.close()
        except Exception:
            return ""

        if not rows:
            return ""

        triples = [
            f"- ({row['source']}) --[{row['predicate']}]--> ({row['target']})"
            for row in rows
        ]
        block = (
            "<relevant_memory>\n"
            "Recalled facts from earlier sessions (may be stale, verify before relying):\n"
            + "\n".join(triples)
            + "\n</relevant_memory>"
        )
        return block[: max(200, int(max_tokens) * 4)]

    # --- maintenance -------------------------------------------------------
    def compact(self, now: Optional[float] = None) -> Dict[str, Any]:
        """Prune ephemeral noise, decay stale candidates, then reclaim space."""
        stats: Dict[str, Any] = {
            "ephemeral_pruned": 0,
            "candidates_dropped": 0,
            "candidates_decayed": 0,
            "orphan_entities_pruned": 0,
            "vacuumed": False,
        }
        if not self.db_path.exists():
            stats["skipped"] = "no database"
            return stats

        reference = time.time() if now is None else now

        def _compact() -> None:
            conn = self.connect()
            try:
                with conn:
                    cursor = conn.execute(
                        "DELETE FROM relations WHERE status = 'ephemeral' AND (? - last_observed) > ?",
                        (reference, self.ephemeral_ttl_seconds),
                    )
                    stats["ephemeral_pruned"] = max(0, cursor.rowcount or 0)

                    for row in conn.execute(
                        "SELECT id, confidence, last_observed FROM relations WHERE status = 'candidate'"
                    ).fetchall():
                        decayed = ebbinghaus_decay(
                            row["confidence"], row["last_observed"], now=reference
                        )
                        if decayed < self.decay_floor:
                            conn.execute("DELETE FROM relations WHERE id = ?", (row["id"],))
                            stats["candidates_dropped"] += 1
                        else:
                            conn.execute(
                                "UPDATE relations SET confidence = ? WHERE id = ?",
                                (decayed, row["id"]),
                            )
                            stats["candidates_decayed"] += 1

                    cursor = conn.execute(
                        "DELETE FROM entities WHERE name NOT IN (SELECT source FROM relations) "
                        "AND name NOT IN (SELECT target FROM relations)"
                    )
                    stats["orphan_entities_pruned"] = max(0, cursor.rowcount or 0)
            finally:
                conn.close()

        self._retry(_compact)

        # VACUUM cannot run inside a transaction: use a clean autocommit connection.
        try:
            conn = self.connect()
            try:
                conn.isolation_level = None
                conn.execute("VACUUM;")
                stats["vacuumed"] = True
            finally:
                conn.close()
        except sqlite3.Error:
            pass

        return stats

    # --- introspection -----------------------------------------------------
    def journal_mode(self) -> str:
        conn = self.connect()
        try:
            return str(conn.execute("PRAGMA journal_mode;").fetchone()[0])
        finally:
            conn.close()

    def relation(
        self, source: str, predicate: str, target: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Fetch a stored relation (tests and debugging)."""
        conn = self.connect()
        try:
            if target is None:
                row = conn.execute(
                    "SELECT * FROM relations WHERE source = ? AND predicate = ? LIMIT 1",
                    (source, predicate),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT * FROM relations
                    WHERE source = ? AND predicate = ? AND target = ?
                    """,
                    (source, predicate, target),
                ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def stats(self) -> Dict[str, Any]:
        conn = self.connect()
        try:
            by_status = {
                row["status"]: row["n"]
                for row in conn.execute(
                    "SELECT status, COUNT(*) AS n FROM relations GROUP BY status"
                ).fetchall()
            }
            return {
                "db_path": str(self.db_path),
                "exists": self.db_path.exists(),
                "relations": conn.execute("SELECT COUNT(*) AS n FROM relations").fetchone()["n"],
                "entities": conn.execute("SELECT COUNT(*) AS n FROM entities").fetchone()["n"],
                "by_status": by_status,
                "journal_mode": str(conn.execute("PRAGMA journal_mode;").fetchone()[0]),
            }
        finally:
            conn.close()


# ------------------------------------------------------------------------------
# Process-wide memory
# ------------------------------------------------------------------------------
_MEMORY: Optional[GraphMemory] = None


def configure(memory: GraphMemory) -> GraphMemory:
    global _MEMORY
    _MEMORY = memory
    return memory


def get_memory(settings: Optional[Any] = None) -> GraphMemory:
    global _MEMORY
    if _MEMORY is None:
        if settings is None:
            from .config import load_settings

            settings = load_settings()
        _MEMORY = GraphMemory(settings.db_path)
        try:
            _MEMORY.init_schema()
        except Exception:
            pass
    return _MEMORY
