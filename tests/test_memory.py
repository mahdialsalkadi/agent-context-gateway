"""
Graph memory lifecycle.

The concurrency test is the important one. The gateway writes after every turn
while the sentinel compacts on a schedule, so the store must survive 50
simultaneous writers without a single `database is locked` -- which is exactly
what WAL plus a busy timeout (and a bounded retry) is there to guarantee.
"""

from __future__ import annotations

import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.memory import (
    GRADUATION_SESSIONS,
    STABILITY_SECONDS,
    GraphMemory,
    classify_epistemic_stance,
    ebbinghaus_decay,
    extract_triples,
)


def insert_raw(memory: GraphMemory, **values) -> None:
    """Write a row with exact control over timestamps and confidence."""
    defaults = {
        "source": "a",
        "predicate": "uses",
        "target": "b",
        "context": "test",
        "status": "candidate",
        "confidence": 0.5,
        "session_count": 1,
        "last_session_id": "s1",
        "first_seen": time.time(),
        "last_observed": time.time(),
    }
    defaults.update(values)
    conn = memory.connect()
    try:
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO relations
                    (source, predicate, target, context, status, confidence,
                     session_count, last_session_id, first_seen, last_observed)
                VALUES (:source, :predicate, :target, :context, :status, :confidence,
                        :session_count, :last_session_id, :first_seen, :last_observed)
                """,
                defaults,
            )
            for name in (defaults["source"], defaults["target"]):
                conn.execute(
                    "INSERT OR REPLACE INTO entities (name, type, updated_at) VALUES (?, 'concept', ?)",
                    (name, time.time()),
                )
    finally:
        conn.close()


# ------------------------------------------------------------------------------
# Storage
# ------------------------------------------------------------------------------
def test_database_is_wal_with_a_busy_timeout(memory):
    conn = memory.connect()
    try:
        assert conn.execute("PRAGMA journal_mode;").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA busy_timeout;").fetchone()[0] == 5000
    finally:
        conn.close()


def test_schema_initialisation_is_idempotent(memory):
    memory.init_schema()
    memory.init_schema()
    assert "relations" in memory.stats()


def test_fifty_concurrent_writes_never_hit_database_is_locked(memory):
    def worker(index: int) -> int:
        return memory.ingest_sync(
            f"session-{index}", f"service{index} uses store{index}", "noted"
        )

    with ThreadPoolExecutor(max_workers=50) as pool:
        results = list(pool.map(worker, range(50)))

    # ingest_sync reports 0 if it swallowed a lock error, so this catches any.
    assert all(result == 1 for result in results), f"some writes failed: {results}"
    assert memory.stats()["relations"] == 50


def test_concurrent_writers_survive_concurrent_compaction(memory):
    """The gateway writes while the sentinel compacts: neither may fail."""

    def writer(index: int) -> int:
        return memory.ingest_sync(f"s{index}", f"item{index} uses bucket{index}", "n")

    def compactor() -> bool:
        try:
            memory.compact()
            return True
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=24) as pool:
        futures = [pool.submit(writer, i) for i in range(20)]
        futures += [pool.submit(compactor) for _ in range(4)]
        results = [future.result() for future in futures]

    assert all(result == 1 for result in results[:20]), "a writer failed under compaction"
    assert all(result is True for result in results[20:]), "compaction raised"


# ------------------------------------------------------------------------------
# Decay maths
# ------------------------------------------------------------------------------
def test_ebbinghaus_decay_follows_the_curve():
    now = 1_000_000.0
    assert ebbinghaus_decay(1.0, now, now=now) == 1.0
    one_window = ebbinghaus_decay(1.0, now - STABILITY_SECONDS, now=now)
    assert abs(one_window - 0.3679) < 0.001
    assert ebbinghaus_decay(0.8, now - 5 * STABILITY_SECONDS, now=now) < 0.01


def test_ebbinghaus_decay_is_monotonic_and_bounded():
    now = 1_000_000.0
    previous = 1.1
    for age in range(0, 200_000, 10_000):
        value = ebbinghaus_decay(1.0, now - age, now=now)
        assert 0.0 <= value <= 1.0
        assert value <= previous
        previous = value


# ------------------------------------------------------------------------------
# Epistemic filter and extraction
# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "statement",
    [
        "the build is slow",
        "I hate this library",
        "just testing, ignore this",
        "temporary workaround",
    ],
)
def test_transient_statements_are_ephemeral(statement):
    assert classify_epistemic_stance(statement) == "ephemeral"


@pytest.mark.parametrize(
    "statement",
    ["Hermes uses sqlite", "the service runs on port 8090", "backend depends on redis"],
)
def test_technical_statements_are_candidates(statement):
    assert classify_epistemic_stance(statement) == "candidate"


def test_triple_extraction():
    triples = extract_triples(
        "The gateway is built with FastAPI and the store runs on postgres16"
    )
    sources = {triple[0] for triple in triples}
    assert "store" in sources
    assert ("store", "runs on", "postgres16") in triples


def test_extraction_ignores_self_reference_and_noise():
    assert extract_triples("redis uses redis") == []
    assert extract_triples("") == []
    assert extract_triples("no relation in this sentence at all") == []


# ------------------------------------------------------------------------------
# Ingestion lifecycle
# ------------------------------------------------------------------------------
def test_triple_is_stored_with_entities(memory):
    written = memory.ingest([("api", "uses", "postgres")], "s1", "api uses postgres")
    assert written == 1

    row = memory.relation("api", "uses", "postgres")
    assert row is not None
    assert row["status"] == "candidate"
    assert row["session_count"] == 1
    assert set(memory.match_entities("tell me about api and postgres")) == {"api", "postgres"}


def test_ephemeral_statement_is_marked_ephemeral(memory):
    memory.ingest([("build", "is", "slow")], "s1")
    assert memory.relation("build", "is", "slow")["status"] == "ephemeral"


def test_graduation_after_three_distinct_sessions(memory):
    for session in ("s1", "s2", "s3"):
        memory.ingest([("api", "uses", "postgres")], session)

    row = memory.relation("api", "uses", "postgres")
    assert row["session_count"] == GRADUATION_SESSIONS
    assert row["status"] == "permanent"
    assert row["confidence"] > 0.3


def test_repeated_observation_in_one_session_does_not_graduate(memory):
    for _ in range(5):
        memory.ingest([("api", "uses", "postgres")], "same-session")

    row = memory.relation("api", "uses", "postgres")
    assert row["session_count"] == 1
    assert row["status"] == "candidate"


def test_confidence_saturates_at_one(memory):
    for index in range(10):
        memory.ingest([("api", "uses", "postgres")], f"s{index}")
    assert memory.relation("api", "uses", "postgres")["confidence"] == 1.0


def test_conflict_overwrites_when_superseded(memory):
    memory.ingest([("svc", "uses", "pg8")], "s1")
    memory.ingest([("svc", "uses", "pg16")], "s2", supersedes={("svc", "uses"): True})

    conn = memory.connect()
    try:
        targets = {
            row["target"]
            for row in conn.execute(
                "SELECT target FROM relations WHERE source='svc' AND predicate='uses'"
            ).fetchall()
        }
    finally:
        conn.close()
    assert targets == {"pg16"}


def test_conflict_keeps_both_when_not_superseded(memory):
    """A classifier saying "these coexist" must not delete the old value."""
    memory.ingest([("svc", "uses", "pg8")], "s1")
    memory.ingest([("svc", "uses", "pg16")], "s2", supersedes={("svc", "uses"): False})

    conn = memory.connect()
    try:
        count = conn.execute(
            "SELECT COUNT(*) AS n FROM relations WHERE source='svc' AND predicate='uses'"
        ).fetchone()["n"]
    finally:
        conn.close()
    assert count == 2


# ------------------------------------------------------------------------------
# Async ingestion (conflict resolution before the write lock)
# ------------------------------------------------------------------------------
async def test_async_ingest_uses_the_classifier_and_fails_open(memory):
    class StubClassifier:
        def __init__(self, verdict):
            self.verdict = verdict
            self.calls = 0

        async def value_supersedes(self, source, predicate, old, new):
            self.calls += 1
            return self.verdict

    memory.ingest([("svc", "uses", "pg8")], "s1")

    superseding = StubClassifier(True)
    await memory.ingest_async("s2", "svc uses pg16", "noted", superseding)
    assert superseding.calls == 1
    assert memory.relation("svc", "uses", "pg16") is not None
    assert memory.relation("svc", "uses", "pg8") is None

    # Fail-open: a classifier that raises must cost us the decision, never the
    # write. Losing the fact would be a silent memory hole.
    class Broken:
        async def value_supersedes(self, *args):
            raise RuntimeError("classifier down")

    written = await memory.ingest_async("s3", "svc uses pg17", "noted", Broken())
    assert written == 1, "a raising classifier must not lose the write"
    assert memory.relation("svc", "uses", "pg17") is not None


async def test_async_ingest_never_raises_on_bad_input(memory):
    assert await memory.ingest_async("s1", "", "", None) == 0
    assert await memory.ingest_async("s1", None, None, None) == 0


# ------------------------------------------------------------------------------
# Retrieval
# ------------------------------------------------------------------------------
def test_retrieval_matches_on_token_boundaries(memory):
    memory.ingest([("postgres", "runs on", "host1")], "s1")
    memory.ingest([("post", "runs on", "host2")], "s1")

    block = memory.retrieve("does postgres need tuning?")
    assert "postgres" in block
    assert "host2" not in block


def test_retrieval_excludes_ephemeral_and_low_confidence(memory):
    insert_raw(memory, source="redis", predicate="uses", target="tls", status="ephemeral", confidence=0.99)
    insert_raw(memory, source="redis", predicate="uses", target="acls", status="candidate", confidence=0.05)
    assert memory.retrieve("tell me about redis") == ""


def test_retrieval_orders_by_confidence(memory):
    # Six rows, so all fit inside the default LIMIT and ordering is observable.
    for index in range(6):
        insert_raw(
            memory,
            source="cache",
            predicate=f"p{index}",
            target=f"t{index}",
            confidence=0.4 + index * 0.05,
        )
    block = memory.retrieve("what about cache", max_tokens=500)
    assert block.startswith("<relevant_memory>")
    assert block.endswith("</relevant_memory>")
    # Highest confidence is listed first.
    assert block.index("(t5)") < block.index("(t0)")


def test_retrieval_respects_the_token_budget(memory):
    for index in range(12):
        insert_raw(
            memory,
            source="cache",
            predicate=f"p{index}",
            target=f"t{index}",
            confidence=0.9,
        )
    block = memory.retrieve("what about cache", max_tokens=50)
    assert len(block) <= 50 * 4
    assert block.startswith("<relevant_memory>")


def test_retrieval_with_no_match_is_empty(memory):
    assert memory.retrieve("nothing matches this") == ""
    assert memory.retrieve("") == ""


# ------------------------------------------------------------------------------
# Compaction
# ------------------------------------------------------------------------------
def test_compaction_prunes_decays_and_vacuums(memory):
    """Regression: VACUUM inside a transaction raises in SQLite."""
    now = time.time()
    insert_raw(memory, source="old", predicate="uses", target="a", status="ephemeral", last_observed=now - 200_000)
    insert_raw(memory, source="stale", predicate="uses", target="b", status="candidate", confidence=0.01, last_observed=now)
    insert_raw(memory, source="solid", predicate="uses", target="c", status="candidate", confidence=0.9, last_observed=now)

    conn = memory.connect()
    try:
        with conn:
            conn.execute(
                "INSERT OR IGNORE INTO entities (name, type, updated_at) VALUES ('orphan_entity', 'concept', ?)",
                (now,),
            )
    finally:
        conn.close()

    stats = memory.compact(now=now)

    assert stats["ephemeral_pruned"] == 1
    assert stats["candidates_dropped"] >= 1
    assert stats["vacuumed"] is True
    assert memory.relation("solid", "uses", "c") is not None
    assert memory.relation("stale", "uses", "b") is None
    assert memory.relation("old", "uses", "a") is None
    # Still queryable after VACUUM, and the orphan is gone.
    assert memory.stats()["relations"] == 1
    assert "orphan_entity" not in memory.match_entities("orphan_entity")


def test_compaction_on_a_missing_database_is_safe(tmp_path):
    graph = GraphMemory(tmp_path / "does-not-exist" / "graph.db")
    stats = graph.compact()
    assert stats["skipped"] == "no database"


def test_compaction_preserves_fresh_high_confidence_rows(memory):
    memory.ingest([("api", "uses", "postgres")], "s1")
    memory.compact()
    row = memory.relation("api", "uses", "postgres")
    assert row is not None
    assert row["confidence"] > 0.2


# ------------------------------------------------------------------------------
# Graceful degradation
# ------------------------------------------------------------------------------
def test_ingest_with_no_triples_is_a_no_op(memory):
    assert memory.ingest([], "s1") == 0
    assert memory.ingest([("", "", "")], "s1") == 0


def test_unwritable_database_degrades_instead_of_raising(memory, tmp_path, monkeypatch):
    """A broken store must not take the request path down with it."""
    broken = GraphMemory(tmp_path / "readonly" / "graph.db")

    def explode(*args, **kwargs):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(broken, "connect", explode)
    assert broken.ingest([("a", "uses", "b")], "s1") == 0
    assert broken.retrieve("a uses b") == ""
