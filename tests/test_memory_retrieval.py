"""
Memory retrieval enhancements: fuzzy entity matching, budget-packing, auditing.

The matcher must stay precise -- a memory system that injects the wrong facts is
worse than one that injects none -- so these tests pin both sides: typos and
prefixes are forgiven, but unrelated short names and lookalike words are not.
"""

from __future__ import annotations

import json
import os

import pytest

from src.memory import _levenshtein, GraphMemory


@pytest.fixture
def graph(tmp_path):
    memory = GraphMemory(tmp_path / "graph.db")
    memory.init_schema()
    memory.audit_hook = lambda record: records.append(record)
    records = []
    memory.ingest_sync(
        "s1",
        "The billing service uses PostgreSQL 16. The api depends on Redis 7.",
        "",
    )
    return memory, records


# ------------------------------------------------------------------------------
# Fuzzy matching
# ------------------------------------------------------------------------------
def test_exact_boundary_still_matches(graph):
    memory, _ = graph
    assert "PostgreSQL" in memory.match_entities("what does postgres use?")


def test_single_character_typo_is_forgiven(graph):
    memory, _ = graph
    # 'postgresl' is one insertion from 'postgres' in the stored entity name.
    matched = memory.match_entities("how is the postgresl migration?")
    assert "PostgreSQL" in matched


def test_two_edits_are_not_forgiven(graph):
    """Beyond the edit budget, only exact boundaries may fire."""
    memory, _ = graph
    # 'postgress' is two edits from 'postgres' (s->l, +s): too far.
    assert "PostgreSQL" not in memory.match_entities("how is postgress configured?")


def test_token_prefix_is_forgiven(graph):
    memory, _ = graph
    matched = memory.match_entities("check the postgresl migration")
    assert "PostgreSQL" in matched


def test_prefix_matching_fires_for_redis(graph):
    memory, _ = graph
    # 'redis' is stored; a query token one character longer is a prefix match.
    assert "Redis" in memory.match_entities("rediss latency is high")


def test_short_entities_stay_exact_only(graph):
    """'api' is 3 letters; one edit would make 'api' match almost anything."""
    memory, _ = graph
    # Exact match still works.
    assert "api" in memory.match_entities("is the api up?")
    # One edit away must NOT match.
    assert "api" not in memory.match_entities("is the apt up?")
    assert "api" not in memory.match_entities("the ape service")


def test_unrelated_words_do_not_match(graph):
    memory, _ = graph
    assert memory.match_entities("tell me about ancient rome") == []


def test_match_limit_is_respected(graph):
    memory, _ = graph
    assert len(memory.match_entities("postgres redis api billing service", limit=2)) <= 2


def test_levenshtein_budget_is_honoured():
    assert _levenshtein("postgres", "postgres") == 0
    assert _levenshtein("postgres", "postgress") == 1
    assert _levenshtein("postgres", "postgresql") > 1  # beyond the cap
    assert _levenshtein("redis", "redis") == 0
    # Different lengths beyond the cap short-circuit.
    assert _levenshtein("postgres", "pg") > 1


# ------------------------------------------------------------------------------
# Token-budget packing
# ------------------------------------------------------------------------------
def test_block_respects_the_token_budget(graph):
    memory, _ = graph
    memory.ingest_sync(
        "s2",
        "The search service uses Elasticsearch 8. The cache requires Memcached. "
        "The web tier runs on Nginx 1.24. The queue is deployed on RabbitMQ.",
        "",
    )

    block = memory.retrieve(
        "postgres redis elasticsearch cache nginx queue", max_tokens=60
    )

    assert block.startswith("<relevant_memory>")
    assert block.endswith("</relevant_memory>")
    assert len(block) <= 60 * 4


def test_budget_never_splits_a_triple(graph):
    """A partial line would inject a mangled fact."""
    memory, _ = graph
    memory.ingest_sync(
        "s2",
        "The search service uses Elasticsearch 8. The cache requires Memcached. "
        "The web tier runs on Nginx 1.24. The queue is deployed on RabbitMQ.",
        "",
    )
    block = memory.retrieve(
        "postgres redis elasticsearch cache nginx queue", max_tokens=60
    )

    for line in block.splitlines():
        if line.startswith("- ("):
            assert line.rstrip().endswith(")"), f"truncated fact: {line!r}"


def test_budget_too_small_for_a_single_fact_returns_empty(graph):
    memory, _ = graph
    assert memory.retrieve("postgres", max_tokens=1) == ""


def test_full_block_at_a_generous_budget(graph):
    memory, _ = graph
    block = memory.retrieve("postgres redis", max_tokens=200)

    assert "(PostgreSQL)" in block
    assert "(Redis)" in block


# ------------------------------------------------------------------------------
# Audit hook
# ------------------------------------------------------------------------------
def test_retrieval_writes_an_audit_record(graph):
    memory, records = graph
    block = memory.retrieve("postgres redis", max_tokens=200)

    assert block, "expected facts to be recalled"
    assert len(records) == 1
    record = records[0]
    assert record["surface"] == "memory"
    assert record["triples_injected"] == 2
    assert record["entities_matched"] == 2
    assert record["block_chars"] == len(block)


def test_no_match_writes_no_audit_record(graph):
    memory, records = graph
    assert memory.retrieve("ancient rome") == ""
    assert records == []


def test_default_hook_appends_to_the_audit_log(tmp_path, monkeypatch):
    saved = dict(os.environ)
    try:
        monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
        monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
        from src.config import load_settings

        memory = GraphMemory(load_settings(env=dict(os.environ)).db_path)
        memory.init_schema()
        memory.ingest_sync("s1", "The billing service uses PostgreSQL 16.", "")
        memory.retrieve("postgres")

        log = tmp_path / "logs" / "audit.log"
        assert log.exists()
        record = json.loads(log.read_text().splitlines()[-1])
        assert record["surface"] == "memory"
    finally:
        os.environ.clear()
        os.environ.update(saved)
