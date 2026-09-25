# agent-context-gateway

An **agent-agnostic** gateway that sits between your coding agent and any
OpenAI-compatible LLM provider, and quietly fixes the three things that make
long agent sessions expensive and flaky:

| Problem | What the gateway does |
| --- | --- |
| Every turn carries every tool schema, even `hi` | **Prunes tool schemas** on turns that provably don't need them — sub-millisecond heuristics first, an optional LLM classifier only for genuinely ambiguous prompts |
| Old tool output bloats the context until the model loses the thread | **Spills oversized output** to a RAM disk behind a short `fetch_log` handle the model can read on demand |
| Naive proxies buffer the whole stream to inspect it | **Streams chunk-in/chunk-out** with a bounded look-ahead, so pruning costs you no latency |
| Facts learned in one session are gone in the next | **Builds a self-pruning knowledge graph** that decays, graduates and gets injected into later prompts |

Works with **Claude Code, Aider, Cursor, Hermes, or any OpenAI SDK client** — no
vendor lock-in, no agent-specific code, no hardcoded paths.

```bash
cp .env.example .env      # set UPSTREAM_BASE_URL + UPSTREAM_API_KEY
pip install -r requirements.txt
python -m src.gateway
# -> listening on http://127.0.0.1:8090/v1
```

> **Compatibility note.** The OpenAI surface (`/v1/chat/completions`) works with
> every OpenAI-compatible client. The Anthropic surface (`/v1/messages`) exists
> so **Claude Code** can point `ANTHROPIC_BASE_URL` at the gateway; see
> [`configs/claude_code.md`](configs/claude_code.md).

---

## Contents

- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Connecting your agent](#connecting-your-agent)
- [How it works](#how-it-works)
- [Testing](#testing)
- [Security](#security)
- [Limitations](#limitations)

---

## Architecture

```text
   Claude Code ──┐
   Aider ────────┤
   Cursor ───────┤   /v1/chat/completions  (OpenAI)
   LangChain ────┼──▶  /v1/messages        (Anthropic, translated)
   Hermes ───────┘     /v1/models | /health
                              │
                    ┌─────────▼──────────┐
                    │   src/gateway.py   │  routing · streaming · telemetry
                    └─────────┬──────────┘
              ┌───────────────┼───────────────┬────────────────┐
              ▼               ▼               ▼                ▼
      classifier.py     artifacts.py      memory.py       bridge.py
      heuristics +      RAM-disk spill    SQLite graph    Anthropic ⇄
      LLM fallback      + fetch_log       (WAL+decay)     OpenAI
                              │
                    ┌─────────▼──────────┐
                    │  src/sentinel.py   │  watchdog · prune · compact · rotate
                    └────────────────────┘
                              │
                    any OpenAI-compatible upstream
```

| Module | Responsibility |
| --- | --- |
| `src/config.py` | Every path and credential, resolved from the environment |
| `src/classifier.py` | Heuristics, LLM classifier, routing policy, decision cache |
| `src/artifacts.py` | Spillover store, content-hash handles, `fetch_log` resolution |
| `src/memory.py` | Knowledge graph: extraction, decay, graduation, compaction |
| `src/bridge.py` | Anthropic ⇄ OpenAI request/response/SSE translation |
| `src/messages.py` | Message content plumbing shared by the above |
| `src/gateway.py` | FastAPI app, protocol surfaces, streaming engine |
| `src/sentinel.py` | Maintenance daemon (timer-friendly, or `--loop`) |

> `config.py`, `messages.py` and `bridge.py` are additions to the originally
> specified layout. They exist to keep configuration, message handling and
> protocol translation from collapsing into `gateway.py`, and to avoid a
> circular import between the classifier and the memory engine.

### Design invariants

These are enforced by the test suite, not just documented:

1. **No forwarding loop.** The gateway refuses to start if the upstream resolves
   to its own address or a legacy proxy port — that would be an infinite loop.
2. **Fail open.** A missing, slow, broken or lying classifier means *keep the
   tools*. Dropping a needed schema is a hard failure; keeping an unneeded one
   just costs a few hundred tokens.
3. **Protocol transparency.** Unknown JSON fields are forwarded untouched, so
   provider-specific options (`provider`, `reasoning_effort`, `seed`, …) survive.
4. **No full-response buffering.** Streaming is chunk-in/chunk-out. The escape
   look-ahead is bounded and exits as soon as the opening characters cannot be
   the sentinel.

---

## Quickstart

### Local

```bash
git clone <your-fork-url> agent-context-gateway
cd agent-context-gateway
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
$EDITOR .env                      # set UPSTREAM_BASE_URL and UPSTREAM_API_KEY

python -m src.gateway
```

Verify:

```bash
curl -s http://127.0.0.1:8090/health | python -m json.tool
curl -s http://127.0.0.1:8090/v1/models | head -c 300
```

### Docker

```bash
cp .env.example .env
docker compose -f docker/docker-compose.yml up --build
curl -s http://127.0.0.1:8090/health
```

### Try it with zero credentials

The repository ships an offline mock upstream, so you can exercise every route
without an API key or a cent of spend:

```bash
# terminal 1
python -m tests.mock_upstream --port 9099

# terminal 2
UPSTREAM_BASE_URL=http://127.0.0.1:9099/v1 UPSTREAM_API_KEY=test \
  python -m src.gateway

# terminal 3
curl -s -D - -o /dev/null -X POST http://127.0.0.1:8090/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"mock-model","stream":true,
       "messages":[{"role":"user","content":"hello"}],
       "tools":[{"type":"function","function":{"name":"terminal","parameters":{"type":"object"}}}]}'
```

Look for `X-Proxy-Tool-Action: Stripped-ZeroTokens` — the greeting was pruned.

### Maintenance daemon

```bash
python -m src.sentinel              # one cycle (use with cron/systemd timer)
python -m src.sentinel --status     # report only, change nothing
python -m src.sentinel --loop       # self-scheduling, for hosts with no cron
```

A ready-made systemd user timer pair is not included, but `--loop` and the
one-shot mode cover both scheduler and no-scheduler hosts.

---

## Configuration

Everything is environment-driven. A `.env` file is read if present; a real
environment variable always wins over it. See [`.env.example`](.env.example)
for the annotated template.

### Core

| Variable | Default | Meaning |
| --- | --- | --- |
| `GATEWAY_HOST` | `127.0.0.1` | Bind address |
| `GATEWAY_PORT` | `8090` | Bind port |
| `DATA_DIR` | `~/.agent-gateway/data` | SQLite graph lives here |
| `LOG_DIR` | `~/.agent-gateway/logs` | Audit log, rotated logs, lock file |
| `SHM_CACHE_DIR` | `/dev/shm/agent_gateway` (falls back to `/tmp/agent_gateway`) | Spilled context |
| `UPSTREAM_BASE_URL` | `https://api.openai.com/v1` | Any OpenAI-compatible endpoint |
| `UPSTREAM_API_KEY` | — | Falls back to `OPENAI_API_KEY`, `OPENROUTER_API_KEY` |
| `GATEWAY_API_KEY` | unset | When set, clients must authenticate |

### Classifier (optional)

Unset `CLASSIFIER_API_URL` to disable it entirely — routing then uses local
heuristics only and fails open.

| Variable | Default | Meaning |
| --- | --- | --- |
| `CLASSIFIER_API_URL` | unset | OpenAI-compatible chat-completions URL |
| `CLASSIFIER_MODEL` | `gpt-4o-mini` | Cheap, fast model |
| `CLASSIFIER_API_KEY` / `CLASSIFIER_API_KEY_ENV` | — | Key, or the *name* of another variable holding it |
| `CLASSIFIER_PROTOCOL` | `openai` | `openai` or `blueprint` |
| `CLASSIFIER_TIMEOUT_SECONDS` | `2.5` | Above this, fail open |
| `CLASSIFIER_NEEDS_TOOLS_THRESHOLD` | `0.15` | Strip only below this confidence |

`CLASSIFIER_API_KEY_ENV=UPSTREAM_API_KEY` reuses the upstream credential, so a
secret never has to be duplicated into a second file.

### Behaviour tuning

| Variable | Default | Meaning |
| --- | --- | --- |
| `MEMORY_INJECTION` | `1` | Inject recalled triples into prompts |
| `MEMORY_INJECT_TOOL_TURNS` | `0` | Also inject mid tool-loop |
| `TRUNCATE_THRESHOLD_CHARS` | `800` | Spill tool output above this size |
| `SNIFF_LIMIT_BYTES` | `512` | Escape look-ahead ceiling |
| `ESCAPE_SCAN_CHARS` | `64` | Give up sniffing this early |
| `REASONING_MODEL_REGEX` | unset | Extra comma-separated regex for models never to prune |

### Migrating an existing deployment

`HERMES_PROXY_HOST`, `HERMES_PROXY_PORT`, `REAL_UPSTREAM_BASE_URL`, `JEV_*` are
accepted as aliases, so an existing config keeps working during a migration.

---

## Connecting your agent

Detailed, copy-pasteable guides live in [`configs/`](configs):

| Agent | Guide | One-liner |
| --- | --- | --- |
| Claude Code | [`configs/claude_code.md`](configs/claude_code.md) | `export ANTHROPIC_BASE_URL=http://127.0.0.1:8090` |
| Aider | [`configs/aider.md`](configs/aider.md) | `aider --openai-api-base http://127.0.0.1:8090/v1` |
| Generic / OpenAI SDK, LangChain, CrewAI | [`configs/generic_agent.md`](configs/generic_agent.md) | `base_url="http://127.0.0.1:8090/v1"` |

### Aider

```bash
export OPENAI_API_BASE=http://127.0.0.1:8090/v1
export OPENAI_API_KEY=dummy          # the gateway injects the real upstream key
aider --model gpt-4o
```

### Generic OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8090/v1", api_key="dummy")
```

The gateway always replaces the client's `Authorization` header with the
upstream credential, so the agent never needs to know the real key.

---

## How it works

### Tool routing

```text
                     ┌─ reasoning model? ────▶ keep everything (never prune)
   turn ──▶ tools? ──┼─ greeting / <25 chars ─▶ strip (send no schema at all)
                     ├─ action words / a path ─▶ keep
                     └─ ambiguous ────────────▶ classifier ─▶ keep | strip
                                                     │
                                        unavailable / bad reply ─▶ keep (fail open)
```

The classifier is only consulted for prompts the heuristics cannot resolve, and
its answers are cached for five minutes.

### Early-stream escape

Pruning is a bet. When the bet is wrong, the model is told how to say so:

```text
[SYSTEM INSTRUCTION: If you cannot fulfill this request without external tools,
 start output with '[ESCAPE_NEED_TOOLS]']
```

The gateway looks at only the first ~64 characters of the reply. If the sentinel
appears, it aborts that stream, restores the tool schema, removes the
instruction, and re-dispatches — all before the client receives a byte. If it
does not appear, the buffered prefix is flushed as-is and normal streaming
resumes. Worst case added latency is the time to receive 64 characters.

### Context spillover

Historical tool output over `TRUNCATE_THRESHOLD_CHARS` (default 800) is written
to the RAM disk and replaced by a notice carrying the errors and a handle:

```text
[Output truncated | errors detected]
fatal: not a git repository
[Full output (41233 chars) preserved as handle LOG_A94F12.
 Call 'fetch_log(id="LOG_A94F12")' to inspect the raw buffer.]
```

The gateway registers a synthetic `fetch_log` tool, and then **answers it
itself**: when the client returns an unknown-tool error for that call, the
gateway substitutes the real slice. Your agent needs no implementation of it.

The two most recent messages are never touched — the model is actively reasoning
about them.

### Graph memory

Facts are extracted with deterministic regex (no model call, no latency), filtered
by an epistemic stance check that separates durable statements from fleeting
opinion, and stored in SQLite:

- **Decay** — un-reinforced candidates lose confidence on the Ebbinghaus curve.
- **Graduation** — a triple seen in ≥ 3 distinct sessions becomes `permanent`.
- **Conflict** — a new value replaces an old one only if the classifier agrees it
  supersedes it; failing that, newest wins.
- **Injection** — matching triples are wrapped in `<relevant_memory>` and added
  to the system prompt, marked as possibly stale.

Classifiers are resolved *before* the write transaction opens, so a slow network
call never holds a database lock.

---

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

```text
168 passed
```

The suite is **fully offline**: `tests/mock_upstream.py` provides both a real
local HTTP server and an in-process `httpx.MockTransport`, so no API keys, no
network and no spend. Coverage highlights:

| Area | What is asserted |
| --- | --- |
| Heuristics | Greeting/keyword/regex detection, including Arabic greetings |
| Reasoning passthrough | `o1`/`o3`/`r1`/`reasoner` never have tools stripped |
| Escape interception | Sentinel detected in the first chunk, abort asserted **< 50ms**, replay verified to carry the restored schema |
| Streaming | Verbatim relay; the sniff is bounded and provably does not consume the whole stream |
| Memory concurrency | **50 concurrent writers** in WAL mode with zero `database is locked` |
| Memory lifecycle | Ebbinghaus curve, `N ≥ 3` graduation, conflict overwrite, compaction + `VACUUM` |
| Anthropic bridge | Request/response/SSE translation, tool_use streaming, forced `tool_choice`, `count_tokens` |
| Invariants | Loop guard, fail-open classification, unknown-field passthrough, no secret leakage in `/health` |

---

## Security

- **`.env` is gitignored**, along with `*.db`, `*.db-wal`, `*.db-shm`, `*.log`,
  `.venv/` and `__pycache__/`.
- The gateway **never forwards the client's `Authorization` header**; it injects
  the upstream credential itself.
- `/health` returns a **redacted** snapshot — presence booleans, never values.
- Bind to `127.0.0.1` unless you also set `GATEWAY_API_KEY`. The Docker compose
  file publishes on loopback only for the same reason.
- Spilled context is written to a shared-memory path readable only by the
  running user.

## Limitations

Documented rather than hidden:

- **Anthropic thinking blocks are not translated.** Extended-thinking content is
  dropped rather than mangled; the token-usage numbers the bridge reports are
  estimates, because asking for exact usage would mean sending `stream_options`,
  which some OpenAI-compatible servers reject.
- **The Anthropic surface requires an OpenAI-compatible upstream.** It translates
  Claude Code's request into chat completions; it is not a passthrough to
  Anthropic's own API.
- **Triple extraction is regex-based**, so targets are single tokens. It favours
  precision over recall: it will miss facts rather than invent them.
- **The classifier is optional by design.** With it disabled, ambiguous prompts
  keep their tools, which is correct but less token-efficient.
- The SQLite graph is single-node. There is no multi-process coordination beyond
  WAL.

## License

[MIT](LICENSE)
