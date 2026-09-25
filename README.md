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

Works with **Claude Code, Aider, Cursor, Codex/Copilot bridges, Google
Antigravity, Hermes, or any OpenAI SDK client** — no vendor lock-in, no
agent-specific code, no hardcoded paths.

**And it can run for $0.** Every routing decision is available in a mode that
reuses a subscription you already pay for, or a model already running on your
own machine — no second provider, no second key. See
[Zero-cost mode](#zero-cost-mode).

```bash
cp .env.example .env      # set UPSTREAM_BASE_URL + UPSTREAM_API_KEY
pip install -r requirements.txt
python -m src.gateway
# -> listening on http://127.0.0.1:8090/v1
```

Or start from a ready-made provider profile:

```bash
python -m src.gateway --profile antigravity   # Google Antigravity bridge (8080)
python -m src.gateway --profile claude        # Claude Code / Anthropic surface
python -m src.gateway --profile codex         # Codex / GitHub Copilot bridge
python -m src.gateway --profile hermes        # Hermes Agent
python -m src.gateway --profile openrouter    # OpenRouter direct
python -m src.gateway --list-profiles         # what is available
```

No `--profile` flag means `.env`, exactly as before.

> **Compatibility note.** The OpenAI surface (`/v1/chat/completions`) works with
> every OpenAI-compatible client. The Anthropic surface (`/v1/messages`) exists
> so **Claude Code** can point `ANTHROPIC_BASE_URL` at the gateway; see
> [`configs/claude_code.md`](configs/claude_code.md).

---

## Contents

- [Architecture](#architecture)
- [Supported providers](#supported-providers)
- [Zero-cost mode](#zero-cost-mode)
- [Profiles](#profiles)
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
   Copilot bridge┤   /v1/messages        (Anthropic, translated)
   Antigravity ──┼──▶  /v1/models | /health
   LangChain ────┤
   Hermes ───────┘
                              │
                    ┌─────────▼──────────┐
                    │   src/gateway.py   │  routing · streaming · telemetry
                    └─────────┬──────────┘
              ┌───────────────┼───────────────┬────────────────┐
              ▼               ▼               ▼                ▼
      classifier.py     artifacts.py      memory.py       bridge.py
      4 routing modes   RAM-disk spill    SQLite graph    Anthropic ⇄
      (see below)       + fetch_log       (WAL+decay)     OpenAI
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
| `src/classifier.py` | Four routing modes, heuristics, sub-tool ranker, decision cache |
| `src/analytics.py` | Token & dollar savings and latency percentiles from the audit log |
| `src/cli.py` | The `agent-gateway` command: start/stop/status/stats/test/service |
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
   to its own address, or to a legacy proxy port unless
   `ALLOW_LEGACY_UPSTREAM_PORT=1` explicitly claims that port (needed when a local
   subscription bridge genuinely lives on 8080). Pointing at *itself* is always
   fatal.
2. **Fail open.** A missing, slow, broken or lying classifier means *keep the
   tools*. Dropping a needed schema is a hard failure; keeping an unneeded one
   just costs a few hundred tokens.
3. **Protocol transparency.** Unknown JSON fields are forwarded untouched, so
   provider-specific options (`provider`, `reasoning_effort`, `seed`, …) survive.
4. **No full-response buffering.** Streaming is chunk-in/chunk-out. The escape
   look-ahead is bounded and exits as soon as the opening characters cannot be
   the sentinel.
5. **The watchdog verifies identity, not reachability.** A bare HTTP 200 on the
   port only proves *something* is listening. `sentinel.probe()` requires the
   health payload to carry the gateway's own markers and to report the same
   `DATA_DIR`; anything else is surfaced as `foreign_service_on_port` rather
   than silently accepted as healthy.
6. **Pruning never gambles.** Selective sub-tool ranking only shrinks a schema
   when the prompt gives positive evidence for some tool; an unmatched prompt
   keeps everything. Mission-critical I/O (`bash`, `read_file`, `fetch_log`)
   is always preserved, and an explicit `tool_choice` is never ranked.

---

## Supported providers

Every row below is a first-class, tested path. "Free" means no second paid
credential is introduced by the gateway.

| Agent / host | Transport into the gateway | Profile & guide | Classifier mode |
| --- | --- | --- | --- |
| **Hermes Agent** | OpenAI `/v1/chat/completions` | `.env.hermes` · [guide](configs/hermes.md) | `upstream_reused` (free) |
| **Google Antigravity** (Google One Pro) | OpenAI, via the local bridge on `:8080` | `.env.antigravity` · [guide](configs/antigravity.md) | `upstream_reused` (free) |
| **Claude Code** | Anthropic `/v1/messages` (translated) | `.env.claude` · [guide](configs/claude_code.md) | `heuristics` (free) |
| **Codex / GitHub Copilot bridge** | OpenAI, via a local bridge on `:4141` | `.env.codex` · [guide](configs/codex.md) | `heuristics` (free) |
| **Aider / Cursor / generic OpenAI SDK, LangChain, CrewAI** | OpenAI | [guide](configs/generic_agent.md) | any |
| **Ollama / vLLM / llama.cpp** | OpenAI | [guide](configs/generic_agent.md) | `local_ollama` (free) |
| **OpenRouter / OpenAI direct** | OpenAI | `.env.openrouter` | `upstream_reused` or `external_jev` |

Anything that speaks the OpenAI chat-completions API works without a guide. The
Anthropic surface exists so Claude Code needs no shim.

---

## Zero-cost mode

There are three ways to run the whole system for $0, and they compose:

| Mode | Extra cost | Trade-off |
| --- | --- | --- |
| `CLASSIFIER_MODE=heuristics` | **none** — zero network calls | Only unambiguous turns are settled; ambiguous ones keep their tools |
| `CLASSIFIER_MODE=upstream_reused` | **none beyond what you already pay** | Spends a few tokens of the subscription you're already using |
| `CLASSIFIER_MODE=local_ollama` | **none** — stays on your machine | Needs a local model pulled (default `qwen2.5:0.5b`) |

```bash
# Cheapest of all: local rules, identical transport behaviour.
CLASSIFIER_MODE=heuristics python -m src.gateway

# Judge with the subscription you already have -- no second key.
CLASSIFIER_MODE=upstream_reused CLASSIFIER_MODEL=gemini-2.5-flash \
  python -m src.gateway

# Judge locally; nothing leaves the machine.
CLASSIFIER_MODE=local_ollama python -m src.gateway
```

**The bigger saving is orthogonal to the classifier.** Tool-schema pruning and
artifact spillover happen on *every* turn regardless of mode, and they are what
actually protects a subscription quota — a turn that sends no tool schema costs a
fraction of one that does. On a quota-limited plan, `heuristics` plus pruning is
strictly free, and no request ever leaves your machine to decide.

### Antigravity example, end to end

```bash
python -m src.gateway --profile antigravity
# listening on 127.0.0.1:8091 -> 127.0.0.1:8080 (Antigravity)
# classifier: upstream_reused -> gemini-2.5-flash (same subscription)
```

Antigravity occupies `8080`, so the gateway takes `8091`. If you change that,
do not reuse 8080 — the startup guard will (correctly) refuse to start.

---

## Profiles

A profile is just `.env.<name>`. Nothing about `.env` changes: the flag decides
which file is read, and a real environment variable still wins over both.

| Profile | Port | Upstream | Classifier |
| --- | --- | --- | --- |
| `antigravity` | `8091` | Antigravity bridge `:8080` | `upstream_reused` · `gemini-2.5-flash` |
| `claude` | `8091` | your OpenAI-compatible provider | `heuristics` |
| `codex` | `8091` | Copilot bridge `:4141` | `heuristics` |
| `hermes` | `8091` | your OpenAI-compatible provider | `upstream_reused` |
| `openrouter` | `8090` | `https://openrouter.ai/api/v1` | `upstream_reused` |

```bash
python -m src.gateway --profile antigravity     # load .env.antigravity
python -m src.gateway                           # fall back to .env
python -m src.gateway --list-profiles           # discover what exists

# The same switch, without the flag -- also honoured by
# `uvicorn src.gateway:app` and by a sentinel-spawned gateway.
AGENT_GATEWAY_PROFILE=claude python -m src.gateway
```

Unknown profile? The process exits `2` and lists what it found, rather than
silently starting with the wrong upstream.

**Switching profiles never moves your state.** `DATA_DIR`, `LOG_DIR` and
`SHM_CACHE_DIR` are deliberately *not* set by any shipped profile, so the SQLite
graph stays in WAL mode with its rows intact and every spilled artifact remains
readable by its content-hash handle. That is asserted by the test suite, one test
per profile file.

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

`.env` is optional: the compose file loads it only when it exists, so a fresh
clone validates and starts without it (`UPSTREAM_BASE_URL=... docker compose
-f docker/docker-compose.yml up --build` works too). The container runs as a
non-root user, publishes the port on loopback only, and gets a 256MB tmpfs at
`/dev/shm` for the spillover cache — the Docker default of 64MB sits below the
eviction ceiling and would churn.

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

### Classifier modes

`CLASSIFIER_MODE` selects where a routing verdict comes from. All four are
first-class; none is required. The local heuristics always run first, and every
mode fails open.

| `CLASSIFIER_MODE` | Endpoint it uses | Credential | Extra cost |
| --- | --- | --- | --- |
| `heuristics` | *none — no network call at all* | — | **$0** |
| `upstream_reused` | `UPSTREAM_BASE_URL/chat/completions` | borrowed from `UPSTREAM_API_KEY` | **$0** (same subscription) |
| `local_ollama` | `OLLAMA_BASE_URL`, default `http://127.0.0.1:11434/v1` | none needed | **$0** (local) |
| `external_jev` | `CLASSIFIER_API_URL` / `JEV_API_URL` | `CLASSIFIER_API_KEY`, or the name of another variable | whatever that endpoint bills |

Leaving `CLASSIFIER_MODE` unset is `auto`, which preserves the original
behaviour exactly: an explicitly configured URL *and* key means `external_jev`,
otherwise `heuristics`. An unrecognised value degrades to `auto` rather than
crashing; `/health` reports the effective mode so a typo is visible.

`heuristics` blanks the endpoint outright, so no code path — and no future
refactor — can reach out. The suite asserts this by poisoning the HTTP client and
requiring the call to never happen.

| Variable | Default | Meaning |
| --- | --- | --- |
| `CLASSIFIER_MODE` | `auto` | `heuristics`, `upstream_reused`, `local_ollama`, `external_jev` |
| `CLASSIFIER_API_URL` | unset | Dedicated endpoint (`external_jev`) |
| `CLASSIFIER_MODEL` | mode-dependent | `gpt-4o-mini`, or `qwen2.5:0.5b` for `local_ollama` |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434/v1` | Local runner for `local_ollama` |
| `CLASSIFIER_API_KEY` / `CLASSIFIER_API_KEY_ENV` | — | Key, or the *name* of another variable holding it |
| `CLASSIFIER_PROTOCOL` | `openai` | `openai`, or `blueprint` (a `external_jev`-only shape) |
| `CLASSIFIER_TIMEOUT_SECONDS` | `2.5` | Above this, fail open |
| `CLASSIFIER_NEEDS_TOOLS_THRESHOLD` | `0.15` | Strip only below this confidence |
| `CLASSIFIER_SUPERSEDE_THRESHOLD` | `0.75` | Confidence needed to overwrite a stored fact |

`upstream_reused` needs no credential of its own — that is the point. A local
bridge that ignores auth still gets classification, because a missing bearer
token does not silently disable the mode.

`CLASSIFIER_API_KEY_ENV=UPSTREAM_API_KEY` reuses the upstream credential even in
`external_jev`, so a secret never has to be duplicated into a second file.

### Behaviour tuning

| Variable | Default | Meaning |
| --- | --- | --- |
| `MEMORY_INJECTION` | `1` | Inject recalled triples into prompts |
| `MEMORY_INJECT_TOOL_TURNS` | `0` | Also inject mid tool-loop |
| `TRUNCATE_THRESHOLD_CHARS` | `800` | Spill tool output above this size |
| `ENABLE_SELECTIVE_PRUNING` | `1` | Rank sub-tools when the schema is large |
| `SELECTIVE_TOOL_LIMIT` | `5` | Prompt-matched tools kept (mission-critical always survive) |
| `SNIFF_LIMIT_BYTES` | `512` | Escape look-ahead ceiling |
| `ESCAPE_SCAN_CHARS` | `64` | Give up sniffing this early |
| `REASONING_MODEL_REGEX` | unset | Extra comma-separated regex for models never to prune |
| `ALLOW_LEGACY_UPSTREAM_PORT` | `0` | Allow an upstream on 8080/8090 (needed by local bridges). Never relaxes the self-forward guard |
| `ANTHROPIC_THINKING_PASSTHROUGH` | `0` | Re-emit upstream reasoning as Anthropic `thinking` blocks (unsigned) |

### Migrating an existing deployment

`HERMES_PROXY_HOST`, `HERMES_PROXY_PORT`, `REAL_UPSTREAM_BASE_URL`, `JEV_*` are
accepted as aliases, so an existing config keeps working during a migration.

---

## Connecting your agent

Detailed, copy-pasteable guides live in [`configs/`](configs):

| Agent | Guide | One-liner |
| --- | --- | --- |
| Claude Code | [`configs/claude_code.md`](configs/claude_code.md) | `export ANTHROPIC_BASE_URL=http://127.0.0.1:8091` |
| Aider | [`configs/aider.md`](configs/aider.md) | `aider --openai-api-base http://127.0.0.1:8090/v1` |
| Google Antigravity | [`configs/antigravity.md`](configs/antigravity.md) | `python -m src.gateway --profile antigravity` |
| Codex / Copilot bridge | [`configs/codex.md`](configs/codex.md) | `python -m src.gateway --profile codex` |
| Hermes Agent | [`configs/hermes.md`](configs/hermes.md) | `python -m src.gateway --profile hermes` |
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
                                                (mode-dependent)  │
                                          unavailable / bad reply ─▶ keep (fail open)
```

The classifier is only consulted for prompts the heuristics cannot resolve, and
its answers are cached for five minutes. With `CLASSIFIER_MODE=heuristics` there
is no network step at all: the ambiguous branch resolves straight to `keep`,
which is why that mode is both free and impossible to break.

### Selective sub-tool pruning

All-or-nothing is a blunt instrument once an agent offers 20+ tools. When the
schema exceeds `SELECTIVE_PRUNING_MIN_TOOLS` (default 8), a local BM25 ranker
scores each tool's name and description against the prompt and keeps only the
`SELECTIVE_TOOL_LIMIT` (default 5) best — plus the mission-critical I/O tools
(`bash`, `read_file`, `fetch_log`), which are never dropped.

Measured on a 25-tool catalog with 3 properties each:

```text
full catalog sent to the upstream : ~2625 tokens per turn
after selective pruning           : 25 -> 4-5 tools (~105 tokens per kept tool)
ranker overhead                   : 0.083 ms per call   (budget: 2 ms)
```

```bash
ENABLE_SELECTIVE_PRUNING=1     # default
SELECTIVE_TOOL_LIMIT=5         # how many prompt-matched tools survive
```

The ranker is honest by construction: a prompt that matches no tool at all keeps
the whole schema (no evidence, no gamble), stop words carry no weight, payload
order is preserved so prompt caches stay warm, and an explicit `tool_choice`
disables ranking entirely. Telemetry: `X-Proxy-Tools-Before` / `X-Proxy-Tools-After`
headers, and `tools_before` / `tools_after` / `selective_dropped` in the audit log.

### Seeing the savings: `agent-gateway stats`

```text
agent-context-gateway -- savings dashboard
==============================================================
  audit source     ~/.agent-gateway/logs/audit.log

  requests         3        sessions        1
  spills           0        fetch_log hits  0

  routes
    Classifier-FailOpen                     2  ########################
    FastPath-Keep                           1  ############............

  estimated savings (see note)
    tool schemas pruned        4,960 tokens   (62 selective drops)
    spilled tool output            0 tokens   (0 chars)
    escape replays                 0
    ----------------------------------------------
    TOTAL                      4,960 tokens
    ~ $0.0149 at $3.00/M prompt tokens

  latency (gateway overhead, ms)
    p50    14.53   p90    50.67   p99    50.67

  note: token and dollar figures are estimates; exact counts require the
  upstream's tokenizer, which no OpenAI-compatible API exposes.
```

```bash
agent-gateway stats            # one snapshot
agent-gateway stats --live     # refreshes in place, Ctrl-C to stop
agent-gateway stats --json     # for your own dashboards
```

### Global CLI

```bash
pip install -e .               # puts `agent-gateway` on your PATH

agent-gateway start [--profile NAME] [--port N] [--daemon]
agent-gateway stop              # stops what this CLI started; never guesses PIDs
agent-gateway status            # pid, health, mode, profile, foreign-service flag
agent-gateway stats [--live|--json]
agent-gateway test              # the offline suite, no keys needed
agent-gateway service install   # systemd user unit, written AND enabled
```

`agent-gateway stop` only ever signals a PID it recorded itself; if something
else answers on the port it says so and exits non-zero rather than killing an
innocent process.

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
290 passed
```

The suite is **fully offline**: `tests/mock_upstream.py` provides both a real
local HTTP server and an in-process `httpx.MockTransport`, so no API keys, no
network and no spend. It also answers the gateway's *classifier* probes, which is
how `upstream_reused` is proven over real HTTP. `tests/conftest.py` wires the
fixtures together; `tests/test_sentinel.py` covers the watchdog;
`tests/test_classifier_modes.py` covers the four modes and profiles;
`tests/test_selective_pruning.py`, `tests/test_analytics.py`, `tests/test_cli.py`
and `tests/test_memory_retrieval.py` cover the newer phases. Coverage
highlights:

| Area | What is asserted |
| --- | --- |
| Heuristics | Greeting/keyword/regex detection, including Arabic greetings |
| Reasoning passthrough | `o1`/`o3`/`r1`/`reasoner` never have tools stripped |
| Escape interception | Sentinel detected in the first chunk, abort asserted **< 50ms**, replay verified to carry the restored schema |
| Streaming | Verbatim relay; the sniff is bounded and provably does not consume the whole stream |
| Memory concurrency | **50 concurrent writers** in WAL mode with zero `database is locked` |
| Memory lifecycle | Ebbinghaus curve, `N ≥ 3` graduation, conflict overwrite, compaction + `VACUUM` |
| Anthropic bridge | Request/response/SSE translation, tool_use streaming, forced `tool_choice`, `count_tokens` |
| Watchdog identity | A 200 from a *foreign* service on the port is rejected, so the sentinel cannot be fooled into reporting a dead gateway as healthy |
| Classifier modes | `heuristics` provably makes **zero** network calls (HTTP client poisoned); `upstream_reused` routes over real HTTP carrying the upstream's own bearer token; `local_ollama` payload shape; `external_jev` blueprint probability |
| Profiles | Every shipped profile loads, resolves a valid mode and cannot forward to itself; unknown profile exits `2`; **switching profiles leaves the WAL graph and the spilled artifacts byte-for-byte intact** |
| Thinking blocks | Dropped by default; opt-in passthrough emits `thinking` before text, for `reasoning_content` / `reasoning` / `thinking` and part-lists |
| Selective pruning | 25-tool catalog shrinks to the tools the prompt implies; mission-critical I/O always kept; unmatched prompts keep everything; **ranking measured < 2 ms** |
| Analytics | Synthetic audit rows in, savings and percentiles out; torn lines skipped; legacy rows estimated |
| CLI | pidfile lifecycle against a really-spawned daemon; `stop` refuses to signal a PID it did not record; service unit generation |
| Memory retrieval | Typos and prefixes forgiven, unrelated words not; budget packing never splits a fact; audit hook writes one row per retrieval |
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
- **The port is exclusive.** If another service already holds `GATEWAY_PORT`
  (a previous proxy, another checkout), the gateway cannot bind. The sentinel
  reports this as `foreign_service_on_port: true` and deliberately does *not*
  respawn, because a second process cannot win the bind either. Change
  `GATEWAY_PORT` to resolve it.
- **Anthropic thinking blocks are opt-in and unsigned.** An OpenAI-shaped
  upstream cannot supply the signature Anthropic expects, and a strict client
  rejects an unsigned thinking block, so reasoning is dropped unless
  `ANTHROPIC_THINKING_PASSTHROUGH=1` asks for it. Fabricating a signature is not
  an option; dropping never breaks a stream.
- **`ALLOW_LEGACY_UPSTREAM_PORT` is a heuristic waiver, not a safety switch.** It
  permits an upstream on 8080/8090 so a local bridge can be used. It never
  permits forwarding to the gateway's own address.
- The SQLite graph is single-node. There is no multi-process coordination beyond
  WAL.

## License

[MIT](LICENSE)
