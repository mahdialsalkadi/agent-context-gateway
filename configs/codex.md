# Codex / GitHub Copilot bridge → agent-context-gateway

Local Copilot bridges expose an OpenAI-compatible surface (commonly
`copilot-gpt4-service` on port `4141`). Point the gateway at it and your Copilot
quota stops paying for tool schemas no turn can use.

## 1. Start the gateway

```bash
cd agent-context-gateway
python -m src.gateway --profile codex
```

```text
[agent-context-gateway] listening on http://127.0.0.1:8091 -> http://127.0.0.1:4141/v1
[agent-context-gateway] notice: classifier mode 'heuristics' issues no network calls
```

`.env.codex` defaults to `CLASSIFIER_MODE=heuristics`, which costs nothing at all.
To get judgement calls from the Copilot model instead, flip to
`CLASSIFIER_MODE=upstream_reused` in that file — the key and endpoint are borrowed
from the upstream, so no second credential is needed.

## 2. Ports

| Service | Port |
| --- | --- |
| Copilot bridge | `4141` |
| **this gateway** | **`8091`** |

4141 is not a reserved proxy port, so no loop-guard relaxation is required here.
If you move the gateway onto the bridge's port, startup fails fatally — that is
the forwarding-loop guard, not a bug.

## 3. Point your client at the gateway

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8091/v1
export OPENAI_API_KEY=dummy
```

For an agent that takes explicit flags:

```bash
aider --openai-api-base http://127.0.0.1:8091/v1 --openai-api-key dummy
```

## 4. Verify

```bash
curl -s http://127.0.0.1:8091/health | python -m json.tool
# "classifier_mode": "heuristics", "upstream": "http://127.0.0.1:4141/v1"
```

## 5. Why a bridge benefits most

A subscription bridge has a hard request/context ceiling rather than a
per-token bill. Pruning the schema and spilling oversized tool output therefore
buys *more conversations per window*, which is usually the real constraint:

- Conversational turns spend **zero** tokens on a tool schema (`FastPath-Strip`).
- Historical tool output above `TRUNCATE_THRESHOLD_CHARS` is replaced by a handle
  and fetched back only when the model asks for it.
- If a stripped turn genuinely needed tools, the gateway aborts early and replays
  with the schema restored — the agent never notices.

Set `TRUNCATE_THRESHOLD_CHARS` lower (e.g. `400`) on a bridge with a small context
window to start spilling sooner.
