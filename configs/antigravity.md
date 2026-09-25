# Google Antigravity → agent-context-gateway

Run the whole stack on a **Google One / Gemini subscription at $0 extra**. The
gateway sits in front of the local Antigravity server, prunes tool schemas that a
turn does not need, and classifies using a fast Gemini model on the *same*
subscription — so the daily quota buys far more work.

## 1. Start the gateway

```bash
cd agent-context-gateway
python -m src.gateway --profile antigravity
```

Watch for these two lines — they are the whole configuration:

```text
[agent-context-gateway] listening on http://127.0.0.1:8091 -> http://127.0.0.1:8080/v1
[agent-context-gateway] profile file: .../.env.antigravity
[agent-context-gateway] classifier mode 'upstream_reused' -> http://127.0.0.1:8080/v1/chat/completions (gemini-2.5-flash)
```

## 2. Port layout — the collision you must avoid

| Service | Port |
| --- | --- |
| Antigravity bridge | `8080` |
| **this gateway** | **`8091`** |

The shipped profile already uses 8091. If you change it, **do not reuse 8080** —
the gateway would be listening on the same port its own upstream lives on, which
is the infinite forwarding loop the startup guard exists to prevent:

```text
[FATAL] Upstream 'http://127.0.0.1:8080/v1' resolves to a local gateway port.
```

Two settings make this safe rather than fragile:

- `GATEWAY_PORT=8091` keeps the two apart.
- `ALLOW_LEGACY_UPSTREAM_PORT=1` says "8080 really is my upstream". Without it the
  loop guard treats 8080 as a likely proxy and refuses to start. Relaxing it does
  **not** disable the real guard — pointing the upstream at *this* gateway still
  fails fatally.

## 3. Point your agent at the gateway

Any OpenAI-compatible client works. In your agent's config:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8091/v1
export OPENAI_API_KEY=dummy        # the gateway injects the real upstream key
```

Or from the SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8091/v1", api_key="dummy")
```

## 4. Verify it is working

```bash
curl -s -D - -o /dev/null -X POST http://127.0.0.1:8091/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"gemini-2.5-flash","stream":true,
       "messages":[{"role":"user","content":"hello"}],
       "tools":[{"type":"function","function":{"name":"terminal",
                 "parameters":{"type":"object","properties":{"command":{"type":"string"}}}}]}'
```

| Header | Reading |
| --- | --- |
| `X-Proxy-Route` | `FastPath-Strip` / `Classifier-Strip` means the schema was dropped |
| `X-Proxy-Tool-Action` | `Stripped-ZeroTokens` — no tool schema sent at all |
| `X-Proxy-Spill-Count` | Large historical tool results moved to a retrievable handle |

If a pruned turn turns out to need its tools, the model signals it and the gateway
aborts that stream mid-flight and replays with the full schema. You never see a
broken turn; you only see the `Classifier-Strip->EarlyEscapeAbort` route header.

## 5. Where the 70-90% saving comes from

Every turn an agent sends carries the full tool schema, and most turns cannot use
most of it. Three mechanisms cut that down:

1. **Schema pruning** — greetings and conversational turns carry no schema at all;
   ambiguous turns get a cheap verdict (here, from `gemini-2.5-flash`).
2. **Artifact spillover** — a 40 KB `ls -R` becomes a 400-byte notice plus a handle
   the model can page back in, instead of riding in every later prompt.
3. **Early-stream escape** — a wrongly pruned turn is corrected in the first ~64
   characters, so the wasted generation is bounded rather than the whole turn.

## 6. Zero-cost checklist

- `CLASSIFIER_MODE=upstream_reused` — no second provider, no second key.
- `CLASSIFIER_MODE=heuristics` — zero tokens, zero latency; ambiguous turns keep
  their tools.
- `CLASSIFIER_MODE=local_ollama` — move the verdict to a local runner on
  `http://127.0.0.1:11434/v1`; nothing leaves the machine.

Switching never touches `DATA_DIR` or the RAM-disk cache, so your graph memory and
spilled artifacts survive a profile change untouched.
