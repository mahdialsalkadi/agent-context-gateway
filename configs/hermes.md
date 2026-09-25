# Hermes Agent → agent-context-gateway

This profile keeps the setup that was verified against Hermes CLI working, and
adds the gateway as an **additional** provider.

> **Do not repoint `model.base_url`.** Add a second provider block and switch to
> it when you want gateway-managed context. Leaving the existing provider alone
> means a gateway problem can never take your agent down — you switch back with
> one config edit.

## 1. Start the gateway

```bash
cd agent-context-gateway
python -m src.gateway --profile hermes
```

```text
[agent-context-gateway] listening on http://127.0.0.1:8091 -> https://openrouter.ai/api/v1
[agent-context-gateway] classifier mode 'upstream_reused' -> https://openrouter.ai/api/v1/chat/completions (google/gemini-2.5-flash-lite)
```

## 2. Ports

| Service | Port |
| --- | --- |
| Hermes' existing provider | `8080` |
| **this gateway** | **`8091`** |

8091 keeps the gateway clear of any provider Hermes already has. If a process
already holds the gateway's port, the sentinel reports
`foreign_service_on_port: true` rather than assuming all is well — see
`python -m src.sentinel --status`.

## 3. Add the provider to `~/.hermes/config.yaml`

Insert a new entry alongside the existing provider — the change is **additive
only**, so nothing about the current setup moves:

```yaml
providers:
  # ... your existing provider stays exactly as it is ...
  agent_gateway:
    base_url: http://127.0.0.1:8091/v1
    api_key: dummy          # the gateway injects the real upstream key
    model: google/gemini-2.5-flash-lite
```

Then select it:

```yaml
model:
  provider: agent_gateway
  # base_url is NOT set here on purpose -- the provider block above owns it
```

To roll back, switch `model.provider` back to the original name. No other key
changes.

## 4. Verify before switching over

```bash
curl -s http://127.0.0.1:8091/health | python -m json.tool
curl -s -N -X POST http://127.0.0.1:8091/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"google/gemini-2.5-flash-lite","stream":true,
       "messages":[{"role":"user","content":"hello"}]}'
```

You should see SSE frames and, in the response headers, the route the gateway
chose (`X-Proxy-Route`).

## 5. Classifier and token reuse

`.env.hermes` sets `CLASSIFIER_MODE=upstream_reused` with
`google/gemini-2.5-flash-lite`. Classification runs on the same OpenRouter key as
the upstream, so there is no second credential to manage — the key is borrowed
from `UPSTREAM_API_KEY`, never copied into a second file. Use
`CLASSIFIER_API_KEY_ENV=UPSTREAM_API_KEY` explicitly if you prefer to be verbose
about it.

For a hard zero-token setup, set `CLASSIFIER_MODE=heuristics` instead.

## 6. Graph memory

`MEMORY_INJECTION=1` (the default in this profile) makes the gateway recall
durable facts from earlier sessions and inject them into later prompts. The graph
lives at `DATA_DIR/graph.db` in SQLite WAL mode, shared by every profile, and is
compacted by the sentinel.

Facts graduate from `candidate` to `permanent` after they are seen in
`GRADUATION_SESSIONS` (default 3) sessions; un-reinforced candidates decay on an
Ebbinghaus curve and are eventually dropped. Run one maintenance cycle with:

```bash
python -m src.sentinel
```
