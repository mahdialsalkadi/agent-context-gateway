# Claude Code → agent-context-gateway

Claude Code speaks the **Anthropic Messages API** (`POST /v1/messages`), not
`/v1/chat/completions`. The gateway implements that surface and translates each
request into an OpenAI chat-completions call, so all the context management
(tool pruning, spillover, memory) applies to Claude Code too.

## 1. Start the gateway

```bash
cd agent-context-gateway
python -m src.gateway --profile claude
curl -s http://127.0.0.1:8091/health | python -m json.tool
```

The `claude` profile (`.env.claude`) listens on **8091** and is the fastest route.
Edit it to set `UPSTREAM_BASE_URL` and `UPSTREAM_API_KEY`; everything else already
works. Prefer a plain file? `cp .env.example .env && python -m src.gateway`.

## 2. Point Claude Code at it

`ANTHROPIC_BASE_URL` is an **origin, not a path** — Claude Code appends
`/v1/messages` itself.

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8091
export ANTHROPIC_API_KEY=dummy        # gateway injects the real upstream key
claude
```

If you would rather use a bearer token than `x-api-key`, Claude Code accepts:

```bash
export ANTHROPIC_AUTH_TOKEN=dummy
```

Both are supported. Neither reaches the upstream: the gateway replaces the
credential with `UPSTREAM_API_KEY`.

### Protecting the gateway

If the port is reachable by anything other than you, set a real key:

```bash
# in the gateway's .env
GATEWAY_API_KEY=choose-a-long-random-string
```

```bash
# in Claude Code's environment
export ANTHROPIC_API_KEY=choose-a-long-random-string
```

Unauthenticated requests then get `401` with an Anthropic-shaped error body.

## 3. Model names — the one thing that will bite you

Claude Code asks for `claude-sonnet-4`, `claude-opus-4`, and so on. The gateway
forwards the model name to your upstream **as-is**. So:

- **OpenRouter or an Anthropic-compatible upstream** — these serve `claude-*`
  ids, so it works with no extra configuration.
- **A plain OpenAI, vLLM, Ollama or llama.cpp upstream** — those do not, and you
  will get a 4xx. Map the name instead:

  ```bash
  # gateway .env
  ANTHROPIC_MODEL_OVERRIDE=gpt-4o-mini
  ```

  This overrides the model on the Anthropic surface only. Your OpenAI surface
  keeps passing through whatever the client asked for.

## 4. Verify it is working

```bash
curl -s -D - -o /dev/null -X POST http://127.0.0.1:8090/v1/messages \
  -H 'content-type: application/json' \
  -H 'x-api-key: dummy' \
  -d '{"model":"claude-sonnet-4","max_tokens":64,"stream":true,
       "messages":[{"role":"user","content":"hello"}],
       "tools":[{"name":"Bash","description":"Run a command",
                 "input_schema":{"type":"object","properties":{"command":{"type":"string"}}}}]}'
```

The telemetry headers tell you what happened:

| Header | Reading |
| --- | --- |
| `X-Proxy-Route` | `FastPath-Strip`, `Classifier-Keep`, `Reasoning-Passthrough`, … |
| `X-Proxy-Tool-Action` | `Stripped-ZeroTokens` means the tool schema was not sent |
| `X-Proxy-Latency-MS` | Time to first byte from the upstream |
| `X-Proxy-Spill-Count` | How many large tool results were moved out of the prompt |

Streamed output is real Anthropic SSE — `message_start`, `content_block_delta`,
`message_stop` — so Claude Code parses it normally.

## 5. Token counting

Claude Code probes `POST /v1/messages/count_tokens` before sending. The gateway
answers with a character-based estimate. It is approximate by design: exact
counts would require the upstream to expose a tokenizer for a model it may not
even host. This only affects the display, not billing.

## 6. Routing mode (and cost)

The profile ships with `CLASSIFIER_MODE=heuristics`: routing decisions come from
local regex and keyword rules in well under a millisecond, with no network call
at all. Ambiguous turns fail open, so a tool schema is never dropped by mistake.

To get judgement calls from a model instead, reuse whichever provider you already
pay for:

```bash
# classify on the same subscription as the upstream -- no second provider
CLASSIFIER_MODE=upstream_reused
CLASSIFIER_MODEL=claude-3-5-haiku
```

A local runner works the same way with `CLASSIFIER_MODE=local_ollama`, and a
dedicated endpoint with `CLASSIFIER_MODE=external_jev`. See the README matrix.

Context handling under this profile: bash/file tool outputs are pruned and
spilled to disk with a retrievable handle, and learned facts are injected into
later turns. Thinking blocks are covered below.

## 7. Thinking blocks

Reasoning text from the upstream is **not** re-emitted as Anthropic `thinking`
blocks by default. Anthropic models expect a `signature` alongside thinking
content, an OpenAI-shaped upstream cannot supply one, and a strict client rejects
an unsigned block — so emitting one would break the stream it is meant to enrich.

Dropping reasoning is therefore the safe default and never breaks a client. If
your client tolerates unsigned blocks, opt in:

```bash
ANTHROPIC_THINKING_PASSTHROUGH=1
```

`reasoning_content`, `reasoning` and `thinking` fields are all recognised, and
the thinking block is emitted before the answer text, in the order the Anthropic
event model requires.

## What is not translated

- **`cache_control` hints** are ignored (OpenAI has no equivalent).
- **Audio and document content blocks** are dropped; text and base64/URL images
  are translated.

If your workflow depends on either, keep Claude Code pointed at Anthropic
directly and use the gateway for an OpenAI-compatible agent instead.
