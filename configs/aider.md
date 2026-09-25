# Aider → agent-context-gateway

Aider speaks the OpenAI chat-completions API, so it needs no translation — just
repoint the base URL.

## 1. Start the gateway

```bash
cd agent-context-gateway
cp .env.example .env
$EDITOR .env      # set UPSTREAM_BASE_URL + UPSTREAM_API_KEY
python -m src.gateway
```

## 2. Run Aider through it

### Command-line flags

```bash
aider \
  --openai-api-base http://127.0.0.1:8090/v1 \
  --openai-api-key dummy \
  --model gpt-4o
```

The API key is a placeholder. The gateway always replaces the incoming
`Authorization` header with `UPSTREAM_API_KEY`, so Aider never holds the real
credential.

### Environment variables

```bash
export OPENAI_API_BASE=http://127.0.0.1:8090/v1
export OPENAI_API_KEY=dummy
aider --model gpt-4o
```

Beware: `OPENAI_API_BASE` is read by many tools at once. If you set it globally
in your shell profile, everything OpenAI-compatible in that shell will route
through the gateway. Prefer the CLI flags, or scope the exports to one shell.

### Per-project

Commit a `.aider.conf.yml` next to your repo:

```yaml
openai-api-base: http://127.0.0.1:8090/v1
openai-api-key: dummy
model: gpt-4o
```

## 3. What you should see

Aider sends a large system prompt plus its tool/function definitions and the
file context. The gateway's behaviour on a typical session:

- **"thanks" / "ok" style turns** → `X-Proxy-Tool-Action: Stripped-ZeroTokens`.
  Aider's edit/read tools are not re-sent, which is a real saving on a turn that
  cannot possibly need them.
- **"add tests to src/main.py"** → kept, because the heuristics see a path.
- **Large `@@` search/replace blocks or verbose test output** arriving back as
  messages → anything over `TRUNCATE_THRESHOLD_CHARS` in the *history* is moved
  out of the prompt behind a `fetch_log` handle.

```bash
# watch routing decisions live
tail -f ~/.agent-gateway/logs/audit.log | python -m json.tool --json-lines
```

## 4. If Aider starts failing

1. **Check the gateway is up and pointed at the right upstream.**

   ```bash
   curl -s http://127.0.0.1:8090/health | python -m json.tool
   ```

2. **Check the upstream key is present.** `"upstream_key": false` in `/health`
   means the gateway has no credential to inject.

3. **Rule out the proxy entirely** by bypassing it for one request:

   ```bash
   curl -s -X POST http://127.0.0.1:8090/v1/chat/completions \
     -H 'content-type: application/json' \
     -H 'x-agent-gateway-bypass: true' \
     -d '{"model":"gpt-4o","messages":[{"role":"user","content":"ping"}]}'
   ```

   `X-Proxy-Route: Bypass` means the gateway forwarded the request untouched.

4. **Turn off memory injection** if you suspect stale recalled facts are
   confusing the model:

   ```bash
   MEMORY_INJECTION=0 python -m src.gateway
   ```

5. **Disable the classifier** if a slow classifier is adding latency. Routing
   then relies on heuristics only and always fails open:

   ```bash
   # simply leave CLASSIFIER_API_URL unset
   ```

## Streaming

Aider streams by default and the gateway streams straight through — it never
buffers a response to inspect it. The only added latency is the bounded escape
look-ahead (~64 characters) on turns where the tool schema was pruned.
