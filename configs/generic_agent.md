# Any OpenAI-compatible agent → agent-context-gateway

If your framework lets you set an OpenAI `base_url`, it works here. No SDK
patches, no monkey-patching, no agent-specific code.

## 1. Start the gateway

```bash
cd agent-context-gateway
cp .env.example .env
$EDITOR .env      # set UPSTREAM_BASE_URL + UPSTREAM_API_KEY
python -m src.gateway
```

## 2. Point your framework at it

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8090/v1",
    api_key="dummy",          # the gateway injects the real upstream key
)

stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "hello"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

### LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    base_url="http://127.0.0.1:8090/v1",
    api_key="dummy",
    streaming=True,
)
```

### CrewAI

```python
import os

os.environ["OPENAI_API_BASE"] = "http://127.0.0.1:8090/v1"
os.environ["OPENAI_API_KEY"] = "dummy"

from crewai import Agent

researcher = Agent(role="Researcher", goal="Find facts", backstory="...")
```

### Anything reading `OPENAI_BASE_URL` / `OPENAI_API_BASE`

```bash
export OPENAI_API_BASE=http://127.0.0.1:8090/v1
export OPENAI_API_KEY=dummy
```

Scope this to one shell. Setting it in your profile routes *every*
OpenAI-compatible tool you run through the gateway — which is often what you
want, but should be a deliberate choice.

### Hermes Agent

Add a provider entry to `~/.hermes/config.yaml` and select it; do not repoint
your existing default until you have verified the gateway works.

```yaml
custom_providers:
  - name: context-gateway
    base_url: http://127.0.0.1:8090/v1
    api_mode: chat_completions
    model: gpt-4o
    models:
      gpt-4o: {}
    models_discovered: true
```

The gateway also accepts the legacy `HERMES_PROXY_HOST` / `HERMES_PROXY_PORT` /
`REAL_UPSTREAM_BASE_URL` / `JEV_*` variable names, so an existing Hermes-oriented
`.env` keeps working during a migration.

## 3. HTTP / curl

```bash
curl -N -X POST http://127.0.0.1:8090/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
        "model": "gpt-4o",
        "stream": true,
        "messages": [{"role": "user", "content": "run the tests"}],
        "tools": [{
          "type": "function",
          "function": {"name": "bash", "parameters": {"type": "object", "properties": {}}}
        }]
      }'
```

## 4. Routes

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness + redacted config snapshot |
| `GET` | `/v1/models` | Passthrough to the upstream model list |
| `POST` | `/v1/chat/completions` | OpenAI-compatible (use this one) |
| `POST` | `/v1/messages` | Anthropic-compatible (Claude Code) |
| `POST` | `/v1/messages/count_tokens` | Anthropic token estimate |

## 5. Header-based control

| Header | Effect |
| --- | --- |
| `x-agent-gateway-bypass: true` | Forward the request completely untouched. Use this to rule the proxy out when debugging. |
| `x-session-id: <id>` | Group turns into one session for memory graduation (`N ≥ 3` distinct sessions promotes a fact to permanent). |

Response headers are the observable surface:

| Header | Meaning |
| --- | --- |
| `X-Proxy-Route` | Which policy fired, e.g. `FastPath-Strip`, `Classifier-Keep`, `Reasoning-Passthrough`, `…->EarlyEscapeAbort` |
| `X-Proxy-Tool-Action` | `Stripped-ZeroTokens` \| `Retained-Heuristic` \| `Retained-Full` \| `Reverted-To-Full` |
| `X-Proxy-Latency-MS` | Time to first byte from the upstream |
| `X-Proxy-Spill-Count` | Large historical tool results moved out of the prompt |
| `X-Proxy-Intercepted` | Synthetic `fetch_log` calls answered from cache |
| `X-Proxy-Request-Id` | Correlates with the audit log line |

## 6. Agent frameworks that inject their own tools

Frameworks like CrewAI, AutoGen and OpenAI Assistants send a fixed tool list on
every turn. Pruning helps most on conversational turns and does nothing harmful
otherwise, because:

- forced tool choice is never pruned,
- reasoning models are never pruned,
- the model can always ask for its tools back with `[ESCAPE_NEED_TOOLS]`,
- any classifier failure keeps the tools.

If you would rather never prune, disable the classifier and make your prompts
look like work — or simply send `x-agent-gateway-bypass: true`.
