"""
Tool-routing decisions.

Two layers, deliberately ordered cheapest-first:

1. **Local heuristics** (`evaluate_fast_path`) -- pure regex, sub-millisecond,
   no network. They resolve the overwhelming majority of turns.
2. **An optional LLM classifier** (`Classifier.needs_tools`) -- consulted only
   when the heuristics cannot decide. Any failure (no key, no URL, timeout,
   non-200, unparseable reply) resolves to "keep the tools".

The asymmetry is intentional: dropping a tool schema the model needed is a hard
failure, while keeping one it did not need merely costs a few hundred tokens.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

import httpx

from .config import (
    DEFAULT_SELECTIVE_TOOL_LIMIT,
    CLASSIFIER_MODE_EXTERNAL_JEV,
    CLASSIFIER_MODE_HEURISTICS,
    CLASSIFIER_MODE_LOCAL_JEV,
    CLASSIFIER_MODE_LOCAL_OLLAMA,
    CLASSIFIER_MODE_UPSTREAM_REUSED,
    ESCAPE_INSTRUCTION,
    ESCAPE_TOKEN,
    Settings,
)

# ------------------------------------------------------------------------------
# Heuristics
# ------------------------------------------------------------------------------
# Reasoning models are never pruned. Their chain-of-thought is exactly where
# tool intent lives, and stripping the schema degrades them sharply.
REASONING_MODEL_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"^o[13](-mini|-preview)?", re.I),
    re.compile(r"deepseek[-/](?:ai[-/])?deepseek-r1", re.I),
    re.compile(r"deepseek-r1", re.I),
    re.compile(r"reasoner", re.I),
    re.compile(r"thinking", re.I),
    re.compile(r"-r1\b", re.I),
)

# Prompts that are clearly asking for work: keep every tool.
FAST_PATH_ACTION_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"(\/|\.py|\.sh|\.json|\.ts|\.js|\.md|\.cpp|\.go|\.rs|\.java|\.yml|\.yaml)\b"),
    re.compile(
        r"\b(git|run|test|tests|build|compile|deploy|install|cat|grep|rg|find|curl|wget"
        r"|docker|kubectl|python|pip|npm|pnpm|yarn|bash|make|sed|awk|chmod|ls)\b",
        re.I,
    ),
    re.compile(r"```[a-z]*\n"),
    re.compile(r"\b(refactor|debug|fix the|implement|migrate|revert|commit|repo|codebase)\b", re.I),
)

# Short pure pleasantries: no tool can help, so do not pay to send a schema.
FAST_PATH_CONVERSATIONAL_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(
        r"^(hi|hello|hey|yo|sup|thanks|thank you|ty|ok|okay|cool|understood|great"
        r"|good morning|good night|good evening|bye|مرحبا|اهلا|أهلا|شكرا|تمام|صباح الخير)"
        r"[.!?,…]*$",
        re.I,
    ),
)

CONVERSATIONAL_MAX_LEN = 25

# Refusing to answer is only meaningful below this length; longer prompts get a
# real decision rather than a magic string.
CLASSIFIER_SYSTEM_PROMPT = (
    "You are a strict binary classifier. Reply with ONLY a compact JSON object "
    'and no prose, of the form {"<key>": true} or {"<key>": false}.'
)

# --- local Jev GGUF, single-pass logprob protocol ----------------------------
# The model answers with exactly one letter; the verdict is read from its log
# probabilities rather than parsed from prose. A 2B model is far more reliable
# as a probability source than as a JSON emitter.
JEV_DECISION_TEMPLATE = (
    "You are a decision function. Read the state, then answer the question by "
    "choosing exactly one option.\n"
    "[State] {state}\n"
    "[Question] {question}\n"
    "[Options]\n"
    "A. Yes\n"
    "B. No\n"
    "Answer:"
)
JEV_NEEDS_TOOLS_QUESTION = (
    "Does this request require running commands, editing files, searching web, "
    "or code execution?"
)
JEV_SUPERSEDE_QUESTION = "Does the new value supersede the existing value?"

# Default verdict budget for the local GGUF. Generous on purpose: the mode's
# value is a real verdict, so the budget covers a cold KV-cache prefill on long
# prompts rather than bailing out at the first hiccup. Tunable with
# LOCAL_JEV_TIMEOUT_SECONDS; latency spikes are logged, not silently swallowed.
LOCAL_JEV_TIMEOUT_SECONDS = 0.8


def is_reasoning_model(
    model_name: str, extra_patterns: Sequence[str] = ()
) -> bool:
    """True for models whose schema must never be touched."""
    name = model_name or ""
    if any(pattern.search(name) for pattern in REASONING_MODEL_PATTERNS):
        return True
    for raw in extra_patterns:
        try:
            if re.search(raw, name, re.I):
                return True
        except re.error:
            continue
    return False


def evaluate_fast_path(prompt: str) -> Optional[str]:
    """Local routing decision.

    Returns `"strip_tools"`, `"keep_tools"`, or `None` when a judgement call is
    needed. Never raises: an unreadable prompt falls through to the classifier.
    """
    try:
        clean = (prompt or "").strip()
        if not clean:
            return None

        if len(clean) < CONVERSATIONAL_MAX_LEN:
            for pattern in FAST_PATH_CONVERSATIONAL_PATTERNS:
                if pattern.search(clean):
                    return "strip_tools"

        for pattern in FAST_PATH_ACTION_PATTERNS:
            if pattern.search(clean):
                return "keep_tools"

        return None
    except Exception:
        return None


def strip_escape_instruction(text: str) -> str:
    """Remove the escape instruction that pruning injected."""
    return (
        text.replace(ESCAPE_INSTRUCTION, "")
        .replace(ESCAPE_INSTRUCTION.strip(), "")
    )


# ------------------------------------------------------------------------------
# Selective sub-tool pruning
# ------------------------------------------------------------------------------
# Tools an agent cannot work without, even when the prompt does not name them:
# the escape-replay contract promises a shell, `fetch_log` is the gateway's own
# retrieval surface, and reading files is how an agent re-establishes context
# after a strip. Kept verbatim, always.
ALWAYS_KEEP_TOOLS = frozenset({"bash", "shell", "terminal", "fetch_log", "read_file"})

# The absolute floor for the hard cap. When the gateway offers a large schema
# and has no evidence at all about the turn, these are the only names it is
# allowed to leave behind: a shell for the escape-replay contract, file I/O for
# re-establishing context, and the gateway's own retrieval surface. The set has
# exactly DEFAULT_SELECTIVE_TOOL_LIMIT members so a capped turn can never exceed
# the configured ceiling.
CORE_TOOLS = frozenset({"bash", "read_file", "edit_file", "write_file", "fetch_log"})


class RoutedToolList(list):
    """A list of routed tool definitions with semantic routing metadata."""

    def __init__(self, tools: Sequence[Dict[str, Any]] = ()):
        super().__init__(tools)
        self.selected_names: List[str] = [
            _tool_name(t) for t in tools if isinstance(t, dict) and _tool_name(t)
        ]
        self.error: Optional[str] = None
        self.raw_response: str = ""


def _tool_name(tool: Any) -> str:
    """Extract tool name in either OpenAI or Anthropic payload shape."""
    if not isinstance(tool, dict):
        return ""
    function = tool.get("function")
    if isinstance(function, dict) and "name" in function:
        return str(function.get("name") or "")
    return str(tool.get("name") or "")


def _tool_description(tool: Any) -> str:
    """Extract a 1-line description in either OpenAI or Anthropic shape."""
    if not isinstance(tool, dict):
        return ""
    function = tool.get("function")
    if isinstance(function, dict) and "description" in function:
        desc = str(function.get("description") or "")
    else:
        desc = str(tool.get("description") or "")
    first_line = desc.strip().splitlines()[0] if desc.strip() else ""
    return first_line[:160]


def parse_jev_tool_selection(text: str, candidate_names: Set[str]) -> List[str]:
    """Parse Jev output into a list of valid candidate tool names."""
    if not text:
        return []
    cleaned = text.strip()

    # 1. Try finding a JSON array in the text
    match = re.search(r"\[.*?\]", cleaned, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                result = []
                for item in parsed:
                    name = str(item).strip().strip("'\"`")
                    if name in candidate_names and name not in result:
                        result.append(name)
                return result
        except Exception:
            pass

    # 2. Try JSON object with "tools" or "selected_tools"
    match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                tools_list = parsed.get("tools") or parsed.get("selected_tools") or []
                if isinstance(tools_list, list):
                    result = []
                    for item in tools_list:
                        name = str(item).strip().strip("'\"`")
                        if name in candidate_names and name not in result:
                            result.append(name)
                    return result
        except Exception:
            pass

    # 3. Comma-separated or tokenized output fallback
    tokens = re.findall(r"[a-zA-Z0-9_\-]+", cleaned)
    result = []
    for token in tokens:
        if token in candidate_names and token not in result:
            result.append(token)
    return result


async def route_tools_via_jev(
    prompt: str,
    tools: List[Dict[str, Any]],
    client: Optional[httpx.AsyncClient] = None,
    settings: Optional[Settings] = None,
    context: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Semantically route candidate tools via Jev without BM25 or arbitrary caps.

    Constructs a structured routing prompt for Jev with candidate tool descriptions
    and user request / execution context. Jev outputs strictly the needed tool names.
    Returns the exact tool subset in original payload order.
    """
    if not tools:
        return RoutedToolList([])

    from .config import load_settings

    cfg = settings or load_settings()

    candidate_map: Dict[str, Dict[str, Any]] = {}
    tool_lines: List[str] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        name = _tool_name(t)
        if not name:
            continue
        candidate_map[name] = t
        desc = _tool_description(t)
        tool_lines.append(f"- {name}: {desc}" if desc else f"- {name}")

    if not candidate_map:
        return RoutedToolList(tools)

    tools_summary = "\n".join(tool_lines)

    system_prompt = (
        "You are a semantic tool router. Given candidate tools and user request/context, "
        "select ONLY the tool names strictly required to fulfill this turn.\n"
        "Rules:\n"
        "1. Return ONLY a valid JSON array of tool name strings, e.g. [\"tool1\", \"tool2\"].\n"
        "2. If no tools are needed (e.g. conversational questions, explanations, creative writing), output [].\n"
        "3. Do not include markdown codeblocks, thoughts, or explanations. Only the raw JSON array."
    )

    user_content = f"Candidate Tools:\n{tools_summary}\n\n"
    if context:
        user_content += f"Conversation Context:\n{context}\n\n"
    user_content += f"User Request: {prompt}\n\nSelected Tools (JSON array):"

    payload = {
        "model": cfg.classifier_model,
        "temperature": 0.0,
        "max_tokens": 128,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }

    headers = {"Content-Type": "application/json"}
    if cfg.classifier_api_key:
        headers["Authorization"] = f"Bearer {cfg.classifier_api_key}"

    target_url = cfg.classifier_api_url
    if not target_url and getattr(cfg, "local_jev_url", None):
        target_url = cfg.local_jev_url
    if not target_url or not target_url.startswith(("http://", "https://")):
        target_url = DEFAULT_LOCAL_JEV_URL

    budget = float(getattr(cfg, "local_jev_timeout", None) or LOCAL_JEV_TIMEOUT_SECONDS)
    timeout = max(5.0, budget * 3)

    try:
        if client is not None:
            res = await client.post(target_url, json=payload, headers=headers, timeout=timeout)
        else:
            async with httpx.AsyncClient(timeout=timeout) as cl:
                res = await cl.post(target_url, json=payload, headers=headers)

        if res.status_code != 200:
            err_msg = f"HTTP {res.status_code}: {res.text[:200]}"
            sys.stderr.write(f"[JEV-ROUTER-ERROR] semantic tool routing failed: {err_msg}\n")
            out = RoutedToolList(tools)
            out.error = err_msg
            return out

        data = res.json()
        raw_text = reply_text(data)
        chosen_names = parse_jev_tool_selection(raw_text, set(candidate_map.keys()))

        selected_set = set(chosen_names)
        filtered = [t for t in tools if _tool_name(t) in selected_set]
        out = RoutedToolList(filtered)
        out.selected_names = chosen_names
        out.raw_response = raw_text
        return out

    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {exc}"
        sys.stderr.write(f"[JEV-ROUTER-ERROR] semantic tool routing failed: {err_msg}\n")
        out = RoutedToolList(tools)
        out.error = err_msg
        return out


def inject_escape_instruction(messages: List[Dict[str, Any]]) -> bool:
    """Tell the model how to ask for its tools back.

    Returns True if a brand-new system message had to be created (so the caller
    knows it may be removed wholesale on revert).
    """
    from .messages import set_text, text_of

    for message in messages:
        if message.get("role") == "system":
            set_text(message, text_of(message.get("content")) + ESCAPE_INSTRUCTION)
            return False
    messages.insert(0, {"role": "system", "content": ESCAPE_INSTRUCTION.strip()})
    return True


# ------------------------------------------------------------------------------
# LLM classifier
# ------------------------------------------------------------------------------
def reply_text(payload: Any) -> str:
    """Extract assistant text from an OpenAI-compatible reply."""
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or choices[0].get("delta") or {}
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
    return ""


def parse_classifier_bool(text: str, key: str) -> Optional[bool]:
    """Coerce a classifier reply into a boolean, or None if it is not a verdict.

    Small models are not reliably syntactically obedient, so fenced JSON, bare
    JSON, prose wrappers and bare yes/no are all accepted -- but anything
    genuinely ambiguous returns None rather than guessing.
    """
    if not text:
        return None

    cleaned = re.sub(r"```(?:json)?", "", text).strip()

    match = re.search(r"\{.*?\}", cleaned, re.S)
    if match:
        try:
            obj = json.loads(match.group(0))
        except Exception:
            obj = None
        if isinstance(obj, dict):
            for candidate in (key, "answer", "result", "value"):
                if candidate not in obj:
                    continue
                value = obj[candidate]
                if isinstance(value, bool):
                    return value
                if isinstance(value, (int, float)):
                    return float(value) >= 0.5
                if isinstance(value, str):
                    token = value.strip().lower()
                    if token in ("true", "yes", "y", "1"):
                        return True
                    if token in ("false", "no", "n", "0"):
                        return False

    lowered = cleaned.lower()
    if re.search(r"\b(yes|true)\b", lowered):
        return True
    if re.search(r"\b(no|false)\b", lowered):
        return False
    return None


def parse_jev_logprobs(data: Any) -> Optional[float]:
    """P("A") from a `logprobs=True` completion, or None if unusable.

    llama-server returns the chosen token plus its `top_logprobs` alternates.
    Both letters are looked for in that first position, including the chosen
    token itself, so a verdict is available whether the model chose A or B.
    """
    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    logprobs = first.get("logprobs") if isinstance(first, dict) else None
    if not isinstance(logprobs, dict):
        return None
    content = logprobs.get("content")
    if not isinstance(content, list) or not content:
        return None
    entry = content[0]
    if not isinstance(entry, dict):
        return None

    candidates: List[Dict[str, Any]] = [
        item for item in (entry.get("top_logprobs") or []) if isinstance(item, dict)
    ]
    candidates.append(entry)

    logprob_a: Optional[float] = None
    logprob_b: Optional[float] = None
    for item in candidates:
        token = str(item.get("token") or "").strip().strip('"').upper()
        value = item.get("logprob")
        if not isinstance(value, (int, float)):
            continue
        if token.startswith("A") and logprob_a is None:
            logprob_a = float(value)
        elif token.startswith("B") and logprob_b is None:
            logprob_b = float(value)

    if logprob_a is None and logprob_b is None:
        return None
    if logprob_b is None:
        return 1.0
    if logprob_a is None:
        return 0.0

    # Softmax over just the two options: P(A) = e^a / (e^a + e^b).
    exp_a = math.exp(logprob_a)
    exp_b = math.exp(logprob_b)
    total = exp_a + exp_b
    if total <= 0.0:
        return None
    return exp_a / total


@dataclass
class Decision:
    """Outcome of a tool-routing evaluation."""

    route: str
    tool_action: str
    reason: str = ""

    @property
    def stripped(self) -> bool:
        return self.tool_action == "Stripped-ZeroTokens"


class Classifier:
    """Optional LLM classifier with a short-lived decision cache.

    One class, four strategies, chosen by `settings.effective_classifier_mode`:

    * `heuristics`      -- never touches the network. `ask()` returns None, which
      makes every ambiguous turn fail open and keep its tools.
    * `upstream_reused` -- the endpoint and credential are the upstream's own, so
      a verdict costs nothing beyond the subscription already in use.
    * `local_ollama`    -- a local runner; same OpenAI payload shape, no internet.
    * `local_jev`       -- the Jev-Style Qwen3.5-2B GGUF on llama-server. One
      forward pass, one letter, read from `logprobs` well inside the budget.
    * `external_jev`    -- a dedicated endpoint (OpenRouter/OpenCode), where the
      `blueprint` protocol is also available.

    Construct once per process. All network access is funnelled through `_post`
    so tests can substitute a transport without patching httpx globally.
    """

    def __init__(self, settings: Settings, cache_ttl: float = 300.0) -> None:
        self.settings = settings
        self.mode = settings.effective_classifier_mode
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[float, bool]] = {}
        self.calls = 0

    # --- cache -------------------------------------------------------------
    def _cache_key(self, kind: str, state: str) -> str:
        raw = f"{kind}|{self.settings.classifier_model}|{state}"
        return hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()

    def _cache_get(self, key: str) -> Optional[bool]:
        entry = self._cache.get(key)
        if not entry:
            return None
        stamp, verdict = entry
        if time.time() - stamp > self.cache_ttl:
            self._cache.pop(key, None)
            return None
        return verdict

    def _cache_put(self, key: str, verdict: bool) -> None:
        if len(self._cache) > 4096:
            self._cache.clear()
        self._cache[key] = (time.time(), verdict)

    # --- transport seam ----------------------------------------------------
    async def _post(
        self, url: str, payload: Dict[str, Any], timeout: Optional[float] = None
    ) -> Optional[Any]:
        """POST JSON to the classifier. Returns None on any failure."""
        headers = {"Content-Type": "application/json"}
        if self.settings.classifier_api_key:
            headers["Authorization"] = f"Bearer {self.settings.classifier_api_key}"
        budget = self.settings.classifier_timeout if timeout is None else timeout
        # When CLASSIFIER_MODE=local_jev, prioritize getting the actual verdict
        # rather than bailing out aggressively: give the socket generous tolerance
        # (up to 3.5s) so latency spikes on long context or cold slots complete and
        # return a real verdict instead of prematurely dropping into fail-open.
        if self.mode == CLASSIFIER_MODE_LOCAL_JEV:
            client_timeout = max(10.0, budget * 5)
        else:
            client_timeout = budget
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=client_timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                if response.status_code != 200:
                    return None
                data = response.json()
                elapsed_ms = (time.perf_counter() - started) * 1000
                budget_ms = budget * 1000
                # A slow verdict is still a verdict: log the spike and keep the
                # answer rather than discarding it for being late.
                if elapsed_ms > budget_ms:
                    sys.stderr.write(
                        f"[classifier] latency spike: {elapsed_ms:.0f}ms "
                        f"(budget {budget_ms:.0f}ms) -- keeping the result\n"
                    )
                return data
        except httpx.TimeoutException:
            sys.stderr.write(
                f"[classifier] verdict timed out after {client_timeout:.2f}s "
                "-- failing open\n"
            )
            return None
        except Exception:
            return None

    # --- core --------------------------------------------------------------
    def _build_payload(self, instruction: str, state: str, key: str) -> Dict[str, Any]:
        if self.settings.classifier_protocol == "blueprint" and self.mode == CLASSIFIER_MODE_EXTERNAL_JEV:
            # The original spec's custom shape. No standard endpoint returns it,
            # which is why `openai` is the default; it is kept for compatible
            # self-hosted classifiers.
            return {
                "model": self.settings.classifier_model,
                "state": state,
                "questions": {key: {"type": "no", "instruction": instruction}},
            }

        return {
            "model": self.settings.classifier_model,
            "temperature": 0,
            "max_tokens": 32,
            # Explicit, so a server that streams by default cannot turn a 32-token
            # verdict into an SSE body the parser would then have to unwrap.
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": CLASSIFIER_SYSTEM_PROMPT.replace("<key>", key),
                },
                {
                    "role": "user",
                    "content": (
                        f"{instruction}\n\nInput:\n{state}\n\n"
                        f'Reply as {{"{key}": true}} or {{"{key}": false}}.'
                    ),
                },
            ],
        }

    def _build_jev_payload(self, state: str, question: str) -> Dict[str, Any]:
        """One-letter decision payload for the local Jev GGUF."""
        # Bound state to avoid huge KV cache prefill overhead on long context prompts
        clean_state = (state or "").strip()
        if len(clean_state) > 2000:
            clean_state = clean_state[:1200] + "\n...[truncated]...\n" + clean_state[-600:]

        payload: Dict[str, Any] = {
            "model": self.settings.classifier_model,
            "temperature": 0,
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": 10,
            "stream": False,
            "cache_prompt": True,
            "messages": [
                {
                    "role": "user",
                    "content": JEV_DECISION_TEMPLATE.format(
                        state=clean_state, question=question
                    ),
                }
            ],
        }
        # A base URL pointing at llama-server's *raw* completion endpoint
        # (…/v1/completions, i.e. not the chat path) receives the decision
        # prompt verbatim: no chat template can inject its own preamble and
        # shift the answer distribution, and `cache_prompt` keeps the KV cache
        # warm across the repeated verdicts of a session.
        if self.settings.classifier_api_url.rstrip("/").endswith(
            "/v1/completions"
        ):
            payload["prompt"] = payload["messages"][0]["content"]
            payload["cache_prompt"] = True
            del payload["messages"]
        return payload

    async def classify_via_local_jev(
        self,
        prompt: str,
        question: str = JEV_NEEDS_TOOLS_QUESTION,
        threshold: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Single-pass Yes/No read of the Jev GGUF's log probabilities.

        Returns `{"needs_tools": bool, "probability": float}` or None when the
        endpoint is unreachable, times out, or returns an unusable shape. Never
        raises: a local classifier that is down must fail open.
        """
        limit = (
            self.settings.classifier_needs_tools_threshold
            if threshold is None
            else threshold
        )
        budget = float(
            getattr(self.settings, "local_jev_timeout", None)
            or LOCAL_JEV_TIMEOUT_SECONDS
        )
        try:
            data = await self._post(
                self.settings.classifier_api_url,
                self._build_jev_payload(prompt, question),
                budget,
            )
        except Exception:
            return None
        if data is None:
            return None
        probability = parse_jev_logprobs(data)
        if probability is None:
            return None
        return {"needs_tools": probability >= limit, "probability": probability}

    async def ask(
        self, instruction: str, state: str, key: str, threshold: float
    ) -> Optional[bool]:
        """Ask the classifier a yes/no question. None means "could not ask"."""
        settings = self.settings
        if self.mode == CLASSIFIER_MODE_HEURISTICS or not settings.classifier_enabled:
            # Belt and braces: `heuristics` must issue zero network calls even if an
            # endpoint somehow leaked into the configuration.
            return None

        cache_key = self._cache_key(key, state)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        self.calls += 1
        if self.mode == CLASSIFIER_MODE_LOCAL_JEV:
            question = (
                JEV_NEEDS_TOOLS_QUESTION
                if key == "needs_tools"
                else JEV_SUPERSEDE_QUESTION
            )
            result = await self.classify_via_local_jev(state, question, threshold)
            if result is None:
                return None
            verdict = bool(result["needs_tools"])
            self._cache_put(cache_key, verdict)
            return verdict

        data = await self._post(
            settings.classifier_api_url, self._build_payload(instruction, state, key)
        )
        if data is None:
            return None

        # Blueprint shape first (a calibrated probability), then the OpenAI
        # shape (a boolean in the assistant text).
        verdict = self._interpret_probability(data, key, threshold)
        if verdict is None:
            verdict = parse_classifier_bool(reply_text(data), key)
        if verdict is not None:
            self._cache_put(cache_key, verdict)
        return verdict

    @staticmethod
    def _interpret_probability(
        data: Any, key: str, threshold: float
    ) -> Optional[bool]:
        """Read the `blueprint` reply shape, or None if it is not that shape.

        Split out from `ask` so the calibrated-probability protocol can be
        exercised without a network round trip.
        """
        answers = data.get("answers") if isinstance(data, dict) else None
        if not isinstance(answers, dict):
            return None
        entry = answers.get(key)
        if not isinstance(entry, dict):
            return None
        probability = entry.get("probability")
        if isinstance(probability, (int, float)):
            return float(probability) >= threshold
        return None

    # --- domain questions --------------------------------------------------
    async def needs_tools(self, prompt: str) -> Optional[bool]:
        """Does this turn strictly require tools? None if it could not be asked."""
        return await self.ask(
            "Does this query strictly require running terminal commands, editing "
            "files, searching the web, or executing code?",
            prompt,
            "needs_tools",
            self.settings.classifier_needs_tools_threshold,
        )

    async def value_supersedes(
        self, source: str, predicate: str, old_target: str, new_target: str
    ) -> Optional[bool]:
        """Does a new value invalidate a previously stored one?"""
        return await self.ask(
            "Does the proposed new value directly update, replace, or invalidate "
            "the existing value?",
            (
                f"Subject: {source}\nProperty: {predicate}\n"
                f"Existing Value: {old_target}\nProposed New Value: {new_target}"
            ),
            "is_superseded",
            self.settings.classifier_supersede_threshold,
        )

    async def route_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        client: Optional[httpx.AsyncClient] = None,
        context: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantically route candidate tools via Jev."""
        return await route_tools_via_jev(
            prompt, tools, client=client, settings=self.settings, context=context
        )


# ------------------------------------------------------------------------------
# Routing policy
# ------------------------------------------------------------------------------
async def decide_route(
    settings: Settings,
    classifier: Classifier,
    model: str,
    prompt: str,
    has_tools: bool,
    is_tool_turn: bool,
) -> Decision:
    """Decide how much of the tool schema this turn actually needs.

    Never raises: any unexpected error degrades to forwarding the request intact.
    """
    if not has_tools:
        return Decision("Default", "Unmodified", "no_tools_offered")
    if is_tool_turn:
        # Mid tool-loop: the model is waiting on a result, not choosing tools.
        return Decision("ToolLoop", "Unmodified", "mid_tool_loop")

    try:
        if is_reasoning_model(model, settings.extra_reasoning_patterns):
            return Decision("Reasoning-Passthrough", "Retained-Full", "reasoning_model")

        fast_path = evaluate_fast_path(prompt)
        if fast_path == "strip_tools":
            return Decision("FastPath-Strip", "Stripped-ZeroTokens", "greeting")
        if fast_path == "keep_tools":
            return Decision("FastPath-Keep", "Retained-Heuristic", "action_prompt")

        verdict = await classifier.needs_tools(prompt)
        if verdict is None:
            return Decision("Classifier-FailOpen", "Retained-Classifier", "fail_open")
        if verdict:
            return Decision("Classifier-Keep", "Retained-Classifier", "needs_tools")
        return Decision("Classifier-Strip", "Stripped-ZeroTokens", "no_tools")
    except Exception as exc:  # pragma: no cover - defensive
        return Decision("FailOpen", "Unmodified", type(exc).__name__)


# ------------------------------------------------------------------------------
# Process-wide classifier
# ------------------------------------------------------------------------------
_CLASSIFIER: Optional[Classifier] = None
_CLASSIFIER_KEY: Optional[tuple] = None


def _classifier_key(settings: Settings) -> tuple:
    return (
        settings.effective_classifier_mode,
        settings.classifier_api_url,
        settings.classifier_api_key,
        settings.classifier_model,
        settings.classifier_protocol,
        settings.classifier_timeout,
        settings.classifier_needs_tools_threshold,
        settings.classifier_supersede_threshold,
    )


def configure(classifier: Classifier) -> Classifier:
    global _CLASSIFIER, _CLASSIFIER_KEY
    _CLASSIFIER = classifier
    _CLASSIFIER_KEY = _classifier_key(classifier.settings)
    return classifier


def get_classifier(settings: Optional[Settings] = None) -> Classifier:
    """Return the process-wide classifier for the active mode.

    The mode is resolved from configuration, so switching profiles changes the
    strategy on the next call without restarting anything. Keyed on the
    classifier-relevant settings (mode included): a cached instance built from a
    different configuration would silently route with the wrong policy.
    """
    global _CLASSIFIER, _CLASSIFIER_KEY
    if settings is None:
        from .config import load_settings

        settings = load_settings()
    key = _classifier_key(settings)
    if _CLASSIFIER is None or _CLASSIFIER_KEY != key:
        _CLASSIFIER = Classifier(settings)
        _CLASSIFIER_KEY = key
    return _CLASSIFIER


__all__ = [
    "ALWAYS_KEEP_TOOLS",
    "CORE_TOOLS",
    "Decision",
    "Classifier",
    "RoutedToolList",
    "route_tools_via_jev",
    "parse_jev_tool_selection",
    "_tool_name",
    "_tool_description",
    "CLASSIFIER_MODE_EXTERNAL_JEV",
    "CLASSIFIER_MODE_HEURISTICS",
    "CLASSIFIER_MODE_LOCAL_JEV",
    "CLASSIFIER_MODE_LOCAL_OLLAMA",
    "CLASSIFIER_MODE_UPSTREAM_REUSED",
    "JEV_DECISION_TEMPLATE",
    "JEV_NEEDS_TOOLS_QUESTION",
    "JEV_SUPERSEDE_QUESTION",
    "LOCAL_JEV_TIMEOUT_SECONDS",
    "ESCAPE_INSTRUCTION",
    "ESCAPE_TOKEN",
    "decide_route",
    "evaluate_fast_path",
    "get_classifier",
    "inject_escape_instruction",
    "is_reasoning_model",
    "parse_classifier_bool",
    "parse_jev_logprobs",
    "reply_text",
    "strip_escape_instruction",
]
