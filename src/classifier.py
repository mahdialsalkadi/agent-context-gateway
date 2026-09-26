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

_TOKEN_RE = re.compile(r"[a-z0-9_]{2,}")

# A prompt with no English lexical overlap -- Arabic, or anything the BM25
# tokenizer cannot read -- used to score 0.0 on every tool, which hit the
# historical "no evidence, so keep everything" fallback and leaked the whole
# catalog back to the model. This table gives such prompts a cheap,
# deterministic way to name the two or three tools they actually need.
# Matching is a plain substring test on the lowered prompt, so Arabic morphology
# ("وشغل" contains "شغل") still fires.
INTENT_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "bash": ("شغل", "نفذ", "اوامر", "تيرمينال", "كوماند", "run", "exec", "terminal", "test", "build"),
    "read_file": ("اقرا", "اقرأ", "افحص", "شوف", "ملف", "كود", "read", "inspect", "view", "open", "show"),
    "edit_file": ("عدل", "اكتب", "غير", "صلح", "write", "edit", "modify", "patch", "fix"),
    "write_file": ("اكتب", "write", "create", "save"),
    "web_search": ("ابحث", "دور", "انترنت", "جوجل", "search", "lookup", "google", "find"),
}

# One intent hit is worth more than any single BM25 term, because it is a direct
# statement about which tool category the turn belongs to.
INTENT_BOOST = 5.0

# The blueprint's name for the same table, kept as an alias so both names in the
# docs (`INTENT_KEYWORDS`, `ARABIC_INTENT_MAP`) resolve to one source of truth.
ARABIC_INTENT_MAP = INTENT_KEYWORDS


def intent_boosts(prompt: str) -> Dict[str, float]:
    """Per-tool score boosts for intent the BM25 tokenizer cannot see.

    Pure and allocation-light: a few dozen substring checks on the hot path.
    """
    text = (prompt or "").lower()
    if not text:
        return {}
    return {
        name: INTENT_BOOST
        for name, keywords in INTENT_KEYWORDS.items()
        if any(keyword in text for keyword in keywords)
    }

# Words too common to carry signal about which tool a turn needs. Without this,
# "search the web for tutorials" ranks `screenshot` because its description
# contains "the".
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in into is it its of on or that "
    "the their them then there these they this to was were will with without you your"
    .split()
)

# Sublinear IDF weights rare terms up; a term in every document is worthless for
# ranking. The +1 inside the log keeps the weight positive and defined for a term
# that somehow appears in every tool.
def _bm25_scores(query_tokens: Sequence[str], documents: Sequence[Sequence[str]]) -> List[float]:
    """BM25 ranking of `documents` against `query_tokens`, k1=1.2, b=0.75.

    Implemented inline rather than pulled in as a dependency: the whole ranker
    must stay well under 2ms for a 30-tool payload, and BM25 over a few dozen
    short documents is a handful of array passes.
    """
    total_docs = len(documents)
    if total_docs == 0 or not query_tokens:
        return [0.0] * total_docs

    doc_tokens = [list(document) for document in documents]
    doc_counts = [len(tokens) or 1 for tokens in doc_tokens]
    average_length = sum(doc_counts) / total_docs

    document_frequency: Dict[str, int] = {}
    for tokens in doc_tokens:
        for term in set(tokens):
            document_frequency[term] = document_frequency.get(term, 0) + 1

    scores: List[float] = []
    for tokens, count in zip(doc_tokens, doc_counts):
        score = 0.0
        for term in query_tokens:
            frequency = tokens.count(term)
            if not frequency:
                continue
            df = document_frequency.get(term, 0)
            idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
            tf_component = (frequency * 2.2) / (
                frequency + 1.2 * (1.0 - 0.75 + 0.75 * count / average_length)
            )
            score += idf * tf_component
        scores.append(score)
    return scores


def _tool_text(tool: Dict[str, Any]) -> Tuple[str, str]:
    """(name, searchable text) for either the OpenAI or Anthropic tool shape."""
    if not isinstance(tool, dict):
        return "", ""
    function = tool.get("function")
    if isinstance(function, dict):
        name = str(function.get("name") or "")
        description = str(function.get("description") or "")
        return name, f"{name} {description}"
    # Anthropic shape: the fields sit directly on the tool object.
    name = str(tool.get("name") or "")
    description = str(tool.get("description") or "")
    return name, f"{name} {description}"


def _hard_cap(
    tools: List[Dict[str, Any]],
    scores: List[float],
    boosts: Dict[str, float],
    keep_limit: int,
) -> List[Dict[str, Any]]:
    """An absolute ceiling on a large schema: never the catalog, never > limit.

    Core I/O tools are served first; relevance fills whatever is left. When the
    prompt produced no evidence at all -- every BM25 score 0.0 and no intent
    match -- only the core tools survive, instead of the historical
    "keep everything" fallback that leaked 40+ schemas to the model.
    """
    core_indices = [
        index
        for index, tool in enumerate(tools)
        if _tool_text(tool)[0].lower() in CORE_TOOLS
    ]
    combined = [
        score + boosts.get(_tool_text(tool)[0].lower(), 0.0)
        for tool, score in zip(tools, scores)
    ]
    positive = [index for index, value in enumerate(combined) if value > 0.0]
    if not positive:
        return [tools[index] for index in core_indices[:keep_limit]]

    selected = sorted(positive, key=lambda index: combined[index], reverse=True)[
        :keep_limit
    ]
    # Guarantee the core set inside the ceiling: displace the weakest non-core
    # pick rather than overflowing the limit.
    core_selected = set(core_indices)
    for index in core_indices:
        if index in selected:
            continue
        if len(selected) < keep_limit:
            selected.append(index)
            continue
        for candidate in reversed(selected):
            if candidate not in core_selected:
                selected[selected.index(candidate)] = index
                break

    keep = set(selected)
    return [tool for index, tool in enumerate(tools) if index in keep]


def rank_tools(
    prompt: str,
    tools: List[Dict[str, Any]],
    keep_limit: int = DEFAULT_SELECTIVE_TOOL_LIMIT,
    hard_cap: bool = False,
) -> List[Dict[str, Any]]:
    """The tools worth sending, ranked by relevance to the prompt.

    Mission-critical I/O tools are always preserved; everything else competes on
    a BM25 score of the prompt against each tool's name and description. Pure
    and local: no model call, well under 2ms for realistic payload sizes.

    `hard_cap` switches from the conservative default (when there is no evidence
    the whole schema is returned) to the gateway's absolute ceiling: a large
    schema is never returned whole, multilingual intent is boosted, and the
    result is always `<= keep_limit`. The gateway path passes `hard_cap=True`.
    """
    try:
        keep_limit = max(1, int(keep_limit))
        query_tokens = [
            token
            for token in _TOKEN_RE.findall((prompt or "").lower())
            if token not in STOPWORDS
        ]
        if not query_tokens:
            if hard_cap:
                return _hard_cap(tools, [0.0] * len(tools), intent_boosts(prompt), keep_limit)
            return list(tools)

        documents = [_TOKEN_RE.findall(_tool_text(tool)[1].lower()) for tool in tools]
        scores = _bm25_scores(query_tokens, documents)

        if hard_cap:
            return _hard_cap(tools, scores, intent_boosts(prompt), keep_limit)

        # Only shrink the schema when there is actual evidence. A prompt that
        # matches no tool at all says nothing about which tools matter, so the
        # safe move is to keep everything rather than gamble on an arbitrary cut.
        positive = [index for index, score in enumerate(scores) if score > 0.0]
        if not positive:
            return list(tools)

        ranked = sorted(positive, key=lambda i: scores[i], reverse=True)
        kept_indices = set(ranked[:keep_limit])

        # Always-keep tools come along even when they did not match the prompt.
        for index, tool in enumerate(tools):
            if _tool_text(tool)[0].lower() in ALWAYS_KEEP_TOOLS:
                kept_indices.add(index)

        # Preserve payload order: schemas are compared across turns by some
        # providers for prompt-cache reuse, and reordering them defeats it.
        return [tool for index, tool in enumerate(tools) if index in kept_indices]
    except Exception:
        # Ranking is an optimisation. Any surprise means keep everything.
        return list(tools)


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
        effective_timeout = self.settings.classifier_timeout if timeout is None else timeout
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=effective_timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                if response.status_code != 200:
                    return None
                data = response.json()
                elapsed_ms = (time.perf_counter() - started) * 1000
                budget_ms = effective_timeout * 1000
                # A slow verdict is still a verdict: log the spike and keep the
                # answer rather than discarding it for being late.
                if elapsed_ms > max(500.0, budget_ms / 2):
                    sys.stderr.write(
                        f"[classifier] slow verdict: {elapsed_ms:.0f}ms "
                        f"(budget {budget_ms:.0f}ms) -- keeping the result\n"
                    )
                return data
        except httpx.TimeoutException:
            sys.stderr.write(
                f"[classifier] verdict timed out after {effective_timeout:.2f}s "
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
                        state=state, question=question
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
    "ARABIC_INTENT_MAP",
    "CORE_TOOLS",
    "INTENT_BOOST",
    "INTENT_KEYWORDS",
    "Decision",
    "Classifier",
    "CLASSIFIER_MODE_EXTERNAL_JEV",
    "CLASSIFIER_MODE_HEURISTICS",
    "CLASSIFIER_MODE_LOCAL_JEV",
    "CLASSIFIER_MODE_LOCAL_OLLAMA",
    "CLASSIFIER_MODE_UPSTREAM_REUSED",
    "JEV_DECISION_TEMPLATE",
    "JEV_NEEDS_TOOLS_QUESTION",
    "JEV_SUPERSEDE_QUESTION",
    "LOCAL_JEV_TIMEOUT_SECONDS",
    "rank_tools",
    "intent_boosts",
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
