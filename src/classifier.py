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
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

import httpx

from .config import ESCAPE_INSTRUCTION, ESCAPE_TOKEN, Settings

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

    Construct once per process. All network access is funnelled through `_post`
    so tests can substitute a transport without patching httpx globally.
    """

    def __init__(self, settings: Settings, cache_ttl: float = 300.0) -> None:
        self.settings = settings
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
    async def _post(self, url: str, payload: Dict[str, Any]) -> Optional[Any]:
        """POST JSON to the classifier. Returns None on any failure."""
        headers = {"Content-Type": "application/json"}
        if self.settings.classifier_api_key:
            headers["Authorization"] = f"Bearer {self.settings.classifier_api_key}"
        try:
            async with httpx.AsyncClient(timeout=self.settings.classifier_timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                if response.status_code != 200:
                    return None
                return response.json()
        except Exception:
            return None

    # --- core --------------------------------------------------------------
    def _build_payload(self, instruction: str, state: str, key: str) -> Dict[str, Any]:
        if self.settings.classifier_protocol == "blueprint":
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

    async def ask(
        self, instruction: str, state: str, key: str, threshold: float
    ) -> Optional[bool]:
        """Ask the classifier a yes/no question. None means "could not ask"."""
        settings = self.settings
        if not settings.classifier_enabled:
            return None

        cache_key = self._cache_key(key, state)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        self.calls += 1
        data = await self._post(
            settings.classifier_api_url, self._build_payload(instruction, state, key)
        )
        if data is None:
            return None

        # Blueprint shape: a calibrated probability.
        answers = data.get("answers") if isinstance(data, dict) else None
        if isinstance(answers, dict):
            entry = answers.get(key)
            if isinstance(entry, dict):
                probability = entry.get("probability")
                if isinstance(probability, (int, float)):
                    verdict = float(probability) >= threshold
                    self._cache_put(cache_key, verdict)
                    return verdict

        # OpenAI shape: a boolean in the assistant text.
        verdict = parse_classifier_bool(reply_text(data), key)
        if verdict is not None:
            self._cache_put(cache_key, verdict)
        return verdict

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
    """Return the process-wide classifier, rebuilding it if the config changed.

    Keyed on the classifier-relevant settings: a cached instance built from a
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
    "Decision",
    "Classifier",
    "ESCAPE_INSTRUCTION",
    "ESCAPE_TOKEN",
    "decide_route",
    "evaluate_fast_path",
    "get_classifier",
    "inject_escape_instruction",
    "is_reasoning_model",
    "parse_classifier_bool",
    "reply_text",
    "strip_escape_instruction",
]
