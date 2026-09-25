"""
agent-context-gateway
=====================

An agent-agnostic, OpenAI-compatible gateway that sits in front of any LLM
provider and does four things well:

1. **Prunes tool schemas** on turns that provably do not need them, using
   sub-millisecond local heuristics first and an optional LLM classifier only
   for genuinely ambiguous prompts.
2. **Spills oversized context** out of the prompt into a shared-memory cache,
   replacing it with a short handle the model can retrieve on demand.
3. **Streams without buffering**, so pruning never costs the user latency, with
   a bounded look-ahead that aborts the pruned route if the model asks for tools.
4. **Learns durable facts** into a self-pruning SQLite graph that is injected
   back into later prompts.

Works with any client that speaks the OpenAI `/v1` API, plus an Anthropic
`/v1/messages` bridge for Claude Code.

Design invariants (the tests enforce these):

* Loop prevention -- the upstream may never resolve to the gateway itself.
* Fail-open -- a broken/slow/missing classifier keeps the tools, never drops them.
* Protocol transparency -- unknown JSON fields are forwarded untouched.
* Zero-thread-block streaming -- no full-response buffering, ever.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
