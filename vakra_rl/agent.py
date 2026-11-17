"""The agent loop: model proposes tool calls, the environment executes them,
results go back to the model, repeat until it answers or runs out of steps.

Every step is recorded. These trajectories are the raw material for all three
research questions: failure analysis now, SFT data next, GRPO rollouts later.
"""
from __future__ import annotations

import asyncio
import math
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field

from .llm import ChatLLM
from .mcp_env import MCPEnv

SYSTEM_PROMPT = """You are a careful enterprise data assistant with access to tools.
- Use the tools to look up facts. Never invent a value that a tool could return.
- If the question is missing a required input (for example an ID) that no tool can find, ask the user for it instead of guessing.
- When one tool's output is needed by another tool, pass the exact value you received.
- When you have enough information, reply with a short, direct final answer and no tool call."""


@dataclass
class ToolStep:
    name: str
    arguments: dict | None
    arguments_raw: str
    response: str
    is_error: bool
    parse_error: str | None
    latency_s: float


@dataclass
class TurnResult:
    query: str
    answer: str = ""
    steps: list[ToolStep] = field(default_factory=list)
    stop_reason: str = ""            # answer | step_limit | llm_error
    llm_calls: int = 0
    prompt_tokens: list[int] = field(default_factory=list)
    text_fallback_calls: int = 0     # tool calls recovered from plain text
    error: str = ""
    latency_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} characters]"


async def run_dialogue(
    llm: ChatLLM,
    env: MCPEnv,
    queries: list[str],
    additional_instructions: str = "",
    max_steps: int = 8,
    tools: list[dict] | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    max_tool_chars: int = 6000,
    keep_response_chars: int = 20000,
) -> list[TurnResult]:
    """Run one (possibly multi-turn) dialogue. Returns one TurnResult per user turn."""
    tools = tools if tools is not None else env.openai_tools()
    system = system_prompt
    if additional_instructions:
        system += "\n\nAdditional instructions you must follow:\n" + additional_instructions.strip()
    messages: list[dict] = [{"role": "system", "content": system}]
    results: list[TurnResult] = []

    for query in queries:
        turn = TurnResult(query=query)
        t_turn = time.perf_counter()
        messages.append({"role": "user", "content": query})
        for _ in range(max_steps):
            try:
                # The OpenAI client is synchronous; run it in a thread so the MCP
                # session's background reader keeps running.
                reply = await asyncio.to_thread(llm.chat, messages, tools)
            except Exception as e:
                turn.stop_reason, turn.error = "llm_error", f"{type(e).__name__}: {e}"
                break
            turn.llm_calls += 1
            if reply.prompt_tokens is not None:
                turn.prompt_tokens.append(reply.prompt_tokens)
            turn.text_fallback_calls += len(reply.tool_calls) if reply.from_text_fallback else 0
            messages.append(reply.assistant_message())

            if not reply.tool_calls:
                turn.answer, turn.stop_reason = reply.content, "answer"
                break

            for tc in reply.tool_calls:
                if tc.arguments is None:
                    text, is_err, lat = f"Error: {tc.parse_error}. Send arguments as a JSON object.", True, 0.0
                else:
                    r = await env.call(tc.name, tc.arguments)
                    text, is_err, lat = r.text, r.is_error, r.latency_s
                turn.steps.append(ToolStep(tc.name, tc.arguments, tc.arguments_raw,
                                           _truncate(text, keep_response_chars), is_err,
                                           tc.parse_error, lat))
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": _truncate(text, max_tool_chars)})
        else:
            turn.stop_reason = "step_limit"
        turn.latency_s = time.perf_counter() - t_turn
        results.append(turn)
        if turn.stop_reason == "llm_error":
            break
    return results


def to_vakra_record(uuid: str, domain: str, turns: list[TurnResult], turn_ids: list[int] | None = None,
                    status: str = "success", duration_s: float = 0.0) -> dict:
    """Convert a run into VAKRA's prediction format so IBM's evaluator can score it."""
    turn_ids = turn_ids or list(range(len(turns)))
    output = []
    for tid, t in zip(turn_ids, turns):
        output.append({
            "turn_id": tid,
            "query": t.query,
            "answer": t.answer,
            "sequence": {
                "tool_call": [{"name": s.name, "arguments": s.arguments or {}} for s in t.steps],
                "tool_response": [s.response for s in t.steps],
            },
        })
    return {"uuid": uuid, "domain": domain, "status": status,
            "error": next((t.error for t in turns if t.error), ""),
            "duration_s": round(duration_s, 3), "output": output}


# ---------------------------------------------------------------------------
# Tool shortlisting (capability 2 has ~100+ tools per domain, too many to put in
# a small model's context). This is a deliberately simple lexical baseline:
# BM25 over tool name + description. Replace it with embeddings later and
# measure the difference; that comparison is itself a result.
# ---------------------------------------------------------------------------
_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower().replace("_", " "))


def shortlist_tools(query: str, tools: list[dict], k: int) -> list[dict]:
    if k <= 0 or len(tools) <= k:
        return tools
    docs = [_tokens(t["function"]["name"] + " " + t["function"].get("description", "")) for t in tools]
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / max(n, 1)
    df = Counter(w for d in docs for w in set(d))
    q = _tokens(query)
    k1, b = 1.5, 0.75

    def score(doc: list[str]) -> float:
        tf = Counter(doc)
        s = 0.0
        for w in q:
            if w not in tf:
                continue
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(doc) / max(avgdl, 1e-9)))
        return s

    ranked = sorted(range(n), key=lambda i: score(docs[i]), reverse=True)
    return [tools[i] for i in ranked[:k]]
