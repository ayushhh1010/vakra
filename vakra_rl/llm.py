"""One client for every model server that speaks the OpenAI chat API.

Ollama (laptop), vLLM (rented GPU, later also used inside GRPO), Groq and
OpenRouter (free/cheap reference models) all accept the same request, so the
agent code never changes when the model does.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field

from openai import OpenAI

PROVIDERS = {
    #  name        default base_url                         env var holding the key
    "ollama":     ("http://localhost:11434/v1",             None),
    "vllm":       ("http://localhost:8000/v1",              None),
    "groq":       ("https://api.groq.com/openai/v1",        "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1",          "OPENROUTER_API_KEY"),
}

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments_raw: str
    arguments: dict | None          # None when the model produced invalid JSON
    parse_error: str | None = None


@dataclass
class LLMReply:
    content: str
    tool_calls: list[ToolCallRequest]
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_s: float
    from_text_fallback: bool = False
    raw: dict = field(default_factory=dict)

    def assistant_message(self) -> dict:
        """The message to append to the conversation history."""
        msg: dict = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.name, "arguments": tc.arguments_raw}}
                for tc in self.tool_calls
            ]
        return msg


def _parse_args(raw: str | dict | None) -> tuple[dict | None, str | None]:
    if raw is None or raw == "":
        return {}, None
    if isinstance(raw, dict):
        return raw, None
    try:
        val = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"invalid JSON arguments: {e}"
    if not isinstance(val, dict):
        return None, "arguments must be a JSON object"
    return val, None


class ChatLLM:
    def __init__(self, provider: str, model: str, base_url: str | None = None,
                 api_key: str | None = None, temperature: float = 0.0,
                 max_tokens: int = 1024, seed: int | None = None, timeout_s: float = 600):
        if provider not in PROVIDERS:
            raise ValueError(f"provider must be one of {sorted(PROVIDERS)}")
        default_url, key_env = PROVIDERS[provider]
        key = api_key or (os.environ.get(key_env) if key_env else None) or "not-needed"
        if key_env and key == "not-needed":
            raise SystemExit(f"Set {key_env} in your environment to use provider '{provider}'.")
        self.provider, self.model = provider, model
        self.temperature, self.max_tokens, self.seed = temperature, max_tokens, seed
        self.client = OpenAI(base_url=base_url or default_url, api_key=key,
                             timeout=timeout_s, max_retries=5)

    def chat(self, messages: list[dict], tools: list[dict] | None) -> LLMReply:
        kwargs = dict(model=self.model, messages=messages, temperature=self.temperature,
                      max_tokens=self.max_tokens)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if self.seed is not None:
            kwargs["seed"] = self.seed
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(**kwargs)
        latency = time.perf_counter() - t0

        msg = resp.choices[0].message
        content = _THINK.sub("", msg.content or "").strip()
        calls: list[ToolCallRequest] = []
        for tc in msg.tool_calls or []:
            args, err = _parse_args(tc.function.arguments)
            calls.append(ToolCallRequest(tc.id or f"call_{uuid.uuid4().hex[:8]}",
                                         tc.function.name, tc.function.arguments or "{}", args, err))

        # Some servers return Qwen-style <tool_call>{...}</tool_call> as plain text
        # instead of structured tool_calls. Recover those so the run is not wasted,
        # and record that it happened (it is a formatting failure worth counting).
        fallback = False
        if not calls and "<tool_call>" in content:
            for m in _TOOL_CALL_TAG.finditer(content):
                try:
                    obj = json.loads(m.group(1))
                    name = obj.get("name", "")
                    args_raw = obj.get("arguments", {})
                    args, err = _parse_args(args_raw)
                    raw = args_raw if isinstance(args_raw, str) else json.dumps(args_raw)
                except json.JSONDecodeError as e:
                    name, raw, args, err = "", m.group(1), None, f"unparseable tool_call tag: {e}"
                calls.append(ToolCallRequest(f"call_{uuid.uuid4().hex[:8]}", name, raw, args, err))
            if calls:
                fallback = True
                content = _TOOL_CALL_TAG.sub("", content).strip()

        usage = resp.usage
        return LLMReply(
            content=content,
            tool_calls=calls,
            prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            latency_s=latency,
            from_text_fallback=fallback,
            raw={"finish_reason": resp.choices[0].finish_reason},
        )
