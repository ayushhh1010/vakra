"""A thin, explicit wrapper around one VAKRA MCP tool server.

This is the "environment" in the RL sense: the agent sends a tool call, the
environment executes it against real data and returns the result. Later, the
same object becomes the environment inside GRPO training, so keep it simple.

Two ways to start a server:

* bpo_server_params(vakra_root)
    VAKRA's BPO (recruiting analytics) domain runs as a plain local Python
    process. No Docker, no database download. Good for day-1 testing.

* docker_server_params(capability, domain)
    Everything else runs inside VAKRA's Docker containers, reached with
    `docker exec -i ... python /app/mcp_dispatch.py`, exactly as VAKRA's own
    benchmark/mcp_connection_config.yaml does it.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

CAPABILITY_CONTAINERS = {
    1: "capability_1_bi_apis",
    2: "capability_2_dashboard_apis",
    3: "capability_3_multihop_reasoning",
    4: "capability_4_multiturn",
}


def _field(obj, *names, default=None):
    """The MCP SDK renamed fields between v1 (camelCase) and v2 (snake_case). Accept both."""
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return default


@dataclass
class ToolResult:
    text: str
    is_error: bool
    latency_s: float


class MCPEnv:
    """Usage:

        async with MCPEnv(bpo_server_params("../vakra")) as env:
            print(env.tool_names)
            result = await env.call("skills_skill_analysis", {"requisition_id": "05958BR"})
    """

    def __init__(self, params: StdioServerParameters, call_timeout_s: float = 120.0,
                 server_log: str | Path | None = None):
        """server_log: file that receives the server's own log output (stderr).
        None prints it to your terminal."""
        self.params = params
        self.call_timeout_s = call_timeout_s
        self.server_log = Path(server_log) if server_log else None
        self.session: ClientSession | None = None
        self.tools: list = []
        self.tool_names: set[str] = set()
        self._stack: AsyncExitStack | None = None

    async def __aenter__(self) -> "MCPEnv":
        self._stack = AsyncExitStack()
        errlog = sys.stderr
        if self.server_log is not None:
            self.server_log.parent.mkdir(parents=True, exist_ok=True)
            errlog = self._stack.enter_context(open(self.server_log, "a", encoding="utf-8"))
        try:
            read, write = await self._stack.enter_async_context(stdio_client(self.params, errlog=errlog))
            self.session = await self._stack.enter_async_context(ClientSession(read, write))
            await self.session.initialize()
            listed = await self.session.list_tools()
        except BaseException as e:
            await self._stack.aclose()
            raise RuntimeError(
                "Could not start the MCP tool server. Run its command by hand to see the error:\n  "
                f"{self.params.command} {' '.join(self.params.args)}"
                + (f"\nServer log: {self.server_log}" if self.server_log else "")
            ) from e
        self.tools = list(listed.tools)
        self.tool_names = {t.name for t in self.tools}
        return self

    async def __aexit__(self, *exc) -> None:
        if self._stack is not None:
            await self._stack.aclose()

    def openai_tools(self, only: set[str] | None = None) -> list[dict]:
        """MCP tool specs -> OpenAI function-calling format (what Ollama/vLLM/Groq accept)."""
        specs = []
        for t in self.tools:
            if only is not None and t.name not in only:
                continue
            specs.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": (t.description or "").strip(),
                    "parameters": _field(t, "input_schema", "inputSchema",
                                         default={"type": "object", "properties": {}}),
                },
            })
        return specs

    async def call(self, name: str, arguments: dict) -> ToolResult:
        t0 = time.perf_counter()
        if name not in self.tool_names:
            return ToolResult(f"Error: unknown tool '{name}'.", True, 0.0)
        try:
            res = await asyncio.wait_for(
                self.session.call_tool(name, arguments or {}), timeout=self.call_timeout_s
            )
        except asyncio.TimeoutError:
            return ToolResult(f"Error: tool '{name}' timed out after {self.call_timeout_s}s.",
                              True, time.perf_counter() - t0)
        except Exception as e:  # server-side exceptions arrive here on some MCP versions
            return ToolResult(f"Error: {type(e).__name__}: {e}", True, time.perf_counter() - t0)

        parts = [c.text for c in (res.content or []) if getattr(c, "text", None) is not None]
        text = "\n".join(parts)
        structured = _field(res, "structured_content", "structuredContent")
        if not text and structured is not None:
            text = json.dumps(structured, ensure_ascii=False, default=str)
        return ToolResult(text, bool(_field(res, "is_error", "isError", default=False)),
                          time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Server launchers
# ---------------------------------------------------------------------------
def _child_env(extra: dict | None = None) -> dict:
    env = dict(os.environ)
    # Windows consoles default to cp1252; force UTF-8 so tool outputs with
    # non-ASCII text do not crash the server's stdout.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env.update(extra or {})
    return env


def bpo_server_params(vakra_root: str | Path) -> StdioServerParameters:
    root = Path(vakra_root).resolve()
    server = root / "environment" / "bpo" / "mcp" / "server.py"
    if not server.exists():
        raise FileNotFoundError(
            f"BPO server not found at {server}. Pass --vakra-root pointing at your clone of "
            "https://github.com/IBM/vakra"
        )
    return StdioServerParameters(
        command=sys.executable,  # the Python of your current venv
        args=[str(server)],
        cwd=str(root),
        env=_child_env({"MCP_DOMAIN": "bpo"}),
    )


def docker_server_params(capability: int, domain: str, runtime: str = "docker",
                         container: str | None = None) -> StdioServerParameters:
    container = container or CAPABILITY_CONTAINERS[capability]
    exec_env = {"MCP_DOMAIN": domain, "CAPABILITY_ID": str(capability), "MCP_DB_ROOT": "/app/db"}
    if capability == 1:
        exec_env["MCP_SERVER_TYPE"] = "router"
    if capability == 4:
        exec_env["PRELOAD_COLLECTIONS"] = "false"
    env_args: list[str] = []
    for k, v in exec_env.items():
        env_args += ["-e", f"{k}={v}"]
    return StdioServerParameters(
        command=runtime,
        args=["exec", "-i", *env_args, container, "python", "/app/mcp_dispatch.py"],
        env=_child_env(),
    )
