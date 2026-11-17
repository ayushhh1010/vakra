"""Day-1 baseline: run a model on VAKRA's BPO domain and find where it fails.

BPO (recruiting analytics) ships inside the VAKRA repo with its own tasks, so
this needs no Docker and no dataset download. It exercises the full loop:
real MCP server -> real tool execution -> model -> scored trajectory.

    python scripts/run_bpo_baseline.py --vakra-root ../vakra \
        --provider ollama --model qwen3:4b-instruct-2507-q4_K_M

Add --repeats 3 --temperature 0.7 to measure reliability (pass^k).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vakra_rl.agent import run_dialogue  # noqa: E402
from vakra_rl.llm import ChatLLM  # noqa: E402
from vakra_rl.mcp_env import MCPEnv, bpo_server_params  # noqa: E402
from vakra_rl.report import write_report  # noqa: E402
from vakra_rl.scoring import classify_failure, keyword_hits, match_tool_calls, norm_text  # noqa: E402

TASK_FILES = ["tasks", "tasks_edge_cases", "tasks_undocumented", "tasks_http_errors",
              "tasks_type_mismatch", "tasks_schema_violations"]


def load_bpo_tasks(vakra_root: Path, files: list[str]) -> list[dict]:
    tasks = []
    for name in files:
        path = vakra_root / "environment" / "bpo" / "data" / f"{name}.json"
        if not path.exists():
            print(f"  [skip] {path} not found")
            continue
        for suite in json.loads(path.read_text(encoding="utf-8")):
            for tc in suite.get("test_cases", []):
                exp = tc.get("expected_output", {})
                tasks.append({
                    "task_id": f"{name}/{tc['name']}",
                    "group": name,
                    "difficulty": tc.get("difficulty", "-"),
                    "query": tc["intent"],
                    "expected_response": exp.get("response", ""),
                    "keywords": exp.get("keywords", []),
                    "expected_calls": [{"name": c["name"], "arguments": c.get("args", {})}
                                       for c in exp.get("tool_calls", [])],
                })
    return tasks


def answer_correct(answer: str, task: dict) -> tuple[bool, list[bool]]:
    if task["keywords"]:
        hits = keyword_hits(answer, task["keywords"])
        return all(hits), hits
    exp = norm_text(task["expected_response"])
    ok = bool(exp) and exp in norm_text(answer)
    return ok, [ok]


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")[:60]


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vakra-root", default="../vakra", help="your clone of github.com/IBM/vakra")
    ap.add_argument("--provider", default="ollama", choices=["ollama", "vllm", "groq", "openrouter"])
    ap.add_argument("--model", default="qwen3:4b-instruct-2507-q4_K_M")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--task-files", nargs="+", default=TASK_FILES, choices=TASK_FILES)
    ap.add_argument("--limit", type=int, default=None, help="only the first N tasks (smoke test)")
    ap.add_argument("--repeats", type=int, default=1, help="attempts per task, for pass^k")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None, help="output dir (default runs/<time>_bpo_<model>)")
    args = ap.parse_args()

    vakra_root = Path(args.vakra_root).resolve()
    tasks = load_bpo_tasks(vakra_root, args.task_files)[: args.limit]
    if not tasks:
        sys.exit("No BPO tasks found. Check --vakra-root.")
    if args.repeats > 1 and args.temperature == 0:
        print("Note: --repeats > 1 at temperature 0 mostly repeats the same answer. "
              "Use --temperature 0.7 to measure reliability.")

    out_dir = Path(args.out or f"runs/{datetime.now():%Y%m%d-%H%M}_bpo_{slug(args.model)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    llm = ChatLLM(args.provider, args.model, base_url=args.base_url, temperature=args.temperature,
                  max_tokens=args.max_tokens, seed=args.seed)

    records: list[dict] = []
    traj_path = out_dir / "trajectories.jsonl"
    print(f"Starting BPO MCP server from {vakra_root} ...")
    async with MCPEnv(bpo_server_params(vakra_root), server_log=out_dir / "server.log") as env:
        tools = env.openai_tools()
        print(f"  {len(tools)} tools available. Running {len(tasks)} tasks x {args.repeats} "
              f"with {args.provider}:{args.model}\n")
        with open(traj_path, "w", encoding="utf-8") as tf:
            for i, task in enumerate(tasks, 1):
                for attempt in range(args.repeats):
                    t0 = time.perf_counter()
                    turn = (await run_dialogue(llm, env, [task["query"]], max_steps=args.max_steps,
                                               tools=tools))[0]
                    predicted = [{"name": s.name, "arguments": s.arguments, "is_error": s.is_error,
                                  "parse_error": s.parse_error} for s in turn.steps]
                    ok, hits = answer_correct(turn.answer, task)
                    stage = classify_failure(task["expected_calls"], predicted, ok, turn.stop_reason)
                    valid = [p for p in predicted if not p["parse_error"]]
                    m = match_tool_calls(task["expected_calls"], valid)
                    rec = {
                        "task_id": task["task_id"], "group": task["group"],
                        "difficulty": task["difficulty"], "attempt": attempt,
                        "query": task["query"], "answer": turn.answer,
                        "expected_answer": task["expected_response"],
                        "keyword_hits": hits, "success": stage == "success", "stage": stage,
                        "expected_tools": [c["name"] for c in task["expected_calls"]],
                        "called_tools": [p["name"] for p in predicted],
                        "name_recall": m["name_recall"], "full_recall": m["full_recall"],
                        "n_calls": len(predicted), "n_errors": m["n_predicted_errors"],
                        "stop_reason": turn.stop_reason, "llm_calls": turn.llm_calls,
                        "prompt_tokens_max": max(turn.prompt_tokens) if turn.prompt_tokens else None,
                        "text_fallback_calls": turn.text_fallback_calls,
                        "latency_s": round(time.perf_counter() - t0, 2),
                    }
                    records.append(rec)
                    tf.write(json.dumps({**rec, "trajectory": turn.to_dict()}, ensure_ascii=False) + "\n")
                    tf.flush()
                    mark = "OK  " if rec["success"] else "FAIL"
                    print(f"[{i:>3}/{len(tasks)}] {mark} {stage:<22} {task['task_id']:<38} "
                          f"calls={rec['n_calls']} {rec['latency_s']:.1f}s")

    config = {"environment": "VAKRA BPO (local MCP)", "provider": args.provider, "model": args.model,
              "temperature": args.temperature, "repeats": args.repeats, "max_steps": args.max_steps,
              "task_files": ",".join(args.task_files), "date": f"{datetime.now():%Y-%m-%d %H:%M}"}
    notes = ["13 of the 26 core BPO tasks expect NO tool call (the agent should ask for a missing ID "
             "or decline); those test restraint, not tool use.",
             "Success = every expected keyword group appears in the final answer."]
    s = write_report(records, out_dir, f"BPO baseline: {args.model}", config, notes)
    print(f"\nSuccess {100 * s['success_rate']:.1f}% over {s['attempts']} attempts. "
          f"Stages: {s['failure_stages']}")
    if s["max_prompt_tokens"] and args.provider == "ollama":
        print(f"Largest prompt: {s['max_prompt_tokens']} tokens. If that is close to your Ollama "
              "context length, raise OLLAMA_CONTEXT_LENGTH (see docs/SETUP_WINDOWS.md).")
    print(f"Report: {out_dir / 'report.md'}")


if __name__ == "__main__":
    asyncio.run(main())
