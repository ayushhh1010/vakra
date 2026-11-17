"""Day-2 baseline: run a model on VAKRA's own tasks through the Docker tool servers.

Needs: VAKRA containers running (docker compose up -d in the VAKRA repo), the
databases for the domains you pick, and the train split downloaded.

    python scripts/run_vakra_baseline.py --capability 3 --domain authors --limit 10

Writes, under runs/<time>_cap<N>_<model>/:
    prediction/<domain>.json  VAKRA's submission format -> feed to IBM's evaluator
    trajectories.jsonl        every step, for your own failure analysis
    report.md / summary.json  proxy scores against the train ground truth

Proxy scores here compare against ground truth with simple rules. The official
number comes from IBM's evaluator (LLM judge); see docs/SETUP_WINDOWS.md.
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

from mcp import StdioServerParameters  # noqa: E402

from vakra_rl.agent import run_dialogue, shortlist_tools, to_vakra_record  # noqa: E402
from vakra_rl.llm import ChatLLM  # noqa: E402
from vakra_rl.mcp_env import MCPEnv, _child_env, docker_server_params  # noqa: E402
from vakra_rl.report import write_report  # noqa: E402
from vakra_rl.scoring import answer_contains_gt, classify_failure, match_tool_calls  # noqa: E402
from vakra_rl.vakra_io import CAPABILITY_DIRS, final_turn, input_queries, load_json, turn_tool_calls  # noqa: E402


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")[:60]


async def run_domain(args, llm, cap: int, domain: str, out_dir: Path, records: list[dict]) -> None:
    cap_dir = Path(args.data_root) / args.split / CAPABILITY_DIRS[cap]
    inputs = load_json(cap_dir / "input" / f"{domain}.json")[: args.limit]
    gt_path = cap_dir / "output" / f"{domain}.json"
    gt = {r["uuid"]: r for r in load_json(gt_path)} if gt_path.exists() else {}

    if args.server_python:  # a local Python MCP server instead of Docker (testing, BPO)
        params = StdioServerParameters(command=sys.executable, args=[args.server_python],
                                       env=_child_env({"MCP_DOMAIN": domain}))
    else:
        params = docker_server_params(cap, domain, runtime=args.runtime)

    predictions = []
    async with MCPEnv(params, server_log=out_dir / "server.log") as env:
        all_tools = env.openai_tools()
        print(f"\n== cap {cap} / {domain}: {len(inputs)} dialogues, {len(all_tools)} tools"
              + (f" (shortlisting to {args.top_k_tools})" if 0 < args.top_k_tools < len(all_tools) else ""))
        for i, rec in enumerate(inputs, 1):
            queries = input_queries(rec)
            tools = shortlist_tools(" ".join(queries), all_tools, args.top_k_tools)
            t0 = time.perf_counter()
            turns = await run_dialogue(llm, env, queries,
                                       additional_instructions=rec.get("additional_instructions") or "",
                                       max_steps=args.max_steps, tools=tools)
            dur = time.perf_counter() - t0
            status = "error" if any(t.stop_reason == "llm_error" for t in turns) else "success"
            predictions.append(to_vakra_record(rec["uuid"], domain, turns, status=status, duration_s=dur))

            last = turns[-1]
            g = gt.get(rec["uuid"])
            g_final = final_turn(g) if g else None
            expected = turn_tool_calls(g_final) if g_final else []
            predicted = [{"name": s.name, "arguments": s.arguments, "is_error": s.is_error,
                          "parse_error": s.parse_error} for s in last.steps]
            ok = answer_contains_gt(last.answer, g_final.get("answer")) if g_final else False
            stage = classify_failure(expected, predicted, ok, last.stop_reason) if g_final else "no_ground_truth"
            m = match_tool_calls(expected, [p for p in predicted if not p["parse_error"]])
            shortlisted = {t["function"]["name"] for t in tools}
            r = {
                "task_id": f"{domain}/{rec['uuid']}", "group": domain, "attempt": 0,
                "query": queries[-1] if queries else "", "answer": last.answer,
                "expected_answer": g_final.get("answer") if g_final else None,
                "success": stage == "success", "stage": stage,
                "expected_tools": [c["name"] for c in expected],
                "called_tools": [p["name"] for p in predicted],
                "gt_tools_in_shortlist": (sum(c["name"] in shortlisted for c in expected) / len(expected)
                                          if expected else None),
                "name_recall": m["name_recall"], "full_recall": m["full_recall"],
                "n_calls": len(predicted), "n_errors": m["n_predicted_errors"],
                "stop_reason": last.stop_reason, "n_turns": len(queries),
                "prompt_tokens_max": max((p for t in turns for p in t.prompt_tokens), default=None),
                "text_fallback_calls": sum(t.text_fallback_calls for t in turns),
                "latency_s": round(dur, 2),
            }
            records.append(r)
            with open(out_dir / "trajectories.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({**r, "turns": [t.to_dict() for t in turns]}, ensure_ascii=False,
                                   default=str) + "\n")
            print(f"[{i:>3}/{len(inputs)}] {'OK  ' if r['success'] else 'FAIL'} {stage:<22} "
                  f"calls={r['n_calls']} {dur:.1f}s")

    pred_dir = out_dir / "prediction"
    pred_dir.mkdir(parents=True, exist_ok=True)
    (pred_dir / f"{domain}.json").write_text(json.dumps(predictions, indent=2, ensure_ascii=False,
                                                        default=str), encoding="utf-8")


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="../vakra/data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--capability", type=int, required=True, choices=[1, 2, 3, 4])
    ap.add_argument("--domain", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=10, help="dialogues per domain")
    ap.add_argument("--provider", default="ollama", choices=["ollama", "vllm", "groq", "openrouter"])
    ap.add_argument("--model", default="qwen3:4b-instruct-2507-q4_K_M")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--top-k-tools", type=int, default=20,
                    help="show the model only the K most query-relevant tools (0 = all)")
    ap.add_argument("--runtime", default="docker", help="docker or podman")
    ap.add_argument("--server-python", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out or f"runs/{datetime.now():%Y%m%d-%H%M}_cap{args.capability}_{slug(args.model)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    llm = ChatLLM(args.provider, args.model, base_url=args.base_url, temperature=args.temperature,
                  max_tokens=args.max_tokens)
    records: list[dict] = []
    for domain in args.domain:
        try:
            await run_domain(args, llm, args.capability, domain, out_dir, records)
        except FileNotFoundError as e:
            print(f"[skip {domain}] {e}")
    if not records:
        sys.exit("Nothing ran.")

    shortlist_cov = [r["gt_tools_in_shortlist"] for r in records if r["gt_tools_in_shortlist"] is not None]
    config = {"environment": f"VAKRA cap {args.capability} ({args.split} split)", "domains": ",".join(args.domain),
              "provider": args.provider, "model": args.model, "temperature": args.temperature,
              "max_steps": args.max_steps, "top_k_tools": args.top_k_tools,
              "date": f"{datetime.now():%Y-%m-%d %H:%M}"}
    notes = ["Success = every ground-truth answer value appears in the final answer (numbers within 1%). "
             "Stricter than VAKRA's LLM judge on wording, looser on grounding.",
             "Only the last turn is scored, as in VAKRA's evaluator."]
    if shortlist_cov:
        notes.append(f"Ground-truth tools that survived shortlisting: {100 * sum(shortlist_cov) / len(shortlist_cov):.0f}% "
                     "(if this is low, failures are a retrieval problem, not a reasoning problem).")
    if args.capability == 1:
        notes.append("Capability 1 ground-truth arguments include file paths and $label references, so "
                     "argument recall is not meaningful there; read tool-name recall.")
    write_report(records, out_dir, f"VAKRA cap {args.capability} baseline: {args.model}", config, notes)
    print(f"\nReport: {out_dir / 'report.md'}\nPredictions for IBM's evaluator: {out_dir / 'prediction'}")


if __name__ == "__main__":
    asyncio.run(main())
