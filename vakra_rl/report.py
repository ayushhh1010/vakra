"""Turn a list of per-attempt records into summary.json and a readable report.md."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from .scoring import FAILURE_DESCRIPTIONS, FAILURE_STAGES, pass_at_k_and_hat_k


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return mean(vals) if vals else None


def _fmt(x: float | None, nd: int = 2) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def _table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def summarize(records: list[dict]) -> dict:
    n = len(records)
    by_task: dict[str, list[bool]] = defaultdict(list)
    for r in records:
        by_task[r["task_id"]].append(bool(r["success"]))
    prompt_max = [r["prompt_tokens_max"] for r in records if r.get("prompt_tokens_max")]
    return {
        "attempts": n,
        "tasks": len(by_task),
        "success_rate": mean(r["success"] for r in records) if n else 0.0,
        "mean_name_recall": _mean(r["name_recall"] for r in records),
        "mean_full_recall": _mean(r["full_recall"] for r in records),
        "mean_tool_calls": mean(r["n_calls"] for r in records) if n else 0.0,
        "mean_latency_s": mean(r["latency_s"] for r in records) if n else 0.0,
        "median_latency_s": median(r["latency_s"] for r in records) if n else 0.0,
        "max_prompt_tokens": max(prompt_max) if prompt_max else None,
        "text_fallback_calls": sum(r.get("text_fallback_calls", 0) for r in records),
        "failure_stages": dict(Counter(r["stage"] for r in records)),
        "reliability": pass_at_k_and_hat_k(list(by_task.values())),
    }


def _group_rows(records: list[dict], key: str) -> list[list]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[str(r.get(key, "-"))].append(r)
    rows = []
    for g, rs in sorted(groups.items()):
        rows.append([g, len(rs), _pct(mean(r["success"] for r in rs)),
                     _fmt(_mean(r["name_recall"] for r in rs)),
                     _fmt(_mean(r["full_recall"] for r in rs)),
                     f"{mean(r['n_calls'] for r in rs):.1f}"])
    return rows


def write_report(records: list[dict], out_dir: Path, title: str, config: dict,
                 notes: list[str] | None = None, n_examples: int = 8) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(records)
    summary["config"] = config
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    L: list[str] = [f"# {title}", ""]
    L.append("Proxy metrics computed by vakra-rl (no LLM judge). Not the official VAKRA score.")
    L.append("")
    L.append(_table(["Setting", "Value"], [[k, f"`{v}`"] for k, v in config.items()]))
    L.append("")
    L.append("## Overall")
    L.append("")
    rows = [
        ["Attempts (tasks x repeats)", f"{summary['attempts']} ({summary['tasks']} tasks)"],
        ["Success rate", _pct(summary["success_rate"])],
        ["Tool-name recall (mean, tasks that need tools)", _fmt(summary["mean_name_recall"])],
        ["Tool name+argument recall (mean, same tasks)", _fmt(summary["mean_full_recall"])],
        ["Tool calls per attempt (mean)", f"{summary['mean_tool_calls']:.1f}"],
        ["Latency per attempt, mean / median (s)",
         f"{summary['mean_latency_s']:.1f} / {summary['median_latency_s']:.1f}"],
        ["Largest prompt seen (tokens)", summary["max_prompt_tokens"]],
        ["Tool calls recovered from plain text", summary["text_fallback_calls"]],
    ]
    rel = summary["reliability"]
    if rel and rel["k"] > 1:
        k = rel["k"]
        rows += [[f"pass@{k} (any of {k} attempts right)", _pct(rel[f"pass@{k}"])],
                 [f"pass^{k} (all {k} attempts right)", _pct(rel[f"pass^{k}"])]]
    L.append(_table(["Metric", "Value"], rows))
    L.append("")

    L.append("## Where it breaks (first failing stage)")
    L.append("")
    counts = summary["failure_stages"]
    stage_rows = [[f"`{s}`", desc, counts.get(s, 0), _pct(counts.get(s, 0) / max(len(records), 1))]
                  for s, desc in FAILURE_STAGES if counts.get(s, 0)]
    L.append(_table(["Stage", "Meaning", "Count", "Share"], stage_rows))
    L.append("")

    for key, label in (("group", "By task group"), ("difficulty", "By difficulty")):
        if any(key in r for r in records):
            L.append(f"## {label}")
            L.append("")
            L.append(_table([label.split()[-1].capitalize(), "Attempts", "Success", "Name recall",
                             "Name+arg recall", "Calls"], _group_rows(records, key)))
            L.append("")

    fails = [r for r in records if not r["success"]][:n_examples]
    if fails:
        L.append("## Example failures")
        L.append("")
        for r in fails:
            L.append(f"### {r['task_id']}: `{r['stage']}`")
            L.append("")
            L.append(f"- **Query:** {r['query']}")
            L.append(f"- **Expected tools:** {', '.join(r['expected_tools']) or '(none: should ask or refuse)'}")
            L.append(f"- **Called:** {', '.join(r['called_tools']) or '(none)'}")
            L.append(f"- **Answer:** {r['answer'][:400] or '(empty)'}")
            if r.get("expected_answer"):
                L.append(f"- **Expected:** {str(r['expected_answer'])[:300]}")
            L.append("")

    L.append("## My interpretation")
    L.append("")
    L.append("_Write this yourself after reading the trajectories: which stage dominates, "
             "why you think it happens, and what you will try next._")
    L.append("")
    if notes:
        L.append("## Notes")
        L.append("")
        L += [f"- {n}" for n in notes]
        L.append("")
    (out_dir / "report.md").write_text("\n".join(L), encoding="utf-8")
    return summary
