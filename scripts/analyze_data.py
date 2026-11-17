"""Describe the VAKRA training split: what the tasks look like before any model runs.

Reads <data-root>/<split>/capability_*/{input,output}/<domain>.json and writes
CSV tables, figures and a markdown report to --out.

    python scripts/analyze_data.py --data-root ../vakra/data --split train

Questions it answers (each one matters for SFT/GRPO design):
  * How many samples per capability and domain?  -> how to split train/dev by domain
  * How many tool calls does the final turn need? -> max_steps, reward shaping
  * How often does a call consume an earlier call's output? -> real multi-hop share
  * How often are documents (retrievers) involved in capability 4? -> multi-source share
  * How often are there tool-use policies (additional_instructions)?
  * What do answers look like (number, text, list)? -> what a proxy reward can check
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vakra_rl.vakra_io import (CAPABILITY_DIRS, CAPABILITY_NAMES, answer_kind,  # noqa: E402
                               dependent_call_count, domain_files, final_turn, input_queries,
                               is_retriever, load_json, turn_tool_calls, turn_tool_responses)

# Reference palette (light mode) from the dataviz guidance: one hue for magnitude,
# recessive ink for axes and grid.
SERIES = "#2a78d6"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


def load_capability(data_root: Path, split: str, cap: int, max_mb: float):
    rows, skipped = [], []
    for f in domain_files(data_root, split, cap):
        recs_in, recs_out = {}, {}
        for side, path in (("input", f["input"]), ("output", f["output"])):
            if path is None:
                continue
            mb = path.stat().st_size / 1024 ** 2
            if max_mb and mb > max_mb:
                skipped.append(f"{CAPABILITY_DIRS[cap]}/{side}/{path.name} ({mb:.0f} MB)")
                continue
            data = load_json(path)
            target = recs_in if side == "input" else recs_out
            for r in data:
                target[r.get("uuid")] = r
        for uid in sorted(set(recs_in) | set(recs_out), key=str):
            rin, rout = recs_in.get(uid, {}), recs_out.get(uid, {})
            queries = input_queries(rin) if rin else [t.get("query", "") for t in rout.get("output", [])]
            instr = (rin.get("additional_instructions") or "").strip()
            ft = final_turn(rout) if rout else None
            calls = turn_tool_calls(ft) if ft else []
            resps = turn_tool_responses(ft) if ft else []
            all_turn_calls = [turn_tool_calls(t) for t in (rout.get("output") or [])]
            names = [c["name"] for c in calls]
            n_ret = sum(is_retriever(n) for n in names)
            rows.append({
                "capability": cap,
                "domain": f["domain"],
                "uuid": uid,
                "has_input": bool(rin),
                "has_output": bool(rout),
                "n_turns": len(queries),
                "final_query_chars": len(queries[-1]) if queries else 0,
                "has_policy": bool(instr),
                "policy_chars": len(instr),
                "final_calls": len(calls) if ft else None,
                "final_unique_tools": len(set(names)) if ft else None,
                "final_dependent_calls": dependent_call_count(calls, resps) if ft else None,
                "final_retriever_calls": n_ret if ft else None,
                "final_api_calls": (len(names) - n_ret) if ft else None,
                "total_calls_all_turns": sum(len(c) for c in all_turn_calls) if rout else None,
                "answer_kind": answer_kind(ft.get("answer")) if ft else None,
                "final_tools": "|".join(names),
            })
    return rows, skipped


def source_mix(row: dict) -> str:
    a, r = row["final_api_calls"] or 0, row["final_retriever_calls"] or 0
    if a and r:
        return "APIs + documents"
    if r:
        return "documents only"
    if a:
        return "APIs only"
    return "no tool call"


def summarize(rows: list[dict]) -> dict:
    with_out = [r for r in rows if r["has_output"]]
    calls = [r["final_calls"] for r in with_out]
    tools = Counter(t for r in with_out for t in r["final_tools"].split("|") if t)
    return {
        "domains": len({r["domain"] for r in rows}),
        "dialogues": len(rows),
        "with_ground_truth": len(with_out),
        "mean_turns": mean(r["n_turns"] for r in rows) if rows else 0,
        "max_turns": max((r["n_turns"] for r in rows), default=0),
        "policy_share": mean(r["has_policy"] for r in rows) if rows else 0,
        "final_calls_mean": mean(calls) if calls else 0,
        "final_calls_median": median(calls) if calls else 0,
        "final_calls_max": max(calls, default=0),
        "share_2plus_calls": mean(c >= 2 for c in calls) if calls else 0,
        "share_dependent": mean((r["final_dependent_calls"] or 0) > 0 for r in with_out) if with_out else 0,
        "share_uses_documents": mean((r["final_retriever_calls"] or 0) > 0 for r in with_out) if with_out else 0,
        "distinct_tools": len(tools),
        "answer_kinds": dict(Counter(r["answer_kind"] for r in with_out)),
        "top_tools": tools.most_common(10),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _style(ax, title: str):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def make_figures(rows_by_cap: dict[int, list[dict]], out: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    caps = [c for c in sorted(rows_by_cap) if rows_by_cap[c]]

    # 1. Samples per domain (top 15 per capability)
    fig, axes = plt.subplots(1, len(caps), figsize=(4.2 * len(caps), 5.2), facecolor=SURFACE, squeeze=False)
    for ax, cap in zip(axes[0], caps):
        cnt = Counter(r["domain"] for r in rows_by_cap[cap]).most_common(15)[::-1]
        ax.barh([d for d, _ in cnt], [n for _, n in cnt], color=SERIES, height=0.6)
        for i, (_, n) in enumerate(cnt):
            ax.text(n, i, f" {n}", va="center", fontsize=7, color=INK_2)
        _style(ax, f"Cap {cap}: samples per domain (top 15)")
    fig.tight_layout()
    fig.savefig(out / "fig_samples_per_domain.png", dpi=160)
    plt.close(fig)
    made.append("fig_samples_per_domain.png")

    # 2. Tool calls needed in the final (scored) turn
    fig, axes = plt.subplots(1, len(caps), figsize=(3.6 * len(caps), 3.2), facecolor=SURFACE, squeeze=False)
    for ax, cap in zip(axes[0], caps):
        vals = [min(r["final_calls"], 10) for r in rows_by_cap[cap] if r["final_calls"] is not None]
        cnt = Counter(vals)
        xs = list(range(0, (max(vals) if vals else 0) + 1))
        ax.bar(xs, [cnt.get(x, 0) for x in xs], color=SERIES, width=0.7)
        ax.set_xticks(xs, [str(x) if x < 10 else "10+" for x in xs])
        _style(ax, f"Cap {cap}: tool calls in final turn")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out / "fig_calls_per_final_turn.png", dpi=160)
    plt.close(fig)
    made.append("fig_calls_per_final_turn.png")

    # 3. Capability 4: turns per dialogue and source mix of the final turn
    if rows_by_cap.get(4):
        r4 = rows_by_cap[4]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2), facecolor=SURFACE)
        tc = Counter(r["n_turns"] for r in r4)
        xs = sorted(tc)
        a1.bar(xs, [tc[x] for x in xs], color=SERIES, width=0.7)
        a1.set_xticks(xs)
        _style(a1, "Cap 4: turns per dialogue")
        a1.grid(axis="y", color=GRID, linewidth=0.8)
        a1.grid(axis="x", visible=False)
        order = ["APIs only", "documents only", "APIs + documents", "no tool call"]
        mix = Counter(source_mix(r) for r in r4 if r["has_output"])
        a2.barh(order[::-1], [mix.get(k, 0) for k in order[::-1]], color=SERIES, height=0.6)
        for i, k in enumerate(order[::-1]):
            a2.text(mix.get(k, 0), i, f" {mix.get(k, 0)}", va="center", fontsize=8, color=INK_2)
        _style(a2, "Cap 4: sources used in final turn")
        fig.tight_layout()
        fig.savefig(out / "fig_cap4_turns_and_sources.png", dpi=160)
        plt.close(fig)
        made.append("fig_cap4_turns_and_sources.png")
    return made


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="../vakra/data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--capabilities", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--max-file-mb", type=float, default=150, help="skip files bigger than this (RAM)")
    ap.add_argument("--out", default="reports/data")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    data_root, out = Path(args.data_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_by_cap, skipped_all, summaries = {}, [], {}
    for cap in args.capabilities:
        rows, skipped = load_capability(data_root, args.split, cap, args.max_file_mb)
        skipped_all += skipped
        if not rows:
            print(f"cap {cap}: no files under {data_root / args.split / CAPABILITY_DIRS[cap]}")
            continue
        rows_by_cap[cap] = rows
        summaries[cap] = summarize(rows)
        s = summaries[cap]
        print(f"cap {cap}: {s['dialogues']} dialogues, {s['domains']} domains, "
              f"final-turn calls mean {s['final_calls_mean']:.2f}")
    if not rows_by_cap:
        sys.exit("Nothing found. Did you download the train split? See scripts/download_vakra.py")

    all_rows = [r for rows in rows_by_cap.values() for r in rows]
    with open(out / "turns.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)

    dom = defaultdict(list)
    for r in all_rows:
        dom[(r["capability"], r["domain"])].append(r)
    with open(out / "by_domain.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["capability", "domain", "dialogues", "mean_final_calls", "share_dependent",
                    "share_policy", "distinct_tools"])
        for (cap, d), rs in sorted(dom.items()):
            fc = [r["final_calls"] for r in rs if r["final_calls"] is not None]
            tools = {t for r in rs for t in r["final_tools"].split("|") if t}
            w.writerow([cap, d, len(rs), f"{mean(fc):.2f}" if fc else "",
                        f"{mean((r['final_dependent_calls'] or 0) > 0 for r in rs):.2f}",
                        f"{mean(r['has_policy'] for r in rs):.2f}", len(tools)])

    figs = [] if args.no_figures else make_figures(rows_by_cap, out)

    # ---- report
    L = [f"# VAKRA {args.split} split: data analysis", ""]
    L.append(f"Source: `{data_root / args.split}`. Generated by `scripts/analyze_data.py`.")
    L.append("")
    head = ["", *[f"Cap {c}" for c in summaries]]
    def row(label, fn):
        return f"| {label} | " + " | ".join(fn(summaries[c]) for c in summaries) + " |"
    L.append("| " + " | ".join(head) + " |")
    L.append("|" + "---|" * len(head))
    L.append("| Capability | " + " | ".join(CAPABILITY_NAMES[c] for c in summaries) + " |")
    L.append(row("Domains", lambda s: str(s["domains"])))
    L.append(row("Dialogues", lambda s: str(s["dialogues"])))
    L.append(row("With ground truth loaded", lambda s: str(s["with_ground_truth"])))
    L.append(row("Turns per dialogue (mean / max)", lambda s: f"{s['mean_turns']:.2f} / {s['max_turns']}"))
    L.append(row("Has tool-use policy", lambda s: f"{100 * s['policy_share']:.0f}%"))
    L.append(row("Final-turn tool calls (mean / median / max)",
                 lambda s: f"{s['final_calls_mean']:.2f} / {s['final_calls_median']:.0f} / {s['final_calls_max']}"))
    L.append(row("Final turn needs 2+ calls", lambda s: f"{100 * s['share_2plus_calls']:.0f}%"))
    L.append(row("Final turn has a dependent call (heuristic)", lambda s: f"{100 * s['share_dependent']:.0f}%"))
    L.append(row("Final turn uses documents", lambda s: f"{100 * s['share_uses_documents']:.0f}%"))
    L.append(row("Distinct tools in ground truth", lambda s: str(s["distinct_tools"])))
    L.append("")
    L.append("Answer types in the final turn:")
    L.append("")
    for c, s in summaries.items():
        kinds = ", ".join(f"{k} {v}" for k, v in sorted(s["answer_kinds"].items(), key=lambda kv: -kv[1]))
        L.append(f"- Cap {c}: {kinds}")
    L.append("")
    for fname in figs:
        L.append(f"![{fname}]({fname})")
        L.append("")
    L.append("Most frequent ground-truth tools (final turn):")
    L.append("")
    for c, s in summaries.items():
        L.append(f"- Cap {c}: " + ", ".join(f"`{t}` ({n})" for t, n in s["top_tools"][:6]))
    L.append("")
    if skipped_all:
        L.append("Files skipped for size (the numbers above exclude them):")
        L.append("")
        L += [f"- {s}" for s in skipped_all]
        L.append("")
    L.append("## Check by hand before trusting the heuristics")
    L.append("")
    L.append("- Open `turns.csv`, pick 10 random capability-3 rows, read the ground truth, and label the "
             "hop pattern yourself (single API / API1 -> API2 / API1 -> API2 -> API3). Compare with "
             "`final_dependent_calls`. Write the agreement rate here.")
    L.append("- Read 5 capability-4 policies (`additional_instructions`). What kinds of rules are they?")
    L.append("")
    L.append("## My interpretation")
    L.append("")
    L.append("_What do these numbers imply for max_steps, the reward, and how you split train/dev by domain?_")
    L.append("")
    (out / "data_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\nWrote {out / 'data_report.md'}, turns.csv, by_domain.csv and {len(figs)} figures.")


if __name__ == "__main__":
    main()
