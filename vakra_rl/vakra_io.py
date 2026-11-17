"""Read VAKRA's JSON files without assuming a single exact shape.

VAKRA ships one pair of files per domain and capability:

    <split>/<capability_dir>/input/<domain>.json   -> what the agent sees
    <split>/<capability_dir>/output/<domain>.json  -> ground truth

An input record looks like:
    {"uuid", "domain", "num_turns",
     "dialogue": {"turns": [{"turn_id", "query"}, ...]},
     "additional_instructions": "..."}            # cap 4 tool-use policies live here

An output record looks like:
    {"uuid", "domain",
     "output": [{"turn_id", "query", "answer", "sequence"}, ...]}

`sequence` is not always the same shape. The repo's own sample files contain both
    {"tool_call": [...], "tool_response": [...]}
and
    [{"tool_call": [...]}, ...]
so everything here goes through helpers that accept either.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

CAPABILITY_DIRS = {
    1: "capability_1_bi_apis",
    2: "capability_2_dashboard_apis",
    3: "capability_3_multihop_reasoning",
    4: "capability_4_multiturn",
}
CAPABILITY_NAMES = {
    1: "API chaining",
    2: "Tool selection",
    3: "Multi-hop API reasoning",
    4: "Multi-hop, multi-source, with policies",
}


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def as_list(x: Any) -> list:
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


# ---------------------------------------------------------------------------
# Input side
# ---------------------------------------------------------------------------
def input_queries(record: dict) -> list[str]:
    """The user turns of one input dialogue, in order."""
    turns = (record.get("dialogue") or {}).get("turns") or []
    turns = sorted(turns, key=lambda t: t.get("turn_id", 0))
    return [t.get("query", "") for t in turns]


# ---------------------------------------------------------------------------
# Ground-truth side
# ---------------------------------------------------------------------------
def _sequence_blocks(sequence: Any) -> Iterator[dict]:
    for block in as_list(sequence):
        if isinstance(block, dict):
            yield block


def turn_tool_calls(turn: dict) -> list[dict]:
    """Tool calls of one ground-truth (or predicted) turn as
    [{"name", "arguments", "label"}], in the order they appear."""
    calls = []
    for block in _sequence_blocks(turn.get("sequence")):
        for tc in as_list(block.get("tool_call")):
            if isinstance(tc, dict) and "name" in tc:
                args = tc.get("arguments", tc.get("args")) or {}
                calls.append({"name": tc["name"], "arguments": args, "label": tc.get("label")})
    return calls


def turn_tool_responses(turn: dict) -> list:
    responses = []
    for block in _sequence_blocks(turn.get("sequence")):
        responses.extend(as_list(block.get("tool_response")))
    return responses


def final_turn(record: dict) -> dict | None:
    """VAKRA's evaluator scores the LAST turn only, so most analysis keys off it."""
    turns = record.get("output") or []
    if not turns:
        return None
    return sorted(turns, key=lambda t: t.get("turn_id", 0))[-1]


def is_retriever(tool_name: str) -> bool:
    # Capability 4 document retrievers are exposed as tools named retriever_<collection>.
    return tool_name.startswith("retriever")


_LABEL_REF = re.compile(r"\$([A-Za-z0-9_]+)\.")


def dependent_call_count(calls: list[dict], responses: list | None = None) -> int:
    """Heuristic count of calls that consume an earlier call's output.

    A call counts as dependent if
      * one of its arguments references an earlier call's label ("$FILTERED_DF_0.data_path$",
        capability 1 style), or
      * one of its argument values (len >= 3) appears verbatim in an earlier tool response
        (capability 3/4 style, where API1's output is passed into API2).

    This is an estimate for describing the data, not a ground-truth hop label.
    """
    responses = responses or []
    seen_labels: set[str] = set()
    n_dep = 0
    for i, call in enumerate(calls):
        arg_text = json.dumps(call.get("arguments", {}), ensure_ascii=False, default=str)
        dep = any(lbl in seen_labels for lbl in _LABEL_REF.findall(arg_text))
        if not dep and i > 0 and responses:
            earlier = json.dumps(responses[:i], ensure_ascii=False, default=str)
            for v in _leaf_values(call.get("arguments", {})):
                s = str(v)
                if len(s) >= 3 and s in earlier:
                    dep = True
                    break
        n_dep += int(dep)
        if call.get("label"):
            seen_labels.add(call["label"])
    return n_dep


def _leaf_values(x: Any) -> Iterator[Any]:
    if isinstance(x, dict):
        for v in x.values():
            yield from _leaf_values(v)
    elif isinstance(x, list):
        for v in x:
            yield from _leaf_values(v)
    elif x is not None and x != "":
        yield x


def answer_kind(answer: Any) -> str:
    """Rough type of a ground-truth answer, for describing the data."""
    flat = flatten_answer(answer)
    if not flat:
        return "empty"
    if all(_is_number(v) for v in flat):
        return "number" if len(flat) == 1 else "number_list"
    return "text" if len(flat) == 1 else "text_list"


def flatten_answer(answer: Any) -> list[str]:
    out: list[str] = []
    for v in _leaf_values(answer):
        out.append(str(v).strip())
    return [v for v in out if v]


def _is_number(s: str) -> bool:
    try:
        float(str(s).replace(",", "").rstrip("%"))
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Files on disk
# ---------------------------------------------------------------------------
def domain_files(data_root: str | Path, split: str, capability: int) -> list[dict]:
    """[{domain, input, output}] for one capability. Missing files are None."""
    cap_dir = Path(data_root) / split / CAPABILITY_DIRS[capability]
    domains = set()
    for sub in ("input", "output"):
        d = cap_dir / sub
        if d.exists():
            domains |= {p.stem for p in d.glob("*.json")}
    rows = []
    for dom in sorted(domains):
        inp = cap_dir / "input" / f"{dom}.json"
        out = cap_dir / "output" / f"{dom}.json"
        rows.append({
            "domain": dom,
            "input": inp if inp.exists() else None,
            "output": out if out.exists() else None,
        })
    return rows
