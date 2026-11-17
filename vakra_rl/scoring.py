"""Scoring that runs without an LLM judge.

These are PROXY metrics for fast iteration and, later, RL rewards. They are not
VAKRA's official score (that uses IBM's evaluator with an LLM judge). Report
both, and say which is which.

Two pieces:
  1. match_tool_calls / answer checks  -> numbers
  2. classify_failure                  -> one stage label per attempt, so you can
     say *where* the agent breaks, not just how often.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .vakra_io import flatten_answer

# ---------------------------------------------------------------------------
# Value normalisation
# ---------------------------------------------------------------------------
_WS = re.compile(r"\s+")


def norm_text(s: Any) -> str:
    return _WS.sub(" ", str(s)).strip().lower()


def _to_float(s: Any) -> float | None:
    try:
        return float(str(s).replace(",", "").strip().rstrip("%"))
    except ValueError:
        return None


def values_equal(expected: Any, actual: Any, rel_tol: float = 1e-3) -> bool:
    if isinstance(expected, (dict, list)) or isinstance(actual, (dict, list)):
        return json.dumps(expected, sort_keys=True, default=str).lower() == \
               json.dumps(actual, sort_keys=True, default=str).lower()
    fe, fa = _to_float(expected), _to_float(actual)
    if fe is not None and fa is not None:
        return abs(fe - fa) <= rel_tol * max(1.0, abs(fe))
    return norm_text(expected) == norm_text(actual)


def args_match(expected: dict, actual: dict | None) -> bool:
    """Every expected argument is present with an equal value. Extra args are allowed."""
    if actual is None:
        return False
    for k, v in (expected or {}).items():
        if k not in actual or not values_equal(v, actual[k]):
            return False
    return True


# ---------------------------------------------------------------------------
# Tool-call matching (order-invariant, each predicted call used at most once,
# mirroring the "subset, order-invariant" spirit of VAKRA's exact-match stage)
# ---------------------------------------------------------------------------
def match_tool_calls(expected: list[dict], predicted: list[dict]) -> dict:
    """expected/predicted: [{"name", "arguments"}]. predicted may also carry "is_error"."""
    used_name: set[int] = set()
    used_full: set[int] = set()
    name_hits = full_hits = 0
    for e in expected:
        args = e.get("arguments", e.get("args")) or {}
        for i, p in enumerate(predicted):
            if i not in used_full and p["name"] == e["name"] and args_match(args, p.get("arguments")):
                used_full.add(i)
                full_hits += 1
                break
        for i, p in enumerate(predicted):
            if i not in used_name and p["name"] == e["name"]:
                used_name.add(i)
                name_hits += 1
                break
    n_exp = len(expected)
    return {
        "n_expected": n_exp,
        "n_predicted": len(predicted),
        "n_predicted_errors": sum(1 for p in predicted if p.get("is_error")),
        # None when no call was expected: recall is undefined there, and counting it
        # as 1.0 would inflate averages with tasks that test restraint, not tool use.
        "name_recall": name_hits / n_exp if n_exp else None,
        "full_recall": full_hits / n_exp if n_exp else None,
        "name_precision": len(used_name) / len(predicted) if predicted else (1.0 if n_exp == 0 else 0.0),
    }


# ---------------------------------------------------------------------------
# Answer checks
# ---------------------------------------------------------------------------
def _alt_hit(alt: str, answer: str) -> bool:
    alt = alt.strip()
    if not alt:
        return False
    if alt.startswith("re:"):  # BPO files mark regex alternatives with a "re:" prefix
        try:
            return re.search(alt[3:], answer or "", flags=re.IGNORECASE) is not None
        except re.error:
            return False
    return norm_text(alt) in norm_text(answer)


def keyword_hits(answer: str, keywords: list[str]) -> list[bool]:
    """BPO-style keywords: each entry is 'alt1|alt2|re:<regex>'; a group is hit if any
    alternative appears in the answer (case-insensitive substring, or regex search)."""
    return [any(_alt_hit(alt, answer) for alt in kw.split("|")) for kw in keywords]


def answer_contains_gt(answer: str, gt_answer: Any, rel_tol: float = 1e-2) -> bool:
    """Proxy for VAKRA answers: every ground-truth value appears in the predicted
    answer text (numbers compared with tolerance). Strict on purpose; the official
    LLM judge is more lenient about phrasing."""
    gts = flatten_answer(gt_answer)
    if not gts:
        return False
    a = norm_text(answer)
    nums_in_answer = [_to_float(x) for x in re.findall(r"-?\d[\d,]*\.?\d*", answer or "")]
    nums_in_answer = [x for x in nums_in_answer if x is not None]
    for g in gts:
        fg = _to_float(g)
        if fg is not None:
            if not any(abs(fg - x) <= rel_tol * max(1.0, abs(fg)) for x in nums_in_answer):
                return False
        elif norm_text(g) not in a:
            return False
    return True


# ---------------------------------------------------------------------------
# Failure taxonomy: the first stage that broke
# ---------------------------------------------------------------------------
FAILURE_STAGES = [
    ("success", "Answer correct"),
    ("llm_error", "Model server error or timeout"),
    ("format_error", "Tool calls were malformed (bad JSON), nothing valid executed"),
    ("should_not_call_tools", "Expected a clarifying question or refusal; agent called tools instead"),
    ("no_tool_call", "Needed tools, called none (answered from memory)"),
    ("tool_errors_only", "Every tool call returned an error"),
    ("wrong_tool", "None of the needed tools were called"),
    ("incomplete_chain", "Some but not all needed tools were called"),
    ("bad_arguments", "Right tools, wrong argument values"),
    ("step_limit", "Ran out of steps before answering"),
    ("readout_failure", "Right tools and arguments, but the final answer is wrong"),
    ("wrong_answer", "No tools expected, answer still wrong"),
]
FAILURE_DESCRIPTIONS = dict(FAILURE_STAGES)


def classify_failure(expected_calls: list[dict], predicted: list[dict], answer_ok: bool,
                     stop_reason: str) -> str:
    """predicted: [{"name", "arguments", "is_error", "parse_error"}] in call order."""
    if answer_ok and stop_reason == "answer":
        return "success"
    if stop_reason == "llm_error":
        return "llm_error"
    if not expected_calls:
        # Any attempted call (even a malformed one) is the main mistake here.
        return "should_not_call_tools" if predicted else "wrong_answer"
    valid = [p for p in predicted if not p.get("parse_error")]
    if predicted and not valid:
        return "format_error"
    if not valid:
        return "step_limit" if stop_reason == "step_limit" else "no_tool_call"
    if all(p.get("is_error") for p in valid):
        return "tool_errors_only"
    m = match_tool_calls(expected_calls, valid)
    if m["name_recall"] == 0:
        return "wrong_tool"
    if m["name_recall"] < 1:
        return "incomplete_chain"
    if m["full_recall"] < 1:
        return "bad_arguments"
    if stop_reason == "step_limit":
        return "step_limit"
    return "readout_failure"


def pass_at_k_and_hat_k(successes_per_task: list[list[bool]]) -> dict:
    """pass@k: at least one of k attempts succeeds. pass^k: all k succeed
    (tau-bench's reliability metric). Uses every attempt available per task."""
    tasks = [s for s in successes_per_task if s]
    if not tasks:
        return {}
    k = min(len(s) for s in tasks)
    return {
        "k": k,
        "pass@1": sum(sum(s[:k]) / k for s in tasks) / len(tasks),
        f"pass@{k}": sum(any(s[:k]) for s in tasks) / len(tasks),
        f"pass^{k}": sum(all(s[:k]) for s in tasks) / len(tasks),
    }
