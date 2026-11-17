"""Run with:  python -m pytest -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vakra_rl.agent import shortlist_tools
from vakra_rl.scoring import (answer_contains_gt, classify_failure, keyword_hits, match_tool_calls,
                              pass_at_k_and_hat_k)
from vakra_rl.vakra_io import dependent_call_count, turn_tool_calls


def test_keyword_hits_with_regex_and_alternatives():
    kws = ["Dice", "SLA|service level", r"re:\b80\b|re:\b95\b"]
    assert keyword_hits("Dice has an SLA of 95%", kws) == [True, True, True]
    assert keyword_hits("Dice has an SLA of 950", kws) == [True, True, False]


def test_match_tool_calls_is_order_invariant_and_one_to_one():
    exp = [{"name": "a", "arguments": {"x": 1}}, {"name": "b", "arguments": {}}]
    pred = [{"name": "b", "arguments": {}}, {"name": "a", "arguments": {"x": "1.0"}}]
    m = match_tool_calls(exp, pred)
    assert m["name_recall"] == 1 and m["full_recall"] == 1
    m = match_tool_calls(exp + [{"name": "a", "arguments": {"x": 1}}], pred)
    assert m["name_recall"] == 2 / 3  # one predicted call cannot satisfy two expected calls


def test_recall_undefined_when_no_calls_expected():
    assert match_tool_calls([], [{"name": "a", "arguments": {}}])["name_recall"] is None


def test_answer_contains_gt_numbers_and_text():
    assert answer_contains_gt("There are 23 breweries.", ["23"])
    assert answer_contains_gt("Ratio is about 1.2587", [[1.2587021916630856]])
    assert not answer_contains_gt("There are 24.", ["23"])
    assert answer_contains_gt("It was the 2012 Summer games", ["2012 Summer"])


def test_failure_stages():
    exp = [{"name": "get_x", "arguments": {"id": "7"}}, {"name": "get_y", "arguments": {}}]
    call = lambda n, a=None, err=False, pe=None: {"name": n, "arguments": a or {}, "is_error": err, "parse_error": pe}
    assert classify_failure(exp, [], False, "answer") == "no_tool_call"
    assert classify_failure(exp, [call("other")], False, "answer") == "wrong_tool"
    assert classify_failure(exp, [call("get_x", {"id": "7"})], False, "answer") == "incomplete_chain"
    assert classify_failure(exp, [call("get_x", {"id": "8"}), call("get_y")], False, "answer") == "bad_arguments"
    assert classify_failure(exp, [call("get_x", {"id": "7"}), call("get_y")], False, "answer") == "readout_failure"
    assert classify_failure(exp, [call("get_x", pe="bad json")], False, "answer") == "format_error"
    assert classify_failure([], [call("get_x")], False, "answer") == "should_not_call_tools"
    assert classify_failure(exp, [call("get_x", {"id": "7"}), call("get_y")], True, "answer") == "success"


def test_pass_hat_k():
    r = pass_at_k_and_hat_k([[True, True, True], [True, False, True], [False, False, False]])
    assert r["k"] == 3 and abs(r["pass@3"] - 2 / 3) < 1e-9 and abs(r["pass^3"] - 1 / 3) < 1e-9


def test_sequence_shapes_and_dependency_heuristic():
    list_shape = {"sequence": [{"tool_call": [
        {"name": "init", "arguments": {}, "label": "T"},
        {"name": "filter", "arguments": {"data_source": "$T.data_path$"}, "label": "F"}]}]}
    calls = turn_tool_calls(list_shape)
    assert [c["name"] for c in calls] == ["init", "filter"]
    assert dependent_call_count(calls) == 1
    dict_shape = {"sequence": {"tool_call": [{"name": "get_id", "arguments": {"t": "X"}},
                                             {"name": "count", "arguments": {"pid": "P1234"}}]}}
    assert dependent_call_count(turn_tool_calls(dict_shape), [{"pid": "P1234"}, 3]) == 1


def test_shortlist_prefers_relevant_tools():
    tools = [{"type": "function", "function": {"name": n, "description": d}} for n, d in [
        ("get_brewery_count_by_country", "Count breweries in a country"),
        ("get_player_goals", "Goals scored by a hockey player"),
        ("get_movie_rating", "Average rating of a movie")]]
    top = shortlist_tools("How many breweries are in the United States?", tools, 1)
    assert top[0]["function"]["name"] == "get_brewery_count_by_country"
