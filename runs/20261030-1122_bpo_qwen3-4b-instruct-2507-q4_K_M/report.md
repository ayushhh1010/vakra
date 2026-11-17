# BPO baseline: qwen3:4b-instruct-2507-q4_K_M

Proxy metrics computed by vakra-rl (no LLM judge). Not the official VAKRA score.

| Setting | Value |
|---|---|
| environment | `VAKRA BPO (local MCP)` |
| provider | `ollama` |
| model | `qwen3:4b-instruct-2507-q4_K_M` |
| temperature | `0.0` |
| repeats | `1` |
| max_steps | `8` |
| task_files | `tasks,tasks_edge_cases,tasks_undocumented,tasks_http_errors,tasks_type_mismatch,tasks_schema_violations` |
| date | `2026-10-30 11:25` |

## Overall

| Metric | Value |
|---|---|
| Attempts (tasks x repeats) | 3 (3 tasks) |
| Success rate | 66.7% |
| Tool-name recall (mean, tasks that need tools) | 0.75 |
| Tool name+argument recall (mean, same tasks) | 0.75 |
| Tool calls per attempt (mean) | 2.0 |
| Latency per attempt, mean / median (s) | 63.8 / 50.7 |
| Largest prompt seen (tokens) | 3295 |
| Tool calls recovered from plain text | 0 |

## Where it breaks (first failing stage)

| Stage | Meaning | Count | Share |
|---|---|---|---|
| `success` | Answer correct | 2 | 66.7% |
| `incomplete_chain` | Some but not all needed tools were called | 1 | 33.3% |

## By task group

| Group | Attempts | Success | Name recall | Name+arg recall | Calls |
|---|---|---|---|---|---|
| tasks | 3 | 66.7% | 0.75 | 0.75 | 2.0 |

## By difficulty

| Difficulty | Attempts | Success | Name recall | Name+arg recall | Calls |
|---|---|---|---|---|---|
| easy | 2 | 100.0% | 1.00 | 1.00 | 2.0 |
| medium | 1 | 0.0% | 0.50 | 0.50 | 2.0 |

## Example failures

### tasks/task_3: `incomplete_chain`

- **Query:** For requisitions like 05958BR, which sources provided the most candidates, and how effective were they at converting to hires?
- **Expected tools:** candidate_source_candidate_volume_by_source, candidate_source_source_recommendation_summary
- **Called:** candidate_source_candidate_volume_by_source, candidate_source_funnel_conversion_by_source
- **Answer:** For requisition 05958BR, the sources that provided the most candidates were:

- **LinkedIn** and **Dice**, each contributing 519 candidates (18% of the total).
- **GitHub** with 468 candidates (16% of the total).
- **Indeed** and **Internal** with 400 candidates each (14% of the total).

In terms of effectiveness in converting candidates to hires, the key funnel metrics are as follows:

- **Linked
- **Expected:** LinkedIn: 519 candidates (18%), 7 hires. Offer acceptance rate: 70%. Dice: 516 candidates (18%), 11 hires. Offer acceptance rate: 79%. GitHub: 468 candidates (16%), 10 hires. Offer acceptance rate: 77%.

## My interpretation

_Write this yourself after reading the trajectories: which stage dominates, why you think it happens, and what you will try next._

## Notes

- 13 of the 26 core BPO tasks expect NO tool call (the agent should ask for a missing ID or decline); those test restraint, not tool use.
- Success = every expected keyword group appears in the final answer.
