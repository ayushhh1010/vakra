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
| date | `2026-10-30 11:43` |

## Overall

| Metric | Value |
|---|---|
| Attempts (tasks x repeats) | 45 (45 tasks) |
| Success rate | 53.3% |
| Tool-name recall (mean, tasks that need tools) | 0.79 |
| Tool name+argument recall (mean, same tasks) | 0.79 |
| Tool calls per attempt (mean) | 1.5 |
| Latency per attempt, mean / median (s) | 19.4 / 15.0 |
| Largest prompt seen (tokens) | 4084 |
| Tool calls recovered from plain text | 0 |

## Where it breaks (first failing stage)

| Stage | Meaning | Count | Share |
|---|---|---|---|
| `success` | Answer correct | 24 | 53.3% |
| `llm_error` | Model server error or timeout | 1 | 2.2% |
| `should_not_call_tools` | Expected a clarifying question or refusal; agent called tools instead | 6 | 13.3% |
| `wrong_tool` | None of the needed tools were called | 3 | 6.7% |
| `incomplete_chain` | Some but not all needed tools were called | 6 | 13.3% |
| `readout_failure` | Right tools and arguments, but the final answer is wrong | 1 | 2.2% |
| `wrong_answer` | No tools expected, answer still wrong | 4 | 8.9% |

## By task group

| Group | Attempts | Success | Name recall | Name+arg recall | Calls |
|---|---|---|---|---|---|
| tasks | 26 | 23.1% | 0.48 | 0.48 | 1.8 |
| tasks_edge_cases | 5 | 80.0% | 1.00 | 1.00 | 1.0 |
| tasks_http_errors | 4 | 100.0% | 1.00 | 1.00 | 1.0 |
| tasks_schema_violations | 4 | 100.0% | 1.00 | 1.00 | 1.0 |
| tasks_type_mismatch | 3 | 100.0% | 1.00 | 1.00 | 1.7 |
| tasks_undocumented | 3 | 100.0% | 1.00 | 1.00 | 1.0 |

## By difficulty

| Difficulty | Attempts | Success | Name recall | Name+arg recall | Calls |
|---|---|---|---|---|---|
| easy | 22 | 36.4% | 0.56 | 0.56 | 1.3 |
| hard | 11 | 63.6% | 0.89 | 0.89 | 2.3 |
| medium | 12 | 75.0% | 0.88 | 0.88 | 1.3 |

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

### tasks/task_5: `should_not_call_tools`

- **Query:** Should I include the skill Python? What is its impact on SLA, fill rate, and overall relevance?
- **Expected tools:** (none: should ask or refuse)
- **Called:** skills_skill_relevance_justification
- **Answer:** The requisition ID "R12345" was not found in the system. However, I have a list of suggested requisition IDs: REQ00123, REQ00124, REQ00125, and REQ00134.

To proceed accurately, I need to know which specific requisition you are referring to. Could you please confirm the correct requisition ID?
- **Expected:** Can you share the requisition ID (e.g., 05958BR or 05959BR)?

### tasks/task_6: `incomplete_chain`

- **Query:** What are the best sources to prioritize for 05959BR?
- **Expected tools:** candidate_source_sla_per_source, candidate_source_candidate_volume_by_source, candidate_source_funnel_conversion_by_source
- **Called:** candidate_source_list_all_sources, candidate_source_source_sla_score, candidate_source_source_sla_score, candidate_source_source_sla_score, candidate_source_funnel_conversion_by_source, candidate_source_total_hires_by_source, candidate_source_candidate_volume_by_source, candidate_source_source_recommendation_summary
- **Answer:** (empty)
- **Expected:** You should prioritize Dice, GitHub, and LinkedIn. Dice and LinkedIn both met SLA 100% of the time and brought in 18% of all candidates. Dice had a strong offer conversion rate (2.7%), and GitHub had the highest conversion (2.8%) despite slightly lower SLA. Indeed should be avoided due to 0% SLA and 

### tasks/task_7: `wrong_answer`

- **Query:** Out of these skills — Python, Quantum Physics, Cyber Engineering, Risk Analysis, Wireshark — which ones negatively affect SLA performance?
- **Expected tools:** (none: should ask or refuse)
- **Called:** (none)
- **Answer:** To determine which skills negatively affect SLA (Service Level Agreement) performance, I need to analyze the impact of each skill on SLA achievement rates. This requires using the `skills_skill_impact_sla` function for each skill, which evaluates how the inclusion of a specific skill affects SLA achievement.

I will proceed step by step for each skill: Python, Quantum Physics, Cyber Engineering, R
- **Expected:** Can you share the requisition ID (e.g., 05958BR or 05959BR)?

### tasks/task_8: `wrong_tool`

- **Query:** How is the SLA metric defined for 05958BR?
- **Expected tools:** candidate_source_definitions_and_methodology
- **Called:** candidate_source_metadata_and_timeframe
- **Answer:** The SLA metric for requisition ID 05958BR is defined within the context of the data timeframe from October 9, 2023, to March 15, 2025. This period covers the analysis of 40 total requisitions, with the data last updated on April 29, 2025. However, the specific definition of the SLA metric (e.g., target completion time, success thresholds) is not explicitly provided in the metadata. For a detailed 
- **Expected:** SLA is defined as 'Percentage of candidates reviewed within the defined SLA window (e.g., 48 hours)'.

### tasks/task_9: `wrong_tool`

- **Query:** How many requisitions were used to compute these metrics for 05958BR?
- **Expected tools:** candidate_source_definitions_and_methodology
- **Called:** candidate_source_metadata_and_timeframe
- **Answer:** The metrics for requisition ID 05958BR were computed using data from 40 requisitions.
- **Expected:** Metrics are computed from 1047 requisitions.

### tasks/task_10: `wrong_tool`

- **Query:** What are the top metrics considered when evaluating candidate sources for 05958BR?
- **Expected tools:** candidate_source_definitions_and_methodology
- **Called:** candidate_source_source_recommendation_summary
- **Answer:** The top metrics considered when evaluating candidate sources for requisition ID 05958BR are:

1. **Jobs Filled Percentage**: Measures the proportion of job postings successfully filled by each source.
2. **First Round Review Percentage**: Indicates the percentage of candidates who passed the initial review stage.
3. **Offer Acceptance Rate**: Reflects the rate at which offers made to candidates we
- **Expected:** The top metrics considered are: SLA %, First round review %, Offer acceptance rate, Candidate volume, Total hires.

### tasks/task_13: `incomplete_chain`

- **Query:** For 05958BR, when evaluating the SLA impact of Python, which models were used, what was the SLA delta, and what is the system definition of SLA?
- **Expected tools:** skills_skill_impact_sla, skills_data_sources_used, candidate_source_definitions_and_methodology
- **Called:** skills_model_registry, skills_skill_impact_sla, skills_successful_posting_criteria
- **Answer:** For requisition 05958BR:

- The models used to evaluate the SLA impact of Python are:
  1. Skill relevance classifier (version 2.1.0)
  2. SLA impact regression model (version 1.4.2)
  3. Funnel conversion recommender (version 3.0.0-beta)

- The SLA delta for Python is 0% (the SLA achievement rate is 90% with and without Python).

- The system definition of SLA (Service Level Agreement) is:
  - Mi
- **Expected:** 'SLA impact regression model' was used. The SLA delta for Python was 0.0%. SLA is defined as 'Percentage of candidates reviewed within the defined SLA window (e.g., 48 hours)'.

## My interpretation

_Write this yourself after reading the trajectories: which stage dominates, why you think it happens, and what you will try next._

## Notes

- 13 of the 26 core BPO tasks expect NO tool call (the agent should ask for a missing ID or decline); those test restraint, not tool use.
- Success = every expected keyword group appears in the final answer.
