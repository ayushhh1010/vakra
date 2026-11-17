# BPO baseline: qwen3:4b-instruct-2507-q4_K_M

Proxy metrics computed by vakra-rl (no LLM judge). Not the official VAKRA score.

| Setting | Value |
|---|---|
| environment | `VAKRA BPO (local MCP)` |
| provider | `ollama` |
| model | `qwen3:4b-instruct-2507-q4_K_M` |
| temperature | `0.7` |
| repeats | `3` |
| max_steps | `8` |
| task_files | `tasks,tasks_edge_cases,tasks_undocumented,tasks_http_errors,tasks_type_mismatch,tasks_schema_violations` |
| date | `2026-10-30 12:33` |

## Overall

| Metric | Value |
|---|---|
| Attempts (tasks x repeats) | 135 (45 tasks) |
| Success rate | 54.1% |
| Tool-name recall (mean, tasks that need tools) | 0.77 |
| Tool name+argument recall (mean, same tasks) | 0.77 |
| Tool calls per attempt (mean) | 1.5 |
| Latency per attempt, mean / median (s) | 17.8 / 12.8 |
| Largest prompt seen (tokens) | 4084 |
| Tool calls recovered from plain text | 0 |
| pass@3 (any of 3 attempts right) | 57.8% |
| pass^3 (all 3 attempts right) | 51.1% |

## Where it breaks (first failing stage)

| Stage | Meaning | Count | Share |
|---|---|---|---|
| `success` | Answer correct | 73 | 54.1% |
| `llm_error` | Model server error or timeout | 3 | 2.2% |
| `should_not_call_tools` | Expected a clarifying question or refusal; agent called tools instead | 17 | 12.6% |
| `wrong_tool` | None of the needed tools were called | 11 | 8.1% |
| `incomplete_chain` | Some but not all needed tools were called | 16 | 11.9% |
| `readout_failure` | Right tools and arguments, but the final answer is wrong | 5 | 3.7% |
| `wrong_answer` | No tools expected, answer still wrong | 10 | 7.4% |

## By task group

| Group | Attempts | Success | Name recall | Name+arg recall | Calls |
|---|---|---|---|---|---|
| tasks | 78 | 26.9% | 0.45 | 0.45 | 1.9 |
| tasks_edge_cases | 15 | 80.0% | 1.00 | 1.00 | 1.0 |
| tasks_http_errors | 12 | 100.0% | 0.92 | 0.92 | 0.9 |
| tasks_schema_violations | 12 | 100.0% | 1.00 | 1.00 | 1.0 |
| tasks_type_mismatch | 9 | 77.8% | 1.00 | 1.00 | 1.4 |
| tasks_undocumented | 9 | 100.0% | 1.00 | 1.00 | 1.0 |

## By difficulty

| Difficulty | Attempts | Success | Name recall | Name+arg recall | Calls |
|---|---|---|---|---|---|
| easy | 66 | 39.4% | 0.52 | 0.52 | 1.5 |
| hard | 33 | 66.7% | 0.85 | 0.85 | 1.9 |
| medium | 36 | 69.4% | 0.88 | 0.88 | 1.3 |

## Example failures

### tasks/task_1: `wrong_tool`

- **Query:** For requisition 05958BR, which source has the lowest SLA performance?
- **Expected tools:** candidate_source_sla_per_source
- **Called:** candidate_source_sla_extended
- **Answer:** The source with the lowest SLA performance for requisition 05958BR is not explicitly identified in the provided data. However, the SLA performance metric for all sources combined shows a percentage of 0%, which indicates a significant issue. Since no specific source is assigned a non-zero SLA percentage, and the overall SLA is 0%, it suggests that no sourcing channel met the SLA target for this re
- **Expected:** CyberSec Jobs with 67%

### tasks/task_3: `incomplete_chain`

- **Query:** For requisitions like 05958BR, which sources provided the most candidates, and how effective were they at converting to hires?
- **Expected tools:** candidate_source_candidate_volume_by_source, candidate_source_source_recommendation_summary
- **Called:** candidate_source_candidate_volume_by_source, candidate_source_funnel_conversion_by_source
- **Answer:** The sources that provided the most candidates for requisition 05958BR were LinkedIn, Dice, GitHub, Indeed, Internal, and Referral, each contributing approximately 18% to 14% of the total candidate volume.  

In terms of conversion effectiveness to hires, the performance of the sources in the funnel stages is as follows:  
- **LinkedIn**: 75.1% first-round review rate, 20.4% interview rate, 1.9% of
- **Expected:** LinkedIn: 519 candidates (18%), 7 hires. Offer acceptance rate: 70%. Dice: 516 candidates (18%), 11 hires. Offer acceptance rate: 79%. GitHub: 468 candidates (16%), 10 hires. Offer acceptance rate: 77%.

### tasks/task_3: `incomplete_chain`

- **Query:** For requisitions like 05958BR, which sources provided the most candidates, and how effective were they at converting to hires?
- **Expected tools:** candidate_source_candidate_volume_by_source, candidate_source_source_recommendation_summary
- **Called:** candidate_source_candidate_volume_by_source, candidate_source_funnel_conversion_by_source
- **Answer:** For requisition 05958BR:

- The sources that provided the most candidates were **LinkedIn** and **Dice**, each contributing 18% of the total candidate volume (519 and 516 candidates, respectively).
- In terms of conversion effectiveness to hires:
  - **GitHub** had the highest conversion rates, with 76.1% first-round review approval, 23.7% interview rate, and 2.8% offer acceptance rate.
  - **Inde
- **Expected:** LinkedIn: 519 candidates (18%), 7 hires. Offer acceptance rate: 70%. Dice: 516 candidates (18%), 11 hires. Offer acceptance rate: 79%. GitHub: 468 candidates (16%), 10 hires. Offer acceptance rate: 77%.

### tasks/task_3: `incomplete_chain`

- **Query:** For requisitions like 05958BR, which sources provided the most candidates, and how effective were they at converting to hires?
- **Expected tools:** candidate_source_candidate_volume_by_source, candidate_source_source_recommendation_summary
- **Called:** candidate_source_candidate_volume_by_source, candidate_source_funnel_conversion_by_source
- **Answer:** For requisition 05958BR, the sources that provided the most candidates were:

- **LinkedIn**: 519 candidates (18% of total)
- **Dice**: 516 candidates (18% of total)
- **GitHub**: 468 candidates (16% of total)

In terms of effectiveness at converting candidates to hires, the key funnel conversion metrics are as follows:

- **LinkedIn**: 75.1% first-round review rate, 20.4% interview rate, 1.9% off
- **Expected:** LinkedIn: 519 candidates (18%), 7 hires. Offer acceptance rate: 70%. Dice: 516 candidates (18%), 11 hires. Offer acceptance rate: 79%. GitHub: 468 candidates (16%), 10 hires. Offer acceptance rate: 77%.

### tasks/task_6: `wrong_tool`

- **Query:** What are the best sources to prioritize for 05959BR?
- **Expected tools:** candidate_source_sla_per_source, candidate_source_candidate_volume_by_source, candidate_source_funnel_conversion_by_source
- **Called:** candidate_source_list_all_sources, candidate_source_source_recommendation_summary
- **Answer:** The best sources to prioritize for requisition 05959BR are:

1. **Dice** - With 11 total hires, a high offer acceptance rate (271%), and strong performance in first-round reviews (76%), Dice demonstrates the highest hiring success and candidate engagement.

2. **GitHub** - With 10 total hires, a balanced offer acceptance rate (77%), and solid first-round review performance (72%), GitHub is a stron
- **Expected:** You should prioritize Dice, GitHub, and LinkedIn. Dice and LinkedIn both met SLA 100% of the time and brought in 18% of all candidates. Dice had a strong offer conversion rate (2.7%), and GitHub had the highest conversion (2.8%) despite slightly lower SLA. Indeed should be avoided due to 0% SLA and 

### tasks/task_6: `incomplete_chain`

- **Query:** What are the best sources to prioritize for 05959BR?
- **Expected tools:** candidate_source_sla_per_source, candidate_source_candidate_volume_by_source, candidate_source_funnel_conversion_by_source
- **Called:** candidate_source_list_all_sources, candidate_source_source_metrics_lite, candidate_source_source_sla_score, candidate_source_source_sla_score, candidate_source_source_sla_score, candidate_source_funnel_conversion_by_source, candidate_source_total_hires_by_source, candidate_source_candidate_volume_by_source
- **Answer:** (empty)
- **Expected:** You should prioritize Dice, GitHub, and LinkedIn. Dice and LinkedIn both met SLA 100% of the time and brought in 18% of all candidates. Dice had a strong offer conversion rate (2.7%), and GitHub had the highest conversion (2.8%) despite slightly lower SLA. Indeed should be avoided due to 0% SLA and 

### tasks/task_7: `should_not_call_tools`

- **Query:** Out of these skills — Python, Quantum Physics, Cyber Engineering, Risk Analysis, Wireshark — which ones negatively affect SLA performance?
- **Expected tools:** (none: should ask or refuse)
- **Called:** skills_skill_impact_sla, candidate_source_list_all_sources, skills_skill_impact_sla, skills_skill_impact_sla, skills_skill_impact_sla, skills_skill_impact_sla, skills_skill_impact_sla
- **Answer:** All five skills — Python, Quantum Physics, Cyber Engineering, Risk Analysis, and Wireshark — negatively affect SLA performance when included in the requisition. 

For each skill, the data shows a delta of -50, meaning that including the skill reduces SLA achievement from 50% without the skill to 0% with the skill. This indicates a significant negative impact on SLA performance for all listed skill
- **Expected:** Can you share the requisition ID (e.g., 05958BR or 05959BR)?

### tasks/task_7: `should_not_call_tools`

- **Query:** Out of these skills — Python, Quantum Physics, Cyber Engineering, Risk Analysis, Wireshark — which ones negatively affect SLA performance?
- **Expected tools:** (none: should ask or refuse)
- **Called:** skills_skill_impact_sla, candidate_source_metadata_and_timeframe, skills_skill_impact_sla, skills_skill_impact_sla, skills_skill_impact_sla, skills_skill_impact_sla, skills_skill_impact_sla
- **Answer:** All of the skills—Python, Quantum Physics, Cyber Engineering, Risk Analysis, and Wireshark—have a negative impact on SLA performance when included in the requisition. Each skill reduces SLA achievement from 50% without the skill to 0% with the skill, with a delta of -50%. This indicates that all these skills negatively affect SLA performance.
- **Expected:** Can you share the requisition ID (e.g., 05958BR or 05959BR)?

## My interpretation

_Write this yourself after reading the trajectories: which stage dominates, why you think it happens, and what you will try next._

## Notes

- 13 of the 26 core BPO tasks expect NO tool call (the agent should ask for a missing ID or decline); those test restraint, not tool use.
- Success = every expected keyword group appears in the final answer.
