# vakra-rl

Training small open models (4B parameters or fewer) to use tools reliably, with rewards that come from **executing** the tool calls, on IBM's [VAKRA](https://github.com/IBM/vakra) benchmark.

VAKRA gives an agent 8,000+ real APIs over 62 domains, plus document collections, and asks multi-hop, multi-source questions with natural-language tool-use policies. This repository builds the pieces needed to study three questions:

- **RQ1.** For models of 4B parameters or fewer, does reinforcement learning (GRPO) with execution-verified rewards beat supervised fine-tuning on the same data?
- **RQ2.** Which reward works best: binary final-answer reward, a gated reward (policy, then tool-response match, then answer), or turn-level reward? Which invites reward hacking?
- **RQ3.** Where in the pipeline do small models fail (choosing tools, filling arguments, or reading the answer out of tool results), and does RL change that? Does it survive 8-bit and 4-bit quantization?

This is also the basis of a SemEval-2027 Task 11 (VAKRA-Advanced) system.

## Status

| Stage | State |
|---|---|
| Agent loop over VAKRA's MCP tool servers, any OpenAI-compatible model | done |
| Proxy scoring, stage-wise failure taxonomy, pass^k reliability | done |
| Selective dataset download and training-split analysis | done |
| Baselines on the BPO domain and capability 3 | in progress |
| SFT data from ground-truth trajectories | next |
| GRPO with execution-verified rewards (TRL) | planned |
| Quantization and internal analysis (RQ3) | planned |

## How it works

```
 VAKRA task ──► agent loop (vakra_rl/agent.py) ──► model server (Ollama / vLLM / Groq)
                   │    ▲                              OpenAI-compatible API
     tool call     ▼    │ tool result
              MCP tool server (vakra_rl/mcp_env.py)
              BPO: local Python process   caps 1–4: VAKRA Docker containers
                   │
                   ▼
   trajectory ──► scoring (vakra_rl/scoring.py) ──► report.md, failure stages, pass^k
              └─► prediction/<domain>.json ──► IBM's official evaluator (LLM judge)
```

The same environment object is what the GRPO trainer will call during rollouts, so baseline numbers and training rewards come from the same code path.

## Layout

| Path | What it is |
|---|---|
| `vakra_rl/mcp_env.py` | Starts a VAKRA tool server and executes tool calls (the RL environment) |
| `vakra_rl/llm.py` | One client for Ollama, vLLM, Groq and OpenRouter, with recovery of tool calls the model wrote as plain text |
| `vakra_rl/agent.py` | The tool-use loop, trajectory records, VAKRA output format, BM25 tool shortlisting |
| `vakra_rl/scoring.py` | Tool-call matching, answer checks, failure stages, pass@k and pass^k |
| `vakra_rl/vakra_io.py` | Readers for VAKRA's JSON files, robust to their two `sequence` shapes |
| `vakra_rl/report.py` | Summary tables and example failures as markdown |
| `scripts/download_vakra.py` | Download only what you need, with a size plan first |
| `scripts/analyze_data.py` | Statistics and figures for the training split |
| `scripts/run_bpo_baseline.py` | Baseline on VAKRA's BPO domain (no Docker needed) |
| `scripts/run_vakra_baseline.py` | Baseline on capabilities 1–4 through VAKRA's Docker servers |
| `docs/SETUP_WINDOWS.md` | Step-by-step setup for a Windows laptop with a 4 GB GPU |
| `docs/RESEARCH_LOG.md` | Dated log of runs and findings |
| `docs/EXERCISES.md` | Exercises in build order that turn this scaffold into your own results |

## Quick start

Clone [IBM/vakra](https://github.com/IBM/vakra) next to this repository, then:

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: see docs/SETUP_WINDOWS.md
pip install -r requirements.txt && pip install -e .
python -m pytest -q

ollama pull qwen3:4b-instruct-2507-q4_K_M
python scripts/run_bpo_baseline.py --vakra-root ../vakra --limit 3     # smoke test
python scripts/run_bpo_baseline.py --vakra-root ../vakra               # full BPO run

python scripts/download_vakra.py --parts train --max-file-mb 50        # size plan
python scripts/download_vakra.py --parts train --max-file-mb 50 --yes
python scripts/analyze_data.py --data-root ../vakra/data
```

## Results

Numbers go here only after a run produced them. Each row links to its `runs/.../report.md`.

| Model | Setting | Success | Main failure stage | Report |
|---|---|---|---|---|
| | | | | |

## Metrics, honestly

`scoring.py` computes **proxy** metrics without an LLM judge, for fast iteration and as RL rewards. VAKRA's official score comes from IBM's evaluator, which replays tool calls and uses an LLM judge for correctness and groundedness. Reported results state which one they use.

Sanity check: replaying the reference tool calls and answers on the 45 BPO tasks scores 44/45 with the proxy metric. The miss is a task whose own reference answer does not contain its required keywords.

## Data and licence

VAKRA's data is released by IBM under CC BY-NC-SA 4.0 and is not redistributed here. Download it from [ibm-research/VAKRA](https://huggingface.co/datasets/ibm-research/VAKRA).

```bibtex
@misc{vakra-bench,
  title  = {VAKRA: A Benchmark for Evaluating Multi-Hop, Multi-Source Tool-Calling in AI Agents},
  author = {Naik, Ankita Rajaram and Murthi, Anupama and Elder, Benjamin and Huo, Siyu and
            Venkateswaran, Praveen and Contractor, Danish},
  year   = {2026},
  url    = {https://ibm-research-vakra.hf.space}
}
```
