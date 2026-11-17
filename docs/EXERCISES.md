# Make it yours: exercises in build order

Each exercise teaches one idea you will be asked about in interviews and by reviewers, and each produces a result for the papers. Do them roughly in order; write what you find in `RESEARCH_LOG.md`.

## Week 1: understand the loop

1. **Trace one task by hand.** Pick a failed BPO task from `trajectories.jsonl`. Rebuild the exact message list the model saw (system, user, assistant tool calls, tool results). Where did it go wrong? This is how you learn what a "trajectory" is.
2. **Read `classify_failure`, then try to break it.** Find one attempt where the assigned stage is wrong. Fix the rule or add a stage, and add a test in `tests/test_scoring.py`.
3. **Prompt sensitivity.** Change one line of `SYSTEM_PROMPT` and rerun BPO. How much does success move? This tells you how noisy your baselines are before any training.

## Week 2: measurement you can defend

4. **Reliability.** Run BPO with `--repeats 5 --temperature 0.7`. Plot pass^k for k = 1..5. Read the tau-bench paper section on pass^k first.
5. **Proxy against official.** On 20 capability-3 tasks, compare `answer_contains_gt` against IBM's evaluator. Where do they disagree, and why? This decides whether the proxy is good enough as an RL reward.
6. **Shortlisting.** Capability 2 has 100+ tools per domain. Compare `--top-k-tools 10, 20, 40` and report the share of ground-truth tools that survive shortlisting. Then replace BM25 with an embedding model and compare.

## Week 3: towards training

7. **SFT data.** Convert ground-truth trajectories from the train split into chat-format examples (system, user, assistant tool call, tool result, ..., final answer). Hold out whole domains for evaluation, never random samples.
8. **Reward function.** Write `vakra_rl/rewards.py` with `binary_reward`, `gated_reward` and `turn_reward`, each taking a trajectory and its ground truth and returning a float. Unit-test each one on hand-made trajectories, including one that tries to cheat (calls every tool, or copies a tool result without answering).
9. **One GRPO step.** Before renting a GPU, run TRL's GRPOTrainer for a few steps with a 0.5B model on CPU or Kaggle, using your environment and reward. The goal is a pipeline that runs, not a good model.
