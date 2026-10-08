# Contributing

This repository is a small, verifiable lab bench for RL on LLM agents: environments with a four-method interface, graders with hidden tests and sandbox defenses, a harness, evaluation scripts, and fifteen experiments with their predictions and results. Contributions that fit it best are **new environments, new graders or cheating probes, and replications at other scales**. The trainer itself is not the contribution surface; a change to it needs a replicated score to be reviewed.

## Run it first

```bash
make test     # six self-checks, fake model + real sandbox, no GPU, under a minute
make smoke    # five-minute tour of the environments, harness and graders
```

Then pick one of the scripts in `experiments/` and run it on your GPUs; each one ends with the numbers recorded in the lab log so you can reconcile your run against them.

## Adding an environment

An environment is a class with four methods and two attributes; the harness calls nothing else (`pkg_env.py` and `guess_env.py` are the smallest examples):

| Member | Role |
|---|---|
| `stops` | the closing tags generation stops on, e.g. `["</test>", "</write>", "</run>"]` |
| `fake_tags` | the opening tags only the environment may write, e.g. `["<result>"]`; a model that writes one is caught and penalized |
| `reset(problem) -> state` | start an episode: build the working directory, load the initial files, zero the counters |
| `initial_obs(problem) -> str` | optional first observation shown in the prompt |
| `step(state, full_text) -> (injected_text, done)` | the harness stopped on a tag: execute it, mutate the state, return the observation to inject |
| `score(state, full_text) -> (reward, diagnostics)` | the grader: run the hidden tests on the final state, return a number and the readouts the evaluators print |

Design decisions to state explicitly (see §2.3 of the report): what the state contains, which actions exist and what each costs, what the observation returns and how strong it is, where the problems come from and whether they can be generated without limit, what counts as success and how the grader is defended, and the budget (calls, tokens, timeouts). Add a test under `tests/` in the style of the existing ones: a scripted policy, a real sandbox, assertions on the readouts.

## Adding an experiment

Follow the seven-step template the fifteen experiments used (§2.1 of the report):

1. One question, one variable, and **predictions written down before the run** with the numbers you expect.
2. The bench: problem generator, verifier, environment, grader, harness, templates, held-out set, defenses, unit test, readouts.
3. Two probes before training: a capability probe for `p`, and a cheating probe for holes in the grader (run again after RL).
4. Teacher data, with the list of what it is meant to teach and where that list came from.
5. SFT, then evaluation: pass@1 at ×8 and pass@32, held-out problems and held-out templates, failure modes.
6. RL, run twice with the same recipe so the noise floor is known; same evaluation.
7. Conclusions, the prediction reconciliation (what was predicted, what happened, hit or miss), and the debts it leaves.

A pull request that adds an experiment should contain the script under `experiments/`, the raw evaluation JSON or a link to it, and a short ledger entry in the PR description in the same shape as Appendix C of the report.

## Open experiments

Ranked by cost on four 24 GB GPUs; all of them are debts named in §8 of the report.

| | Experiment | Why it matters | Cost |
|---|---|---|---|
| 1 | Second RL runs for ②-B and ③ with the same recipe | every price-of-observation number is a single run; this puts a noise floor under them | 2 × 4 h |
| 2 | `bench_observation_price.py` on other models: Qwen2.5-Coder 7B / 32B, open agentic models | turns one 1.5B point into a curve: does the price shrink with scale? | 1 h per model on an 80 GB GPU |
| 3 | The middle arm: the environment runs only the assertions from the problem statement's examples | separates "has a verifier" from "has the author's verifier" | 1 day |
| 4 | Textbook v2 for stage ③: fewer, more accurate assertions | tests whether the 17–22% false-positive rate of self-written checks is a data problem | 1 day |
| 5 | The teacher as the grader: on-policy distillation with a local 7B | can judgment be installed into a 1.5B at all? | 2 days |
| 6 | 4B (LoRA) and 7B (80 GB, full parameter) replications of the coding stages | the report's conclusions are limited to 1.5B; this is the first test of whether they hold | 1–3 days |

## Conventions

- Scripts stay flat in the repository root and import each other by file name.
- Every model is reported with pass@1 (temperature 1, 8 samples) and pass@32; differences under one standard error (about .03 on 100 problems) are read as flat.
- The DeepSeek key is read only from the environment variable `DEEPSEEK_API_KEY`; never write it into a file.
- Chinese is the working language of the lab log (`PLAN.md`); English is fine for everything else.
