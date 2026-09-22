---
title: From pretraining from scratch to a coding agent
description: Four RTX 4090s, one person, four weeks, fifteen experiments: one variable at a time, predictions written before each run, reconciled after
lang: en
---

English · [中文](../zh/)

# From pretraining from scratch to a coding agent

**Fifteen experiments on four RTX 4090s: one variable at a time, predictions written before each run, reconciled after**

Author: Qirun Li　·　Period: August 26 to September 22, 2026　·　Compiled by: Claude (conversational collaboration)

> This report is an edited version of the experiment log `PLAN.md` (eleven thousand lines). Each experiment gives its setup, data, charts, and conclusions; every number comes from the original log. The predictions written before each run and the reconciliation after it are kept together, with both ✓ and ✗. All conclusions are limited to the model sizes used here (95M to 1.5B) and to these tasks; they are not extrapolated to larger models.

---

## 0. One page

**What was done.** Starting from Stanford CS336, pretrain two English models from scratch (95M, 201M) on four 4090s, do SFT, then move to RL: first reproduce GRPO on GSM8K, then use the Countdown number game to chase DeepSeek-R1's "aha moment" (顿悟), raising this task's pass@1 from 0.02 to 0.75, and to 0.98 with the expert exit open; then build the first stateful environment (number guessing); finally move to coding agents: bug fixing, bundling, writing real modules from scratch, and taking away the tests so the model has to verify itself, four stages in all.

**How.** Each experiment changes one variable; predictions are written down before the run and reconciled after it; every model reports two numbers, pass@1 (reliability) and pass@32 (ceiling); a cheating probe is run at every stage; the hit record of more than forty predictions is kept in the appendix.

**What came out.** Six regularities that run through the whole project, each with its own numbers:

| Regularity | One sentence | Evidence |
|---|---|---|
| RL moves within the support set | pass@1 rises, the pass@k ceiling does not move; opening new paths happens only at the 1/k margin | GSM8K pass@64 is .980 for all three models; ②-B held-out pass@32 actually drops |
| Three-stage division of labor | pretraining gives the parts, SFT gives the procedure, RL gives the preference | 200M SFT "knowing when to stop ≠ knowing how to answer"; three arms: self-distillation = RL only, seeding +.09 |
| The shape of the scale decides what gets learned | process rewards get gamed; final-outcome reward only + pool screening + anchor turns RL from learning bad habits into a safe small gain | Arm 3 grader gamed at step 126; Arm S, Conclusion ⑯ |
| The price of observation | same problems, same scale, only the tests taken away: SFT −.09, RL −.16, ceiling −.06, and the whole price falls on the fix loop | Stage ③ |
| Fixing is a part, not a habit | the improved rate per fix round is 10 to 20%, unchanged across three settings, before and after SFT, before and after RL | ②-A, ②-B, ③ |
| Opening new paths = support-set density × feedback density × sampling budget | the denser the feedback, the weaker the prior that can still open a path | Countdown cannot walk blind and needs seeding; number guessing grows binary search from 4% with pure RL |

**Three main charts.**

![Countdown 4-number pass@1 ladder](../assets/en/countdown_ladder.svg)

![GSM8K pass@k convergence](../assets/en/gsm8k_passk.svg)

![The price of observation](../assets/en/observation_price.svg)

---

## 1. Roadmap

### 1.1 Timeline

| # | Date | Experiment | Base model | Key numbers | Conclusion |
|---|---|---|---|---|---|
| 1 | from 08-26 | CS336 self-study, no videos, all by conversation | | Covered L1 L2 L3 L9 L11 | |
| 2 | 08-27 / 08-28 | Pretraining from scratch twice: 95M / 1.5B tokens, 201M / 10B tokens | self-trained | perplexity 30.17 / 18.10 | Beats Chinchilla, loses to GPT-2 small by .148 |
| 3 | 08-30 | SFT on 200M, Alpaca, 2×2 comparison | self-trained 201M | appropriate responses 0/7 → 5/7 | Knowing when to stop ≠ knowing how to answer; post-training does not look at val loss |
| 4 | 09-01 / 09-02 | GSM8K GRPO, two rounds | Qwen2.5-1.5B-Instruct | greedy .687 → .767; pass@64 .980 unchanged | RL only reorders, does not create; entropy collapse |
| 5 | 09-02 | Countdown 3-number, R1-Zero reproduction | Qwen2.5-1.5B Base | .035 → .405, length 186 → 48 | Shortcut; conditional policy binding |
| 6 | 09-02 to 09-04 | Mixed training Countdown + GSM8K, four segments | same | CD-3 .405 → .61; mix4-300 CD-4 .393 | Shortcut blocked; the functional form of the aha moment appears, long chains of thought do not |
| 7 | 09-04 to 09-06 | Three-arm CD-4 + Arms 3/3′ changing only the reward | mix4-300 | RL only .471 / self-distillation .473 / seeding .563 | New parts come only from the teacher; process reward gamed |
| 8 | 09-07 to 09-08 | Calculator tool, Arm T, the first harness | same | SFT .506 → .710; RL .711 | ⑮ once the parts are fixed, RL has only shaping-away left |
| 9 | 09-08 | Arm S, stripped-down scale + pool screening + anchor | same | .754; CD-3 ceiling .92 → 1.00 | ⑯ RL moves within the reachable set |
| 10 | 09-10 to 09-11 | Arm A, asking an expert with a budget | same | with help .979, hard problems .946; wording change .966 → .036 | ⑰ the method binds to the prompt wording; the expert = all the gain on hard problems |
| 11 | 09-11 | E number guessing, three arms, the first stateful environment | hf_armA | pure RL .63 / curriculum .84 / seeding .976 | ⑱ explicit state comes from the textbook |
| 12 | 09-11 to 09-13 | ① short-horizon strong observation: bug fixing | Qwen2.5-Coder-1.5B | SFT +.55, RL +.12, pass@32 unchanged | ⑲ ⑳ |
| 13 | 09-14 to 09-15 | ②-A long-horizon strong observation 1: independent bundling | same | p^K; file-level pass@32 +.02 | ㉑ to ㉖ |
| 14 | 09-15 to 09-16 | ②-B long-horizon strong observation 2: exercism from scratch | same | SFT .329 → RL .446; held-out pass@32 .600 → .533 | ㉗ gain in the first version, problem binding, tail cutting |
| 15 | 09-17 | ③ long-horizon weak observation: write your own tests | same | SFT .186 → RL .282; strong-room control .446 | ㉘ the price of observation |

### 1.2 Chain of motivation: each next step is the previous step's debt

1. Train 95M from scratch: walk the pretraining pipeline by hand; perplexity 30 against GPT-2 small's 26 → ask how post-training is done.
2. 200M SFT: install the dialogue format; find that post-training cannot pick a model by val loss → need a task that can be scored automatically.
3. GSM8K GRPO: pass@1 .53 → .74, greedy only +.08, the format emerges on its own, entropy collapse → "RL eliminates bad paths, it does not improve the best path", but what was learned cannot be stated → need a more controllable task.
4. Countdown 3-number: after RL the answers get shorter and the reasoning disappears → what RL changes is behavior in a specific context → ask what the conditions for the aha moment are.
5. Mixed training: the task must genuinely need search and the starting point must be nonzero; CD-3 is too easy and RL eliminates the reasoning → switch to CD-4.
6. Three-arm CD-4: self-distillation = RL only, seeding +.09 → new parts come only from the teacher → ask whether a reward can teach a process.
7. Arm 3, changing only the reward: the process grader is gamed at step 126 → from now on only final-outcome rewards → where are the remaining errors.
8. Calculator / Arm T: the failures are all on lines with arithmetic errors → outsource to a calculator, value from .45 to 1.00, build the harness; after RL the model actually learns bad habits → three options: protect the seed, change the scale, change the teacher.
9. Arm S: reward only, no penalty + problem screening + anchor → safe small gain, pass@64 unchanged → RL moves within the reachable set; hard problems still all wrong → need external information.
10. Arm A: the textbook installs properly, RL only trims runaway behavior, the expert is all the gain on hard problems, the cost is tool dependence; the first multi-turn action → set the "agent + RL" direction.
11. E number guessing: the smallest agent environment; pure RL grows binary search from 4%, curriculum beats training directly on the hard setting, seeding is best, N=1000 relies on "writing down the interval" → explicit state comes from the textbook, the harness is generalized → ready for real tasks.
12. Define the coding line: two axes, observation strength × horizon length; a coding environment can be tightened all the way, and the product sits on this line → four-cell curriculum.
13. ① bug fixing: get the loop, the sandbox, and the scale running; zero-shot ≈ 0 → a cold start is required; SFT +.55, RL +.12; pass@32 unchanged; strict scale + cheating probe → ask what happens when the path gets longer.
14. ②-A bundling: deliberately independent, to compare against p^K; the multiplicative effect, the state line has no effect, bin and frac reach the same endpoint, file-level pass@32 +.02 → the original plan was to inject coupling, changed to real modules.
15. ②-B exercism from scratch: one step toward the real thing; SFT .33, RL .45, the gain is in the first version; the pass@32 gain is entirely on seen problems, half of it is problem binding, held-out shows tail cutting → debts: the problem pool is small, fixing did not move.
16. ③ weak observation: take away the tests to examine whether the model can build its own verification, and expand the pool along the way to treat problem binding; the action is installed, RL keeps it, but it does not earn points; judging and fixing are parts → three options: fewer but more accurate assertions / the teacher as the scale / 4B.

In one sentence: the aha moment cannot be explained → find a controllable task; self-distillation does nothing → find a teacher; process reward gamed → final outcome only; arithmetic errors → add a tool; learning bad habits → fix the scale; hard problems unsolvable → ask the expert; a single turn is not enough → multi-turn; toys are done → real tasks; short horizon measured → lengthen it; long-horizon strong observation measured → remove the observation; action installed → parts missing. The next debt: where do parts come from.

---

## 2. Method

### 2.1 Experiment procedure template

Each experiment follows seven steps. The seven were summarized after all fifteen experiments were done; the earlier experiments did not all go through them completely:

1. **Question, variable, prediction.** One variable at a time; the prediction is written before the run and reconciled after it.
2. **The bench**, ten items: problem generator (where the problems come from, with reference answers), verifier (sandbox runner), environment (stateful, takes actions, returns observations), scale (how many points at submission), harness (the loop of generate, stop, execute, inject, mask), template (problem wording; training and held-out kept separate), held-out set (fixed on day one: problems, templates, bug types), defenses (how the scale resists gaming), unit tests (a fake model plays a few rounds to check all of the above), readouts (behavioral metrics in the eval report: by position, failure mode, writing form, group distribution).
3. **Two probes.** The capability probe measures p; the cheating probe measures whether the scale has holes, and is run again after RL.
4. **Textbook.** Source (program teacher / distillation / self-sampling with filtering); the list of what to teach comes from the probe's failure modes; held-out templates do not enter the textbook.
5. **SFT + eval**, K curve, held-out, failure modes. Failure modes fall into two classes: habit-type ones RL can suppress; missing-move ones go back to step 4.
6. **RL, two runs** (repeating the same recipe once gives a noise floor) + eval; smoke test before the run; infrastructure changes change only speed, not scores, and after each change first verify "same score, only faster".
7. **Conclusion, prediction reconciliation, debts.** The debts are the source of the next round's step 4.

### 2.2 Evaluation protocol

- **pass@1**: temperature 1, 8 samples per problem, 100 out-of-pool problems, reported as the mean over 800 samples. 1σ is about 0.03 to 0.04; any difference smaller than that is read as flat.
- **pass@k ceiling**: the same 100 problems, 32 samples each (64 on the Countdown line), unbiased estimator. pass@1 measures reliability, pass@32 measures the ceiling, and the gap between the two decides whether to run RL or change the base model.
- **Greedy** is only a training dashboard and does not enter conclusions: greedy decoding hides the gain (GSM8K greedy +.03, temperature 1 +.22).
- **Held-out** comes in three kinds: held-out problems, held-out templates (wording), held-out bug types, fixed on day one.
- **Long-horizon tasks** are measured at the atomic granularity: bundles report file-level and bundle-level results separately.

### 2.3 Definition of an environment

An environment = the world the model acts in, with three features, none of which may be missing: it has state (it remembers how things are now), it takes actions (it can only be changed through the defined actions), and it returns observations (the response depends on the current state). The one-sentence test: if the same action is done twice, will the response differ? Yes → environment; no → tool (a calculator or a verifier computes whatever it is given and remembers nothing). The scale reads the final state and scores it; it is not itself the environment. The harness is the pipe connecting the model and the environment. Environment design is six decisions: state, actions and their price, observation strength, problem source, scale and defenses, budget.

---

## 3. Experiment details
