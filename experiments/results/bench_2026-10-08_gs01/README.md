# Re-evaluation of the four coding checkpoints, 2026-10-08

Machine: the lab machine (4x RTX 4090), one checkpoint per GPU, `bench_observation_price.py --hf-dir <ckpt> --model hf_coder_1.5b`
from a clean clone of this repository at commit 0a1c4c4. 100 training problems x 8 samples, temperature 1, visible tests available
(strong) and taken away (weak). Each `summary_*.json` is the script's output; raw trajectories stayed on the machine.

| checkpoint | with tests | tests removed | price | recorded in the report |
|---|---|---|---|---|
| hf_sft_ex (SFT ②-B) | .343 | .204 | -.139 | .329 / .240 / -.089 |
| hf_exRL_bin (RL ②-B) | .459 | .314 | -.145 | .446 / .295 / -.151 |
| hf_sft_exw (SFT ③) | .210 | .196 | -.014 | weak .186 |
| hf_exwRL_bin (RL ③) | .265 | .296 | +.031 | weak .282 |
