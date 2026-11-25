# Generated-map imitation experiments

The experiment uses `PlatformerState-v2` observations and actions. Training requests are built from the six frozen families with separate seeds, and every training geometry is checked against both frozen evaluation splits. The student sees only the 15-element observation vector. Saved run artifacts under `runs/` include the demonstrations, model, geometry hashes, training settings, and development results; `runs/` is intentionally ignored by Git.

## Stage 1: behavior cloning

Run `python -m scripts.train_pcg_bc --output runs/pcg_bc_v1`. The default pool is eight training seeds × six families = 48 maps, with 80 supervised epochs and training seed 17. The teacher finished all 48 maps; 6,880 observation/action pairs were recorded. The learned MLP student reached the goal and met the per-map collection target on 12/12 frozen development maps; mean coin ratio was 0.9722. The scripted teacher's development coin ratio was 1.0. These are static map results from one policy seed, not independent training trials. The 12-map holdout remains unopened.

The 80 epochs were chosen after inspecting development performance, so development is a model-selection set. Early 10-epoch behavior cloning mostly learned to run right (2/12 goal reaches). The experiment tests learning from demonstrations, not PPO improvement or broad generalization beyond the defined six families.

## Stage 2: DAgger on learner states

Run `python -m scripts.train_pcg_dagger --bc-run runs/pcg_bc_v1 --output runs/pcg_dagger_v1 --epochs-per-round 5 --learning-rate 0.0001`. Each round runs the current student on the same 48 training maps, records observations it actually visits, obtains action labels from the scripted teacher, and refits on the accumulated dataset. The teacher discards its pending short-jump action when the student executes a different action. The initial BC model and every DAgger round are evaluated on development; the selected checkpoint maximizes collection targets, then goal reaches, then mean coin ratio.

In the first two-round run, the student reached all 48 training-map goals but disagreed with the teacher on 241 actions in round one and 71 in round two. Both refitted rounds reached 12/12 development goals but met only 10/12 collection targets, below the BC baseline's 12/12. A gentler five-epoch, `1e-4` run also ended at 10/12. The selected checkpoint therefore remains round 0 (BC). This is a negative DAgger result, possibly reflecting rare-action imbalance or limits of a scripted teacher labeling off-policy states; it does not establish the cause. The aggregated data and per-round outcomes are retained for diagnosis. Holdout remains unopened.

## Stage 3: equal-budget PPO comparison

Run `python -m scripts.compare_pcg_ppo --bc-run runs/pcg_bc_v1 --output runs/pcg_ppo_equal_v1`. Scratch and BC-initialized models use the same 48-map training pool, network, PPO settings, three independent training seeds (17, 23, 31), and 16,384 environment transitions per run. Only the actor weights transfer from BC; the critic and PPO optimizer start fresh. Development results:

| Training seed | Scratch goal / target | BC init goal / target |
| --- | ---: | ---: |
| 17 | 2/12 · 0/12 | 12/12 · 12/12 |
| 23 | 2/12 · 2/12 | 7/12 · 6/12 |
| 31 | 2/12 · 0/12 | 12/12 · 12/12 |

BC initialization helps at this short budget, but one seed regressed from the original BC student's 12/12 collection target result. This is a paired feasibility comparison, not evidence that PPO consistently improves on BC. No holdout results were used to choose settings.

## Stage 4: adaptive map sampling

Run `python -m scripts.compare_pcg_sampling --bc-run runs/pcg_bc_v1 --output runs/pcg_sampling_v1`. This compares uniform sampling with a simple failure-prioritized replay rule under the same BC initialization, three seeds, training pool, PPO settings, and 16,384-transition budget. After each **training** episode, the rule updates that map's exponential moving average of failure (1) or coin deficit after success (0–1). Half the sampling probability stays uniform. Development and holdout episodes never update priorities. This is a pragmatic level-sampling prototype inspired by [Prioritized Level Replay](https://proceedings.mlr.press/v139/jiang21b.html), not the paper's full learning-potential scoring method.

| Training seed | Uniform goal / target | Prioritized goal / target |
| --- | ---: | ---: |
| 17 | 12/12 · 12/12 | 12/12 · 11/12 |
| 23 | 7/12 · 6/12 | 12/12 · 12/12 |
| 31 | 12/12 · 12/12 | 12/12 · 12/12 |

The total targets met across these three development runs increased from 30/36 to 35/36, but the effect varies by seed and there are only three training seeds. Episode counts differ because policies survive for different lengths even at equal transition budgets. The run report preserves per-map sampling counts. Holdout remains unopened at this stage.
