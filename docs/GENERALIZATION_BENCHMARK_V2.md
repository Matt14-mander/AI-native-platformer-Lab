# Frozen geometry development benchmark

Run `python -m scripts.evaluate_generalization --output runs/generalization_v2_development_scripted.json` after installing the benchmark dependencies. The runner evaluates only the `development` split by default. Opening the frozen `holdout` split requires both `--split holdout` and `--open-holdout`; do this only for a final evaluation after choosing a policy.

The manifest contains 12 development and 12 holdout geometries, two generation seeds for each of six families in each split. Geometry hashes are checked before each run. One policy seed is used per static map, so the denominator is 12 maps, not independent policy trials. The scripted baselines use `PlatformerState-v2`, four physics ticks per action, and a 768-action limit. `--model path/to/model.zip` adds a compatible PPO checkpoint when available.

## Initial development results

| Agent | Reached goal | Met collection target | Mean coin ratio |
| --- | ---: | ---: | ---: |
| Move right | 2/12 | 2/12 | 0.3854 |
| Rule jump | 12/12 | 4/12 | 0.6632 |
| Coin jump | 12/12 | 10/12 | 0.9167 |

Coin jump misses the collection target on both `mixed_ground` maps (0.875 coin ratio). This is a development result from scripted agents only. PPO v9 was not evaluated because its checkpoint is unavailable. The holdout split remains unopened.
