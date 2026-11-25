# PCG imitation experiment v1 artifacts

These small checkpoints and reports are committed so the first learned generalization result remains available even if local `runs/` directories are removed.

| File | Purpose |
| --- | --- |
| `bc_model.zip` | 80-epoch behavior cloning student; SHA-256 `6f9bfd802e6074af5179c118563ead67a4077d80ad58daf099819e338a925713` |
| `bc_demonstrations.npz` | Observation/action training pairs from 48 generated maps |
| `selected_ppo.zip` | Development-selected BC-initialized PPO, seed 17; SHA-256 `2f875044416a539f32fa8193696a91f7e47f378fdeeddad4b3d462c2dfc4ebf1` |
| `*_report.json` | Full per-map training, development, and final holdout results |

Load a checkpoint with `stable_baselines3.PPO.load(path, device="cpu")`. These experimental checkpoints use `PlatformerState-v2`, four physics ticks per action, a 768-action episode limit, and the frozen `segments-v1` physics profile. They are not legacy v9 checkpoints and do not carry the older playback sidecar format. The source code and reproduction commands are in [the experiment record](../../docs/PCG_IMITATION_EXPERIMENTS.md). The 12-map holdout is exposed; freeze a new holdout for subsequent tuning.
