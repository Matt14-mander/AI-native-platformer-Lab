# Ground coin jump teacher

The `GroundCoinJumpAgent` changes one scripted decision: when grounded before a low obstacle, with ground ahead and a ground-height coin signal, it holds jump for three environment actions and then releases until landing. It keeps the existing `CoinJumpAgent` behavior for gaps, tall obstacles, and elevated coins. The rule uses the observation vector, without inspecting the map file or privileged world state.

On the frozen **development** split, `CoinJumpAgent` reached the goal on 12/12 maps and met the collection target on 10/12. `GroundCoinJumpAgent` reached the goal and met the collection target on 12/12; its mean coin ratio was 1.0 versus 0.9167. The two newly solved development maps belong to `mixed_ground`. On the three older diagnostic `mixed_ground` maps, collection improved from 14/16 to 16/16 on each map.

This is a deterministic scripted teacher, not a trained PPO improvement. The frozen holdout remains unopened. A future PPO experiment needs a checkpoint or new independent training runs, a fixed training protocol, and final holdout evaluation after model selection.
