# Generated-map imitation experiments

The experiment uses `PlatformerState-v2` observations and actions. Training requests are built from the six frozen families with separate seeds, and every training geometry is checked against both frozen evaluation splits. The student sees only the 15-element observation vector. Saved run artifacts under `runs/` include the demonstrations, model, geometry hashes, training settings, and development results; `runs/` is intentionally ignored by Git.

## Stage 1: behavior cloning

Run `python -m scripts.train_pcg_bc --output runs/pcg_bc_v1`. The default pool is eight training seeds × six families = 48 maps, with 80 supervised epochs and training seed 17. The teacher finished all 48 maps; 6,880 observation/action pairs were recorded. The learned MLP student reached the goal and met the per-map collection target on 12/12 frozen development maps; mean coin ratio was 0.9722. The scripted teacher's development coin ratio was 1.0. These are static map results from one policy seed, not independent training trials. The 12-map holdout remains unopened.

The 80 epochs were chosen after inspecting development performance, so development is a model-selection set. Early 10-epoch behavior cloning mostly learned to run right (2/12 goal reaches). The experiment tests learning from demonstrations, not PPO improvement or broad generalization beyond the defined six families.
