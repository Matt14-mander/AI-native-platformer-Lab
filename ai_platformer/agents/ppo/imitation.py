"""Optional training-only rule demonstrations and learner-state correction before PPO."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from ai_platformer.agents.scripted import RuleJumpAgent
from ai_platformer.core import Action


def teacher_action(observation: np.ndarray) -> int:
    action = RuleJumpAgent().act(observation)
    # Correct learner states that hold jump on the ground: release before pressing again.
    if len(observation) > 14 and observation[4] > 0.5 and observation[14] > 0.5:
        return int(Action.RIGHT_RUN)
    return action


def warm_start_policy(
    model, *, factory, stages, options: dict, seeds: list[int], seed: int
) -> dict:
    """Fit actor only. First round follows the teacher; later rounds label learner states.

    Collect only prefixes of train episodes. No validation/test/OOD observations or rewards
    are used for supervision; the value network and PPO optimizer are left untouched.
    """
    levels = list(dict.fromkeys(level for stage in stages for level in stage["train_levels"]))
    rng = np.random.default_rng(seed)
    actor_parameters = list(model.policy.mlp_extractor.policy_net.parameters()) + list(
        model.policy.action_net.parameters()
    )
    optimizer = torch.optim.Adam(actor_parameters, lr=options["learning_rate"])
    observations, labels, rounds = [], [], []
    env = factory.make(level_id=levels[0])
    try:
        for iteration in range(options["rounds"]):
            for index, level in enumerate(levels):
                observation, _ = env.reset(
                    seed=seeds[index % len(seeds)], options={"level_id": level}
                )
                for _ in range(options["max_steps_per_episode"]):
                    label = teacher_action(observation)
                    observations.append(observation.copy())
                    labels.append(label)
                    action = (
                        label
                        if iteration == 0
                        else int(model.predict(observation, deterministic=True)[0])
                    )
                    observation, _, terminated, truncated, _ = env.step(action)
                    if terminated or truncated:
                        break
            batch_observations = torch.as_tensor(np.asarray(observations), device=model.device)
            batch_labels = torch.as_tensor(labels, dtype=torch.long, device=model.device)
            model.policy.set_training_mode(True)
            updates = 0
            for _ in range(options["epochs"]):
                for offset in range(0, len(labels), options["batch_size"]):
                    # A single permutation per epoch is generated below, before its first batch.
                    if offset == 0:
                        order = rng.permutation(len(labels))
                    selected = order[offset : offset + options["batch_size"]]
                    logits = model.policy.get_distribution(
                        batch_observations[selected]
                    ).distribution.logits
                    loss = F.cross_entropy(logits, batch_labels[selected])
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor_parameters, 1.0)
                    optimizer.step()
                    updates += 1
            model.policy.set_training_mode(False)
            with torch.no_grad():
                logits = model.policy.get_distribution(batch_observations).distribution.logits
                accuracy = float((logits.argmax(dim=1) == batch_labels).float().mean())
            rounds.append(
                {
                    "round": iteration + 1,
                    "samples": len(labels),
                    "updates": updates,
                    "training_label_accuracy": accuracy,
                }
            )
    finally:
        env.close()
        model.policy.set_training_mode(False)
    return {
        "teacher": "rule_jump_with_release_v1",
        "source": "training layouts only",
        "levels": levels,
        "seed": seed,
        "options": options,
        "rounds": rounds,
        "gradient_updates": sum(item["updates"] for item in rounds),
        "note": "Supervised actor warm start; not pure PPO learning. Critic and PPO optimizer unchanged.",
    }
