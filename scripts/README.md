# Scripts

后续命令行入口放在这里，包括：

- legacy smoke run；
- headless benchmark；
- PPO train/evaluate；
- replay playback；
- ONNX export/validation；
- PCG batch generation/validation。

脚本只负责编排，领域逻辑必须保留在 `ai_platformer/` 中。

课程训练、独立评估和续训命令见 [强化学习训练 v1](../docs/RL_TRAINING_V1.md)。`train_ppo` 支持 `--resume`、`--eval-every`；`benchmark_scripted` 支持 `--config`、`--task`；`validate_rl_readiness --courses` 覆盖课程门禁。
