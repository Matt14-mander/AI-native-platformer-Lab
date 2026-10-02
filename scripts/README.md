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

`train_ppo --resume ... --start-stage gap` 可以显式进入缺口实验，历史会记录手动切换，保持原有 checkpoint 兼容性校验。

第二轮缺口开发见 [训练 v2](../docs/RL_TRAINING_V2.md)：`--init-from` 权重初始化、任意运行的 `--start-stage`、`diagnose_ppo` 三种策略对照、`evaluate_ppo --suite ood`、`validate_rl_readiness --manifest`。
