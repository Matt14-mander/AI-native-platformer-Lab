# V2 gap 前置门槛修复

> 后续 mixed 训练和保留 test/OOD 验收已经完成，当前推荐模型及验收见 [mixed v5 验收](MIXED_V5_ACCEPTANCE.md)。本文保留 gap 修复阶段的模型和证据。

2026-10-04。结论：新 checkpoint 已通过 flat、obstacle、gap 的连续两轮联合门禁，并经重新加载确认，可进入 mixed。没有降低成功率门槛、放宽成功步数、删除失败布局、修改物理或奖励。

## 失败原因与处理

原纯 PPO 模型在 gap 验证集成功 45/54。逐物理帧诊断显示，6 个近缺口失败布局在出生时距边缘 263 px 就起跳，3 个长接近距离布局在首次落地后、距边缘 338 px 时又立即起跳，导致落入缺口。训练中 gap 改善与 obstacle 遗忘交替出现。

新实验使用 `config/ppo_gameplay_v2_gap_recovery.json`：

- 从原 v2 `best_gap.zip` 初始化 actor/critic 权重，重置优化器、计步和课程状态；不是严格续训。
- 在 flat/obstacle/gap 的 106 个训练布局上进行 4 轮规则示范与学习器状态纠正，累计 14,692 个样本、3,650 次监督梯度更新。
- PPO 学习率从 0.0003 降至 0.00001，entropy 系数从 0.01 改为 0，减少偏离已校正动作的探索。
- gap 阶段按 flat 5%、obstacle 45%、gap 50% 的 episode 采样权重回放前置任务。
- 连续两轮门禁仍要求每个布局达到原成功率及成功步数标准；通过后自动停在 mixed 入口。

这是“训练集示范纠正 + PPO”，不是纯 PPO。没有把验证失败布局加入训练，也没有 mixed 示范。示范仅更新 actor，随后运行 PPO；源模型的 actor/critic 都来自之前的 v2 训练。训练 Monitor 的布局归属也已核对，仅包含这三个任务的训练 split。

## 验收

在新实验 8,192 和 16,384 PPO transition 时连续联合通过，最终状态 `ready_for_stage`，target 为 mixed。重载的最终策略 hash 与保存的晋级策略一致。

| 任务 | 重载验证合格布局 | 原门槛 |
| --- | ---: | --- |
| flat | 4/4 | 每布局成功率 ≥95%，成功步数 ≤128 |
| obstacle | 40/40 | 每布局成功率 ≥80%，成功步数 ≤160 |
| gap | 54/54 | 每布局成功率 ≥80%，成功步数 ≤192 |

这些门禁评估采用已固定的环境 seed 100。附加诊断使用种子 200/201/202：gap 确定性 162/162、随机采样 162/162，全部成功且都记录缺口后落地。原失败的 9 个布局首次起跳距离收敛到距边缘 98 或 118 px，全部通过。种子重复不会产生新几何，也不是多个独立训练模型。

对冻结的新模型做 mixed 零样本回归：

| 分组 | 结果 |
| --- | ---: |
| 原 mixed 验证 | 36/36 |
| v5 新 train 布局探针（未用于本模型训练） | 64/64 |
| v5 新组合验证，确定性 | 24/24 |
| v5 新组合验证，3 个采样种子 | 72/72 |

没有运行 PPO 的保留 test/OOD 验收。以上 mixed 结果是迁移评估，不是 mixed 专项训练；此前 560 px 恢复距离失败模式已在这组验证布局上消失。

## 模型、报告与命令

推荐模型：`runs/ppo_gameplay_v2_gap_recovery/model.zip`，须与 `model.json` 配套。训练全过程：同目录 `run.json`。重载晋级证据：`mixed_entry_verified.json`。机读摘要：`docs/reports/gap_v2_recovery.json`。

本次训练命令：

```bash
.venv/bin/python -m scripts.train_ppo \
  --config config/ppo_gameplay_v2_gap_recovery.json \
  --init-from runs/ppo_gameplay_v2_retrain/checkpoints/best_gap.zip \
  --start-stage gap --stop-before-stage mixed \
  --output-dir runs/ppo_gameplay_v2_gap_recovery
```

试玩：

```bash
.venv/bin/python -m scripts.play_ppo \
  --model runs/ppo_gameplay_v2_gap_recovery/model.zip --task gap
```

当前课程状态已经停在 mixed。下一轮可使用本次保存的配置和同一签名从 `model.zip` 严格续训，选择新的输出目录；无需再次运行示范。使用旧的 v5 训练配置（原学习率/采样权重）直接 `--resume` 会因签名不同而拒绝。方案已有一个训练种子的可复现证据；多训练种子和独立 test/OOD 验收仍需后续完成。
