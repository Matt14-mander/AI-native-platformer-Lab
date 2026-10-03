# V2 首轮 PPO 重训

日期：2026-10-04。冻结配置：`config/ppo_gameplay_v2_retrain.json`。新物理为起跳 -13.8、按住上升重力 0.5、松开/下降重力 1.8，最大跑速 10。

本轮从随机参数开始，仅使用 PPO，没有旧模型权重或规则示范。一个训练种子，4 个并行环境；每 8,192 transition 验证一次。训练及评估未使用独立 test/OOD 集。现有课程没有松果，所以本轮验证的是通关能力，不是收集策略。

## 结果

实际训练 240,112 transition。flat 在 16,384 时通过连续两轮门禁，obstacle 在 90,112 时通过。gap 阶段用满 150,000 transition 后仍未逐布局全部合格，状态为 `budget_exhausted`，没有进入 mixed 专项训练。

重新加载联合评估最好的 `checkpoints/best_gap.zip`，确认 sidecar、模型 hash、物理/内容协议及 timestep 一致。该模型保存于 229,376 transition，验证结果如下：

| 任务 | 验证成功数 | 成功率 | 平均成功步数 |
| --- | ---: | ---: | ---: |
| flat | 4/4 | 100% | 19.0 |
| obstacle | 40/40 | 100% | 35.9 |
| gap | 45/54 | 83.33% | 37.78 |
| mixed 迁移探针 | 30/36 | 83.33% | 50.33 |

mixed 是额外的验证集迁移评估，不是 mixed 训练或正式通过。当前门禁逐布局检查，不能以 gap 总成功率超过 80% 代替所有布局合格。失败集中在较靠近出生点的 134/164 px 缺口，以及另一组 164 px 缺口。

训练中出现 gap 改善、obstacle 退化的波动；最终参数模型 `model.zip` 为 gap 43/54，弱于保存的 best_gap。推荐试玩和后续分析使用 `best_gap.zip`，保留最终模型及全部评估记录作对照。

## 文件与复现

- 推荐模型：`runs/ppo_gameplay_v2_retrain/checkpoints/best_gap.zip`，配套 `.json` 不可丢失。
- 模型 SHA256：`0cf78b004c31e213ec5e4abd8381614d52824beb56f4a0fdc977af316fe1a833`。
- 训练完整记录：`runs/ppo_gameplay_v2_retrain/run.json`。
- 重载验证：`runs/ppo_gameplay_v2_retrain/best_gap_verified.json`。
- 可版本控制摘要：`docs/reports/ppo_gameplay_v2_retrain.json`。
- 逐次评估、首个失败轨迹及模型：该 run 下的 `evaluation/`、`checkpoints/`。

本轮执行命令：

```bash
.venv/bin/python -m scripts.train_ppo \
  --config config/ppo_gameplay_v2_retrain.json --stop-before-stage full \
  --output-dir runs/ppo_gameplay_v2_retrain
```

该输出目录已存在，重跑需选新目录。试玩推荐模型：

```bash
.venv/bin/python -m scripts.play_ppo \
  --model runs/ppo_gameplay_v2_retrain/checkpoints/best_gap.zip --task gap
```

下一轮优先解决 gap 的起跳时机/按住时长和 obstacle 遗忘，使用版本化的新实验配置比较回放比例、学习率及训练预算；不改已经确认的物理，也不跳过当前失败布局。若继续相同签名配置从 best_gap 续训，需要增加 gap 阶段预算；原预算仅剩少量，直接续跑很快会再次触发阶段停止。改变采样策略时要作为新的实验记录，不能冒充严格续训。

稳定通过前置门禁后再进入 mixed，随后增加带松果的课程。多训练种子及独立 test/OOD 验收留待方案稳定后执行。本轮不据此宣称 v2 优于旧 v1：旧模型使用规则示范预热，训练方法不同。
