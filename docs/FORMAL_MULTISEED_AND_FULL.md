# 正式多训练 seed 验收与完整关卡

日期：2026-10-04。

## 正式 mixed 验收：通过

预先冻结 20261004、20261005、20261006 三个独立训练 seed，每次从随机网络初始化；没有共享 checkpoint。三次均使用 flat/obstacle/gap 训练布局的规则示范（4 轮、每轮 50 epochs），然后运行 gap → mixed PPO。因此方法是规则示范预热加 PPO，不是纯 PPO，也不是把同一个模型换三个评估 seed。

每次均在 16,384 PPO transition 通过 gap 连续两轮门槛，在 32,768 通过 mixed 连续两轮门槛，自动停在完整关卡入口。flat 4/4、obstacle 40/40、gap 54/54、mixed 60/60 validation 均符合原成功率与步数门槛。保存最终模型后重新加载验收；没有依据 test/OOD 更换 checkpoint。

| 训练 seed | PPO transition | test 确定性 / 采样 | OOD 确定性 / 采样 |
| --- | ---: | ---: | ---: |
| 20261004 | 32,768 | 696/696、696/696 | 192/192、192/192 |
| 20261005 | 32,768 | 696/696、696/696 | 192/192、192/192 |
| 20261006 | 32,768 | 696/696、696/696 | 192/192、192/192 |

共 222 个唯一保留布局，5,328 个 episode，全部成功且步数符合门槛。评估种子 1000–1003 不参与训练。静态地图不会因环境 seed 改变；重复局数不能解释成独立几何数量。三个训练 seed 支持此训练协议的初步稳定性结论，仍不代表任意地图能力。

v5 旧 test/OOD 已在此前实验查看，本轮作为回归。v6 在训练前新增 20 个 test 位置变化和 12 个 OOD 几何（高度 92、缺口宽 188、恢复距离 420/600/800），训练与 validation 集完全不变。新增 32 个唯一布局共 768 局，全部成功。test 仍使用此前已暴露的五个组合单元，不能宣称这些组合单元首次盲测；本轮新增的是保留位置/几何。今后这些 v6 布局也应作为回归集。

冻结配置及内容 hash、随机初始化来源、示范仅来自训练集、自动晋级与模型 hash 均已审计。机器摘要：[formal_training_seeds.json](reports/formal_training_seeds.json)。可复现实验基配置为 `config/ppo_formal_training_seed.json`，依次仅更改 seed 为计划中的三个值。原始计划和逐布局报告在 `runs/formal_training_seed_cohort/`。

## 完整关卡推进

完整关卡主 seed 在查看验收前已指定为 20261004。严格续训配置为 `config/ppo_full_after_multiseed.json`，来源为 `runs/formal_training_seed_cohort/seed_20261004/model.zip`。保留优化器、物理、reward、网络和课程签名；预算是额外 131,072 transition。

进入 full 前的确定性迁移基线：通关率 0%，49 步死亡，最大进度 12.76%，收集 2 个松果；失败位于首个高障碍附近。报告为 `runs/formal_training_seed_cohort/full_transfer_baseline.json`，附首个失败 trace。完整关卡含高障碍和交互实体，mixed 成功并不自动等于完整关卡通关。

完整地图是已知 `level_1`，训练与验证复用同一几何，不能计作未知完整地图泛化。完整关卡结果单独记录，不能改变已冻结的三 seed mixed 验收。

### 严格续训结果与训练集修正

严格续训已运行额外 131,072 transition，总计 163,840，状态 `timesteps_limit`。完整关卡仍在 49 步死亡、进度 12.76%，前置 validation 仍全部成功。失败实验保留在 `runs/ppo_full_after_multiseed/`，摘要为 [full_strict_continuation.json](reports/full_strict_continuation.json)，不将它描述成通过验收。

随后创建独立修正实验 `config/ppo_full_train_correction.json`：仍使用事先指定 seed 20261004 的 mixed 最终权重，`--init-from` 只复制 actor/critic，重新初始化优化器、RNG、PPO 计步和课程。以 full 为当前阶段，4 轮、每轮 50 epochs 的规则示范及 learner-state correction 只采集所有阶段的 train 布局；将示范上限从 128 步改为 768，以包含完整路线。此处 full 的训练地图同时用于 validation，属于已知地图回归。计划最多 65,536 PPO transition，要求连续两轮完整关卡与全部前置门槛通过。没有使用 test/OOD 进行修正。


播放最终 full 模型的命令（训练结果见下方）：

```bash
.venv/bin/python -m scripts.play_ppo \
  --model runs/ppo_full_train_correction/model.zip --task full
```

重跑 mixed 训练时使用新的输出目录：

```bash
.venv/bin/python -m scripts.train_ppo \
  --config config/ppo_formal_training_seed.json \
  --start-stage gap --stop-before-stage full --output-dir runs/formal_seed_rerun
```

该示例为 seed 20261004；三个配置仅 seed 不同，且每次都省略 `--resume` / `--init-from`。

### 完整关卡修正结果

修正实验在 8,192 / 16,384 PPO transition 连续两轮联合验证通过，课程状态为 `completed`。新计步为 16,384；这是新优化器实验的计步，不与来源 32,768 或失败续训 163,840 直接相加。示范共 33,702 个累积训练样本、16,600 次 actor 梯度更新，最后训练标签准确率 100%。PPO Monitor 的 80 个完成 episode 涉及 18 个 train 布局；示范和 PPO 均没有 test/OOD 布局。

最终 checkpoint 重新加载后，全部 159 个 validation 布局通过。完整关卡确定性 4/4、随机策略 4/4，均为 246 步、无死亡/超时、收集 2 个松果。Pygame headless 播放也在 246 步 / 981 core ticks 通关；终局截图为 `runs/ppo_full_train_correction/playback.png`。

当前完整关卡模型为 `runs/ppo_full_train_correction/model.zip`，须与 `model.json` 一起保存。它是训练集规则示范修正加 PPO 的单 seed 已知地图基线；通关不等于松果收集优化（2/25），也不代表其他完整地图泛化。下一阶段应在固定 full 训练协议上补齐多训练 seed，并扩展完整地图与交互覆盖，再设计收集质量指标。

最终完整关卡验收与回归摘要：[full_train_correction.json](reports/full_train_correction.json)。test 确定性 / 采样各 696/696，OOD 各 192/192，共 1,776 个回归 episode，全部符合原门槛。最终模型 hash 在评估期间保持不变。代码检查通过，全套测试 84 passed（另有 8 个 subtest，2 条旧 Gym 环境弃用提示）。
