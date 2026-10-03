# 当前 checkpoint 的 mixed 基线

日期：2026-10-03。模型：`runs/mixed_prerequisites_imitation/model.zip`，SHA256 `7c847b19c44da6a2012ef82cded57f8580d915d5ca85ab47c303f6f582892b55`。

## 结论

当前模型在现有 mixed 分布上已达到成功率上限：36 个验证布局全部通关，随机动作采样 5 个种子共 180 局也全部通关。它没有接受 mixed 专项训练，也没有使用 mixed 示范；这是 flat/obstacle/gap 规则示范预热与 16,384 PPO transition 后的组合迁移基线，不是纯 PPO 从零探索结果。

本次从 checkpoint 恢复完整 `PlatformerState-v1` 协议并验证模型 hash 和内容 hash。跳跃仍使用旧版可变高度物理。新 `PlatformerState-v2` 候选没有被本次结果验证；其初版固定跳跃已在 2026-10-04 改为新的可变高度跳跃。

## 实测结果

| 评估 | 唯一布局 | 局数 | 成功率 | 死亡率 | 超时率 | 平均步数 | 平均进度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PPO 确定性，mixed 验证 | 36 | 36 | 100% | 0% | 0% | 45 | 100% |
| PPO 确定性，mixed 训练布局 | 36 | 36 | 100% | 0% | 0% | 45 | 100% |
| PPO 采样，mixed 验证 | 36 | 180 | 100% | 0% | 0% | 45 | 100% |
| rule-jump，mixed 验证 | 36 | 36 | 100% | 0% | 0% | 45 | 100% |
| move-right，mixed 验证 | 36 | 36 | 0% | 0% | 100% | 1,024 | 10.32% |
| 冻结初始观测后采样，mixed 验证 | 36 | 36 | 0% | 0% | 100% | 1,024 | 10.32% |

成功局均在 44–46 个环境步完成，平均回报 14.955。一个环境步最多推进 4 个物理帧；这些步数不是推理延迟或 FPS。验证集确定性种子为 100，采样种子为 200–204，附加观测诊断种子为 200。课程几何随布局 ID 变化，环境 seed 不产生新几何；5 个采样种子不是 5 个独立训练模型。

现有 mixed 数值门槛为每布局成功率至少 80%、成功步数不超过 256。本次确定性与采样基线均为 36/36 布局合格。这是单 checkpoint 的基线评估，没有修改课程晋级历史，也没有执行 mixed 训练或正式验收。

## 行为与难度判断

确定性策略平均每局实际起跳 2.75 次，采样策略为 2.57 次。冻结观测后不再起跳，卡在首个障碍附近，与直接向右跑的对照一致；这支持成功依赖动态观测，而不是仅靠初始动作分布。

诊断中的 `landed_after_gap_rate` 为 83.33%，而通关率为 100%。部分局在越过缺口后仍处于空中便触发终点，因此没有记录缺口后的落地；这个指标不能单独解释为越过缺口的成功率。

当前 mixed 是“单个障碍 + 足够恢复距离 + 单个缺口”。训练布局障碍高 48 px、缺口宽 100/130/160 px；验证布局障碍高 48 px、缺口宽 104/134/164 px。生成器要求障碍右缘到缺口超过 360 px，宽度均为 2,100 px。训练/验证几何 hash 无重复，但验证主要是这一结构上的位置和缺口宽度变化，组合压力有限。

PPO 与规则策略均为 100%，且成功步数一致，说明现有分布已不足以衡量下一轮训练的明显进步。这不能证明更紧凑的组合、多障碍、多缺口或完整关卡已解决。本次未读取独立 test/OOD 的评估结果。

## 下一步

1. 先确定正式使用 v1 还是新 v2 物理。若采用 v2，重新训练/验证前置任务并测 mixed 基线，不能套用本报告的成功率。
2. 现有 mixed 保留为回归基线。下一版课程增加障碍高度和缺口宽度变化，并逐步缩短恢复距离、增加连续组合；紧凑组合需要同时审查传感器可观测性。新内容单独版本化，先用规则策略和受控动作确认可达性。
3. 用多个独立训练种子比较新增课程的成功率、死亡位置、完成步数和前置任务遗忘。冻结课程和策略选择规则后，再做独立 test/OOD 验收。

## 报告与复现

摘要及协议：`docs/reports/mixed_baseline_v1.json`。完整逐布局/逐局结果：`runs/mixed_baseline_v1/report.json`。跳跃与观测诊断：该目录下的 `stochastic_diagnostics.json`、`observation_diagnostics.json`。本次一次性 runner 为 `runs/mixed_baseline_v1/evaluate.py`，没有更新模型权重。

可用现有脚本复测验证集，输出需使用新路径：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m scripts.evaluate_ppo \
  --model runs/mixed_prerequisites_imitation/model.zip \
  --task mixed --suite validation --seeds 100 \
  --output runs/mixed_baseline_recheck.json

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m scripts.diagnose_ppo \
  --model runs/mixed_prerequisites_imitation/model.zip \
  --task mixed --suite validation --seeds 200 201 202 203 204 \
  --modes stochastic --output runs/mixed_sampling_recheck.json
```
