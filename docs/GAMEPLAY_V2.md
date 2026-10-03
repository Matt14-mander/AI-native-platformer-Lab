# Mixed 前的游戏设置评估

## 结论与范围

旧核心并不会无限升空：只有地面上的按下沿能起跳。长按时上升重力为 0.3，松开后为 1.0，因此长按明显延长上升。实测单帧短按跳高 57.2 像素、滞空 0.367 秒，长按跳高 178.5 像素、滞空 0.95 秒。

本轮提供固定高度跳跃候选设置 `config/gameplay_ai_v2.json`。Pygame 手动游戏默认使用候选；训练显式选择 `PlatformerState-v2`。旧 v0/v1、旧训练配置和模型的物理协议保留。

## 实测对比

以下数据来自共享核心的平地模拟，物理频率为 60 Hz；跑跳距离从已达到最大跑速时起算。

| 项目 | 旧设置 | v2 候选 |
| --- | --- | --- |
| 短按 / 长按跳高 | 57.2 / 178.5 px | 均为 182.7 px |
| 长按到顶点 | 34 帧 | 21 帧 |
| 长按滞空 | 57 帧 / 0.95 s | 45 帧 / 0.75 s |
| 最大跑速 | 12 px/帧 | 10 px/帧 |
| 满速跑跳距离 | 684 px | 450 px |
| 步行 / 奔跑达到最大速度 | 均为 40 帧 | 均为 20 帧 |
| 长按 600 帧连续起跳次数 | 1 | 1 |

v2 起跳速度为 -17.5，上升和下降重力均为 0.8，下落速度上限仍为 11。步行、奔跑、转向加速度分别从 0.15/0.3/0.35 调整为 0.3/0.5/0.7。起跳后松开、再次按下都不会改变空中弧线；落地后必须松开再按才能重新跳跃。

固定弧线让策略主要探索起跳时机和方向，减少持续按跳跃维持高度的要求。较快加速缩短动作反馈延迟，较低跑速减小每次决策的水平位移。这些是设计动机；尚未用 PPO 对照训练证明学习效率提升。

跳高保留约 183 像素，是为了越过原完整关卡的高管道和台阶。较低的 124 像素候选虽然通过短课程，却无法通过原完整关卡，因此未采用。课程最大障碍高度约 80 像素，后续应评估固定高跳是否使 mixed 缺少挑战，再调整课程分布。

## 验证结果

- SB3 环境检查通过；随机动作 1,000 局、238,609 次 transition，观测与奖励有限且符合接口约束。
- curriculum_v4 的训练/验证布局 276 个全部通过规则可达性、move-right 对照、进度奖励守恒和静止负收益检查。该结果不代表 PPO 成功率，也不包含独立测试/OOD 集验收。
- 原完整关卡使用规则策略，起跳阈值 0.34 和 0.5 均通关，分别用 257 和 261 个环境步。
- 奖励审计通过。终局奖励符号改用受控平地/坑洞触发，避免其结果依赖旧规则策略的起跳时机；真实内容上的静止、原地跳、往返和重复收集检查继续保留。课程可达性独立检查。
- 自动测试覆盖短按/长按/空中重复按键、落地重新起跳、Gym 与键盘轨迹一致、旧协议保留及错误终局奖励的拒绝。

原始物理测量见 `docs/reports/gameplay_audit_v2.json`，门禁摘要见 `docs/reports/gameplay_v2_readiness.json`。完整门禁输出在本地 `runs/readiness_gameplay_v2_final.json`。

## 试玩与复现

在仓库根目录执行：

```bash
# 候选设置：方向键移动，Space/A 跳跃，Shift/S 奔跑
.venv/bin/python -m source.main

# 旧设置对照
.venv/bin/python -m source.main --settings config/gameplay.json

# 复测物理；--plot 可选，需 matplotlib
.venv/bin/python -m scripts.audit_gameplay \
  --output runs/gameplay_audit.json --plot runs/gameplay_jump.png

# 新环境课程和随机动作门禁
.venv/bin/python -m scripts.validate_rl_readiness --courses \
  --environment-id PlatformerState-v2 --manifest config/curriculum_v4.json \
  --episodes 1000 --output runs/readiness_gameplay_v2.json
```

试玩重点：落地后再次起跳、管道和台阶的余量、空中转向、跑步停下的距离。此次没有加入跳跃缓冲或离地宽限，避免在观测中引入未暴露的计时状态。奔跑切换到步行仍会立即将速度限制到 6；若后续需要平滑减速，应单独改进并重新验证。

## 接下来再进入 mixed

候选使用 15 维观测、10 个动作、每个动作重复 4 个物理帧，与 v1 的接口形状相同，但环境 ID 和物理协议不同。新训练配置为 `config/ppo_gameplay_v2.json`，解析时会冻结实际物理参数。

本轮未运行新 PPO 训练。建议先试玩确认候选，再用新环境重新完成 flat → obstacle → gap 的学习与遗忘检查，达到门槛后进入 mixed。可先运行 32,768 transition 的独立探索实验：

```bash
.venv/bin/python -m scripts.train_ppo --config config/ppo_gameplay_v2.json \
  --stop-before-stage mixed --output-dir runs/ppo_gameplay_v2_probe
```

该预算是首轮诊断预算，不保证完成全部前置阶段；不足时依据保存的配置和状态续训。正式比较使用相同关卡、预算和多个训练种子，对比 v1/v2 的成功率、成功步数、死亡原因及旧任务遗忘。

现有 v1 checkpoint 不能直接作为 v2 的严格续训或跨协议 `--init-from` 输入。旧模型播放会从模型元数据恢复旧环境；TinyInfer 导出包同样保留旧协议。新物理需要新训练、导出、轨迹一致性与性能验证。目录名 `tinyinfer_actor_v2` 表示之前的导出迭代，并不是本次的 `PlatformerState-v2`。
