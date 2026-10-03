# V2 短按低跳、长按高跳

2026-10-04 根据试玩反馈替代初版固定高跳候选。修改文件为 `config/gameplay_ai_v2.json`，手动游戏默认读取该文件，训练显式使用 `PlatformerState-v2`。

## 参数与实际弧线

| 参数 | 当前值 | 作用 |
| --- | ---: | --- |
| jump_velocity | -13.8 | 初次起跳的向上速度 |
| rising_gravity | 0.5 | 仍在上升且按住跳跃时的重力 |
| falling_gravity | 1.8 | 松开跳跃或下降时的重力 |
| max_fall_speed | 11 | 下落速度上限 |

步行/奔跑/转向加速度仍为 0.3/0.5/0.7，最大步行/奔跑速度仍为 6/10。本次替换试玩中 -20 的起跳速度；该值在相等重力下实测会跳高约 240 px。

共享核心平地实测，按 60 Hz 换算：

| 按住时长 | 高度 | 滞空帧数 |
| --- | ---: | ---: |
| 1 个物理帧，快速点按 | 56.0 px | 17 |
| 4 帧，AI 的一个动作 | 83.2 px | 21 |
| 8 帧，AI 的两个连续动作 | 114.4 px | 27 |
| 持续长按 | 183.6 px | 48 |

长按约 27 帧达到顶点，此后仍受下降重力作用。松开时使用较大重力，加快消耗向上速度；按住不会持续施加新的起跳冲量。空中再次按下不会二段跳，但仍在上升时可恢复较低的上升重力。落地后继续按住也不自动连跳，需松开再按。

这是共享核心已有的按键重力机制，本次通过参数校准启用；没有加入新的状态字段、隐藏计时器、动作或观测维度。v2 保留 15 维观测、10 个动作和默认 action_repeat=4。JUMP_HELD 继续让策略观察上一动作的按键状态。

## 松果和奖励

前段 5 个松果位于 `game_content/levels/level_1.json`，y 分别为 490、458、438、458、490。测试在平地复用这些位置：从 x=460 起步，先步行 20 帧，短跳 4 帧，再向右移动，可收齐 5 个松果；score 增加 500，Gym 收集 reward 累计增加 2.5。再次经过不会重复奖励。

这是受控收集路线验证，不代表自动策略已学会收集，也不代表原完整关卡每条路线都收齐松果。原完整关卡的规则通关探针本次收集了 2 个松果。

`RewardConfig.coin_reward` 原本就是每个松果 +0.5，本轮没有调整 reward 权重，也没有新增带松果的训练课程。现有 mixed 不含松果；收集专项学习仍需要后续加入对应课程。

## 验证与复现

本轮验证短按/长按的高度分级、最大高度、空中不能重新起跳、落地重新起跳、键盘与 Gym 轨迹一致，以及收集奖励一次性。

原完整关卡规则策略通关：246 个环境步、2 个松果。训练/验证布局、随机动作及奖励门禁结果见 `docs/reports/gameplay_variable_v2_readiness.json`；物理测量见 `docs/reports/gameplay_variable_v2.json`。旧固定高度报告保留为历史记录。

```bash
# 试玩：Space/A 跳跃，方向键移动，Shift/S 奔跑
.venv/bin/python -m source.main

# 复测物理，选择新的输出路径
.venv/bin/python -m scripts.audit_gameplay --output runs/variable_jump_recheck.json

# 课程与随机动作门禁
.venv/bin/python -m scripts.validate_rl_readiness --courses \
  --environment-id PlatformerState-v2 --manifest config/curriculum_v4.json \
  --episodes 1000 --output runs/variable_readiness_recheck.json
```

本轮未训练新模型。`config/ppo_gameplay_v2.json` 的新训练会读取并冻结当前参数；已保存的模型使用自己的冻结参数。旧 v1 模型及 TinyInfer 导出包仍保留旧物理。本次候选修改不能套用之前 v1 mixed 的 100% 成功率，也不能直接跨协议续训。
