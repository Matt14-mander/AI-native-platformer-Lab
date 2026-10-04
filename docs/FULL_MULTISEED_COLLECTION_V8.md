# 完整关卡收集与多地图 v8

日期：2026-10-04。原关卡三独立训练 seed 复现已通过；v7 收集/多地图验收暴露两个 seed 的失败，历史与详细指标见 [v7 报告](FULL_MULTISEED_COLLECTION_V7.md)。

## 本轮改动与冻结协议

v8 从每个 seed 自己的 v7 权重初始化，保留三条独立谱系，不丢弃失败 seed、不按保留结果选 checkpoint。只复制 actor/critic，重置优化器和计步；训练集示范修正加 PPO，仍不是纯 PPO。主 seed 20261004 预先指定，三次配置一致，计划最多 98,304 transition。

新增示范拟合增强：对规则 teacher 不读取的 PLAYER_X、PROGRESS、COIN_RATIO、REMAINING_TIME 随机采样 [0,1]；局部几何、速度、松果位置、grounded 和 jump-held 保持原样，标签不变。增强只用于训练梯度，不修改环境/部署观测；PPO 仍接收原始 15 维观测。两个 teacher 都不用这些全局字段，目的是减少依赖地图位置/阶段计数的捷径。此项与顺序扩展同时引入，不做单项因果加速/泛化声明。

新增 12 train 地图覆盖全部六种三段排列，6 validation 地图，4 fresh test 和 4 fresh OOD；完整 full 池现在 train 24、validation 10、test 8、OOD 8。所有旧内容/分组保持，test/OOD 中的 v7 旧布局属于回归，v8 新布局单独统计。

- 新 test：高度 56/104/152、缺口 104/148/168，改变出生点与间距。训练已覆盖全部排列，所以这是新的几何参数组合与位置，不能称为新顺序单元。
- 新 OOD：高度 178、缺口 200，恢复距离 1,040/1,120/1,200，外加新的出生位置；仍在当前物理可达范围。
- 已知 `level_1` 同时保留在训练与 validation 作为回归例外，不计入盲测。

所有原门槛不变：full 每布局成功率 ≥90%、成功步数 ≤768、生成地图成功局逐局收集 ≥75%，原关卡 ≥20%（5/25）；前置课程保持旧门槛。连续两轮联合 validation 通过才保存完成模型，之后重载再检查。新 test/OOD 第一次按冻结三 seed 计划评价，v7/旧 mixed 仅做回归。

配置：[ppo_full_collection_v8.json](../config/ppo_full_collection_v8.json)。冻结计划：[full_collection_v8_plan.json](../config/full_collection_v8_plan.json)。manifest：[curriculum_v8.json](../config/curriculum_v8.json)。验收工具：[accept_full_checkpoint.py](../scripts/accept_full_checkpoint.py)。

## 使用

```bash
# 播放生成的完整 validation 地图，N 切换；HUD 显示松果数
.venv/bin/python -m scripts.play_ppo \
  --model runs/full_collection_v8/seed_20261004/model.zip --task full --suite validation

# 原关卡
.venv/bin/python -m scripts.play_ppo \
  --model runs/full_collection_v8/seed_20261004/model.zip --level-id level_1

# 重跑使用新输出路径；此示例仅 seed 20261004
.venv/bin/python -m scripts.train_ppo \
  --config config/ppo_full_collection_v8.json \
  --init-from runs/full_collection_v7/seed_20261004/model.zip \
  --start-stage full --output-dir runs/full_collection_v8_rerun

.venv/bin/python -m scripts.accept_full_checkpoint \
  --model runs/full_collection_v8/seed_20261004/model.zip \
  --output-dir runs/full_collection_v8_acceptance_rerun
```

该版本是受限的静态生成地图族；没有新增 legacy level_2–4 的实体迁移，也没有验证其他敌人/传送行为。不能将生成地图通过等同于全部原始关卡通关。TinyInfer 数值和性能基准尚未针对新 checkpoint 重测。

## v8 验收结果：未整体通过

新几何和已有课程回归均通过，但三个 seed 的原关卡随机策略都有成功局少收松果，故 **0/3** 满足全部门槛。确定性均为 5/25；采样平均分别为 4.75/25、3/25、4.75/25。详见 [full_collection_v8.json](reports/full_collection_v8.json)。没有用新地图均值掩盖已知地图漏收。

内容审计进一步发现：旧 adapter 的第一主区域 width=9,086、goal=8,470，但 20 个 legacy 松果的 x=9,318..9,582，属于未迁移的后续地图区域。当前 episode 结束前无法到达；可玩主区域只有 overlay 的 5 个松果。因此旧 25 分母包含不可达内容，不适合作为当前主区域的收集率。v9 增加显式主区域版本，保留旧 `level_1`、旧报告和 hash 兼容性。最新结果见 [v9 报告](FULL_MULTISEED_COLLECTION_V9.md)。
