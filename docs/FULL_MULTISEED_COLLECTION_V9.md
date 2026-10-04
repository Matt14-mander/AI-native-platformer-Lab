# 完整关卡多 seed、收集和多地图 v9

日期：2026-10-04。原完整关卡的三独立谱系复现已通过；此前收集优化 v7/v8 的失败均保留，见 [v7](FULL_MULTISEED_COLLECTION_V7.md) / [v8](FULL_MULTISEED_COLLECTION_V8.md)。

## 可玩主区域与收集目标

原第一关 JSON 包含多个地图区域。现有 core 第一主区域 goal=8,470，但旧适配器统计的 25 个松果中有 20 个位于 x=9,318..9,582 的后续区域，当前传送/区域切换未迁移，episode 终点前无法到达。前段 overlay 的 5 个才属于已接入主区域。

新增 `level_1_main` 显式内容版本：保留原物理、出生点、宽度、地形、方块、敌人、道具和终点，只保留终点之前的主区域松果。`level_1` 仍返回原始 25 个松果，旧模型协议/hash 未被修改。renderer 对别名仍复用原地图各类地形分组。

验收要求主区域每个成功 episode 收集 **5/5**。这和 v7/v8 要求实际 5 个的数量相同，只修正分母，不降低收集目标。原通关模型为 246 步、2 个；规则收集 teacher 的训练地图校准为 243 步、5 个。

## 训练与验证

20261004/20261005/20261006 继续各自独立的 v8 谱系，包括之前失败的模型，没有选掉失败 seed。只复制权重，重新初始化优化器、RNG、PPO 计步和课程；方法是训练集规则示范修正 + PPO，不是纯 PPO。主 seed 20261004 事先指定。

保留已有六种排列的生成训练布局及增强；提高主区域训练示范权重到每轮 8 次、4 轮、每轮 100 epochs。主区域作为明确的已知地图回归，豁免全局特征增强，让拟合覆盖其精细松果路线；新生成地图仍随机化 teacher 不用的 x/progress/coin-ratio/time 特征。所有示范、learner-state correction、加权和 PPO 仅使用 train 布局；配置拒绝保留布局出现在加权/豁免清单。

验证门槛保持：全部 full 布局成功率 ≥90%，成功步数 ≤768；主区域逐局 100% 收集，生成地图逐局 ≥75%，全部前置课程保持原门槛。连续两轮联合 validation 通过后，冻结最终模型、重载，再运行确定性与随机策略。三个 seed 都需要通过，不依据保留结果换 checkpoint。

新版本完整地图池：24 个生成 train + 1 主区域；10 个生成 validation + 主区域回归；test/OOD 各 12 个，其中各 8 个旧 v7/v8 集为回归，各 4 个 v9 新几何为首次保留评价。新增 test 高度 60/108/156、缺口 108/152/176，OOD 使用已观察过的 178/200 边界配新出生位置与间距。OOD 新的是几何位置，不声称提高了此前外推的物理极值。

8 个新保留地图 × 4 评估 seed × 2 模式 × 3 训练 seed = **192 个 fresh full episode**。旧 full 保留回归 384 局，旧课程 test/OOD 回归 5,328 局。静态地图的评估 seed 不增加独立几何数量。

冻结协议：[full_collection_v9_plan.json](../config/full_collection_v9_plan.json)。训练基配置：[ppo_full_collection_v9.json](../config/ppo_full_collection_v9.json)。课程：[curriculum_v9.json](../config/curriculum_v9.json)。

## 使用

```bash
# 可玩第一主区域；HUD 和输出均显示 coins
.venv/bin/python -m scripts.play_ppo \
  --model runs/full_collection_v9/seed_20261004/model.zip --level-id level_1_main

# 生成 full validation 地图，按 N 切换
.venv/bin/python -m scripts.play_ppo \
  --model runs/full_collection_v9/seed_20261004/model.zip --task full --suite validation

# 严格重载、检查最终晋级记录，再评价保留布局；使用新输出目录
.venv/bin/python -m scripts.accept_full_checkpoint \
  --model runs/full_collection_v9/seed_20261004/model.zip \
  --output-dir runs/full_collection_v9_acceptance_rerun

# 重训示例仅为主 seed；其他 seed 使用自己的源 checkpoint
.venv/bin/python -m scripts.train_ppo \
  --config config/ppo_full_collection_v9.json \
  --init-from runs/full_collection_v8/seed_20261004/model.zip \
  --start-stage full --output-dir runs/full_collection_v9_rerun
```

目前多地图泛化限于该静态生成族的连续障碍/缺口与松果路线。没有迁移后续区域的 20 个松果和传送流程，也没有验证全部 legacy level_2–4 或新敌人行为。TinyInfer 接口维度不变；预指定主模型已重新导出并通过数值、轨迹与性能回归，另外两个 seed 被既有导出数值门槛拦截，见 [v9 部署回归](DEPLOYMENT_V9.md)。


## 最终验收：3/3 通过

每次均在新增 PPO 计步 8,192 / 16,384 连续两轮联合 validation 通过并结束训练；计步与优化器在本版本初始化时重置，不能和历史失败实验直接相加。完整拟合梯度更新分别为 58,800 / 58,600 / 58,600；这部分示范训练不能忽略后只按 16,384 PPO 步描述训练成本。

| 训练 seed | 主区域确定性 / 采样 | v9 fresh test 收集率（确定性 / 采样） | v9 fresh OOD 收集率（确定性 / 采样） | 最终通过 |
| --- | --- | --- | --- | --- |
| 20261004 | 各 4/4、5/5、243 步 | 97.92% / 96.88% | 95.83% / 96.35% | 是 |
| 20261005 | 各 4/4、5/5、243 步 | 97.92% / 96.88% | 97.92% / 98.44% | 是 |
| 20261006 | 各 4/4、5/5、243 步 | 93.75% / 93.75% | 100% / 99.48% | 是 |

新增 8 个唯一保留布局共 192 局全部通关，收集 2,237/2,304 个松果，平均 **97.09%**；实际最少每局 10/12，超过 9/12 的门槛，最多成功步数 244。所有老 full / 课程回归 5,712 局全部成功、满足原门槛，包含之前失败的 v7 地图。主区域 24 个确定性/采样 episode 全部 5/5；重载 validation 全部通过。训练 Monitor 与示范来源均只包含 train 布局，三份最终模型与来源/config/manifest hash 均审计且未在验收中改变。

Pygame headless 主区域为 243 步 / 969 ticks / 5/5，10 张生成 validation 地图全部通关并达到收集门槛；HUD 和终局 JSON 显示松果数。全套测试 **90 passed、8 subtests**，仅 2 条旧 Gym 环境弃用提示；改动文件 Ruff 和 diff whitespace 检查通过。

完整机器报告：[full_collection_v9.json](reports/full_collection_v9.json)。当前推荐 checkpoint 为 `runs/full_collection_v9/seed_20261004/model.zip`，须与同名 `model.json` 一起保留；其余 seed 模型留在对应目录，不以评分替换主 seed。

主模型 SHA256：`f6f5eabb26446fb7ee999c99e2cd76acfb98bb88c0abd4660e26d71658e42e3c`。
