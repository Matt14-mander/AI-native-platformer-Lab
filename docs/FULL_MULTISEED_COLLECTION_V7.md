# 完整关卡多 seed、松果收集与多地图

日期：2026-10-04。

## 原完整关卡三 seed 复现

20261004、20261005、20261006 各使用自己独立训练的 mixed 谱系，按相同 full 修正配置执行训练集规则示范 + PPO，没有共享单一 checkpoint。20261004 已在上一轮完成；其余两次为固定协议复现，所以是已知地图复现，不是三次从未接触过地图的盲测。

三次均在新 PPO 计步 8,192 / 16,384 连续两轮通过完整关卡和全部前置 validation；最终状态 `completed`。各模型重新加载后确定性 / 采样各 4/4 通关，均为 246 步、2/25 松果。合计完整地图 24 局、已有 test/OOD 回归 5,328 局全部通过。报告：[full_seed_cohort.json](reports/full_seed_cohort.json)。

## 收集质量与训练协议

观测、动作、物理与奖励接口保持 v2（15 维、10 动作、每松果 +0.5），没有修改旧模型语义，也没有声称升级了 TinyInfer 数值/部署验证。已有观测能够定位松果，先改训练目标与示范，比提高奖励数值更容易检查实际收集行为。

新增 `coin_jump_v1` 规则示范：危险障碍/缺口优先采用原高跳，安全地面可见的低空松果采用减速步行短跳，达到松果高度后释放跳跃；按 episode 清空跳跃上下文。后续 learner-state correction 与 PPO 均只用训练集。已知训练关卡的 teacher 校准为 243 步通关、5/25 松果；原规则是 246 步、2/25。示范上限 768 步、4 轮、每轮 50 epochs。

评估 episode 新增 `coins_total`，summary 新增 `mean_coin_ratio`。没有松果或旧报告缺少分母时不虚构比例；配置了收集门槛但 episode 缺分母会拒绝。成功局逐局检查比例，不能用跨地图平均分掩盖某个漏收地图。联合选模先检查全部门槛，再考虑进度、平均收集比例与耗时。

v7 full 门槛：每布局成功率 ≥90%，成功步数 ≤768，新完整地图每个成功 episode 收集 ≥75%；已知 `level_1` 单独保留回归，门槛为 ≥20%（5/25）。保持 flat/obstacle/gap/mixed 原成功率和步数门槛。连续两轮 validation 全部通过才结束训练。test/OOD 用相同门槛，不在观察结果后放宽。

## 多地图范围

`config/curriculum_v7.json` 保留全部 v6 course 内容和分组，新增 24 张生成式完整地图：train 12、validation 4、test 4、OOD 4。每张是三个连续障碍—缺口段，含高低障碍、不同恢复距离和 12 个地面/低跳松果；地图长度超过 6,000 像素。高障碍后留出 800/880/960 像素恢复距离，允许高台跳跃落地后再次越过缺口。

train 的三种循环顺序与 test 的三种反向顺序不同；test 还改变出生位置。OOD 使用 176 像素障碍、196 像素缺口和反向顺序。生成地图几何 hash 全部不同、split 无交叉。已知 `level_1` 同时留在 full train / validation 作为显式回归例外，不能算新地图泛化。

新地图只覆盖固定生成族的静态几何和松果路线；不代表所有原始关卡、未观测敌人行为或复杂交互泛化。新 test/OOD 在冻结三 seed 训练计划前未用于策略调试；结构校验可以加载地图，但示范、PPO 和 teacher 路线校准只使用 train/validation。validation 的 teacher 校准用于路线可行性，未用于监督标签。

三 seed 收集优化各从自己的 full 模型复制 actor/critic，重新初始化优化器、PPO 计步和课程；不是严格优化器续训，也不是纯 PPO。计划最多 98,304 新 transition，主 seed 仍预指定 20261004，报告所有 seed。冻结配置、来源 hash、课程 hash 与训练签名在 `config/full_collection_v7_plan.json`。

## 重跑和播放

```bash
.venv/bin/python -m scripts.train_ppo \
  --config config/ppo_full_collection_v7.json \
  --init-from runs/ppo_full_train_correction/model.zip --start-stage full \
  --output-dir runs/full_collection_v7_rerun

.venv/bin/python -m scripts.accept_full_checkpoint \
  --model runs/full_collection_v7/seed_20261004/model.zip \
  --output-dir runs/full_collection_v7_acceptance_rerun

# 新生成 full validation 地图，可按 N 切换
.venv/bin/python -m scripts.play_ppo \
  --model runs/full_collection_v7/seed_20261004/model.zip --task full --suite validation

# 原始已知关卡
.venv/bin/python -m scripts.play_ppo \
  --model runs/full_collection_v7/seed_20261004/model.zip --level-id level_1
```

输出目录应使用新路径；重新查看 v7 test/OOD 是回归，不再称为新的盲测。当前基配置使用 seed 20261004；其他配置只改变 seed，并初始化自对应谱系，不能都指向主 seed。

## v7 首次验收结果：未全部通过

三次确定性 validation 均通过，但完整保留验收最终为 **1/3**：

| seed | 整体通过 | 实际失败 |
| --- | --- | --- |
| 20261004 | 是 | 无 |
| 20261005 | 否 | 原 `level_1` 采样策略一次少收松果，未达到 5/25 的逐局门槛 |
| 20261006 | 否 | `v7_full_test_02` 确定性策略在最后高障碍前持续向右，4 局均 1,024 步超时 |

其余新 full test/OOD 及旧课程回归的门槛通过。失败不是用平均成功率或降低收集标准掩盖。机器报告：[full_collection_v7.json](reports/full_collection_v7.json)。

对全局位置/进度/收集比例的过拟合是待验证的解释，不是已证实的唯一原因。下一轮同时扩展训练顺序并加入全局特征增强，故结果不能当作该增强单独效果的因果消融。新一轮以 v8 新保留几何验收，v7 全部保留集改作回归。最新开发结果见 [v8 报告](FULL_MULTISEED_COLLECTION_V8.md)。
