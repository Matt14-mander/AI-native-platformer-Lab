# 缺口学习第二轮：诊断、课程分布与权重初始化

本轮保持 `PlatformerState-v1` 的 15 维 observation、动作 ID、物理和 reward 不变。目标是先辨别缺口失败的行为机制，再做可复现、相同新增预算的开发对照。

## 已实现的接口

`diagnose_ppo` 在同一批布局、配对 episode seed 下比较三种策略：

- `deterministic`：每步选择概率最大的动作，与独立 PPO 评估一致。
- `stochastic`：根据实时 observation 的动作概率采样。
- `fixed_observation`：始终输入该 episode 的首帧 observation，按其动作概率采样，作为不接收后续状态信息的对照。

采样使用每个 episode 独立的 CPU Torch generator，不消耗训练的全局随机流。报告包括成功/死亡/超时率、每个布局结果、动作频次、按真实环境状态分组的平均动作概率、有效起跳、落地事件、首次起跳到缺口的距离和落到对岸的比例。事件在每个 core tick 上采集，因此不会漏掉 action repeat 内的起跳与落地。

“落到对岸”指落地时玩家身体的水平范围已触及对岸，不等于最终安全通关；成功也可能发生在空中到达终点时，两个指标应分别解读。边缘距离为缺口起点减去起跳前玩家右边缘，越过边缘后为负值。动作概率按决策时的真实状态分组，固定观测对照也采用同一分组规则。动作分布随状态变化只说明相关性；单凭成功率或概率变化不能证明策略使用了边缘传感器。

`--init-from` 创建新 PPO，仅复制 actor 和 critic 参数，重新创建优化器、随机流、累计步数和课程历史。来源路径、模型 SHA-256、来源步数与协议哈希写入每个 checkpoint 和运行报告；后续严格续训继续保留该来源。允许更换课程内容、训练 seed、PPO 超参数及预算；仍要求非内容环境协议、动作映射和网络结构兼容。

`--resume` 继续恢复优化器、步数和课程状态，仍使用原来的完整兼容性检查；两个选项互斥。`--start-stage` 现在支持冷启动、初始化和严格续训，显式切换写入历史，跳过的阶段不会标为 mastered。

## Curriculum v2

`config/curriculum_v2.json` 包含 272 个唯一布局，由 `scripts/generate_curriculum_v2.py` 可复现生成：

| 课程 | train | validation | final test | OOD |
| --- | ---: | ---: | ---: | ---: |
| flat | 8 | 4 | 4 | — |
| obstacle | 8 | 4 | 4 | — |
| gap | 36 | 36 | 36 | 12 |
| mixed | 36 | 36 | 36 | 12 |

flat/obstacle 保留 v1 的分组和几何，用于旧任务回放与回归。gap/mixed 的训练、验证和最终测试分别覆盖 3 个出生点 × 4 个接近距离 × 3 个宽度区间；小幅偏移防止精确几何重复。OOD 使用更长的接近距离和宽度 170 的缺口，不参与采样、晋级或 best 模型选择。v1 manifest、模型及最终测试保持原样。

`ppo_curriculum_v2.json` 保留 20% 旧课程回放，连续两轮、每个布局成功率达标且保持旧课程才能晋级，新增每个成功 episode 的步数上限：flat 128、obstacle 160、gap 192、mixed 256、full 768。上限同时检查旧课程；无法用平均值掩盖单个布局的长时间停滞。best 选择仍优先成功率与进度，在相同得分时偏向更快通关。只有新配置启用该约束，v1 的续训契约不变。

v1 的地面边缘传感器表示玩家当前脚底高度上的支撑距离；起跳后通常为 0，不能当作空中向下投影得到的缺口距离。本轮加入几何回归测试并保持其语义不变。如果后续需要空中落点信息，应另设 observation 版本。

## 使用方法

在仓库根目录执行；受限环境可设置项目内缓存：

```bash
export MPLCONFIGDIR="$PWD/.venv/matplotlib"
export XDG_CACHE_HOME="$PWD/.venv/cache"
```

复验新课程门禁，覆盖 train/validation，不使用学习型策略的 final test：

```bash
.venv/bin/python -m scripts.validate_rl_readiness \
  --courses --environment-id PlatformerState-v1 \
  --manifest config/curriculum_v2.json --episodes 1000 \
  --output runs/readiness_v2_new.json
```

相同新增预算的两组训练，输出目录须为空：

```bash
.venv/bin/python -m scripts.train_ppo \
  --config config/ppo_curriculum_v2.json --start-stage gap \
  --timesteps 65536 --output-dir runs/gap_v2_cold_new

.venv/bin/python -m scripts.train_ppo \
  --config config/ppo_curriculum_v2.json --start-stage gap \
  --init-from runs/ppo_state_v1_gap_learning/model.zip \
  --timesteps 65536 --output-dir runs/gap_v2_initialized_new
```

初始化组已有来源模型的 90,112 步历史训练，两组只有本轮新增预算相同，不能视为总算力相同。当前为单训练 seed 开发实验，环境 seed 主要改变采样随机流，不能把重复固定布局的 episode 当作独立泛化样本。

策略诊断与 OOD 评估：

```bash
.venv/bin/python -m scripts.diagnose_ppo \
  --model runs/gap_v2_cold_new/model.zip --task gap --suite validation \
  --seeds 200 201 202 203 204 205 206 207 208 209 \
  --output runs/gap_v2_cold_new/gap_diagnostics.json

.venv/bin/python -m scripts.diagnose_ppo \
  --model runs/gap_v2_cold_new/model.zip --task gap --suite ood \
  --output runs/gap_v2_cold_new/gap_ood.json
```

`--manifest config/curriculum_v1.json` 可对旧几何做迁移/回归诊断。入口先验证 checkpoint 的原始内容协议，再使用指定 manifest；报告同时保留来源协议和评估协议。final test 必须在方案冻结后使用 `--suite test`，且 seed 不得与来源训练/验证池重合。

## 本机验证与开发实验

实际结果见 `docs/reports/ppo_gap_stage_v2.json`。运行模型、Monitor、失败轨迹和完整诊断保存在被 Git 忽略的 `runs/`。报告明确区分最终模型与验证挑选的 best，开发结论不会用 final test 结果选方案。

本机结果（2026-10-02，每组新增 65,536 transitions，比较最终模型）：

| 指标 | 冷启动 | 旧模型权重初始化 |
| --- | ---: | ---: |
| gap validation 确定性成功布局 | 9/36（25%） | 36/36（100%） |
| gap validation 实时观测采样成功率 | 53.33% | 77.50% |
| gap validation 固定首帧观测采样成功率 | 56.67% | 29.44% |
| gap OOD 确定性成功布局 | 0/12 | 12/12 |
| v1 gap validation 确定性成功布局 | 0/4 | 4/4 |
| flat validation 确定性成功布局 | 4/4 | 4/4 |
| obstacle validation 确定性成功布局 | 3/4 | 3/4 |

初始化组最终模型为 `runs/ppo_gap_v2_initialized/model.zip`，累计本轮步数从 0 开始到 65,536；来源模型有 90,112 步历史训练。初始化组确定性策略首次起跳距缺口边缘约 104.6–213 像素，起跳位置随布局变化；其固定首帧观测对照显著低于实时观测采样，支持状态输入对通关有贡献，尚不能单独归因到某个边缘特征。

两组均保持在 gap，状态为 `timesteps_limit`。初始化组的障碍失败布局为 `course_obstacle_09`，冷启动为 `course_obstacle_11`；旧任务保持条件未通过，未晋级 mixed。初始化组 final 的成功障碍 episode 为 31 步，未出现此前 928 步才通关的情况，但失败布局仍需修复。

需区分 `best_gap` 与最终模型：初始化组 best checkpoint 保存于最后一次 rollout 优化前，虽然 gap 同样为 100%，障碍回归仅 2/4；最终优化后的模型为 3/4。best 选择以当前课程得分为主，不能把单任务 best 当作部署合格模型。后续候选选择应同时检查旧任务保持及通关速度。

验证记录：49 项单元/集成测试全部通过，无跳过；Ruff 与 diff 空白检查通过。1,000 个随机 episode 的稳定性门禁通过，共 235,478 transitions；168 个 train/validation 布局的 SB3 checker、规则可达性和奖励守恒审计通过。272 个布局完成 scripted 几何校准；学习型策略未运行 final test。以上是单训练 seed 开发结果，正式多 seed 验收尚未完成。

下一轮优先修复障碍回归：记录失败障碍处的动作、跳跃释放与再次起跳行为，补充同分布的训练几何和有权重的 obstacle/gap 回放，并将旧任务保持加入候选模型选择；维持缺口验证表现后再进入 mixed 与多 seed 验收。验证失败布局用于诊断，不直接加入训练集合。
