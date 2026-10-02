# 强化学习训练 v1：实现与使用

开发日期：2026-10-02。对应 `RL_TRAINING_PLAN.md` 的实验基线、课程内容、训练设施及课程状态机；多训练 seed 的正式性能验收尚未完成。

## 本轮实现

- 统一 `EnvironmentFactory`，PPO、scripted benchmark 与独立评估使用相同物理、reward、sensor range、action repeat 和环境内 time limit。配置会冻结实际物理/奖励参数和默认完整关卡 ID。
- 64 个显式课程布局：flat、obstacle、gap、mixed，每类 8 个训练、4 个验证、4 个最终测试布局。关卡内容与 manifest 保存哈希，重复几何或重复分组会被拒绝；legacy 完整关卡继续可用。
- `PlatformerState-v1` 在原有 14 维之后增加 `JUMP_HELD`（索引 14），反映上一个环境动作是否按住跳跃。原 `PlatformerState-v0` 保留 14 维；动作 ID、物理和奖励不变。
- episode 边界采样训练关卡；flat → obstacle → gap → mixed → full 按验证结果晋级。每个布局必须达到阈值，连续两轮达标，同时检查旧课程保持成功率。默认混入约 20% 旧课程 episode。
- 定期评估、按成功率/平均进度选 best、周期 checkpoint、失败轨迹、模型重载与 `--resume`。课程阶段超出预算会停止，避免失败阶段自动升级。
- checkpoint sidecar 记录环境/动作/observation/reward 协议、内容哈希、模型 SHA-256、seed 池、网络、PPO 参数、阶段与历史。加载时拒绝协议不一致的模型；旧无 sidecar 的 v0 模型不支持本次新增的续训入口。
- Monitor CSV 增加 level ID、环境 seed、进度和 outcome；报告包含死亡率、超时率、分关卡结果和成功 episode 的平均通关步数。

新的课程训练配置为 `config/ppo_curriculum_v1.json`，评估配置为 `config/benchmark_v1.json`；两者使用 `PlatformerState-v1`。原 `ppo_state_v0.json` 和 benchmark v0 配置仍使用旧 observation。

## 为什么新增 observation 版本

最初的 v0 短训练在障碍验证布局中通过 3/4，但失败布局的轨迹显示：玩家在 `x=418` 接触障碍后，持续输出 `RIGHT_RUN_JUMP`（动作 9），最后 200 个动作均未释放跳跃。core 仅在从释放转为按下时触发起跳，v0 未暴露这个 latch。

v1 让策略能够观测并学习释放/再次按下。新增信息本身不保证短时间训练就提升成功率，也没有改变 core 的按键规则。测试验证了相同动作下 v0/v1 的前 14 维完全一致，并验证持有、释放、再次起跳和 reset。

## 本机运行环境

已创建项目 `.venv`，使用 Python 3.12.14。当前验证的 macOS x86_64 环境使用 Gymnasium 1.0.0、SB3 2.4.1、PyTorch 2.2.2、NumPy 1.26.4、Pygame 2.6.1。完整依赖快照见 `requirements/training-macos-v1.txt`，包含开发工具。

该快照是本机验证记录。其他平台可以先安装项目 training extras，不应直接把历史 Windows 的 `training-v0.txt` 视为当前机器的依赖锁定文件。

仓库根目录下的 macOS/Linux 命令：

```bash
.venv/bin/python -m pip install -e '.[training,dev]'
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

如果受限环境中 Matplotlib cache 不可写，可在命令前设置项目内路径：

```bash
export MPLCONFIGDIR="$PWD/.venv/matplotlib"
export XDG_CACHE_HOME="$PWD/.venv/cache"
```

## 验证与训练

完整关卡门禁及课程门禁：

```bash
.venv/bin/python -m scripts.validate_rl_readiness \
  --output runs/readiness_full.json
.venv/bin/python -m scripts.validate_rl_readiness \
  --environment-id PlatformerState-v1 --courses \
  --output runs/readiness_courses.json
```

每次默认运行 1,000 个随机 episode。课程门禁覆盖 48 个 train/validation 布局，循环分配 episode；checker、rule-jump 可达性、move-right 对照及进度守恒审计覆盖每个布局。最终测试布局只在几何可达性校准时检查，训练晋级不使用它们。

公平的单障碍 scripted 对照：

```bash
.venv/bin/python -m scripts.benchmark_scripted \
  --config config/benchmark_v1.json --task obstacle --suite validation \
  --output runs/scripted_obstacle.json
```

短训练：

```bash
.venv/bin/python -m scripts.train_ppo \
  --config config/ppo_curriculum_v1.json \
  --timesteps 4096 --eval-every 1024 \
  --output-dir runs/curriculum_smoke_new
```

正式课程运行：

```bash
.venv/bin/python -m scripts.train_ppo \
  --config config/ppo_curriculum_v1.json \
  --output-dir runs/curriculum_seed_20260923
```

正式配置总预算为 1M transitions，五阶段上限依次为 50k、150k、150k、250k、400k。未达到晋级阈值会以 `budget_exhausted` 停止；整体预算用尽为 `timesteps_limit`，全部课程掌握为 `completed`。普通 baseline 训练继续按完整 rollout 执行，所以实际 transitions 可能超过配置。

独立重评验证布局：

```bash
.venv/bin/python -m scripts.evaluate_ppo \
  --model runs/curriculum_smoke_new/model.zip \
  --task obstacle --suite validation \
  --output runs/curriculum_smoke_new/obstacle_validation.json
```

续训使用该运行保存的已解析配置，并选择新的输出目录：

```bash
.venv/bin/python -m scripts.train_ppo \
  --config runs/curriculum_smoke_new/config.json \
  --resume runs/curriculum_smoke_new/model.zip \
  --timesteps 100000 \
  --output-dir runs/curriculum_continued
```

`--timesteps` 表示新增 transitions，阶段预算按累计阶段步数计算。如果之前因阶段预算耗尽而停止，续训前应根据失败报告调整该阶段预算；不允许静默更换训练/验证 seed 池、网络、PPO 参数、物理或 observation 协议。续训恢复模型、优化器、累计步数、阶段与历史，环境及随机流重新初始化，不保证与未中断训练逐步一致。

训练入口拒绝覆盖非空输出目录。每个 checkpoint 的 `.zip` 与同名 `.json` 必须一起保留。

## 手动进入缺口实验

可以从已有兼容 checkpoint 明确选择 `gap`，不必等待 obstacle 自动晋级：

```bash
.venv/bin/python -m scripts.train_ppo \
  --config runs/ppo_state_v1_resume_smoke/config.json \
  --resume runs/ppo_state_v1_resume_smoke/model.zip \
  --start-stage gap --timesteps 16384 --eval-every 4096 \
  --output-dir runs/gap_experiment_new
```

`--start-stage` 仅在续训时可用，保留原来的模型、优化器、协议和 seed 池。切换会重置当前阶段预算起点及连续达标计数，并在历史中写入 `manual_stage_change`；不会把未通过的阶段标为 mastered。旧课程混合采样及回归验证仍然生效，后续自动晋级仍要求先前任务达到阈值。

2026-10-02 已从累计 8,192 步的 checkpoint 手动进入 gap。首轮新增 16,384 步、累计 24,576 步；随后追加 65,536 步，累计达到 90,112 步，缺口阶段实际训练 81,920 transitions。当前结果：

| 任务/集合 | 成功布局 | 成功率 |
| --- | --- | --- |
| flat validation | 4/4 | 100% |
| obstacle validation | 4/4 | 100% |
| gap train（确定性重评） | 3/8 | 37.5% |
| gap validation | 0/4 | 0% |

模型保持在 gap，状态为 `timesteps_limit`，未晋级 mixed。独立验证的失败轨迹仍连续选择动作 9：起点立即跳跃，在靠后的缺口上落空；训练布局中的 `00/01/04` 可以被这种动作序列通过。模型尚未学会根据地面边缘选择起跳时机，增加步数没有改变验证表现。下一轮应先扩展训练几何覆盖并检查探索/动作分布，而非把训练布局成功率当作泛化成功。

当前模型为 `runs/ppo_state_v1_gap_learning/model.zip`，完整结果及失败轨迹也在该目录；可提交的摘要见 `docs/reports/ppo_gap_stage_v1.json`。缺口成功率按独立几何布局计算，不把重复 seed episode 当作独立样本。最终 test 未使用。

## 验证记录与范围

- 全量 42 项测试通过，无跳过；包括真实短 PPO 训练、best/周期模型、重载、续训、协议拒绝、阶段预算和 episode 边界切换。
- 对相同训练 seed，只改变 evaluation 频率的 baseline 训练，最终 policy 参数逐项完全一致，验证评估没有消耗训练随机流或干扰训练状态。
- 64 个课程布局的 rule-jump 可达性校准全部通过；move-right 只能通过 flat。
- 当前完整关卡的 1,000 episode 门禁通过；完整关卡 move-right 进度 12.76%、成功率 0%，rule-jump 成功率 100%。敌人进入后 move-right 会死亡，旧文档中“停到超时”的结果属于历史版本。
- v0 课程短训练 4,096 transitions，flat 晋级于第 3,072 步；obstacle 验证通过 3/4 布局，继续到 6,144 步仍为 3/4。
- v1 训练 4,096 transitions 后 flat 验证成功率 100%、obstacle 0%；续训到累计 8,192 transitions 后，flat 仍为 100%、obstacle 通过 3/4 布局（75%）。未晋级到 gap；这是单个训练 seed 的开发结果，不是正式性能验收。
- v1 的 1,000 episode 课程门禁通过，覆盖 48 个训练/验证布局、233,282 transitions。报告保存在 `runs/readiness_state_v1.json`，模型与训练记录保存在 `runs/ppo_state_v1_smoke/` 和 `runs/ppo_state_v1_resume_smoke/`；这些运行目录被 Git 忽略。关键数字另保存到 `docs/reports/rl_training_v1_validation.json`，便于提交和复核。

固定课程的不同环境 seed 不改变几何，也不改变确定性策略轨迹；重复 seed episode 不能算作独立泛化样本。应按布局汇总成功率，并用多个独立训练 seed 检查学习稳定性。

最终 test 仅在方案冻结后使用 `evaluate_ppo --suite test`。本次没有对学习型策略运行最终 test，也没有完成多 seed 正式实验、完整关卡迁移验收或 ONNX 导出。下一步先做受预算限制的 v1 学习实验，分析是否学会释放跳跃，再进行正式对照。
