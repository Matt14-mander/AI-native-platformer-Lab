# AI-Native Platformer Lab

一个从大一 Pygame 学习项目演进而来的 AI-native 横版平台游戏与智能体实验场。

项目正在把原始马里奥练习代码迁移成一套可复现、可无头运行、可训练 agent、可程序化生成关卡，并最终能部署轻量策略模型的原创游戏基础设施。

> 当前仍包含学习阶段使用的马里奥相关素材，仅用于本地研究和迁移验证。公开发行前会全部替换为原创或明确授权的角色、美术、音乐与音效。

## 当前能力

- 确定性 `BasicPlatformerCore`，不依赖 Pygame、窗口或系统时间；
- 人类玩家与未来 agent 共用同一套离散 `Action`；
- Pygame renderer 只消费 `WorldSnapshot`，不维护第二套物理状态；
- 原创湖畔粉彩主题已接入可玩画面：主角五态动作、湖景、苔土地面、树桩、台阶、橡果、刺猬敌人、互动方块和护盾浆果；
- 固定 seed 的 episode、死亡、通关和重新开始；
- 旧 `level_1.json` 的地形、管道、台阶、出生点、旗杆和金币迁移；
- 金币动态实体、收集状态、score、reward 与 HUD；
- 暂停、快速重开和版本化 gameplay settings；
- headless 单元测试与真实 Pygame 集成测试；
- 已注册的 `PlatformerState-v0`：14 维状态 observation、10 个离散动作与 reward breakdown；
- random、move-right、rule-jump 三个固定 seed scripted benchmark 基线；
- SB3 checker、1,000 episode 稳定性门禁和 reward exploit audit；
- 可复现的 MLP PPO 训练、Monitor 日志、模型保存与确定性评估链路；
- 为 Gymnasium、PPO、PCG、DDA、Jev、LLM 和 ONNX 预留的模块边界。

已迁移第一关前段的一种巡逻敌人、可破坏砖、一次性奖励箱与护盾道具；其余旧地图实体尚未纳入新核心。仍待完成：checkpoint/传送、完整音频流程，以及其他关卡的原创内容。

## 快速开始

推荐使用 64 位官方 CPython 3.11 或 3.12。某些 MSYS Python 发行版没有可用的 Pygame wheel，会被迫本地编译 SDL。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m source.main
```

当前 Codex 工作区也可以使用已经准备好的隔离依赖：

```powershell
$env:PYTHONPATH = (Resolve-Path ".deps").Path
& "C:\Users\Rog\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m source.main
```

## 操作

| 操作 | 按键 |
| --- | --- |
| 移动 | `←` / `→` |
| 跳跃 | `A` / `Space` |
| 奔跑 | `S` / `Shift` |
| 暂停/继续 | `P` / `Esc` |
| 重开当前关卡 | `R` |
| 菜单确认 | `Enter` |

手动游戏默认使用 [`config/gameplay_ai_v2.json`](config/gameplay_ai_v2.json)：固定高度跳跃、较快的移动响应。长按只起跳一次，落地后需松开再按。旧设置保留在 [`config/gameplay.json`](config/gameplay.json)，可用 `python -m source.main --settings config/gameplay.json` 试玩对照。评估结果与训练版本说明见 [游戏设置评估](docs/GAMEPLAY_V2.md)。

## 测试

不安装 Pygame 也能运行 headless core 测试；Pygame 集成测试会自动跳过：

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

安装完整项目依赖后，所有集成测试都应执行并通过：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

## Gymnasium 环境与 Benchmark

安装环境/评估依赖并运行固定验证集：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[benchmark]"
.\.venv\Scripts\platformer-benchmark.exe --suite validation
```

不做 editable install 时，可在仓库根目录直接运行：

```powershell
$env:PYTHONPATH = (Resolve-Path ".deps").Path
& "C:\Users\Rog\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m scripts.benchmark_scripted --suite validation
```

代码中创建环境：

```python
import gymnasium as gym
import ai_platformer.envs  # 注册 PlatformerState-v0

env = gym.make("PlatformerState-v0")
observation, info = env.reset(seed=100)
```

seed 集合和 action-repeat 固定在 [`config/benchmark_v0.json`](config/benchmark_v0.json)。当前关卡本身尚无随机内容，因此 scripted 策略跨 seed 的结果相同；这些 seed 会在 PCG/随机实体进入后继续作为稳定评估协议。

## RL 门禁与 PPO

安装训练依赖后，运行完整的 1,000 episode 前置门禁：

```powershell
python -m scripts.validate_rl_readiness --output runs/readiness_v0.json
```

训练首个 MLP PPO baseline：

```powershell
python -m scripts.train_ppo --output-dir runs/ppo_state_v0_seed_20260923
```

快速验证训练链路可使用：

```powershell
python -m scripts.train_ppo --timesteps 4096 --output-dir runs/ppo_smoke
```

训练配置位于 [`config/ppo_state_v0.json`](config/ppo_state_v0.json)，首轮实验结果与下一步分析见 [`docs/PPO_BASELINE_V0.md`](docs/PPO_BASELINE_V0.md)。当前 PPO v0 尚未通关，其 deterministic policy 与 move-right baseline 同样停在首个管道；下一轮应建立平地/单障碍 curriculum。

## 架构

PPO checkpoint 可直接用 Pygame 播放，支持暂停、重播、关卡切换和速度调节：

```bash
python -m scripts.play_ppo --model runs/mixed_prerequisites_imitation/model.zip --task gap
```

参数和无窗口验证方式见 [PPO 模型播放](docs/PPO_PLAYBACK.md)。

PPO actor 的 ONNX 导出、TinyInfer C++/Python 桥接和数值验证已提供，见 [TinyInfer 部署接入](docs/TINYINFER_DEPLOYMENT.md)。
播放支持 `--backend tinyinfer`，部署耗时与 SB3 对比见 [性能测量](docs/DEPLOYMENT_PERFORMANCE.md)。

```text
Keyboard / Scripted / PPO / Jev
              │
            Action
              │
    Deterministic Game Core  <── LevelDefinition / future LevelSpec
              │
         WorldSnapshot
       ┌──────┼──────────┐
   Pygame   Gymnasium   Replay/Benchmark
```

```text
ai_platformer/
├── core/          # 动作、物理、碰撞、状态和 episode 规则
├── content/       # legacy adapter、未来 LevelSpec/PCG/验证器
├── envs/          # Gymnasium environment（下一阶段）
├── agents/        # scripted、PPO、Jev、LLM adapters
├── adaptive/      # 玩家建模与 DDA
├── rendering/     # Pygame 输入和显示
├── benchmark/     # 固定 seed、指标、replay 和回归评估
└── deployment/    # ONNX 导出和轻量推理
game_content/
└── levels/         # 新内容 overlay，逐步替代 legacy JSON
```

详细文档：

- [架构约束](docs/ARCHITECTURE.md)
- [开发计划](docs/DEVELOPMENT_PLAN.md)
- [强化学习训练开发计划](docs/RL_TRAINING_PLAN.md)
- [游戏内容迁移计划](docs/CONTENT_MIGRATION_PLAN.md)
- [AI 接入门槛与顺序](docs/AI_INTEGRATION_PLAN.md)
- [GitHub 项目元数据](docs/GITHUB.md)

## AI 路线

首个学习型 agent 计划采用 **PPO**，但接入顺序是：

1. 冻结 `Action v1`、结构化 observation 和 reward v1；
2. 实现并注册 `PlatformerState-v0` Gymnasium environment；
3. 通过 Gymnasium/SB3 environment checker；
4. 建立 random、move-right、rule-jump scripted baselines；
5. 再训练 MLP PPO，并在未见 seed/关卡上评估；
6. 稳定后导出 ONNX，最后适配目标轻量推理运行时。

LLM/PCG、DDA 和 Jev 会在 LevelSpec、telemetry 与 benchmark 稳定后接入，避免模型依赖不断变化的游戏协议。

## 内容与版权

`source/` 和 `resources/` 中的旧内容是迁移参考。未来可发布内容放在 `assets/`，通过主题 manifest 引用。发布构建不得包含第三方游戏 IP 素材。

## 项目阶段

当前阶段：**RL readiness gates 完成 + 首个 MLP PPO baseline 已运行**。

下一阶段：**建立平地 → 单管道 → 单缺口 curriculum，让 PPO 明显超过 move-right baseline**。

课程训练设施现已实现：64 个独立布局、统一评估、定期验证、best/周期 checkpoint、续训与自动晋级。新 `PlatformerState-v1` 增加 jump-held observation；v0 保持兼容。使用方式和本机验证结果见 [强化学习训练 v1](docs/RL_TRAINING_V1.md)。正式多 seed 性能验收尚未完成。
