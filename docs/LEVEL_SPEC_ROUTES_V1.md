# LevelSpec 加载、播放与路线验证 v1

新格式关卡现在可以直接通过 `EnvironmentFactory` 加载，复用 Gym 环境、真实核心和现有 Pygame 渲染器。已提供键盘、规则 agent、路线重放与 PPO/TinyInfer 播放。路线搜索找到候选动作后，在全新环境中录制并再次独立重放，才返回 verified。

## 加载与准入

```python
from ai_platformer.envs.factory import EnvironmentFactory

factory = EnvironmentFactory({
    "environment_id": "PlatformerState-v2",
    "level_spec": "game_content/level_specs/full_v1.json",
    "action_repeat": 4,
    "episode_step_limit": 768,
})
env = factory.make()
observation, info = env.reset(seed=100)
```

`level_spec` 与 `curriculum_manifest` 互斥。加载读取固定文档、转换为 LevelDefinition，并按最终解析的 physics 执行静态准入；失败不会进入模拟。此仓库只管理单张候选地图，不分配 train/test split，不自动加入训练池。协议同时固定地图内容、LevelSpec 文档、物理、reward、观测和动作时序。

原课程仓库和协议保持原行为。`factory.for_level_spec(path)` 创建独立候选工厂，继承现有工厂的已解析环境配置，不修改原工厂。`scripts.play_ppo --level-spec` 先验证 checkpoint 自己的原协议，再明确在候选地图上试玩，不宣称新地图属于 checkpoint 原验收集。

安装内容和环境依赖：

```bash
.venv/bin/python -m pip install -e '.[content,benchmark]'
```

PPO 播放另需 training；TinyInfer 运行另需 inference 和已构建动态库。Pydantic 仅在新格式功能使用，旧 core/课程路径不强制导入。

## 搜索与重放

```bash
# 输出必须不存在；使用新目录复跑
.venv/bin/python -m scripts.search_level_route \
  --input game_content/level_specs/full_v1.json \
  --min-coin-ratio 0.75 \
  --max-steps 768 --max-expansions 25000 --beam-width 24 --max-seconds 15 \
  --output runs/level_route_rerun/full.json

.venv/bin/python -m scripts.replay_level_route \
  --input game_content/level_specs/full_v1.json \
  --route runs/level_route_rerun/full.json \
  --output runs/level_route_rerun/replay.json
```

搜索默认先试 move-right、rule-jump、coin-jump 的真实环境候选，再以所有十种 Action 做有限宽度 beam 搜索。每次决策仍推进 `action_repeat` 个真实核心 tick；分支复制完整核心状态，没有另外编写跳跃公式。使用位置/速度量化与 jump latch、已收集松果状态去重，按前进距离及指定收集目标排序。`--no-scripted-probes` 可以只运行 beam 搜索。

这不是完备规划器，也不寻找最短路线。状态量化、beam 截断、有限动作序列和预算都会漏掉可行路线。结果：

| status | 含义 |
| --- | --- |
| verified | 动作序列成功到达终点并满足显式收集目标，独立重放通过 |
| unknown | 预算用完、当前搜索未找到证据，或交互内容暂不支持搜索 |
| invalid | 输入、配置或静态准入失败；查看 reason |

`min_coin_ratio` 默认为 0，只要求通关；空松果地图按收集目标满足处理。使用 0.75/1.0 分别要求至少 75%/全部收集；失败不能证明松果不可达。当前搜索支持静态 solids/collectibles；含 blocks/enemies/powerups 的地图返回 unknown，仍可进行键盘或 agent 播放。

预算包括决策长度、候选扩展数、beam 宽度及搜索时间。默认 768 步、25,000 次扩展、beam=24、15 秒；每次规则决策或 beam 分支计一次扩展。时间预算在扩展之间检查，不打断正在执行的单次环境/core 推进。搜索时间不包含随后证据录制和独立重放，二者分别最多 30 秒并在决策之间检查。路线长度上限 10,000 决策、单决策最多 120 tick，完整证据最多 100,000 tick。

重放文档固定 seed、已解析环境设置、地图/协议 SHA256、动作序列、初始状态和每次决策的完整 WorldSnapshot/观测/reward/info 哈希，以及终局、步数、tick 和松果数。它不是逐物理 tick 的独立日志。每次重新创建环境再比较；地图元数据改变也会使文档协议失配。哈希用于内容一致性检查，不是签名认证。

成功退出 0；unknown/invalid 或重放失败退出 1，同时保留报告；输出已存在退出 2。不覆盖既有输出。

## 播放

```bash
# 键盘：方向键移动，A/Space 跳跃，S/Shift 奔跑
.venv/bin/python -m scripts.play_level_spec \
  --input game_content/level_specs/full_v1.json

# 播放已提交、验证过的 75% 收集目标路线
.venv/bin/python -m scripts.play_level_spec \
  --input game_content/level_specs/full_v1.json --controller replay \
  --route game_content/routes/full_v1.route.json

# 真实 Pygame 无窗口重放与截图
.venv/bin/python -m scripts.play_level_spec \
  --input game_content/level_specs/full_v1.json --controller replay \
  --route game_content/routes/full_v1.route.json --headless --episodes 1 \
  --screenshot runs/level_route_rerun/replay.png

# 使用 PPO 模型自己的环境设置试玩新格式地图
.venv/bin/python -m scripts.play_ppo \
  --model runs/full_collection_v9/seed_20261004/model.zip \
  --level-spec game_content/level_specs/full_v1.json

# TinyInfer 融合后端
.venv/bin/python -m scripts.play_ppo --backend tinyinfer --fuse-relu \
  --model runs/deployment_v9/seed_20261004/actor.onnx \
  --library build/tinyinfer_bridge/libplatformer_tinyinfer.dylib \
  --level-spec game_content/level_specs/full_v1.json
```

独立播放支持 P 暂停、R 重开、Esc 退出、`--speed`、`--episodes` 与 `--screenshot`。`--controller rule` 使用规则 agent；headless 需要 rule/replay 和有限 episode 数。重放先验证证据，再按证据的 seed/environment 运行，每步画面实际推进也核对 witness 哈希。PPO 播放沿用原来的暂停/重播/速度控制。

当前复用湖畔主题与 1000×600 窗口、横向相机，metadata.theme 暂不切换美术，尚无纵向地图相机。游戏核心接受的地图尺寸不等于渲染器已支持全部视野需求。

## 本轮验收

| 地图/模式 | 方法 | 搜索扩展数 | 结果 |
| --- | --- | ---: | --- |
| flat 示例 | move-right 候选 | 18 | 18 决策通关，重放通过 |
| flat 示例 | 仅 beam | 1,291 | 18 决策通关，重放通过 |
| 单障碍 course_obstacle_08 | 仅 beam | 2,991 | 27 决策通关，重放通过 |
| full 示例，目标 0 | rule-jump 候选 | 958 | 190 决策，6/12 松果，重放通过 |
| full 示例，目标 75% | coin-jump 候选 | 1,167 | 209 决策/835 tick，11/12 松果，重放通过 |

full 示例上，SB3、TinyInfer 原始图与融合图逐决策的动作、观测、reward、终止、info 和完整 WorldSnapshot 完全一致，均 209 决策、11/12 松果。真实 Pygame PPO/TinyInfer/路线重放均已执行。此地图是已暴露的 validation 地图，结果为加载与播放回归，不是新策略泛化结论。本轮未重新测性能。

全量测试：130 passed、8 subtests passed，两条既有 Gym 弃用提示；最后增加首个差异步诊断后，针对路线测试 15 passed。Ruff 与 diff whitespace 检查通过。测试覆盖预算耗尽、交互内容 unknown、原协议不变、路线/物理/元数据篡改及真实无窗口重放。

旧 `level_1_main` 仍有三个越界地形，严格新格式准入拒绝加载。原课程/legacy 播放保持可用，未裁剪地图或修改 checkpoint。机器报告：[level_spec_routes_v1.json](reports/level_spec_routes_v1.json)。路线示例：[full_v1.route.json](../game_content/routes/full_v1.route.json)。

下一步可以围绕这个管线建立受约束的生成请求和 PCG，再接 LLM；新候选先验证和报告，不自动用于 PPO 重训。
