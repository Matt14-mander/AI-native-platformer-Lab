# 行为指标与玩家画像 v1

日期：2026-10-26。已实现 Telemetry 日志 → 离线行为统计 → 可解释玩家画像 → JSON/Markdown 报告。全程本地运行，不需要 API Key；不会修改地图、物理、奖励、观察协议或自动调整难度。mixed_ground 漏收仍保留为既有策略缺口。

## 运行方式

先正常记录真人游玩，沿用同一个 player-id，不同进程使用不同输出目录：

```bash
.venv/bin/python -m scripts.play_level_spec \
  --input runs/llm/live_006/level.json --controller keyboard \
  --telemetry-dir runs/telemetry/human_001 --player-id local_player_01
```

可用 `game_content/level_specs/full_v1.json` 或其他已校验地图继续记录。P 暂停、R 重开、Esc 退出；自然通关、死亡和超时才计入有效结果，主动重开与退出会单独保留。无需开启 trace-actions，也能在新记录中获取运动聚合。

退出游戏后按**从旧到新**列出会话：

```bash
.venv/bin/python -m scripts.summarize_play_sessions \
  --input runs/telemetry/human_001 runs/telemetry/human_002 \
  --output-dir runs/telemetry/profile_001
```

输出 `report.json` 与 `report.md`。输出目录必须尚不存在且不能位于输入会话内部。最多输入 128 个不同会话；复制同一 session 不能增加证据。输入日志只读，报告记录 source manifest/events 的 SHA256，并核查分析期间源文件没有变化。

默认窗口为最近 10 次有效尝试，可用 `--window 5..100` 调整。v1 使用显式输入顺序，再使用会话内尝试顺序；不使用文件修改时间猜测新旧，也不跨会话猜测失败重试。相同日志、规则版本、顺序与窗口可重建相同 JSON/Markdown；输出不引入新的墙钟生成时间，build_id 标识输入和规则版本。

## 指标口径

指标定义版本为 `metrics_version=1`。每次尝试保留原 session/attempt ID、actor、身份、地图/协议/环境哈希和排除原因，汇总按身份、actor、地图内容、协议、播放模式分组；非真人还区分控制器来源。不同来源、地图版本与环境不混算技能水平。

| 指标 | 定义及边界 |
| --- | --- |
| 通关率 | 完整会话中 success / (success + death + time_limit)；中断不计死亡 |
| 完成耗时 | 仅成功尝试的模拟秒数；暂停、渲染和播放倍速不改变 tick 时间 |
| 推进效率 | 最大进度与净水平位移/模拟秒分别报告；不作跨地图排名 |
| 收集率 | 收集数 / 当前可玩地图权威松果总数；零松果分母显示 null；不把分母当作已验证可达机会数 |
| 跳跃行为 | 请求、有效起跳、落地分别计数；松开时记录的按跳长度构成完整样本 |
| 截尾按跳 | 尝试结束时仍按住的时长单独记录，不混入完整按跳时长均值 |
| 空中占比 | 每个真实物理 tick 后 grounded=False 的 tick 占比 |
| 回撤距离 | 每 tick 的 max(0, before.x-after.x) 之和，记录真实运动而非按键意图 |
| 水平静止 | 每 tick 水平位移绝对值 ≤0.05px；原地跳跃也可能属于水平静止 |
| 停滞 | 连续水平静止至少 2 秒的完整区间，包含区间全部 tick；阈值按 tick_hz 转换 |
| 失败后重试 | 同一完整会话内，上一尝试 death/time_limit 后紧接着开始同地图同协议尝试；重开本身不算死亡 |
| 死亡位置 | 按地图与协议保存坐标，分入 100px 横向区域；没有明确证据时死因保持 unknown |
| 交互 | 分别统计实际受击、踩敌、方块与护盾事件，不直接推断战斗技能 |

新采集器使用常量空间逐 tick 聚合运动事实，在 attempt_end 的可选 `motion` 字段落盘，独立声明 `version=1`、观察 tick 数与阈值。读取会核查 tick/阈值一致性。旧事件 schema_version=1 日志继续可读；没有 motion 的旧尝试明确显示 unknown/null，不从稀疏跳跃事件伪造距离、空中占比或停滞。部分新旧样本混合时报告 available/missing 覆盖量。

不完整会话中的尝试可用于查看原始事实，但保守地排除出有效通关率与画像证据。损坏 JSON、digest 不匹配等情况直接拒绝生成报告，不静默忽略。

## PlayerProfile v1

画像规则版本为 `profile_version=1`，结构由以下内容组成：

- 身份：显式 player-id 才能跨会话合并；匿名玩家仅在单一 session 内建模。
- 环境：按已记录的环境配置和 tick_hz 分开；不同物理、奖励或步数限制不会共同提升同一画像的证据量。
- 窗口：默认最近 10 次符合条件的真人尝试；累计统计指全部显式输入的数据，不是数据库中的永久历史。
- 可追溯证据：每个窗口、地图分组及排除项均列出 attempt_id；可追到源事件和哈希。
- 行为维度：逐地图的通过稳定性、推进效率、收集表现、跳跃与运动行为，以及同会话失败后的重试事实。

只有完整会话、actor_type=human、非 headless、正常播放速度、自然结束的尝试进入画像有效窗口。倍速变化过的会话也排除；PPO、scripted、replay 不会创建真人身份或改变真人画像。

`ready` 要求窗口内至少 5 次有效尝试、覆盖至少 2 个不同地图内容哈希。相同内容换 level_id 不增加地图数；这里的内容版本数量不等于严格的几何 OOD 或统计独立性证明。未达标为 `insufficient_data`，并列出缺失条件。ready 仅是工程证据门槛，不是置信区间，不代表所有维度都已知。

所有表现指标按地图展示，不给一个不透明的总技能分。收集偏好、跳跃错误率与战斗技能保持 unknown：低收集率不能独自区分偏好、能力与机会限制，空中按跳也不能直接视作错误。没有真实人工数据时不会用模型样本补出真人画像。

## PPO 播放统一采集

```bash
.venv/bin/python -m scripts.play_ppo \
  --backend sb3 \
  --model runs/full_collection_v9/seed_20261004/model.zip \
  --level-spec runs/llm/live_006/level.json --seed 100 \
  --episodes 1 --headless \
  --telemetry-dir runs/telemetry/ppo_001 --trace-actions
```

PPO 播放记录 actor_type=ppo、模型哈希、后端、确定性模式、融合开关、暂停/重放/切图与倍速变化。SB3 与 TinyInfer 复用相同 recorder；后者仍需原有 library/ONNX 参数。模型数据可加入同一批量报告，但始终单独展示。

真实本机烟测报告位于 `runs/telemetry/metrics_profile_v1/report/`，输入两个旧会话与一个新 PPO 会话，共 4 次尝试。没有真人记录，因此 profiles=0；这体现数据隔离，不表示自动测试未覆盖画像规则。验收摘要见 [机器报告](reports/player_profile_v1.json)。

## 验收与下一步

全量测试 248 passed、8 subtests passed；ruff 通过。覆盖手工可算的真实运动轨迹、收集分母、自然终局/中断、同图重试、匿名/玩家/环境/actor 隔离、最近窗口、重复样本拒绝、旧日志、未知维度、PPO CLI 与离线重建。另验证了 reset 中途采集器失败后，游戏和关闭流程仍正常运行，残缺记录不会生成有效画像证据。

真实 v9 在 live_006 保持 211 决策、841 tick、17/20 松果；相较原 Telemetry v1 存档，动作与每个决策的世界状态哈希、累计奖励完全一致。未修改策略、地图或原保留评测成绩。

新增运动聚合后的本机测量：每决策中位耗时关闭采集 0.315ms、默认事件 0.418ms、事件加动作/哈希轨迹 0.728ms。测量为三轮轮换顺序，每轮固定动作重放十局，包含 reset 与记录 I/O，不包含推理与渲染；相对关闭约 +32.7% / +131.3%，仅代表该机器与工作负载。详细样本保留在机器报告中，训练仍默认关闭采集。

下一步需要真实玩家完成少量可比场景的正常游玩，并检查报告是否贴合实际体验。随后再开发独立的规则 DDA 建议模式；本轮不进入自动调难度或 Jev 决策。
