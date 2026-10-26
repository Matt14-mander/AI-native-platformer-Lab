# Telemetry v1：游玩事实记录

本阶段实现可选事件采集、本地存储与读取校验，首先接入 `scripts.play_level_spec` 的键盘、规则和固定路线重放入口。2026-10-26 已补齐 [行为指标与玩家画像 v1](PLAYER_PROFILE_V1.md)、逐 tick 运动聚合和 `scripts.play_ppo --telemetry-dir`。`source.main` 暂未接入；规则 DDA 与 Jev 尚未实施。

## 开始记录真人游玩

在仓库根目录执行，输出目录必须尚不存在：

```bash
.venv/bin/python -m scripts.play_level_spec \
  --input game_content/level_specs/full_v1.json \
  --controller keyboard \
  --telemetry-dir runs/telemetry/human_001 \
  --player-id local_player_01 \
  --trace-actions
```

也可把 input 换成 `runs/llm/live_006/level.json`。方向键移动，Space/A 跳跃，Shift/S 奔跑；P 暂停/继续，R 重开，Esc 退出。每次进程使用新的输出目录；同一个玩家后续可以继续使用同一 player-id。不指定 player-id 时，身份范围仅限当前 session，不把多个匿名会话当作同一玩家。

`--trace-actions` 可省略。开启后除事件外，还保存每个决策的动作、决策后世界状态 SHA256 和累计奖励，供固定 seed、原地图和原环境下独立重放核查；身份与墙钟时间不进入游戏状态哈希。

查看记录：

```bash
.venv/bin/python -m scripts.inspect_telemetry \
  --input runs/telemetry/human_001 \
  --output runs/telemetry/human_001_report.json
```

报告路径必须为新文件且位于源 session 目录之外。读取命令不修改源记录；不需要 Pygame、Gymnasium、Torch 或 API Key。它只展示事件计数、尝试账本和完整性，不生成画像或技能判断。

## 记录格式

每个 session 目录包含：

- `manifest.json`：schema_version=1、session_id、actor_type、可选 player_id、身份范围、环境配置、模拟 tick 频率、播放倍速、headless 标志、来源；另有记录状态、事件数和事件文件 SHA256。
- `events.jsonl`：按 event_seq 追加的事件流。默认每 64 个事件 flush；尝试结束、会话结束和 manifest 更新时也 flush。异常退出仍可能丢失尚未刷出的末尾缓冲，不承诺断电级持久性。
- `attempts.json`：正常关闭时原子生成的尝试摘要缓存。读取校验从原始事件重建账本，不依赖此缓存。

每条事件的公共字段为 schema_version、session_id、attempt_id、event_seq、player_id、actor_type、level_id、level_sha256、protocol_sha256、seed、tick、decision、wall_seconds、type、payload。会话级事件的 attempt/map/tick 上下文为空。

actor_type 明确区分 human、scripted、ppo、replay。每次 reset 生成新的 attempt_id；相同地图和 seed 的重复尝试也不会混淆。地图哈希沿用 content_hash，环境协议哈希沿用 EnvironmentFactory.protocol，采集器不加入原协议。

| 事件 | 含义 |
| --- | --- |
| session_start / session_end | 会话开始、关闭；空会话也可正确关闭 |
| attempt_start / attempt_end | 独立尝试边界；结束原因和最终计数 |
| pause / resume | 入口暂停状态变化，重复设置不重复记事件 |
| jump_requested | 新的按跳边沿；不等同于跳跃成功 |
| jump_started | core 接受的落地状态按跳边沿，包括起跳后立即顶到砖块 |
| jump_released | 松开跳跃，以及此前连续按跳的真实物理 tick 数 |
| landed | 从未落地变为落地 |
| coin_collected | core 实际收集的实体 ID、坐标、计数与玩家位置 |
| interaction | core 的方块、奖励箱、道具、踩敌、受击和护盾事件 |
| action / step | 可选动作轨迹及逐决策世界状态哈希、累计奖励 |
| speed_changed | 播放倍速发生变化，保留变化值；画像排除非正常倍速会话 |

运动、交互和尝试边界事件保存玩家快照。v1 不保存每个物理 tick 的全部世界快照；新增采集在 attempt_end 的可选 motion 字段保存版本化的运动聚合。旧日志仍可读，缺失运动信息时保持未知。

结束原因区分 success、death、time_limit、restart、quit、level_switch、replay 和 error。核心自然终局立即结算，随后退出不会再结算一次。运行中同图 reset 记 restart，固定路线的同图 reset 记 replay，换图 reset 记 level_switch；终局后的新尝试不补造一个重开结果。程序异常时已启动的尝试以 error 关闭；进程被强制结束而未能执行清理时，读取标为 incomplete。

## 时间、完整性和故障处理

模拟耗时为 tick / tick_hz，独立于播放倍速和暂停。墙钟耗时使用 monotonic clock，尝试摘要分开记录 active_wall_seconds 与 paused_wall_seconds；暂停时重开会在新尝试继续累计暂停时间。活跃墙钟包含模拟、渲染及记录开销，不能当作纯玩家反应时间。

采集发生在环境 action_repeat 循环的每个真实物理 tick 之后，因此能获得中间帧的收集和交互；原有观察、奖励、info、core 状态与返回值保持不变。记录器不改变输入动作、不控制模拟时钟、不在训练环境中默认启用。

读取校验覆盖 schema、字段类型、身份、连续事件序号、地图/seed 一致性、单调计数、尝试生命周期、暂停切换、重复松果、收集计数对账和可选动作/step 配对。closed 会话必须匹配最终 SHA256、事件数和结束事件。未关闭文件只有最后一个未写完的 JSON 行可被容忍；其他损坏明确报错。未闭合尝试只标记 incomplete，不猜测其死亡或通关结果。

写入或 observer 运行失败时发出 warning，停用该记录器并尽力标记 incomplete，游戏继续运行。若输出目录已存在则在启动时拒绝覆盖；若启动时发生文件系统故障则提示无法采集并继续游戏。

`eligible_for_profile` 仅是后续画像的输入资格筛选：要求完整会话、human、正常倍速、非 headless，且至少有一个自然结束的尝试。它不代表样本量足够、不代表画像已实现，也不把 restart/quit/error 当作死亡。匿名数据的身份范围仍然仅限当前 session。

## API 接入与后续范围

```python
from ai_platformer.adaptive.telemetry import TelemetryRecorder

recorder = TelemetryRecorder(
    "runs/telemetry/api_001", factory,
    actor_type="human", player_id="local_player_01", trace_actions=True,
)
env = factory.make()
env.telemetry = recorder  # 必须在第一次 reset 前挂载
try:
    observation, info = env.reset(seed=100)
    # 原有动作与 env.step 流程；UI 暂停时同步 recorder.set_paused(True/False)。
finally:
    recorder.close()  # 生命周期所有者负责关闭；异常路径可使用 close("error")。
    env.close()
```

记录器由入口管理，可跨环境尝试复用，但 factory 必须能解析目标地图和协议。PPO 接入时 actor_type 使用 ppo，source 应携带 checkpoint 哈希、后端和确定性模式；固定动作重放使用 replay。关闭和写入故障的会话不应继续复用。

自动验证与本机烟测结果见 [验收记录](reports/telemetry_v1.json)。覆盖真实 core 事件、动作重放一致性、自然终局、暂停/重开/切图、写盘故障、损坏记录、真实离屏键盘事件循环和规则 CLI。离屏自动键盘检查不等于真实玩家人工验收；人工样本与画像有效性仍属于后续迭代。

2026-10-24 首轮全量测试为 236 passed、8 subtests passed；ruff 通过。v9 在 live_006 上开启/关闭记录的动作序列、最终世界状态和总回报一致，均为 211 决策、841 tick、17/20 松果、通关。2026-10-26 增量后的测试与开销见 [玩家画像 v1](PLAYER_PROFILE_V1.md)。

首轮本机固定动作基准（尚无运动聚合；每模式 5 轮，每轮 10 局；无策略推理和渲染，计入 reset、写盘和关闭）测得每决策耗时中位数：关闭 0.335ms，默认事件 0.379ms，可选动作与状态哈希 0.679ms。相对开销约 +13.1% / +102.7%；这是该地图与机器的测量，不能作为所有关卡的性能承诺。默认不记录动作哈希，正常游玩不指定 telemetry-dir 时不创建记录器。详细样本保留在首轮验收 JSON 中。
