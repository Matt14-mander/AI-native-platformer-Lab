# 游戏内容迁移计划

## 目标

把 legacy JSON 和 Pygame 对象逐步迁移为 renderer-independent、可序列化、可复现的 core 内容。每类内容都必须同时支持：

1. headless simulation；
2. `WorldSnapshot`；
3. Pygame rendering；
4. reset/replay；
5. 测试与未来 benchmark observation。

## 数据事实来源

- 当前迁移输入：`source/data/`；
- 新增可玩内容 overlay：`game_content/levels/`；
- 根目录 `data/` 暂时冻结，不再新增内容；
- 迁移期间通过 `LegacyLevelRepository` 读取旧格式；
- 最终目标：版本化 `LevelSpec`，legacy adapter 只作为兼容层；
- 原创可发布资产进入 `assets/`，不覆盖 `resources/` 中的参考素材。

## 当前迁移矩阵

| 内容 | Core | Snapshot | Pygame | Reset/Test | 状态 |
| --- | --- | --- | --- | --- | --- |
| 出生点/边界 | ✓ | ✓ | ✓ | ✓ | 完成 |
| 地面/管道/台阶 | ✓ | 间接 | 背景图 | ✓ | 初版完成 |
| 旗杆/终点 | ✓ | progress | 背景图 | ✓ | 初版完成 |
| 金币 | ✓ | EntitySnapshot | ✓ | ✓ | 本轮完成 |
| score/收集数 | ✓ | metadata/info | HUD | ✓ | 本轮完成 |
| 砖块 | ✓ | ✓ | ✓ | ✓ | 第一关前段可破坏砖已接入 |
| 箱子/奖励 | ✓ | ✓ | ✓ | ✓ | 空箱、橡果箱、护盾箱已接入 |
| 敌人 | ✓ | ✓ | ✓ | ✓ | 一种地面巡逻敌人已接入 |
| power-up | ✓ | ✓ | ✓ | ✓ | 一次性护盾浆果已接入 |
| checkpoint/传送 | — | — | 背景图 | — | 待迁移 |
| 音乐/音效 | — | — | legacy | — | 待抽象 |

## 迁移批次

### Batch A：交互方块

- 定义 `BlockSpawn`、运行时 block state 和稳定 entity ID；
- 迁移砖块与 box；
- 实现顶撞、破坏、一次性触发和奖励生成；
- snapshot 必须表达 active/used/broken；
- 固定动作 replay 的结果必须一致。

验收：至少一个砖块、一个金币箱和一个空箱可在 headless/Pygame 中一致交互。

### Batch B：敌人与伤害

- 先迁移一种地面敌人；
- 定义 patrol、碰撞、踩踏、伤害与死亡；
- 敌人不能直接读取 renderer、键盘或 wall clock；
- 增加无敌帧/生命系统前，先采用单次碰撞死亡的 benchmark 规则。

验收：固定 seed 下敌人轨迹一致，踩踏与玩家死亡都有稳定测试。

### Batch C：道具、生命与 checkpoint

- mushroom/power-up；
- player form 与 hit points；
- checkpoint 和管道/区域切换；
- episode resume 规则与 replay 格式。

验收：死亡重生不会泄漏旧实体状态，checkpoint 行为可复现。

### Batch D：音频、主题与原创资产

- 音频事件由 core 发出语义事件，Pygame adapter 决定播放素材；
- 建立 theme manifest：角色、tileset、背景、UI、音乐、音效；
- 使用原创或明确授权素材替换马里奥资产；
- 发布构建执行第三方 IP 资产检查。

## 与 AI 的边界

- PPO 只依赖稳定 action/observation/reward，不等待所有内容迁移完成；
- PCG/LLM 必须等待 `LevelSpec v1` 和可玩性 validator；
- DDA 必须等待 telemetry 事件、玩家表现指标与难度参数边界；
- Jev 必须等待明确的高级候选动作，不能进入逐帧物理层。

## 下一内容任务

第一关前段已实现 Batch A、B 和一次性护盾道具；其余关卡内容仍可按以下顺序扩展：

1. `PlatformerState-v0` + scripted baselines；
2. PPO baseline；
3. 交互砖块/箱子；
4. 单一地面敌人；
5. LevelSpec v1 与 PCG validator。
