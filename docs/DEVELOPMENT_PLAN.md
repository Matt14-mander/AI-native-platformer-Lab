# 开发计划

## 当前迭代计划（2026-10-29 review）

本节取代下文历史阶段清单中的“下一迭代建议”。下文保留为历史路线，不应把未更新的复选框视为当前能力盘点。

### 现状与判断

- 确定性 core、Gymnasium v2、PPO v9、主模型 TinyInfer 部署已有实现和历史验收报告；本轮没有重跑训练或部署验收。
- LevelSpec、PCG、LLM 生成、路线验证、telemetry、玩家画像与 DDA 建议协议已有实现；DDA 规则决策与应用闭环尚未完成。
- 主游戏已有原创场景、动画、交互反馈、中文菜单、暂停与结算，但仍通过 LegacyLevelRepository 加载固定关卡。
- 生成关卡仍主要通过独立 CLI 游玩，未共享主入口的输入、焦点管理、完整交互反馈和菜单流程。主入口尚未接入 telemetry。
- 下一阶段重点：让已有能力汇入同一条玩家流程，先完成可玩的多关卡体验，再做规则自适应。

### Review 发现与验证边界

1. **P1：测试入口漏检。** README 推荐 unittest discover，但仓库已有 134 个顶层 `def test_*` 定义（含参数化测试），unittest 不会收集这些 pytest 风格用例。应采用 pytest 作为完整测试入口，并明确 content/benchmark/training/deployment 依赖与分层门禁。
2. **P2：生成关卡暂停提示与行为不符。** `scripts/play_level_spec.py` 调用共享暂停页，提示 Enter/Esc/P 继续、M 返回菜单；事件处理却把 Esc 当作退出，并未处理 Enter/M。修复时应让提示由入口能力决定，或统一状态流。
3. **P2：相机回退缺失。** `PygameLevelRenderer._update_camera` 只向右跟随，玩家回头收集可走出左侧画面。最小复现：800px 屏宽，玩家 x 从 1000 返回 600，相机仍为 733.33，角色屏幕 x 为 -133.33。应增加左右死区跟随，保持相机边界约束。
4. **体验衔接缺口。** 生成关卡键盘入口以 action_repeat=4 轮询按键，没有主入口的按键边沿缓存和失焦暂停；也未调用反馈的 consume/advance。保留 AI 评测协议，同时给人类玩家提供统一控制入口，避免混用物理配置和可玩性证据。

本轮验证：中文 UI 流程 9 项、操作手感 7 项，共 16 项通过（SDL dummy）；相机问题由最小调用复现。使用本机 bundled Python 与 `.deps` 执行 README 全量命令未获得完整测试汇总，定向导入确认缺少 pytest。没有进行真人连续游玩、真实 API 请求、模型再训练或完整部署回归；历史报告不等同于本次验证。

### 分阶段任务与验收

2026-10-29 进展：M0 的测试入口、暂停提示和相机回退修复已随 PR #2 合并。M1 首版已实现：共享 PlaySession、本地关卡目录、菜单选关、重复 `--level-spec` 参数、按人类物理配置静态校验与通关路线搜索/重放、统一 HUD/输入/暂停/反馈、结算后下一关、加载失败保留原状态、重试使用已准入地图快照。现支持原关卡与高度 600 的静态 LevelSpec，交互式 LevelSpec 暂不准入；未找到路线证据不等于不可玩。菜单验证同步执行，显示检查提示；更丰富的目录和后台验证可后续迭代。人工连续游玩验收仍待完成，telemetry/存档属于 M2。

| 顺序 | 开发范围 | 验收标准 |
| --- | --- | --- |
| M0 基线修复 | 修正完整测试命令与依赖说明；修复暂停提示、相机回退；更新过时文档状态 | pytest 收集覆盖 unittest 与函数式用例；明确缺依赖跳过项；相关回归通过；生成关卡暂停/退出按提示工作，回头收集不丢失角色 |
| M1 统一关卡游玩（下一功能迭代） | 为主游戏注入关卡仓库/会话配置，支持 legacy 与经验证的 LevelSpec；加入本地关卡选择、重试、结算后下一关 | 菜单进入至少两个不同关卡；复用中文 HUD、输入缓存、失焦暂停、反馈；连续切关与重试无状态泄漏；非法地图给出可理解错误 |
| M2 游玩记录与进度 | 主入口接入 telemetry；明确 human/AI 及物理配置标识；保存解锁、最好成绩和收集记录 | 暂停不计游玩时间；重试产生独立 attempt；正常退出关闭 session；数据写入失败不阻塞游戏；不同环境画像不混合；进度损坏可恢复 |
| M3 规则 DDA 建议 | 基于真实画像实现 hold/ease/challenge；冻结规则阈值和证据要求；输出可审阅建议 | 证据不足或权限锁定时 hold；建议通过现有协议；可重建决策；调整后的地图重新验证；先保持 suggestion_only，不自动修改当前关卡 |
| M4 内容扩展 | checkpoint、第二主题关卡、区域切换、完整音频流程 | checkpoint/reset/replay 确定性；死亡后实体恢复规则明确；玩法变化触发对应模型/生成器兼容性回归 |

M2 首版进展：主入口接入 telemetry v1，使用独立 HumanPlatformer-v1 协议和真实 60 Hz；进入/重试/暂停/终局/离开/退出均有生命周期记录。短按跳跃记录独立输入边沿，实际起跳由 core 的只读观测标志提供，不更改 snapshot、reward 或旧路线哈希。新增本地进度 v1、稳定匿名玩家 ID、按地图内容和物理配置隔离的成绩、上次关卡定位、原子写入、备份恢复和写入互斥；失败降级为内存进度并在 UI 提示。现有关卡仍可自由选择，不新增锁关规则；checkpoint 中途恢复与自动 DDA 不在本轮。

### M1 实施边界

1. 先建立一个共享游玩会话接口，承载关卡来源、物理设置、seed、输入和生命周期；core 继续只负责模拟。
2. 人类默认沿用 gameplay_player，AI/replay 保留各自已声明协议。生成关卡若无目标人类物理配置的验证证据，必须重新验证，不能直接借用 v9 成功率声明可玩。
3. 第一版关卡选择只读取本地已验证内容；联网生成放在后续菜单功能，不能阻塞游戏循环。
4. 不在本迭代同时扩大 observation、重训 PPO、接入 Jev 或新增 checkpoint；避免改变过多契约。
5. 增加关卡切换、输入边沿、焦点管理、非法内容和反馈的集成回归，并安排至少 5 分钟人工游玩验收。

---

以下为历史阶段规划，能力状态以本节及各专项验收文档为准。

## 项目北极星

构建一个既能作为完整横版平台游戏，也能作为 AI agent 研究环境运行的项目。所有实验必须可复现、可比较，并能从训练一路走到轻量部署。

## Phase 0：仓库基线与架构骨架

状态：**核心骨架完成，仓库清理仍在进行**

目标：保住原始项目，同时为新实现建立清晰边界。

- [x] 保留 `source/`、`resources/` 和旧 JSON 数据作为 legacy 基线；
- [x] 建立新 Python package 和模块目录；
- [x] 定义稳定离散动作、状态快照和核心引擎协议；
- [x] 建立依赖分组、测试目录和基础文档；
- [x] 修复 legacy 启动入口并加入有限帧 smoke 能力；
- [ ] 清点重复的 `data/` 与 `source/data/`，确定唯一事实来源；
- [ ] 为旧版录制一个人工操作 smoke path。

验收标准：旧版可启动；新包可导入；基础测试通过；没有移动或删除用户原始素材。

## Phase 1：确定性 Headless Core

状态：**首个 headless 竖切完成，并已接管 Pygame 游戏**

目标：在没有 Pygame 窗口、图片和音频的条件下完整推进基础关卡。

- [x] 定义固定 tick 和数值单位；
- [x] 从旧版迁移玩家移动、跳跃和重力；
- [x] 实现 axis-aligned rect collision；
- [x] 实现出生、坠落死亡、终点和 episode 上限；
- [x] 将旧 `level_1.json` 适配到临时 `LevelDefinition`；
- [x] 支持 `reset(seed, level_id)` 和 `step(action)`；
- [x] 增加 determinism、行走、跳跃、死亡和终点测试；
- [ ] 覆盖斜向顶撞、连续多碰撞体和高速穿透边界用例；
- [x] 迁移金币、score、reward 和动态实体快照；
- [ ] 迁移敌人、砖块、箱子和关卡事件。

验收标准：相同 seed 与 action 序列产生完全相同的 snapshots；10,000 个 headless ticks 不需要初始化 Pygame。

## Phase 1.5：可玩 Pygame 竖切

状态：**代码竖切完成，等待人工游玩验收**

共享 core 已接管移动、碰撞、金币、死亡、通关和 reset。当前未迁移内容集中在砖块、箱子、敌人、power-up、checkpoint/传送和音频事件。

下一迭代只完成以下竖切：

- [x] 实现 `KeyboardController -> Action`；
- [x] 让 Pygame 主循环调用 `BasicPlatformerCore.step(action)`；
- [x] renderer 只读取 `WorldSnapshot`，不再自行维护玩家物理状态；
- [x] 渲染玩家、静态地形、相机和旗杆终点；
- [x] 实现死亡、成功以及全新 episode 重置；
- [x] 保证连续开始两局时状态完全重置；
- [x] 增加按住、跳跃、死亡、成功和重开集成测试；
- [ ] 完成一次至少 5 分钟的人工游玩 smoke test。

验收标准：从菜单进入关卡后可以移动、跳跃、死亡、重开并到达终点；第二局与第一局行为一致；Pygame 和 headless 使用同一套物理状态。

## Phase 2：Pygame Adapter 与 Gymnasium Environment

目标：同一个 core 同时支持人类游玩和 agent 训练。

- [x] 新建 Pygame renderer 和 keyboard controller；
- [x] 实现 `PlatformerState-v0`；
- [x] 定义 14 维状态 observation 和 reward v1；
- [x] 添加 action repeat、time limit 和 episode 统计；
- [x] 通过 Gymnasium environment checker；
- [x] 实现 random、move-right、rule-jump 三个 scripted baselines；
- [x] 记录固定 seed 的 episode 结果和关键指标；
- [ ] 增加逐 tick replay 文件与回放工具；
- [x] 通过 SB3 environment checker 和 1,000 次随机 episode 门禁；
- [x] 通过 reward exploit audit：进度守恒、金币一次性、原地负收益和终局符号。

验收标准：键盘与 scripted agent 走相同 action 接口；environment checker 通过；benchmark 可批量运行。

## Phase 3：PPO 基线与 Benchmark v1

目标：形成第一个可以量化比较的学习闭环。

- [x] 固定训练、验证和未见关卡 seed；
- [x] 训练首个 100k-step MLP PPO 状态基线；
- [x] 实现平地 → 单障碍 → 单缺口 → 组合 → 完整关卡课程训练设施；
- [ ] 完成 curriculum 的多训练 seed 性能验收与完整关卡迁移；
- [x] 保存配置、随机种子、checkpoint、Monitor 日志和评估结果；
- [x] 首版指标包含成功率、进度、episode return、步数和金币；
- [ ] 建立回归阈值，防止引擎改动悄悄破坏策略。

验收标准：PPO 在未见基础关卡上显著超过 scripted move-right baseline，训练过程可复现。

## Phase 4：PCG、LLM 与可玩性验证

目标：让生成内容进入受约束、可验证的数据管线。

- [ ] 定义版本化 LevelSpec JSON Schema；
- [ ] 实现基于 segment/grammar 的生成器；
- [ ] 静态验证几何、出生点和终点；
- [ ] 用图搜索或运动可达性检查过滤明显死图；
- [ ] 用 scripted/PPO rollout 做动态验证；
- [ ] LLM 只生成符合 schema 的主题与关卡设计意图；
- [ ] 保存 prompt、model、seed、spec 和验证报告。

验收标准：生成关卡全部通过 schema；不可玩候选不会进入游戏；同 seed 可重建同一关卡。

## Phase 5：DDA 与 Jev 决策层

目标：验证低频高级决策与自适应难度，而不是把模型塞进逐帧循环。

- [ ] 定义玩家技能画像和事件特征；
- [ ] 建立可解释的 DDA 参数边界；
- [ ] 建立固定、规则 DDA、学习型 DDA 三组实验；
- [ ] Jev 从合法高级策略候选中进行选择；
- [ ] 记录选择、置信度、输入状态和后续结果；
- [ ] 设计消融实验，比较 Jev、规则和 PPO hierarchy。

验收标准：所有难度调整可审计；固定 seed 下能回放决策；网络失败时安全降级到本地策略。

## Phase 6：ONNX、轻量推理与原创资产

目标：完成训练到部署链路，并建立可替换的原创主题系统。

- [ ] 固化 observation preprocessing；
- [ ] 导出 PPO policy 到 ONNX；
- [ ] 对比 PyTorch 与 ONNX 输出及 episode 行为；
- [ ] 明确目标 TinyInfer 项目及支持的 ONNX operators；
- [ ] 必要时简化 policy network 或编写运行时 adapter；
- [ ] 建立角色、tileset、动画、音频和 UI theme manifest；
- [ ] 用原创/AI 辅助资产替换现有第三方 IP 素材。

验收标准：端侧 policy 与 PyTorch policy 在固定测试集上一致；发布构建不包含马里奥版权素材。

## 下一迭代建议

2026-10-02 更新：本机环境、统一评估、课程内容、checkpoint/续训与晋级状态机已实现，并新增暴露 jump-held 的 `PlatformerState-v1`。下一迭代先稳定通过单障碍，再开展 gap/mixed/full 迁移和正式多 seed 验收；具体门槛见 [开发计划](RL_TRAINING_PLAN.md)，当前实现与实验记录见 [实现与使用](RL_TRAINING_V1.md)。

Gymnasium 与 PPO 已接入；人工游玩验收仍需补齐。Jev 与 LLM 按后续阶段推进，不继续扩展 legacy `Player` 的独立物理逻辑。

## 暂不纳入首轮的工作

- 像素 observation 与 CNN policy；
- 多智能体与 PettingZoo；
- 实时 LLM 生成碰撞几何；
- 大规模素材生成；
- 多平台发布和联机功能。

这些能力不是被放弃，而是要等核心接口、可复现性和 benchmark 稳定后再进入。
