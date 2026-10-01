# 初步开发计划

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
- [ ] 建立 curriculum：平地 → 缺口 → 障碍 → 敌人；
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

PPO v0 已完成但只达到 move-right 水平。下一迭代先恢复本机训练环境、统一评估协议，再实现平地、单障碍和单缺口训练关卡，补齐定期评估、best-model checkpoint 与续训后开展 curriculum 和多 seed 正式实验。具体顺序及验收标准见 [强化学习训练开发计划](RL_TRAINING_PLAN.md)。

Gymnasium 与 PPO 已接入；人工游玩验收仍需补齐。Jev 与 LLM 按后续阶段推进，不继续扩展 legacy `Player` 的独立物理逻辑。

## 暂不纳入首轮的工作

- 像素 observation 与 CNN policy；
- 多智能体与 PettingZoo；
- 实时 LLM 生成碰撞几何；
- 大规模素材生成；
- 多平台发布和联机功能。

这些能力不是被放弃，而是要等核心接口、可复现性和 benchmark 稳定后再进入。
