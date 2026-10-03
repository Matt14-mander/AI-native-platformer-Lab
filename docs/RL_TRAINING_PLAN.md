# 强化学习训练开发计划

制定日期：2026-10-02。依据当前仓库代码、配置及 `PPO_BASELINE_V0.md`；本计划不代表新一轮训练已经完成。

## 目标与现状

下一阶段目标：通过可复现的课程训练，让状态 PPO 学会主动跳越障碍和缺口，在独立几何布局上稳定超过 move-right，然后迁移到现有完整关卡。

已有基础可以继续使用，不需要重建游戏核心或更换算法：

| 模块 | 当前实现 | 本轮要补齐 |
| --- | --- | --- |
| 游戏核心 | 确定性 reset/step、碰撞、金币、死亡、成功 | 新课程几何的边界验证 |
| 环境 | `PlatformerState-v0`，14 维 observation、10 个动作、reward breakdown | 可注入关卡仓库，显式关卡分组 |
| 门禁 | checker、随机稳定性、reward exploit audit 的代码已具备；文档记录已通过 | 对课程关卡重新运行，保存本机结果 |
| PPO | MLP、4 个 DummyVecEnv、Monitor、最终保存与确定性评估 | 定期评估、best/周期 checkpoint、续训、课程切换 |
| 基线 | random、move-right、rule-jump | 与 PPO 共用关卡集合及完整环境配置 |
| 实验记录 | 配置、依赖版本、动作 ID、训练步数、最终评估 | 关卡/协议哈希、Git revision、阶段历史、多训练 seed 汇总 |

历史基线：100,352 transitions 后，validation 的成功率为 0%，进度为 12.76%；move-right 也是 12.76%，rule-jump 成功率为 100%。PPO 确定性策略选择 RIGHT_RUN，停在首个管道。这支持优先降低探索难度，但尚不能证明失败仅由探索造成。

当前源码还暴露出几个必须处理的问题：

- 环境与评估均直接使用 `LegacyLevelRepository`；`game_content/levels/level_1.json` 目前只是 collectible overlay，不能直接承载新课程的完整几何。
- `train_ppo()` 的环境工厂及 `evaluate_policy()` 未传入 `level_id`，仅在配置中添加关卡名不会改变训练地图。
- seed 当前不改变地图几何。train/validation/unseen seed 分离只提供随机性协议，不提供未见关卡验证；训练工厂实际使用 `config.seed + rank`，也未消费 benchmark 的 train seed 列表。
- PPO 评估通过环境内的 1024 步时限截断；scripted benchmark 默认使用 core 时限，外部 `max_steps` 也不会改变 observation 中的 remaining time。公平比较必须统一环境构造，而非只把循环次数设为相同。
- 没有定期评估与续训，只有训练结束后的 `model.zip`；现有 checkpoint metadata 没有显式 observation/reward 协议版本及关卡内容哈希。
- 现有训练依赖快照注明 Windows CPython 3.12，不能直接视为 macOS 已验证环境。

## 实施顺序

### P0：恢复本机实验基线，并统一评估协议

交付：可执行的训练环境、统一 evaluation 配置、当前完整关卡的 scripted 报告。

1. 使用 Python 3.11/3.12 创建项目 `.venv`，安装 training 依赖，记录实际解析版本；核查历史依赖快照是否能在本机安装，不盲目沿用。
2. 执行全量测试、Gymnasium/SB3 checker 和现有 1,000 episode readiness gate。依赖缺失导致的 skip 必须显式报告，不能计为通过。
3. 抽取统一环境工厂/评估协议：level、physics、reward、sensor range、action repeat、episode step limit 对 PPO 与 scripted 一致。
4. 在统一协议下重跑三种 scripted baseline；若历史模型可取得，再重评历史 PPO。当前工作区没有发现 `runs/` 中的原始训练产物，历史数据暂以文档记录为准。
5. 配置校验覆盖 seed 集合重叠、空评估集、非法步数、rollout 与 batch 参数、输出目录覆盖风险。

验收：全部应执行测试通过；门禁结果可追溯；PPO/scripted 使用相同 observation 与截断逻辑。输出依赖清单、环境配置和 baseline JSON。

### P1：课程关卡与关卡级数据划分

交付：平地、单障碍、单缺口、障碍与缺口组合四组小型关卡，以及独立 train/validation/test manifest。

建议新增 `ai_platformer/content/curriculum.py`：用确定性工厂直接生成现有 `LevelDefinition`，复用 `BasicPlatformerCore`。给 `PlatformerStateEnv` 增加可注入 repository/loader；默认仍能加载现有完整关卡。本轮只做训练所需的最小关卡接口，不扩展成完整 PCG 系统。

| 阶段 | 训练任务 | 几何变化与校准 |
| --- | --- | --- |
| flat | 短平地移动到终点 | 长度、出生点与终点距离 |
| obstacle | 单个管道等价矩形 | 位置、高度；先保证能用当前物理跳过 |
| gap | 单个缺口 | 位置、宽度及落地区长度 |
| mixed | 障碍与缺口组合 | 间距与顺序，验证跳跃动作释放后可以再次触发 |
| full | 迁移现有 `level_1` | 保留完整关卡作为专项评估 |

每个基础课程建议至少 8 个训练布局、4 个验证布局、4 个最终测试布局；以完整几何哈希检测重复，manifest 记录参数及生成器版本。验证/测试必须包含未用于训练的几何组合，并位于已检查的可达范围。最终 test 不用于晋级、调参或 checkpoint 选择。

验收：相同关卡 ID/seed/action 可重放；碰撞、出生、终点与缺口边界有效；flat 的 move-right 可通关；障碍/缺口的 move-right 不能通关，并有 rule-jump 或固定动作轨迹证明可通关。rule-jump 失败不直接等于地图不可玩。每组先做快速随机检查，正式实验前执行覆盖各课程的 1,000 episode 稳定性门禁与 reward 审计。

重点检查 observation 的障碍距离在接触边界、地面边缘距离在起跳/落地时是否正确。当前 gap sensor 没有缺口宽度/落地区信息；是否需要新特征由失败轨迹决定。保持 v0 契约；若增加维度或改变原有含义，注册 `PlatformerState-v1`，重建基线并禁止混用旧 checkpoint。

### P2：训练过程可观测、可恢复

交付：定期独立评估、模型选择、周期 checkpoint、续训命令与失败轨迹。

主要修改 `ai_platformer/agents/ppo/training.py` 和 `scripts/train_ppo.py`，新增评估模块/脚本与 v1 实验配置。

- 每约 10,000 transitions 在独立验证环境评估；按 vector env 数量换算 callback 频率，记录实际触发步数。
- 以成功率为首要排序指标，平均进度为次要指标，return 为辅助指标；保存每阶段 best 与周期模型。不要仅按平均 reward 选择策略。
- 同时报告成功率、死亡率、超时率、平均进度、金币、步数，以及成功 episode 的通关步数；按关卡组分别汇总。
- 保存训练 seed、环境 seed、关卡分组、环境/动作/observation/reward 版本、几何哈希、依赖、Git revision 和阶段切换原因。
- 增加 `--resume`；恢复兼容的 PPO 参数、优化器状态与累计步数，并恢复阶段 manifest/历史。明确首次续训只保证过程可追溯，不宣称与未中断训练逐步一致。
- 保存典型失败的 action/observation/reward components/终止位置轨迹，重点识别不跳、一直按跳、重复起跳失败、跳过头和原地停滞。

验收：短训练确实触发评估与保存；best 可重载且在固定评估集上结果一致；续训累计步数继续增长；不兼容 checkpoint 被明确拒绝；评估不改变训练环境状态或 seed 流。

### P3：按验证结果晋级的 curriculum

交付：flat → obstacle → gap → mixed → full 的课程训练闭环。

同一 PPO 策略在兼容 observation/action 下连续训练。先实现 episode 边界切换关卡与简单阶段状态机，不引入复杂自动课程算法。阶段切换时不重置模型累计步数，记录训练采样与预算。

- 建议起始晋级阈值：flat ≥95%，obstacle/gap/mixed ≥80%，连续两轮验证达到阈值，并报告每个布局的结果，避免平均分掩盖单一失败布局。
- 下一阶段混入约 20% 已掌握课程 episode，观察遗忘；该比例是初始实验参数。
- 初始预算上限：flat 50k、obstacle 150k、gap 150k、mixed 250k、full 400k transitions，合计约 1M。预算是试验上限，不是达标保证；SB3 完整 rollout 可能使实际步数略超配置。
- 达到阶段预算仍未晋级时保存失败报告，停止自动升级；先检查传感器、动作边沿、可达性和训练日志，再决定是否调参。
- 每次修改课程、reward 或环境协议，都重新校准 baseline；关卡长度变化会改变 progress delta 的尺度，跨关卡不能仅比较 return。

验收：单 seed 的 flat/obstacle/gap 达到建议阈值；阶段历史完整；转入 mixed/full 后仍能通过基础课程回归集。首次能够稳定跳越单障碍是主要里程碑，不以一次完整关卡通关替代稳定性验证。

### P4：正式对照、多 seed 与最终测试

交付：可复现实验报告、候选模型与明确的继续/停止结论。

固定算法/网络/奖励后比较：直接完整关卡训练、仅分阶段课程、课程加旧任务混合采样。先用一个训练 seed 排除无效方案，再让入围方案至少跑三个训练 seed；对比方案使用相同 transitions 预算和评估协议。训练 seed 的差异与几何布局的差异分别汇总。

最终测试仅在方案冻结后执行，评估各基础课程的保留布局；完整 `level_1` 若已参与训练，只报告完整关卡表现，不称作未见关卡泛化。

建议本轮达标标准：

- 至少三个训练 seed 的保留 obstacle/gap 布局成功率各达到 80%，并展示逐 seed 结果。
- obstacle/gap 上成功率较 move-right 至少提高 50 个百分点；同时保留 rule-jump 对照。
- 完整关卡平均进度超过统一协议下的 move-right；完整关卡成功率达到 50% 作为进阶目标，未达到时记录失败点并继续定位。
- 模型重载、基础课程回归、reward 审计均通过；报告训练耗时与 transitions/s，再决定扩大预算或更换向量环境实现。

以上为拟定验收阈值，P1 可达性校准后冻结；不能在看到最终 test 结果后调整通过标准。

## 失败时的调整顺序

1. 检查可达性、sensor 数值、jump 按下/释放边沿与 action repeat，排除环境及协议问题。
2. 缩短课程、增加有效起跳机会，检查训练时采样策略和确定性策略的差距。
3. 单因素比较 entropy、rollout 长度或 action repeat；每次变更记录协议，保持预算公平。
4. 有明确证据后再试 reward 调整，增加新的漏洞审计；不加入“靠近指定管道即奖励”等地图答案。
5. 若变宽缺口需要额外可见信息，再引入 observation v1；若缺少信息，单纯扩大网络不能解决。

## 预期文件与迭代拆分

| 迭代 | 主要文件/产物 | 完成判据 |
| --- | --- | --- |
| 1：实验基线 | 本地 `.venv`、统一评估工厂、baseline 报告 | 本机门禁与公平评估可复现 |
| 2：课程内容 | `content/curriculum.py`、env 注入接口、课程 manifest、几何测试 | 三个基础任务可达且有对照 |
| 3：训练设施 | PPO callback、评估脚本、resume、metadata | 短训练保存/重载/续训闭环 |
| 4：学习实验 | 课程状态机、v1 配置、阶段报告 | 单 seed 学会单障碍和缺口 |
| 5：正式验收 | 多 seed 对照、保留关卡结果、回归报告 | 达到冻结阈值或给出明确失败定位 |

先完成迭代 1–3，再开展长训练。开发迭代可分别提交，长训练耗时在本机测得吞吐后估算。本轮不需要接入敌人、像素 CNN、LLM、DDA 或 Jev；ONNX 在选出稳定候选策略后开展。

## 本次检查的验证范围

系统默认 `python3` 是 Python 3.7，低于项目要求的 Python ≥3.10，首次测试出现 `dataclass(slots=True)` 导入错误；这是运行环境不匹配，不能据此判定 core 实现损坏。改用 Codex 随附 Python 执行 `unittest discover` 后，28 项测试中 14 项通过、14 项因缺少 Gymnasium/Pygame 依赖跳过，没有失败。训练依赖与长训练门禁尚需在项目专用环境中复验。本次只制定计划，未执行新 PPO 训练。


## 2026-10-02 第二轮进展

策略诊断、curriculum v2、`--init-from` 和通关速度门禁已实现，见 [训练 v2](RL_TRAINING_V2.md)。49 项测试、1,000 episode 稳定性门禁及课程奖励审计通过。相同新增 65,536 transitions 的单 seed 对照中，冷启动缺口验证 9/36，初始化最终模型 36/36、OOD 12/12；障碍回归为 3/4，未晋级 mixed。当前优先恢复障碍能力与缺口保持，并改善多任务候选选择，再做正式验收；最终测试未使用。


## 2026-10-03 mixed 前置交付

障碍覆盖扩展、较长缺口桥接、任务加权回放、联合候选选择、晋级即保存停止和独立重载验收已完成。训练关卡规则示范预热后，PPO 16,384 transitions 连续两轮通过 flat 4/4、obstacle 40/40、gap 54/54；OOD 复查 12/12。最终 checkpoint 已处于 mixed 就绪状态，见 [前置验收与交接](MIXED_PREREQUISITES.md)。57 项测试通过；正式多 seed 验收留待后续。
