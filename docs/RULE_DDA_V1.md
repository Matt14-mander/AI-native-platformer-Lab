# 规则 DDA 建议 v1

`ai_platformer.adaptive.rules.suggest` 将已经绑定的真人画像与 DDAContext 转为可审阅的 DDAProposal。规则只在下一关生成前给出建议，不改变当前游戏、存档、玩家物理或地图。输入保持只读；相同输入生成相同 decision ID 和建议。

## 冻结规则

策略版本为 `rules-v1`。阈值是保守的工程启发式，尚未经真人实验验证，不代表统计置信度或技能排名。

1. 沿用画像 v1 门槛：至少 5 次有效自然结束尝试、2 张内容不同的地图；不足返回 `hold/insufficient_data`。
2. 没有调整权限返回 `hold/constraints_locked`。每个地图与协议分组需至少 2 次尝试，并有通关率、平均最大进度及相应观测维度；不足返回 `hold/missing_dimension_evidence`。
3. 所有分组通关率均 ≤ 40% 时建议 `ease`；所有分组通关率均 ≥ 80% 且平均最大进度均 ≥ 90% 时建议 `challenge`。其他情况保持。分组间不混合计算通关率。
4. 每次只调整一个参数，固定先检查 `obstacle_height`，再检查 `gap_width`。ease 降低范围上限，challenge 提高范围下限。步长取 8 像素、授权步长与范围剩余宽度的最小值；不扩大原请求范围。所有获准范围均无法收窄时保持。
5. 调整建议引用整个画像窗口的 attempt/session 证据，绑定画像、环境、上下文和请求哈希，并通过已有 `validate_proposal` 校验。陈旧画像、计数矛盾或非法测量拒绝输出。

沿用协议原因码 `performance_decline` 表示触发降低难度阈值，并不声称存在时间上的表现下降。全局表现也不能定位具体跳跃错误；建议调整几何只是下一轮受控尝试，不是对失败原因的诊断。

## 使用

已有 `summarize_play_sessions` 报告可直接输入；通过 context 中的 profile ID 精确选择一份画像。context 必须符合现有 DDAContext v1：绑定 `profile_reference(profile)`、目标环境、基础 GenerationRequest 及其 request_hash，并显式声明可调整参数。不声明权限只会得到 hold。

```powershell
python -m scripts.suggest_dda --context runs/dda/context.json --profile runs/profiles/report.json --output runs/dda/proposal.json
python -m scripts.validate_dda_proposal --context runs/dda/context.json --profile runs/profiles/report.json --proposal runs/dda/proposal.json --output runs/dda/validation.json
```

输出要求使用新文件，不能覆盖输入或旧建议。也可从 Python 调用：

```python
from ai_platformer.adaptive.rules import suggest

proposal = suggest(context, profile)
print(proposal.action, proposal.reason.summary)
```

离线 CLI 的协议校验只说明建议符合证据绑定和参数权限，不说明地图可玩。CLI 输出保持 `application=not_applied`、`validation_status=pending`，不执行地图生成或 PPO 评估。主游戏提供下面的独立生成与选择流程。

## 主游戏：查看、验证与选择

死亡、超时或通关结算后按 **D** 查看建议。每个结算页面只保留一份建议；有效证据不足时解释原因并保持原关卡。验证成功后按 **A** 明确选择新关卡，也可按 **R/Enter** 重试原关、**M/Esc** 返回菜单；通关后的 **N** 仍为原目录下一关。查看建议不增加尝试数。

- 首次使用固定练习课模板（1 个障碍、1 个缺口、地面橡果，目标为到达终点，不强制收集比例）。这是新练习关，不承诺复刻当前任意地图；后续建议沿用已选练习课的设计范围。模板只允许障碍高度/缺口宽度单次 8px 的范围收窄。
- `generate_player_level` 复用 segments-v1 的几何与 RNG，使用当前玩家完整 PhysicsConfig 进行静态校验。原 GenerationRequest 的历史物理标签仍属于设计请求哈希；实际生成报告明确标注 `human-custom-v1` 和完整物理参数，不改变原 AI 生成器的输出契约。
- 再用玩家物理、action_repeat=1 搜索到达终点的路线，并独立重放检查。搜索预算 2400 步、6000 扩展、2 秒；证据构造和重放另有已有的各 30 秒上限。检查期间页面显示等待提示；当前实现同步执行。预算耗尽表示未找到证据，不证明不可玩。
- 只有通过静态验证与路线重放的不可变地图快照可以接受。接受前检查候选文件未被修改、原关卡仍是建议生成时的关卡，并先保存选择目录；任何失败均保留原关卡。新关卡进入、重试与退出复用现有记录/存档生命周期。
- 已选候选加入菜单，重启从目录恢复，并按现有准入流程重新验证；上次关卡定位沿用存档。未选候选不会加入菜单。

保存目录中的 `candidates/<id>/` 包含地图、设计请求及 report.json（画像、上下文、建议、实际玩家物理、生成/路线/重放报告）。建议文档保持不变；`candidate_catalog.json` 是玩家已确认选择的权威记录，报告中的 `application=not_applied` 表示生成时尚未选择，不冒充自动应用。

画像按 `session_order.json` 的明确创建顺序读取最近最多 128 个会话，窗口沿用最近 10 次有效尝试。查看建议时先封存已结束的当前会话，下次游玩开启新会话；headless、未正常结束、不完整或不同玩家/物理的数据不进入相应画像。历史升级前未登记顺序的会话仍保留并可手动分析，但不会按文件时间猜测顺序加入自动建议。索引损坏或存档不可写时停用建议，原游戏继续；`--no-save` 下不生成候选。

这些路线只证明相同物理下存在有效输入序列，不证明关卡对真人一定更容易或更难；阈值和体验效果仍需真人试玩验证。

## 验证覆盖

方向与阈值边界、地图间意见冲突、样本门槛、缺失维度、权限步长、范围耗尽、陈旧快照、矛盾统计、输入不变性、重复输出确定性、命令行与现有协议兼容及拒绝覆盖。人工游玩效果和候选地图可玩性不属于这些测试的结论。
