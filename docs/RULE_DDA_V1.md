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

协议校验通过只说明建议符合证据绑定和参数权限，不说明地图可玩。当前 `GenerationRequest` 仍声明 `gameplay-ai-v2`；不能直接把生成器的验证结果视为 HumanPlatformer 的可玩性证据。下一步需基于目标玩家物理重新生成、静态检查、搜索并重放路线；成功后再接入结算页供玩家选择。当前输出保持 `application=not_applied`、`validation_status=pending`，没有执行地图生成或 PPO 评估。

## 验证覆盖

方向与阈值边界、地图间意见冲突、样本门槛、缺失维度、权限步长、范围耗尽、陈旧快照、矛盾统计、输入不变性、重复输出确定性、命令行与现有协议兼容及拒绝覆盖。人工游玩效果和候选地图可玩性不属于这些测试的结论。
