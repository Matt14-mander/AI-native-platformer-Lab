# DDA 建议协议 v1

日期：2026-10-26。本轮完成版本化上下文、建议模型、跨文档校验器和本地 CLI。尚未实现根据表现选择难度的规则策略，也没有生成或自动应用新关卡。无需真人样本即可测试协议边界，但测试不证明对真实玩家有效。

## 三个输入与一个校验结果

1. **DDAContext v1**：调用方对下一次关卡生成的授权，绑定完整基础 GenerationRequest、请求哈希、目标环境哈希、ProfileRef 和允许调整的参数。
2. **PlayerProfile v1 快照**：画像原文，用于重算哈希，并核查身份、版本、证据数量、地图分组与 ready 门槛的一致性；可直接从批量画像 report.json 中按 profile_id 选取。
3. **DDAProposal v1**：建议动作、policy_version、理由和证据引用、完整候选请求，以及 context/profile/base/candidate 的哈希。
4. **dda_contract_validation_v1**：校验结论、服务端计算的参数前后值与差值、各阶段状态；不会执行任何修改。

实现位于 `ai_platformer/adaptive/dda.py`；静态交换协议见 [上下文 JSON Schema](../schemas/dda_context_v1.schema.json) 和 [建议 JSON Schema](../schemas/dda_proposal_v1.schema.json)。Schema 为 draft 2020-12，模型严格拒绝未知字段和类型强制转换。哈希、跨文档关系、范围包含关系和方向检查必须执行 Python 校验器，不能仅依赖 JSON Schema。

## 授权与硬约束

上下文的 mode 固定为 `suggestion_only`，boundary 固定为 `before_next_level_generation`。没有当前地图修改、逐帧决策、物理变更或执行命令字段。

`permissions` 默认两项均为空，即锁定全部调整。显式授权的示例：

```json
{
  "obstacle_height": {"max_endpoint_delta_px": 16},
  "gap_width": {"max_endpoint_delta_px": 8}
}
```

v1 只允许这两个参数，每个授权的端点步长为严格整数 1..32px。没有对应障碍或缺口时不能授权该参数。允许调整的候选范围必须完全包含在原请求范围以内，且每个端点相对基础请求的绝对变化不能超出授权步长。

以下字段全部固定：schema/generator/physics 版本、level_id、seed、障碍数量、缺口数量、coin_layout、min_coin_ratio 和 theme。不能为提高验收成绩而降低松果目标、删障碍或重新选择 seed。

基础请求指调用方已确定的**下一关生成请求**，其中 level_id 和 seed 已选定，不是正在游玩的地图。上下文保留用户原始请求，不把候选请求覆盖回用户约束；v1 的步长相对这个固定基准计算，尚未实现多轮控制器、滞回或跨关冷却。

## 建议语义

| action | 约束 |
| --- | --- |
| hold | 候选请求与基础请求完全一致，没有参数变化 |
| ease | 至少一个真实变化；所有变化端点不增加，原请求包络内缩小高度/宽度范围 |
| challenge | 至少一个真实变化；所有变化端点不减少，原请求包络内提高范围下界 |

例如障碍高度基准 48..96，授权步长 16：ease 可建议 48..80，challenge 可建议 64..96。把范围同时缩成 64..80 会使两个端点方向相反，不符合 ease 或 challenge。几何方向是名义参数含义，不保证实际游玩难度一定单调，仍需后续关卡验证。

画像不足 5 次有效尝试或 2 个不同地图内容哈希时，只能 `hold`，reason.code 必须为 `insufficient_data`。画像 ready 但权限全锁定时，只能 `hold/constraints_locked`。ready 不表示偏好等所有维度都已知；后续规则可使用 `hold/missing_dimension_evidence`。

调整建议必须引用当前画像窗口内的至少一个 session_id/attempt_id；不接受窗口外或重复引用。允许的 reason.code 为 insufficient_data、constraints_locked、missing_dimension_evidence、performance_decline、performance_stable、no_change_needed 和 manual_review。摘要用于审阅，v1 协议不判断文字理由是否被某条表现规则支持；规则阈值与证据匹配属于下一阶段。

所有建议的 application 固定为 `not_applied`、validation_status 固定为 `pending`、fallback 固定为 `hold`。没有概率模型，因此不虚构 confidence 分数。

## 快照与环境绑定

ProfileRef 包含原画像完整 JSON 的 SHA256、profile_id、profile/metrics 版本、身份范围、环境哈希、有效尝试数、地图内容哈希与证据引用。校验器核查这些字段与画像原文一致；匿名画像不能混入其他 session 的证据。

target_environment_sha256 必须与画像环境一致；context_sha256 覆盖授权和基础请求，防止建议在上下文变更后被误用。GenerationRequest 哈希使用现有 request_hash，画像/上下文/建议哈希使用 adaptive.metrics.digest，均对补全默认值后的字典做确定性序列化。

哈希是完整性和来源绑定，不是身份认证，也不能单凭一个哈希证明真实运行环境、真人身份或画像统计正确。上游仍需提供由可信事件管线生成的画像；未来执行器必须核对实际环境配置与物理兼容性。当前协议校验不访问旧 source 路径、不加载模型、不调用 API、不运行物理。

## 运行样例

仓库包含明确标记为 synthetic 的低数据量 hold 样例，不代表真实玩家：

```bash
.venv/bin/python -m scripts.validate_dda_proposal \
  --context game_content/dda_fixtures/protocol_v1/context.json \
  --proposal game_content/dda_fixtures/protocol_v1/proposal.json \
  --profile game_content/dda_fixtures/protocol_v1/profile.json \
  --output runs/dda/protocol_v1_review/hold_validation.json
```

输出文件必须为新文件。成功返回码 0，拒绝建议返回码 1，参数或输出错误返回码 2。每个输入限制 4MB；大报告可先选出单个画像快照。JSON 解析拒绝重复键与非有限数值。命令不会覆盖输入，读写报告以外没有副作用。

真实接入时 `--profile` 可以指定 `summarize_play_sessions` 生成的 report.json，CLI 会依据 context.profile.profile_id 找到唯一匹配项；零个或多个匹配都会拒绝。不要用 synthetic fixture 的环境值替代真实画像环境。

API 组装示例：

```python
from ai_platformer.adaptive.dda import (
    DDAContext, context_hash, hold_proposal, profile_reference, validate_proposal,
)
from ai_platformer.content.generation_request import request_hash

context = DDAContext(
    schema_version=1,
    kind="dda_context_v1",
    context_id="next-level-001",
    profile=profile_reference(profile),  # 来自已有画像报告的单个完整对象
    target_environment_sha256=profile["environment_sha256"],
    base_request=request,               # 下一关的 GenerationRequest
    base_request_sha256=request_hash(request),
    permissions={},                    # 默认不授权任何参数变化
)
suggestion = hold_proposal(context, decision_id="decision-001")
receipt = validate_proposal(context, suggestion, profile)
```

hold_proposal 只是协议的确定性保守回退，不读取表现指标，不是已实现的 DDA 行为策略。

## 后续门禁

协议通过只表示建议结构、来源绑定和授权约束正确。校验结果明确列出 generation/static_validation/route_verification/ppo_evaluation 均为 `not_run`。下一阶段依次实现规则建议器、受约束 PCG 候选生成、静态与独立路线验证、PPO 评估和人工审阅，再讨论自动应用。

原 v9 checkpoint、混合关卡验收、游戏物理和玩家画像算法均保持不变。本轮自动测试与命令行烟测证据见 [验收摘要](reports/dda_protocol_v1.json)。

验证结果：全量 289 passed、8 subtests passed，新增 41 项 DDA 测试；ruff 和 diff 检查通过。合成 hold 样例通过 CLI；本地真实画像报告目前不含真人画像，用其校验样例上下文会明确拒绝匹配，未以合成数据替代真人证据。
