# 小规模生成评测集 v1

将真实 LLM 候选生成、确定性 PCG、静态校验、路线搜索/独立重放、v9 PPO 推理纳入同一份报告。当前冻结为6类中文提示×3个生成seed，共18个候选；默认每个候选最多1次API请求。配置见 [small_v1.json](../game_content/generation_suites/small_v1.json)。

这是一组探索评测，不是训练集或正式test/OOD验收集。每张地图使用同一个固定PPO checkpoint和确定性policy seed=100；生成seed变化不等于多训练seed或独立随机策略试验。提示中的地面/短跳布局仅在既有segments-v1静态族内生效，不扩展敌人、交互物或视觉主题。

## 固定覆盖

| 提示ID | 障碍/坑 | 障碍高度 | 坑宽 | 松果布局 | 最低收集率 |
| --- | --- | --- | --- | --- | --- |
| flat_collect | 0 / 0 | 无 | 无 | ground | 100% |
| low_obstacles | 3 / 0 | 32–64 | 无 | low_jump | 75% |
| gaps_no_coins | 0 / 3 | 无 | 96–152 | none | 0% |
| mixed_balanced | 3 / 2 | 48–96 | 96–152 | low_jump | 75% |
| mixed_ground | 2 / 2 | 48–96 | 96–152 | ground | 100% |
| tall_wide | 3 / 3 | 96–128 | 152–176 | low_jump | 75% |

长度单位为像素。生成seed固定为20261024、20261025、20261026；LLM必须保留调用者指定ID和seed。每条提示的 `expected` 仅用于对照显式需求，不发给LLM，不用于裁剪或改写模型输出。模型合法但不符合提示时，保留差异并继续评估实际地图。

## 真实API批量评测

在已经配置 `GEMINI_API_KEY` 的终端运行。保持之前成功的TAG网络设置：

```bash
export MPLCONFIGDIR="$PWD/.venv/matplotlib"
export XDG_CACHE_HOME="$PWD/.venv/cache"

# 查看覆盖和请求预算；不加载checkpoint，不调用API。
.venv/bin/python -m scripts.evaluate_generated_levels --plan

# 新输出目录；每个候选按序保存，最多18次API尝试。
.venv/bin/python -m scripts.evaluate_generated_levels \
  --provider gemini --model gemini-3.5-flash-lite --tls-mode tls12 \
  --checkpoint runs/full_collection_v9/seed_20261004/model.zip \
  --output-dir runs/llm_eval/small_v1_live \
  --cache-dir runs/llm_eval/live_cache
```

默认单次API超时60秒，输出token上限4096，`--max-attempts 1`；可显式设为2或3，但最大请求预算也相应增加。`--limit 2`只跑排序最前的两个候选，适合连接冒烟；它不代表6类提示都被覆盖，输出路径应另选。API token用量仅汇总已报告的有效、未命中缓存的翻译尝试；失败调用可能有未报告消耗，不能视为账单。脚本不估算美元成本。

3次连续翻译失败会暂停批次；遇到400/401/402/403/404立即暂停，保留已有结果。缺少Key则在任何外部调用前退出，不静默切换到离线样本。

## 恢复与结果

网络中断或主动中断后，保持原命令的所有参数，仅增加 `--resume`：

```bash
.venv/bin/python -m scripts.evaluate_generated_levels \
  --provider gemini --model gemini-3.5-flash-lite --tls-mode tls12 \
  --checkpoint runs/full_collection_v9/seed_20261004/model.zip \
  --output-dir runs/llm_eval/small_v1_live \
  --cache-dir runs/llm_eval/live_cache --resume
```

已发布候选（包括失败记录）不重复调用，恢复只运行尚未发布的候选。失败样本不能通过恢复被悄悄替换；若需重测，使用新输出目录，并保留上一轮报告。suite、checkpoint及sidecar、评测代码、关键环境/内容代码、依赖版本、预算和缓存路径都参与恢复身份检查；变更后拒绝恢复。候选逐个原子发布，汇总JSON逐张更新，已有产物哈希与结果快照会被核对。同一输出目录有进程锁。

`run.json`保存运行身份，`suite.json`保存输入快照；`report.json`给出总计、逐提示及逐地图结果，`report.md`给出简表。每张地图在 `cases/<level_id>/` 下保存：

- `intent.json`、`llm.json`：提示、期望、模型/提示版本、结构化输出、缓存、调用次数和usage。
- `request.json`、`level.json`、`generation.json`：实际请求与PCG/静态校验结果。
- `route.json`：搜索状态和独立重放见证；unknown表示预算内未找到达标路线。
- `ppo.json`：真实SB3 policy的结果、动作序列和最终物理状态，便于定位漏收集/失败位置。
- `result.json`：分阶段摘要、耗时和产物SHA256。

路线与PPO复用checkpoint的相同物理、action_repeat和episode_step_limit（v9为4和1024），避免不相等的步数预算。搜索默认15秒/25000次扩展。所有静态有效地图都运行PPO，包括语义不符或路线unknown的地图；不会根据搜索结果挑选PPO样本。

汇总中的 `rates_over_planned` 以计划候选总数为分母，包括失败和未执行项；PPO另给已评估地图分母。通关与收集目标分开计数；零松果地图收集率为null，不进入平均收集率，但最低目标0仍要求成功通关。`all_stage_passes`同时要求需求匹配、路线verified和PPO达到期望目标。重复几何按忽略地图/实体ID的几何哈希分组，不计成唯一地图；目前只审计评测集内部重复，未宣称与所有旧训练/保留地图完全无重合。

退出0表示计划候选全部处理，不表示全部达标；退出1表示批次未完成，退出2表示参数、身份或IO错误。应查看报告而不是只看退出码。

## 已执行的离线覆盖基线

```bash
.venv/bin/python -m scripts.evaluate_generated_levels \
  --provider expected-fixture \
  --output-dir runs/llm_eval/small_v1_offline_final
```

`expected-fixture`直接从每条提示的期望设计构造显式本地响应，再走同一协议、PCG、路线和真实PPO流程。它是管线和几何覆盖基线，**不是LLM真实生成结果**；LLM语义评分和云端延迟为null，API尝试预算为0。初步18张覆盖中，静态校验18/18、PPO通关18/18、路线与PPO收集目标各15/18。三个 `mixed_ground` 地图均通关但只收集14/16（87.5%），未达到该类提示的100%目标；搜索返回budget_exhausted，不能据此断言不存在100%路线。保留样本，不为获得全通过而降低目标或替换seed。

详见 [离线验收记录](reports/generated_level_evaluation_v1.json)。初始 `live_006` 的真实Gemini调用与PPO成功仅作为此前单图闭环证据，不计入这18张新覆盖结果。真实批次现已完成，见下文；不把离线结果冒充真实生成。


## 首次真实18张批次（2026-10-24）

`runs/llm_eval/small_v1_live` 已完成，使用 `gemini-3.5-flash-lite`、TLS1.2和既有v9 checkpoint。18次未缓存调用均产生合法请求且匹配冻结需求断言，未记录HTTP或连接错误。静态有效18/18，PPO通关18/18，路线verified与PPO收集目标各15/18。三个mixed_ground均只收集14/16，保留为未达标样本。

翻译耗时平均7.586秒、P50为6.753秒、P95为13.576秒、最大20.220秒；报告prompt tokens 5199、candidate tokens 3378、total 8577。耗时含校验，不含PCG/路线/PPO；未估算账单。provider未单独报告thoughtsTokenCount，不能将缺失字段当作已验证的0。

与离线对照逐对核验，18张几何和PPO通关/步数/奖励/松果等摘要一致。说明当前三张目标缺口不来自观察到的需求翻译偏差，但不证明100%目标不可达。下一步可分析漏收集的位置、路线预算与PPO动作，不修改已冻结提示/seed来消除失败。

详见 [真实批次验收记录](reports/llm_generation_live_batch_v1.json) 和 [更新后的统一报告](reports/generated_evaluation_unified_v1.md)。本次只支持该小批次和静态族的结论，长期服务稳定性及更广泛的提示泛化尚待评估。
