# 统一生成评测报告 v1

统一读取已有批次和单图证据，输出机器可读JSON及中文Markdown。它不调用LLM、不修改原始批次、地图或checkpoint，不改变批量执行器，因此不影响已有批次的断点恢复身份。

## 本次已汇总

- `small_v1_offline_final`：6类提示×3生成seed，18个不同几何；PPO通关18/18，收集目标15/18。3个mixed_ground样本均收集14/16，未达到100%目标；搜索为unknown。
- `live_006`：真实Gemini单图，报告生成时又运行一次绑定v9 checkpoint的CPU PPO推理，211步、841tick、17/20松果，满足75%目标。
- `small_v1_live`：真实LLM批次18/18有效请求、18/18匹配冻结需求断言；PPO通关18/18、收集达标15/18。18张均未命中缓存，无记录到的连接/HTTP错误。平均翻译7.586秒，P95为13.576秒，报告8577 tokens。
- 真实批次与离线对照18对地图几何相同，PPO所核对摘要一致；这些配对不计为36张独立新地图。三个来源共37条地图记录，对应19个不同几何。失败目录含两份批次各6个重叠事件，共12条事件，不代表12张不同失败地图。

查看 [已生成的统一报告](reports/generated_evaluation_unified_v1.md) 与 [JSON快照](reports/generated_evaluation_unified_v1.json)。原始完整输出保存在 `runs/llm_eval/unified_v1_live/`，快照使用项目相对路径，记录与原始输出对应的来源和产物哈希。

## 生成报告

不需要API Key或VPN；只读取指定输入。使用 `--single-checkpoint` 时会额外进行本地PPO推理，否则单图PPO显示未评估，不从无checkpoint绑定的终端日志推断模型身份。

```bash
export MPLCONFIGDIR="$PWD/.venv/matplotlib"
export XDG_CACHE_HOME="$PWD/.venv/cache"

.venv/bin/python -m scripts.report_generated_levels \
  --batch-dir runs/llm_eval/small_v1_offline_final \
  --single-package runs/llm/live_006 \
  --single-checkpoint runs/full_collection_v9/seed_20261004/model.zip \
  --output-dir runs/llm_eval/unified_next
```

批量结果本身已记录checkpoint哈希，不需要再次推理。只汇总两个已完成批次时：

```bash
.venv/bin/python -m scripts.report_generated_levels \
  --batch-dir runs/llm_eval/small_v1_offline_final runs/llm_eval/small_v1_live \
  --output-dir runs/llm_eval/unified_after_live
```

输入目录必须存在；不静默忽略缺失来源。新输出目录原子发布，已有目录拒绝覆盖。最多32个不同输入目录；重复输入拒绝。输出不能位于原始来源目录内。读取旧批次不要求使用当前评测代码重新执行，只核对来源的运行身份、suite快照、逐样本结果、文件哈希及请求/几何/语义/PPO摘要一致性。

## 报告结构

- 每个来源单独展示：模式、模型、TLS、checkpoint、policy seed、来源文件SHA256、已处理/计划数量。
- 每来源总计、按提示和按生成seed分组，以及逐地图明细。计划分母包含失败和未执行候选；PPO另展示已实际评估的分母。
- 严格请求合法率与显式需求匹配率分开。离线fixture的LLM语义评分为null；历史单图没有冻结的自动断言，因此也不反推其语义正确率。
- 路线verified、PPO通关、PPO收集目标分别计数；unknown仅表示未找到满足目标的路线，不能作为不可通关证明。
- 收集率给出有松果地图的均值及分位数，并另给总松果加权比例。零松果地图不进入收集率均值，不记成100%收集。
- 失败目录保留语义差异、翻译错误、搜索unknown、PPO失败或漏收集，以及可用的最终玩家位置。同一地图的失败事件可能重叠，不把6个事件误报成6张失败地图。
- 记录重复几何组，包括重复运行间的重合；来源各自保留，不合并成模型排名或独立样本数。

## API与性能口径

原始翻译记录可能对同一次尝试同时保存模型输出和验证错误，按 `attempt` 去重后统计。缓存命中保留请求与语义证据，但不重复计入本次API请求、token或云端耗时。fixture不产生API指标。

`observed_json_response_rate_lower_bound` 是**已记录尝试中观测到JSON输出的下界**，不是精确HTTP成功率或Google服务可用性。旧日志对某些失败只保存通用错误，因此无法完整重建HTTP响应状态；报告保留unknown数量，不补造遥测。

耗时是批次中保存的翻译耗时，包含重试/校验/缓存IO，不含PCG、搜索和PPO；分别给全部未缓存翻译和其中有效翻译的样本数、均值、P50、P95。失败请求也进入全部翻译分布。无历史耗时字段显示“未记录”，不借用网络诊断耗时。

Token只累加provider实际报告的字段及覆盖样本数。缺失字段为null，失败调用可能存在未报告消耗。`live_006`记录261个prompt token、124个candidate token、total 385；该单图没有保存API耗时。报告不估算实际费用或把token简单换算成账单，也不包含未选入的历史连接调试请求。

本报告是探索性静态生成族证据，不执行训练、不分配train/test/OOD split、不替代正式多训练seed验收。


本次真实批次的结构化验收记录见 [llm_generation_live_batch_v1.json](reports/llm_generation_live_batch_v1.json)。旧单图385 tokens仍单独展示，不与真实批次8577 tokens混淆。
