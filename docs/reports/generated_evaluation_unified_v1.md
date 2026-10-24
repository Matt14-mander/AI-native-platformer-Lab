# 生成关卡统一批量报告

离线覆盖、真实API批次和真实单图分开展示，不合并为总成功率。
真实API批次：0；真实单图：1；离线批次：1。

| 类型 | 模型 | 已处理/计划 | 请求合法 | LLM需求匹配 | 路线verified | PPO通关/已评估 | PPO收集达标/已评估 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 离线fixture批次 | offline-expected-fixture | 18/18 | 18/18 (100.0%) | 未评估 | 15/18 (83.3%) | 18/18 (100.0%) | 15/18 (83.3%) |
| 真实API单图 | gemini-3.5-flash-lite | 1/1 | 1/1 (100.0%) | 未评估 | 1/1 (100.0%) | 1/1 (100.0%) | 1/1 (100.0%) |

## runs/llm_eval/small_v1_offline_final

模式：expected-fixture；TLS：tls12；策略seed：100。

Checkpoint SHA256：`f6f5eabb26446fb7ee999c99e2cd76acfb98bb88c0abd4660e26d71658e42e3c`

收集率均值（有松果地图）：88.39%；样本数：15；按松果加权：200/228 (87.7%)。

已记录未缓存API尝试：0；观测到JSON输出：0；响应结果不明尝试：0；缓存命中：0；JSON输出观测率下界：未评估。

翻译耗时（秒，含校验/重试；不含路线和PPO）：均值 未记录，P50 未记录，P95 未记录，样本数 0。

报告的total tokens：未记录；费用：未估算，失败调用消耗可能未报告。

| 提示 | 已处理/计划 | 需求匹配/已断言 | 路线verified | PPO通关 | PPO收集达标 |
| --- | --- | --- | --- | --- | --- |
| flat_collect | 3/3 | 3/3 (100.0%) | 3/3 (100.0%) | 3/3 (100.0%) | 3/3 (100.0%) |
| low_obstacles | 3/3 | 3/3 (100.0%) | 3/3 (100.0%) | 3/3 (100.0%) | 3/3 (100.0%) |
| gaps_no_coins | 3/3 | 3/3 (100.0%) | 3/3 (100.0%) | 3/3 (100.0%) | 3/3 (100.0%) |
| mixed_balanced | 3/3 | 3/3 (100.0%) | 3/3 (100.0%) | 3/3 (100.0%) | 3/3 (100.0%) |
| mixed_ground | 3/3 | 3/3 (100.0%) | 0/3 (0.0%) | 3/3 (100.0%) | 0/3 (0.0%) |
| tall_wide | 3/3 | 3/3 (100.0%) | 3/3 (100.0%) | 3/3 (100.0%) | 3/3 (100.0%) |

| 生成seed | 已处理/计划 | 路线verified | PPO通关 | PPO收集达标 | 收集率均值 |
| --- | --- | --- | --- | --- | --- |
| 20261024 | 6/6 | 5/6 (83.3%) | 6/6 (100.0%) | 5/6 (83.3%) | 87.83% |
| 20261025 | 6/6 | 5/6 (83.3%) | 6/6 (100.0%) | 5/6 (83.3%) | 89.50% |
| 20261026 | 6/6 | 5/6 (83.3%) | 6/6 (100.0%) | 5/6 (83.3%) | 87.83% |

| 地图 | 生成seed | 需求匹配 | 路线 | PPO结果 | 松果 | 目标达标 |
| --- | --- | --- | --- | --- | --- | --- |
| eval_flat_collect_s20261024 | 20261024 | True | verified | success | 4/4 | True |
| eval_flat_collect_s20261025 | 20261025 | True | verified | success | 4/4 | True |
| eval_flat_collect_s20261026 | 20261026 | True | verified | success | 4/4 | True |
| eval_low_obstacles_s20261024 | 20261024 | True | verified | success | 9/12 | True |
| eval_low_obstacles_s20261025 | 20261025 | True | verified | success | 9/12 | True |
| eval_low_obstacles_s20261026 | 20261026 | True | verified | success | 9/12 | True |
| eval_gaps_no_coins_s20261024 | 20261024 | True | verified | success | 0/0 | True |
| eval_gaps_no_coins_s20261025 | 20261025 | True | verified | success | 0/0 | True |
| eval_gaps_no_coins_s20261026 | 20261026 | True | verified | success | 0/0 | True |
| eval_mixed_balanced_s20261024 | 20261024 | True | verified | success | 17/20 | True |
| eval_mixed_balanced_s20261025 | 20261025 | True | verified | success | 17/20 | True |
| eval_mixed_balanced_s20261026 | 20261026 | True | verified | success | 17/20 | True |
| eval_mixed_ground_s20261024 | 20261024 | True | unknown | success | 14/16 | False |
| eval_mixed_ground_s20261025 | 20261025 | True | unknown | success | 14/16 | False |
| eval_mixed_ground_s20261026 | 20261026 | True | unknown | success | 14/16 | False |
| eval_tall_wide_s20261024 | 20261024 | True | verified | success | 22/24 | True |
| eval_tall_wide_s20261025 | 20261025 | True | verified | success | 24/24 | True |
| eval_tall_wide_s20261026 | 20261026 | True | verified | success | 22/24 | True |

## runs/llm/live_006

模式：gemini；TLS：tls12；策略seed：100。

Checkpoint SHA256：`f6f5eabb26446fb7ee999c99e2cd76acfb98bb88c0abd4660e26d71658e42e3c`

收集率均值（有松果地图）：85.00%；样本数：1；按松果加权：17/20 (85.0%)。

已记录未缓存API尝试：1；观测到JSON输出：1；响应结果不明尝试：0；缓存命中：0；JSON输出观测率下界：1/1 (100.0%)。

翻译耗时（秒，含校验/重试；不含路线和PPO）：均值 未记录，P50 未记录，P95 未记录，样本数 0。

报告的total tokens：385；费用：未估算，失败调用消耗可能未报告。


| 地图 | 生成seed | 需求匹配 | 路线 | PPO结果 | 松果 | 目标达标 |
| --- | --- | --- | --- | --- | --- | --- |
| llm_test_001 | 20261010 | 未断言 | verified | success | 17/20 | True |

## 失败目录

| 地图 | 类型 | 原因/目标 |
| --- | --- | --- |
| eval_mixed_ground_s20261024 | route_unknown | budget_exhausted |
| eval_mixed_ground_s20261024 | ppo_collection_miss | 最低收集率 1.0 |
| eval_mixed_ground_s20261025 | route_unknown | budget_exhausted |
| eval_mixed_ground_s20261025 | ppo_collection_miss | 最低收集率 1.0 |
| eval_mixed_ground_s20261026 | route_unknown | budget_exhausted |
| eval_mixed_ground_s20261026 | ppo_collection_miss | 最低收集率 1.0 |

跨来源不同几何：19；重复组：0。重复运行不作为新增独立样本。

## 统计边界

- 离线、真实批次、单图和重复运行不合并计算总成功率。
- API JSON输出观测率仅为已记录尝试的下界，不代表HTTP精确成功率或服务可用性。
- 缺失的历史耗时、用量和语义断言保留为未知；只覆盖所选来源，不等于全部调试请求或账单。
- 失败类型可以重叠；路线unknown不代表不可通关。
- 单图PPO目标来自生成请求；没有冻结需求断言时，不推断其LLM语义正确率。
- 本报告限于探索性的静态生成族；未改变训练/保留集，也未重新训练模型。
