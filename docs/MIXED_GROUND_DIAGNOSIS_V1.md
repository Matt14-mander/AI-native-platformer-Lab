# mixed_ground 收集诊断 v1

本轮在没有 v9 checkpoint 的 Windows 工作区重建探索批次的三张 `mixed_ground` 地图。使用与原评测相同的 `segments-v1`、AI v2 物理、action repeat 4 和 policy seed 100；生成报告、每步轨迹和独立重放 witness 在本地 `runs/mixed_ground_diagnosis_v2/`，该目录不跟踪进 Git。

| 地图生成 seed | `rule_jump` | `coin_jump` | 短跳探针 | 搜索器 |
| --- | ---: | ---: | ---: | --- |
| 20261024 | 14/16 | 14/16 | 16/16，重放通过 | unknown，25,000 次展开 |
| 20261025 | 14/16 | 14/16 | 16/16，重放通过 | unknown，25,000 次展开 |
| 20261026 | 14/16 | 14/16 | 16/16，重放通过 | unknown，25,000 次展开 |

三个地图中长按跳跃策略漏掉的松果均为障碍物后的第一个地面松果（`coin-*-2`）。例如 seed 20261024 的首个障碍位于 x=621..685，松果在 x=905、y=508；`rule_jump` 穿过松果水平范围时玩家顶端 y 约 430，整个人仍在松果上方。短跳探针只改变起跳后的按住时长为 3 个决策步，三张图均通关并收集 16/16。三条完整动作序列经 `collect_route` 构造 witness，再由独立环境 `replay_route` 重放校验。因此这些**具体地图**存在满足 100% 收集目标的路线；搜索器原先的 `unknown` 是启发式搜索未找到 witness，不是地图不可达。

历史 v9 PPO 报告同样记录三张图各 14/16，但当前工作区没有对应权重和逐步轨迹，不能断言 PPO 漏掉的实体 ID 与规则策略完全相同，也不能在这里复跑 v9。下一步可将短跳作为独立 scripted 收集基线，在新开发集测试；不要因此直接修改 v9 的奖励、物理或生成器。

重建命令：

```powershell
python -m scripts.diagnose_mixed_ground --output-dir runs/mixed_ground_diagnosis_v2
```

`--output-dir` 必须是尚不存在的目录，脚本不会覆盖旧证据。
