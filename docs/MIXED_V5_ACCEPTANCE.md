# Mixed v5 训练与保留 test/OOD 验收

2026-10-04。扩展 mixed 的训练及本次保留集验收均已完成，全部通过。当前推荐模型是 `runs/ppo_gameplay_v2_mixed_acceptance/model.zip`，配套 `model.json` 必须保留。

## 训练与选模

从已接受的 v2 gap-recovery checkpoint 严格续训，保留优化器、物理、reward、动作/观测接口与课程签名。新增 16,384 PPO transition，计步从 16,384 延续到 32,768。之前的训练集规则示范记录保留在来源中，本轮没有重新运行示范；整个模型谱系不是纯 PPO。

课程为 v5：mixed train 共 100 个布局（旧 36 + 新 64），validation 共 60 个布局（旧 36 + 新 24）。Monitor 已核对：401 个完成 episode，146 个不同训练布局，其中 mixed 96 个；没有保留集布局。示范、旧权重初始化与本轮 PPO 的 provenance 均保留在 run/sidecar 中。

24,576 和 32,768 transition 的两轮联合验证均达到 flat 4/4、obstacle 40/40、gap 54/54、mixed 60/60。最终状态为 `ready_for_stage`，target=`full`；没有启动完整关卡训练。

选定的是自动晋级时保存的最终 checkpoint，选择发生在查看 test/OOD 结果之前。验收脚本重新加载验证课程门槛，然后写出 `frozen_plan.json`，固定模型/sidecar hash、布局清单、模式、种子和标准，再运行保留集。验收结束再次检查文件 hash 未变。没有根据保留结果重训、改参数或换模型。

## 验收标准与结果

两种模式都按布局独立验收；不以全局均值代替某个失败布局：flat 成功率 ≥95%、成功步数 ≤128；obstacle ≥80%、≤160；gap ≥80%、≤192；mixed ≥80%、≤256。OOD 使用相同标准，不在看到结果后放宽。实际所有布局成功率都是 100%。

种子为 1000/1001/1002/1003，与训练、验证 seed 不重叠。每个布局、每种模式运行 4 局。环境 seed 不改变静态几何；采样 seed 改变动作抽样。因此局数不等于独立几何数量，也不是独立训练次数。

| 保留任务 | 唯一布局 | 确定性成功局 | 采样成功局 |
| --- | ---: | ---: | ---: |
| test flat | 4 | 16/16 | 16/16 |
| test obstacle | 40 | 160/160 | 160/160 |
| test gap | 54 | 216/216 | 216/216 |
| test mixed | 56 | 224/224 | 224/224 |
| OOD gap | 12 | 48/48 | 48/48 |
| OOD mixed | 24 | 96/96 | 96/96 |

合计 190 个唯一保留布局、1,520 局，全部通关，无死亡或超时，所有成功步数符合门槛。没有 flat/obstacle OOD 池，因此没有虚构对应结果。原完整关卡不属于独立保留几何，本次不计入验收。

mixed 的保留覆盖进一步拆分：

| mixed 分组 | 唯一布局 | 每模式成功局 |
| --- | ---: | ---: |
| 旧 test | 36 | 144/144 |
| v5 新组合 test | 20 | 80/80 |
| 旧 OOD | 12 | 48/48 |
| v5 新 OOD | 12 | 48/48 |

新组合 test 包含完整留出的 5 个高度/宽度/恢复距离单元，每单元 4 个位置变化；新 OOD 包含高度 88、缺口宽度 180 的 3 个单元。这支持在本次规定分布上的组合泛化与有限外推结论；仍是一个训练谱系，不能据此声称跨训练随机种子的稳定性或任意复杂关卡能力。

## 文件与命令

- 模型 SHA256：`9c28f8b5a5e149346c160c5b70943cc294df35cddc418c7c57f2cacf08610f48`。
- 配置：`config/ppo_gameplay_v2_mixed_acceptance.json`。
- 训练报告：`runs/ppo_gameplay_v2_mixed_acceptance/run.json`。
- 冻结计划：`runs/mixed_v5_reserved_acceptance/frozen_plan.json`。
- 验收总报告：`runs/mixed_v5_reserved_acceptance/report.json`；该目录另有逐布局确定性结果、采样事件诊断和验证重载证据。
- 可版本控制摘要：`docs/reports/mixed_v5_reserved_acceptance.json`。

执行过的训练和验收命令如下，目录已存在，重跑必须选择新目录。未来重新使用这些保留布局时，应标记为回归测试；若依据这些结果调整训练，需另设新的盲测集。

```bash
.venv/bin/python -m scripts.train_ppo \
  --config config/ppo_gameplay_v2_mixed_acceptance.json \
  --resume runs/ppo_gameplay_v2_gap_recovery/model.zip \
  --stop-before-stage full --output-dir runs/ppo_gameplay_v2_mixed_acceptance

.venv/bin/python -m scripts.accept_mixed_checkpoint \
  --model runs/ppo_gameplay_v2_mixed_acceptance/model.zip \
  --output-dir runs/mixed_v5_reserved_acceptance

# 观看 mixed 策略，自动恢复 checkpoint 的环境参数
.venv/bin/python -m scripts.play_ppo \
  --model runs/ppo_gameplay_v2_mixed_acceptance/model.zip --task mixed
```

本轮新增可复用验收工具、隔离/逐布局判定测试，全部 83 项自动测试通过。物理参数保持已确认的短按低跳、长按高跳。mixed 仍是一个障碍加一个缺口；多障碍连续组合、松果收集课程、full 专项训练和新模型 TinyInfer 导出验证属于后续工作。
