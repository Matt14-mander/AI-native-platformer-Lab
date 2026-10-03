# Mixed 前置开发与验收

本轮解决障碍恢复时的缺口遗忘，以及单任务 best 模型不能保证整体能力的问题。使用 `PlatformerState-v1`，通过课程内容、采样比例和候选选择规则改进训练。

## 已实现

- **障碍覆盖扩展**：curriculum v3 保留全部旧分组，新增 108 个障碍布局，覆盖出生点、接近距离和高度。障碍 train/validation/test 分别为 44/40/40 个布局；原验证布局没有加入训练。
- **较长缺口接近距离**：curriculum v4 再新增 54 个缺口布局，使 gap train/validation/test 各有 54 个。新训练距离为 680、740，验证为 687、747；原 12 个 OOD 布局保持独立，最小接近距离仍为 780。manifest 总计 434 个唯一布局。
- **显式任务采样**：每个阶段可设置 `sampling_weights`，仅在 episode 边界按比例选择当前或更早任务，再从对应 train 集均匀采样。禁止引用未来课程或无效权重。较长距离恢复配置为 flat 5%、obstacle 35%、gap 60%；比例指 episode 采样概率，不是 transitions 比例。
- **联合模型选择**：`evaluation.selection=joint_v1` 依次比较全部任务是否达标、最弱任务的合格布局比例、各任务平均合格比例、进度及成功通关速度。任务规模不同也不会掩盖较小任务的退化。采样权重及选择版本写入严格续训签名。
- **统一前置检查**：模型选择、课程晋级和独立验收复用同一组逐布局成功率与通关步数规则。缺失评估、成功率不足、成功但过慢均视为不合格。
- **晋级即停止**：`--stop-before-stage mixed` 在连续两轮自动达标后保存 `ready_for_stage` checkpoint；保留达标时的参数，停止后不再执行最后一轮梯度更新。采样步数包含最后收集的 rollout，模型参数对应实际已完成的更新。
- **可重载验收**：`verify_stage_entry` 校验模型哈希和环境内容协议，检查自动晋级证据，并重新评估前置布局。单次通过、手动跳阶段、协议变化或与就绪记录步数不符的模型不能被当作就绪 checkpoint。

连续合格轮次的步数随 checkpoint 保存，跨 `--resume` 保留。`--stop-before-stage` 是运行控制选项，正式开始 mixed 时省略它。旧配置默认仍按当前课程选 best，原有 checkpoint 的严格续训协议保持兼容。

## 冻结的进入条件

| 前置任务 | 验证布局数 | 每布局成功率阈值 | 每个成功 episode 最大环境步数 |
| --- | ---: | ---: | ---: |
| flat | 4 | 95% | 128 |
| obstacle | 40 | 80% | 160 |
| gap | 54 | 80% | 192 |

配置每 8,192 transitions 做一次确定性评估，连续两轮全部布局达标才能自动进入 mixed。固定几何和确定性策略下，重复环境 seed 不增加独立样本，因此训练时使用一个验证 seed（100）。随机诊断使用 200–209；正式多训练 seed 验收仍是后续工作。

OOD 不进入采样、best 选择或自动晋级；它作为开发回归复查，并非最终盲测。final test 只做规则基线的几何校准，学习型策略未使用。

## 本轮发现

旧模型在 `course_obstacle_09` 连续 1,024 步选择 `RIGHT_RUN`，没有一次有效起跳。第一次恢复训练在第 98,304 和 106,496 步连续达到 flat 4/4、obstacle 40/40、gap 36/36，自动门禁通过；独立 OOD 复查却从上一轮 12/12 退化为 0/12，因此没有把该模型作为最终 mixed 交接模型。

随后保留该模型的已学能力，使用 v4 的较长接近距离课程继续恢复。该轮缺口保持 54/54，但障碍确定性策略退化；在同一失败障碍上，随机采样 10/10 成功，确定性策略 0/10，说明训练的采样表现不足以代表部署时的确定性动作选择。因此停止退化实验，从联合得分最佳的 checkpoint 初始化稳定恢复：学习率降至 5e-5、熵系数为 0，episode 比例为 flat 5%、obstacle 60%、gap 35%。各次实验、自动晋级证据、独立重载验收、OOD 和测试记录汇总在 `docs/reports/mixed_prerequisites.json`。中间运行保留供审计，最终交接使用该报告指定的 `model.zip`。稳定恢复仍未同时通过所有布局，因此增加可选的规则示范预热：仅采集 flat/obstacle/gap 的 train 布局，先跟随规则策略，再采集学习策略访问的状态并标注动作；只更新 actor，保留 critic 和 PPO 优化器。预热后继续执行 PPO，与单纯 PPO 学习的结果明确区分。

## 运行与交接

仓库根目录下设置本机缓存：

```bash
export MPLCONFIGDIR="$PWD/.venv/matplotlib"
export XDG_CACHE_HOME="$PWD/.venv/cache"
```

复验最终课程门禁：

```bash
.venv/bin/python -m scripts.validate_rl_readiness \
  --courses --environment-id PlatformerState-v1 \
  --manifest config/curriculum_v4.json --episodes 1000 \
  --output runs/readiness_v4_new.json
```

复现最终预热与 PPO 实验（源模型和 sidecar 需同时保留）：

```bash
.venv/bin/python -m scripts.train_ppo \
  --config config/ppo_mixed_prerequisites_imitation.json \
  --init-from runs/mixed_prerequisites_v4/checkpoints/best_gap.zip \
  --start-stage gap --stop-before-stage mixed \
  --output-dir runs/mixed_prerequisites_imitation_new
```

独立验收：

```bash
.venv/bin/python -m scripts.verify_stage_entry \
  --model runs/mixed_prerequisites_imitation/model.zip \
  --target mixed --output runs/mixed_entry_recheck.json
```

所有前置通过后，从就绪模型严格续训，课程状态已指向 mixed：

```bash
.venv/bin/python -m scripts.train_ppo \
  --config runs/mixed_prerequisites_imitation/config.json \
  --resume runs/mixed_prerequisites_imitation/model.zip \
  --timesteps 131072 --output-dir runs/mixed_stage_v4
```

mixed 阶段继续以 flat 5%、obstacle 20%、gap 25%、mixed 50% 的 episode 概率训练；前置任务回归仍参与联合选择与后续晋级。输出目录必须为空。

## 最终结果（2026-10-03）

最终模型为 `runs/mixed_prerequisites_imitation/model.zip`，独立重载验收通过。规则示范预热使用 106 个训练布局、13,663 个采样状态和 3,400 次监督梯度更新；随后新增 16,384 个 PPO transitions，在第 8,192 和 16,384 步连续通过全部前置门禁。

| 项目 | 独立验收结果 | 最长成功通关步数 |
| --- | --- | ---: |
| flat validation | 4/4 | 19 |
| obstacle validation | 40/40 | 39 |
| gap validation | 54/54 | 40 |
| gap OOD（确定性） | 12/12 | — |

OOD 随机采样成功率 100%，固定首帧观测对照为 0%；该对照支持后续状态输入对动作选择有贡献。OOD 属于开发复查，不能替代最终盲测或多训练 seed 验收。

57 项单元/集成测试全部通过，无跳过；包含只更新 actor、保留训练 RNG 和 PPO 优化器、预热来源随续训保存、禁止重复预热、联合选择签名拒绝、连续验收跨续训保持、晋级保存参数一致及独立重载验收。1,000 episode 稳定性门禁通过，共 241,037 transitions，276 个 train/validation 布局通过 checker、规则可达性与奖励审计。学习型策略未使用 final test。
