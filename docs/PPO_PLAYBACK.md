# PPO 模型可视化播放

播放入口 `scripts/play_ppo.py` 直接加载 SB3 `model.zip`，不需要导出 `.pt` 或 ONNX。
必须保留同名 `model.json`：脚本检查模型哈希、checkpoint 协议和关卡内容，随后恢复训练时的环境配置。

从项目根目录运行（先安装 `pip install -e '.[training]'`；本机已有 `.venv` 可直接使用）：

```bash
.venv/bin/python -m scripts.play_ppo \
  --model runs/mixed_prerequisites_imitation/model.zip \
  --task gap --suite validation
```

该 checkpoint 完成了 mixed 的前置课程，尚未完成正式 mixed 训练。建议先看 `flat`、`obstacle`、`gap`。
`--task mixed` 可用于查看当前迁移表现；`--task full` 播放原始完整关卡，也是迁移任务。
默认逐个播放所选任务的 validation 布局，到结尾循环；`--episodes 3` 在完成三回合后退出。
每回合结束向终端输出 JSON，包括关卡、seed、outcome、步数、物理 tick、进度和累计回报。
这属于观察工具，正式性能报告仍使用 `scripts.evaluate_ppo`。

| 操作 | 功能 |
| --- | --- |
| P / Space | 暂停或继续 |
| R | 同一关卡、同一 seed 重播 |
| N | 切换下一个关卡 |
| [ / ] | 减速 / 加速（0.1–8 倍） |
| Esc / 关闭窗口 | 退出 |

HUD 显示策略模式、速度、动作、进度、回报和单次 `predict()` 耗时。
默认 deterministic；`--sampled` 启用动作采样。`--seed` 同时控制环境与策略随机数，默认取 checkpoint 的第一个 validation seed。
`--level-id LEVEL_ID` 指定单个布局，覆盖 task/suite 选择；`--suite train` 或 `ood` 可查看对应课程分组。
播放列表不提供 final test 分组，避免日常观察影响最终验收。

```bash
# 半速查看障碍课程，完成两回合后退出
.venv/bin/python -m scripts.play_ppo \
  --model runs/mixed_prerequisites_imitation/model.zip \
  --task obstacle --speed 0.5 --episodes 2 --seed 200

# 无窗口快速验证；仍执行 Pygame 绘制，并保存最后一帧
.venv/bin/python -m scripts.play_ppo \
  --model runs/mixed_prerequisites_imitation/model.zip \
  --task gap --episodes 2 --headless --screenshot /tmp/ppo-playback.png
```

环境仍按 checkpoint 的 `action_repeat` 一次推进多个物理 tick。1 倍速以 gameplay 的 `render_fps` 为物理 tick 播放基准；
渲染展示每次环境决策后的快照，不插入额外物理步。暂停和速度只改变墙钟调度，不修改观测、物理或动作重复次数。
无窗口模式忽略播放速度与终局等待，按回合快速运行；必须指定正数 `--episodes`。
TinyInfer 后端尚未接入，目前推理使用 CPU 上的 SB3/PyTorch。
