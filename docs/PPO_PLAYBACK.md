# PPO 模型可视化播放

播放入口 `scripts/play_ppo.py` 支持 `--backend sb3`（默认）和 `--backend tinyinfer`。
SB3 加载 `model.zip` 与同名 `model.json`；TinyInfer 加载导出的 `actor.onnx`、同名 `actor.json` 与桥接动态库。
两种路径都检查模型哈希、环境协议和关卡内容，随后恢复训练时的环境配置。

当前推荐 SB3 模型为 `runs/full_collection_v9/seed_20261004/model.zip`，须保留同名 sidecar。主区域已修正不可达松果分母，播放时 HUD 与 JSON 都显示收集数：

```bash
.venv/bin/python -m scripts.play_ppo \
  --model runs/full_collection_v9/seed_20261004/model.zip --level-id level_1_main
```

新模型的 `--task full --suite validation` 播放 manifest 中的生成完整地图，按 N 切换。没有 full split 的旧模型仍播放原完整关卡。完整训练/验收见 [v9 报告](FULL_MULTISEED_COLLECTION_V9.md)。新 checkpoint 尚未重测 TinyInfer，使用 TinyInfer 前须重新导出该模型并运行数值验证。以下保留旧 checkpoint 的使用示例。

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

HUD 显示后端、策略模式、速度、动作、进度、松果数、回报和单次 `predict()` 耗时。
默认 deterministic；SB3 可用 `--sampled` 启用动作采样。TinyInfer 暂只支持 deterministic，指定 `--sampled` 会报错。
`--seed` 控制环境与采样随机数；SB3 默认取 checkpoint 的第一个 validation seed，TinyInfer 默认取部署验证 seed。
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

## TinyInfer 播放

先按 [导出与桥接说明](TINYINFER_DEPLOYMENT.md) 准备部署包和动态库，再运行：

```bash
.venv/bin/python -m scripts.play_ppo \
  --backend tinyinfer --model runs/tinyinfer_actor_v2/actor.onnx \
  --library build/tinyinfer_bridge/libplatformer_tinyinfer.dylib \
  --fuse-relu --task gap
```

省略 `--fuse-relu` 使用原始图；两者均采用已准备的 CPU 执行计划。
模型和执行上下文在整个播放过程中复用，重播与切换关卡不会重新加载模型，退出时释放会话。
TinyInfer 播放不导入 Torch、SB3、ONNX 或 ONNX Runtime；运行依赖可用 `pip install -e '.[inference]'` 安装。
它仍需要本项目的课程 manifest、内容文件和 Pygame 资源，不能只复制 ONNX 后独立显示游戏。
`.dylib` 是 Mac 产物；Linux/Windows 使用相应 `.so`/`.dll`。

部署性能测量使用 [独立 benchmark](DEPLOYMENT_PERFORMANCE.md)，HUD 的单次耗时用于观察，不能替代预热、多轮测量和完整调用比较。
