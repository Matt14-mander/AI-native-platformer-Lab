# v9 模型部署回归

当前推荐主模型 `runs/full_collection_v9/seed_20261004/model.zip` 已纳入部署回归，导出包为 `runs/deployment_v9/seed_20261004/`。主 seed 延续训练前的指定，未根据部署成绩重新选择。CPU FP32，输入 `[1,15]`、输出 `[1,10]`，确定性 argmax，action_repeat=4。

## 正确性验收

- 导出参考覆盖主区域，以及 flat、obstacle、gap、mixed、full validation 各三个代表布局：1,279 个实际观测 + 769 个随机观测，共 2,048 个。
- PyTorch/SB3、ONNX Runtime、TinyInfer 原始图与融合图的动作全部一致。沿用 `atol=1e-5, rtol=1e-5`；TinyInfer 最大绝对误差为 0.000183105，仍满足按参考值计算的相对/绝对误差条件。
- 新增 `scripts.regress_deployment`：主区域 + 12 张基础课程代表地图 + 10 张 full validation + 12 张 full test + 12 张 full OOD，共 47 张地图，环境 seed=100。三后端共 141 局全部成功，每步动作、观测、奖励、终止标志、info 和完整 WorldSnapshot 一致。
- 已经暴露的 test/OOD 在此用于部署回归，不能作为新一轮策略泛化验收；此轮也不验证 TinyInfer 随机动作采样。
- TinyInfer 融合图通过真实 Pygame 无窗口播放：主区域 243 步通关、5/5 松果。截图保存在 `runs/deployment_v9/playback_main.png`。

## 多训练 seed 状态

| 训练 seed | 部署状态 |
| --- | --- |
| 20261004（预指定主模型） | ONNX、两种 TinyInfer 图、47 地图逐步对照和性能测量通过 |
| 20261005 | ONNX 导出数值门槛失败：20,480 个 logit 中一个超容差；未发布部署包 |
| 20261006 | ONNX 导出数值门槛失败：20,480 个 logit 中一个超容差；未发布部署包 |

两个失败的最大绝对误差分别为 0.000183105 和 0.000122070，最大相对误差分别为 8.370531e-5 和 6.310122e-5。它们是 PyTorch 与 ONNX Runtime 的 logit 比较失败；没有进入 TinyInfer 数值、轨迹与性能验收，也不据此断言动作错误。失败日志在 `runs/deployment_v9/export_failure_SEED.log`，内容同时保存在机器报告。数值门槛未放宽，训练权重未改动。这两个 seed 的训练验收结果保持原有结论，但三 seed 部署验收尚未全部通过。

## 本机性能

Mac x86_64，Release，native_arch 关闭，TinyInfer 单线程，Torch intra-op 单线程。复用桥接库，构建 revision `0749e722d0f111d1844fd01cb33090afb7e8ccc2`，库哈希与构建记录一致。测试和其他模型验证结束后单独运行 benchmark；预热 200 次，每轮 5,000 次，共三轮；会话创建重复五次。

| 测量项 | 中位数 μs | P95 μs |
| --- | ---: | ---: |
| TinyInfer 原始图 native run | 12.878 | 13.544 |
| TinyInfer 融合图 native run | 9.371 | 9.858 |
| TinyInfer 原始图 Python predict | 34.434 | 48.209 |
| TinyInfer 融合图 Python predict | 30.097 | 33.584 |
| SB3 predict | 238.773 | 312.481 |
| PyTorch actor tensor forward | 34.251 | 38.699 |
| TinyInfer 融合图无窗口完整帧 | 1,277.177 | 1,767.361 |
| SB3 无窗口完整帧 | 1,563.341 | 2,080.910 |

融合图 native run 比原始图中位数减少约 27.2%；完整 Python predict 比 SB3 快约 7.93 倍。PyTorch actor tensor forward 不含输入输出转换，不能直接用它判断完整 API 加速比。

帧测量包含 predict、环境执行、Pygame 绘制与 flip，不含 HUD、资源加载、限帧等待和轨迹哈希。每个后端 489 帧，主区域、gap validation 和 full validation 三局的终局及轨迹哈希一致；主区域 5/5 松果，full 代表地图 11/12。融合图完整帧中位数减少约 18.3%，不等于窗口 FPS 提高 7.93 倍。历史模型测量保留在原文档，不能作为同模型速度变化。

## 复跑与播放

下面命令在项目根目录执行；输出目录/报告必须使用尚不存在的新路径。

```bash
export MPLCONFIGDIR="$PWD/.venv/matplotlib"
export XDG_CACHE_HOME="$PWD/.venv/cache"

.venv/bin/python -m scripts.export_ppo \
  --model runs/full_collection_v9/seed_20261004/model.zip \
  --output-dir runs/deployment_v9_rerun/seed_20261004 --samples 2048 --seed 100

.venv/bin/python -m scripts.validate_tinyinfer \
  --model runs/deployment_v9_rerun/seed_20261004/actor.onnx \
  --library build/tinyinfer_bridge/libplatformer_tinyinfer.dylib \
  --output runs/deployment_v9_rerun/seed_20261004/numerical.json

.venv/bin/python -m scripts.regress_deployment \
  --checkpoint runs/full_collection_v9/seed_20261004/model.zip \
  --model runs/deployment_v9_rerun/seed_20261004/actor.onnx \
  --library build/tinyinfer_bridge/libplatformer_tinyinfer.dylib \
  --seeds 100 --output runs/deployment_v9_rerun/seed_20261004/trajectory_regression.json

.venv/bin/python -m scripts.benchmark_deployment \
  --checkpoint runs/full_collection_v9/seed_20261004/model.zip \
  --model runs/deployment_v9_rerun/seed_20261004/actor.onnx \
  --library build/tinyinfer_bridge/libplatformer_tinyinfer.dylib \
  --warmup 200 --iterations 5000 --rounds 3 --startup-repeats 5 --episodes 3 --seed 100 \
  --output runs/deployment_v9_rerun/seed_20261004/performance.json

.venv/bin/python -m scripts.play_ppo --backend tinyinfer --fuse-relu \
  --model runs/deployment_v9/seed_20261004/actor.onnx \
  --library build/tinyinfer_bridge/libplatformer_tinyinfer.dylib --level-id level_1_main
```

全量测试：92 passed，8 subtests passed，两条既有 Gym 版本弃用提示。修改的 Python 文件 Ruff 检查通过。模型、ONNX、动态库和原始计时属于本机产物；需保留 checkpoint/sidecar 和整个部署包，跨设备须重新构建库并复跑验收。

机器报告：[deployment_v9.json](reports/deployment_v9.json)，包含来源哈希、数值验证、47 地图轨迹摘要、性能报告及失败记录。完整性能与原始样本在 `runs/deployment_v9/seed_20261004/performance.json` / `performance.samples.npz`。
