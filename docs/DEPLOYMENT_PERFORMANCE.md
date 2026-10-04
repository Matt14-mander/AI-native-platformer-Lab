# TinyInfer 部署性能测量

最新 v9 主模型测量见 [v9 部署回归](DEPLOYMENT_V9.md)，下文实测表为旧 checkpoint 的历史结果。

`scripts.benchmark_deployment` 比较 TinyInfer 原始图、Gemm/ReLU 融合图，可选比较导出来源 SB3 checkpoint 和 PyTorch actor。
数值与动作检查通过后才测量；提供 `--checkpoint` 时必须与导出来源模型的哈希一致。

## 使用

本轮桥接增加了计时 API，先重新构建：

```bash
.venv/bin/python -m scripts.build_tinyinfer_bridge \
  --source /Users/zhengyuanhao/Desktop/Project/TinyInfer

.venv/bin/python -m scripts.benchmark_deployment \
  --model runs/tinyinfer_actor_v2/actor.onnx \
  --library build/tinyinfer_bridge/libplatformer_tinyinfer.dylib \
  --checkpoint runs/mixed_prerequisites_imitation/model.zip \
  --warmup 200 --iterations 5000 --rounds 3 --episodes 3 \
  --output runs/tinyinfer_actor_v2/deployment_performance.json
```

输出 JSON 汇总报告与同名前缀 `.samples.npz` 原始计时。既有文件不被覆盖。
省略 `--checkpoint` 仅测 TinyInfer，不导入训练框架；`--episodes 0` 关闭环境与绘制测量。
默认每轮使用相同观测顺序，每个 Python 计时样本是一个观测的单次调用；`--iterations` 为每轮次数。
奇偶轮反转后端测量顺序，减小固定顺序的影响。显式预热之前已经执行过正确性检查，报告记录这些边界。
默认从导出验证布局中均匀选择三个代表关卡；`--seed` 默认为部署验证 seed。

## 计时边界

| 指标 | 包含 | 排除 |
| --- | --- | --- |
| first_create / session_create | Python 元数据校验、CDLL 加载、ONNX 导入、图优化、执行计划和上下文创建 | 框架导入、环境恢复、会话销毁 |
| first_predict | 创建后的第一次完整 Python 动作调用 | 会话创建 |
| native_run | C++ `context.run()`，包含 TinyInfer 内部派发 | 输入绑定、输出提取、Python/ctypes |
| native_call | 输入 Tensor 分配与复制、绑定、run、输出提取与复制 | 每次 Python/ctypes 调用 |
| python_logits | NumPy 输入检查、ctypes、完整原生推理与输出数组 | argmax |
| predict | 实际策略 API 的完整调用；TinyInfer 包含 argmax | 环境执行、绘制 |
| tensor_forward | PyTorch actor 的 tensor forward 和框架派发 | NumPy 转换、输出提取、动作分布 |
| environment | 动作转换与同一 Gym 环境执行 | 推理、绘制 |
| render | 同一 Pygame 渲染器与 display.flip | 资源加载、HUD、限帧等待 |
| frame | predict + environment + render | HUD、限帧、终局等待、轨迹哈希和统计写入 |

全部时间以微秒记录，汇总中位数、P95、P99、均值和极值；原始样本可用于再次分析。
native_call 内含嵌套计时开销；报告提供空 Python 调用的计时量级，不扣除计时器开销。
不同统计分布的中位数之差不能直接视为精确的桥接开销。
原生 benchmark 一次跨 ctypes 运行整个循环，Python predict 指标逐次调用，二者衡量不同边界。

首次创建是当前进程中的首次会话调用，后续创建会复用动态库和 OS 文件缓存，不代表冷磁盘启动。
TinyInfer 当前单线程 CPU 执行；SB3 对比设置 Torch intra-op 线程数为 1，并记录实际线程设置。
报告记录模型/库/原始计时哈希、平台、Python/NumPy/Torch 版本、构建信息与每轮测量顺序。
模型准确性和部署性能以当前 checkpoint、设备与构建配置为范围；predict 比率不代表整游戏帧率比率。

## 回合正确性

无窗口回合也使用真实环境和渲染器，按 checkpoint 的 action_repeat 推进。
各后端使用相同布局和 seed；在计时之外对每一步完整 WorldSnapshot、动作与回报计算轨迹哈希。
终局结果、步数、物理 tick、回报及轨迹哈希必须全部匹配，否则 benchmark 返回失败并拒绝输出性能比较报告。
这是代表关卡的部署回归检查，不替代正式多 seed 的策略泛化验收。

## 本机实测

使用 `runs/tinyinfer_actor_v2/actor.onnx`，Mac x86_64、Release、native_arch 关闭、Torch intra-op 单线程。
预热 200 次，每轮 5,000 次，共三轮（每项 15,000 个测量样本）；数值容差与 1,024 个固定观测上的动作一致性检查均通过。

| 测量项 | 中位数（微秒） | P95（微秒） |
| --- | ---: | ---: |
| TinyInfer 原始图 native run | 13.50 | 14.45 |
| TinyInfer 融合图 native run | 9.82 | 10.47 |
| TinyInfer 融合图 native call，含输入输出 | 11.11 | 11.86 |
| TinyInfer 原始图 Python predict | 36.75 | 47.20 |
| TinyInfer 融合图 Python predict | 32.39 | 36.95 |
| SB3 predict | 247.65 | 341.87 |
| PyTorch actor tensor forward | 35.91 | 47.05 |

融合后端完整动作调用中位数约为 SB3 的 1/7.65；与 PyTorch actor 的比较需注意上表不同的输入输出边界。
三个代表关卡各后端均通关，三种后端共九回合的完整状态、动作与回报轨迹一致。
每种后端共测 87 个无窗口帧：融合图帧中位数为 893.98 微秒，SB3 为 1,153.61 微秒，约减少 22.5%，不包含窗口限帧等待。
原始图第一次会话创建约 3.28 毫秒；融合图在动态库已加载后的第一次创建约 2.46 毫秒，不能作为对等冷启动比较。

完整报告与原始样本分别保存在 `runs/tinyinfer_actor_v2/deployment_performance.json` 和 `.samples.npz`。
全量测试 69 passed；当前 TinyInfer 源码未修改。以上为当前模型和设备的一轮测量快照。
