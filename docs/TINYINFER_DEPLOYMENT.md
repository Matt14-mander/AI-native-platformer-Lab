# PPO actor 导出与 TinyInfer 桥接

本阶段在本项目实现导出器、C ABI 动态库和 Python `ctypes` 适配器。构建从指定路径读取 TinyInfer 源码，所有构建产物写入本项目；不修改 TinyInfer 文件夹。
当前支持 CPU FP32、静态单观测输入、离散动作、Linear/ReLU actor，以及 deterministic `argmax`。

最新主模型结果见 [v9 部署回归](DEPLOYMENT_V9.md)：主 seed 已通过，另外两个 seed 的导出失败记录一并保留。下文旧模型示例与历史成绩保留供复现。

## 1. 导出真实 checkpoint

导出和 PyTorch/ONNX Runtime 对照需要训练与部署依赖：

```bash
.venv/bin/python -m pip install -e '.[training,deployment]'
.venv/bin/python -m scripts.export_ppo \
  --model runs/mixed_prerequisites_imitation/model.zip \
  --output-dir runs/tinyinfer_actor_v1 --samples 1024 --seed 100
```

输出目录必须不存在，避免覆盖已验收模型。失败不会留下半成品部署包。

| 文件 | 内容 |
| --- | --- |
| `actor.onnx` | actor 权重；当前模型为 `[1,15] observations → [1,10] logits` |
| `actor.json` | 模型哈希、来源 checkpoint、观测与动作协议、环境配置、导出版本和验证结果 |
| `validation.npz` | 固定观测、PyTorch 参考 logits 和 SB3 deterministic 动作 |

导出前检查 checkpoint sidecar、模型哈希、动作映射、环境与课程协议。
只导出 actor，不包含 critic、优化器和动作分布运算；平坦观测的 FlattenExtractor 为恒等变换，导出时省略。
PyTorch 为相同常量生成的 Identity 别名会物化为 initializer，再检查 ONNX 图和输出一致性。
导出器不接受其他网络算子，而是明确报错。

参考观测来自 flat、obstacle、gap、mixed，以及清单包含时的 full validation，各选最多三个布局的完整 deterministic 回合；包含 full 清单时还覆盖当前主区域，再补充 `[-1,1]` 随机输入。
元数据记录真实/随机样本数、布局和 seed；这些是开发验证数据，不能作为最终泛化验收。
导出成功要求 ONNX Runtime 的 logits 满足 `atol=1e-5, rtol=1e-5`，且动作与 SB3 全部一致。

## 2. 构建桥接动态库

需要 CMake ≥ 3.20 和 C++17 编译器；本机命令为：

```bash
.venv/bin/python -m scripts.build_tinyinfer_bridge \
  --source /Users/zhengyuanhao/Desktop/Project/TinyInfer
```

默认 Release、关闭 TinyInfer 的 examples/tests/benchmarks、静态链接 TinyInfer 到桥接库。
`--build-dir` 指定构建目录；`--native` 显式启用当前 CPU 指令集，默认关闭以便比较和移植。
Mac 默认产物为 `build/tinyinfer_bridge/libplatformer_tinyinfer.dylib`；Linux 为 `.so`，Windows 为 `.dll`。
构建脚本打印实际路径，并写入 `bridge_build.json`，记录 TinyInfer revision、工作区状态、平台和构建选项。
Mac 构建会检查 SDK 的 libc++ 头文件路径，以兼容本机 Command Line Tools 布局。

## 3. 验证桥接输出

```bash
.venv/bin/python -m scripts.validate_tinyinfer \
  --model runs/tinyinfer_actor_v1/actor.onnx \
  --library build/tinyinfer_bridge/libplatformer_tinyinfer.dylib \
  --output runs/tinyinfer_actor_v1/tinyinfer_validation.json
```

验证原始图和显式 Gemm/ReLU 融合图两种路径。两者必须均满足数值容差并且动作全部一致。
报告包括模型与桥接库哈希、最大/平均绝对误差、动作匹配率、最小动作 logits 间距和异常样本索引。
验证失败返回非零退出码，并保留已生成的差异报告；既有报告不被覆盖。
最大绝对误差可能略大于 `1e-5`，验收条件是 `abs(error) <= 1e-5 + 1e-5 * abs(reference)`。

## 4. 在 Python 中调用

适配器只依赖 NumPy、标准库和本项目，不导入 Torch、SB3、ONNX 或 ONNX Runtime；动态库运行时也不依赖 Python 训练框架。

```python
from pathlib import Path
import numpy as np
from ai_platformer.deployment.tinyinfer import TinyInferPolicy

with TinyInferPolicy(
    Path("runs/tinyinfer_actor_v1/actor.onnx"),
    Path("build/tinyinfer_bridge/libplatformer_tinyinfer.dylib"),
    fuse_relu=True,
) as policy:
    observation = np.zeros(15, dtype=np.float32)  # 替换为环境产生的真实观测
    logits = policy.logits(observation)
    action, _ = policy.predict(observation, deterministic=True)
```

模型、执行计划和上下文在创建时准备一次，后续调用复用。输入必须为有限 float32，形状 `[N]` 或 `[1,N]`。
输出为独立数组，后续推理不会覆盖旧结果；不同实例拥有独立上下文。
适配器串行化同一实例的推理与关闭，`close()` 可重复调用，关闭后拒绝推理。
C ABI 捕获异常并通过线程本地错误接口返回；直接调用 C ABI 的调用者须管理句柄生命周期，不能在销毁后使用句柄。

## 验证与后续范围

先构建桥接，再运行：

```bash
MPLCONFIGDIR="$PWD/.venv/matplotlib" XDG_CACHE_HOME="$PWD/.venv/cache" \
  .venv/bin/python -m pytest tests/integration/test_tinyinfer_deployment.py -q
```

测试使用临时生成的 PPO checkpoint，覆盖真实导出、两种执行路径、上下文复用、错误输入、模型哈希、命名绑定、释放与训练依赖隔离。
可用 `PLATFORMER_TINYINFER_LIBRARY` 指定测试库路径；未构建动态库时，原生桥接测试会跳过。

导出与桥接已接入 `scripts.play_ppo --backend tinyinfer`，支持部署环境校验及原始图/融合图选择，见 [播放说明](PPO_PLAYBACK.md)。
独立 [部署性能 benchmark](DEPLOYMENT_PERFORMANCE.md) 已提供，分别统计纯执行、输入输出处理、ctypes 调用和无窗口完整帧，并校验各后端回合轨迹。

本机已用 `runs/mixed_prerequisites_imitation/model.zip` 生成 `runs/tinyinfer_actor_v2/` 部署包：
1,024 个观测包含 251 个真实课程观测和 773 个随机输入。原始图与融合图均通过容差检查，动作匹配率均为 100%，最大绝对误差约 `1.53e-5`。
另对九个代表布局、seed 100 进行了两种路径共 18 回合的轨迹冒烟检查，均与 SB3 的动作、状态和回报完全一致并通关。
报告位于该部署目录的 `tinyinfer_validation.json` 和 `trajectory_smoke.json`；这不替代多 seed 泛化验收。
导出与桥接阶段测试为 65 passed；加入后端播放与性能测量后，全量测试为 69 passed，Ruff 与 diff 检查通过。
构建产物与模型在忽略目录中，跨机器运行需重新构建与导出。
