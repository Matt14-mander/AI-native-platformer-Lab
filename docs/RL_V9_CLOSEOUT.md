# 强化学习 v9 基线阶段收尾

日期：2026-10-04。建议里程碑 tag：`rl-v9-baseline`。训练实验版本 v9 与 Python 项目版本 `0.1.0` 分别管理，此次不提升项目包版本。

本阶段冻结 v2 游戏物理、15 维观测、10 个动作、reward、v9 课程及验收协议。方法为训练集规则示范修正 + PPO。三个独立训练 seed（20261004/20261005/20261006）均通过训练验收；主模型始终为预指定的 20261004。主区域 243 步、5/5 松果；8 张新增保留地图共 192 局全部成功、收集率 97.09%；旧课程与 full 回归 5,712 局通过。

主模型部署完成：2,048 个固定观测校验、47 张地图三后端共 141 局逐步轨迹一致、TinyInfer/Pygame 播放通过。融合图 predict 中位数 30.10 μs，相对 SB3 约快 7.93 倍；完整无窗口帧耗时中位数减少 18.3%。最新代码验收为 92 passed、8 subtests passed，两条既有 Gym 弃用提示。

证据入口：[训练验收](FULL_MULTISEED_COLLECTION_V9.md)、[部署回归与完整命令](DEPLOYMENT_V9.md)、[训练机器报告](reports/full_collection_v9.json)、[部署机器报告](reports/deployment_v9.json)。

## 已知问题与下一轮边界

1. seed 20261005/20261006 各一个 ONNX logit 超出既有 `atol=1e-5, rtol=1e-5`，未发布对应部署包。下一轮定位 FP32 累加/算子实现差异，并复跑原验证数据；不能只改容差来宣称通过。主模型部署可作为当前基线。
2. 泛化仅覆盖声明的静态生成地图族。当前主区域分母为实际可达的 5 个松果；后续区域的 20 个、传送、其他 legacy 关卡和新交互尚未迁移。
3. TinyInfer 验收为 CPU FP32 确定性 argmax，未覆盖随机采样、其他架构/设备。主区域为训练中的已知地图；静态环境评估 seed 不增加独立几何数量。
4. 下一轮若改变物理、reward、观测、动作或地图协议，另建版本并重新训练/验收，保留本基线与失败实验。

## 归档与恢复

本地归档为 `runs/archives/rl-v9-baseline/artifacts.zip`；清单与归档 SHA256 见 [归档清单](reports/rl_v9_archive.json)。包含三份完整 v9 训练/验收输出、三份 v8 最终起点及其 sidecar、v9 导出与部署结果、失败日志、课程配置、依赖快照和本机 TinyInfer 桥接库/构建记录。恢复时在项目根目录解压，保留原相对路径。不要覆盖后续实验输出。

Git tag 只保存代码和已跟踪配置，`runs/` 与 `build/` 被忽略，模型不会随 tag 上传。归档需另行备份；其中截图仍可能包含旧学习素材，因此本次只生成本地研究归档，不自动发布 GitHub Release 附件。归档未包含全部更早谱系权重；它支持重载最终模型、复跑验收和从 v8 起点重复 v9 阶段，不声称完整重现此前所有训练实验。

`environment.freeze.txt` 是本机 Python 3.12/macOS x86_64 的安装版本快照，不是跨平台 lockfile。跨设备按项目 extras 安装依赖，并重新构建 TinyInfer 库后复跑验收。本机动态库不能直接用于其他系统。

## 统一入口

在仓库根目录执行；示例输出目录必须尚不存在。其他训练 seed 必须使用自己的 v8 起点与归档中的 `runs/full_collection_v9/config_SEED.json`，不能共享主模型权重；训练 seed 从配置读取。

```bash
export MPLCONFIGDIR="$PWD/.venv/matplotlib"
export XDG_CACHE_HOME="$PWD/.venv/cache"

# 重载训练验收：主模型
.venv/bin/python -m scripts.accept_full_checkpoint \
  --model runs/full_collection_v9/seed_20261004/model.zip \
  --output-dir runs/rl_v9_acceptance_rerun

# PPO 可视化播放
.venv/bin/python -m scripts.play_ppo \
  --model runs/full_collection_v9/seed_20261004/model.zip --level-id level_1_main

# TinyInfer 融合图播放
.venv/bin/python -m scripts.play_ppo --backend tinyinfer --fuse-relu \
  --model runs/deployment_v9/seed_20261004/actor.onnx \
  --library build/tinyinfer_bridge/libplatformer_tinyinfer.dylib --level-id level_1_main

# 从归档中的 v8 主模型开始重复 v9 阶段
.venv/bin/python -m scripts.train_ppo --config config/ppo_full_collection_v9.json \
  --init-from runs/full_collection_v8/seed_20261004/model.zip \
  --start-stage full --output-dir runs/rl_v9_training_rerun
```

导出、数值校验、47 地图部署对照和性能复跑使用 [部署报告](DEPLOYMENT_V9.md) 中的成套命令。性能测量应在训练、测试和其他验证结束后单独运行。

## GitHub tag

建议在收尾文件审核并提交后创建 annotated tag，消息声明“三 seed 训练通过；主模型部署通过；两份辅助 seed 导出待修”。tag 的作用是固定基线代码，GitHub Release 为可选项，本阶段无需公开发行游戏。

```bash
git add README.md docs/RL_V9_CLOSEOUT.md docs/reports/rl_v9_archive.json
git commit -m "docs: close out RL v9 baseline and archive artifacts"
git tag -a rl-v9-baseline -m "RL v9 baseline: 3 training seeds accepted; primary deployment passed; 2 auxiliary exports blocked"
git push origin HEAD
git push origin rl-v9-baseline
```

以上为待执行命令；本次收尾不自动创建 commit、tag 或远端 Release。
