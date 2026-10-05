# GenerationRequest v1 与通用 PCG 接口

日期：2026-10-05。完成 LLM 接入前的生成请求协议与确定性生成管线。本阶段无网络模型调用，未更改 v9 课程或训练权重。

## 请求契约

[JSON Schema](../schemas/generation_request_v1.schema.json) 从 Pydantic `GenerationRequest` 生成，严格拒绝未知字段、数值字符串、布尔数值、NaN/Infinity、重复 JSON 键与越界参数。读取请求最多 1 MB。请求只描述设计参数，具体矩形与实体坐标由 PCG 展开。

| 字段 | v1 范围/含义 |
| --- | --- |
| schema_version | 必填整数 1 |
| generator_version | segments-v1；默认填充并写入规范化请求 |
| physics_profile | gameplay-ai-v2；生成器内冻结的物理配置 |
| level_id | 必填非空标识，最长 128 字符；不作为输出文件路径 |
| seed | 必填 uint32，用于地图生成 |
| obstacle_count / gap_count | 每类 0–8，总计最多 12 |
| obstacle_height.minimum / maximum | 闭区间，整数 16–160；默认 48–96 |
| gap_width.minimum / maximum | 闭区间，整数 64–176；默认 96–152 |
| coin_layout | none / ground / low_jump；默认 low_jump |
| min_coin_ratio | 0–1，默认 0.75；none 必须配 0 |
| theme | 当前仅 lake |

零障碍、零缺口请求会生成一个平地段；每段 ground/low_jump 布局提供四个松果，none 不放松果。low_jump 每段两枚在 y=508，两枚在 y=430。收集比例是待验证目标，不是地图生成时已经得到的保证。

当前示例：[3 障碍 + 2 缺口](../game_content/generation_requests/mixed_v1.json)、[平地全收集](../game_content/generation_requests/flat_v1.json)。

## 通用接口与版本规则

```python
from ai_platformer.content.generation_request import load_generation_request
from ai_platformer.content.pcg import available_generators, generate_level

request = load_generation_request("game_content/generation_requests/mixed_v1.json")
result = generate_level(request)
assert result.report["status"] == "candidate"
spec = result.level       # LevelSpec v1
physics = result.physics # 冻结物理配置
```

`generate_level` 接受严格请求模型或 dict，按 generator_version 分派；当前 `available_generators()` 返回 segments-v1。结果包含规范化 request、LevelSpec、静态报告和目标 physics。纯生成无需 Gym、Torch、SB3、Pygame 初始化或网络，不写文件、不调用全局 RNG。

segments-v1 使用 SHA256 计数器序列、拒绝采样和固定 Fisher–Yates 洗牌，随后逐段抽样；不依赖 Python random 的实现或进程随机状态。同一规范化请求、版本及实现得到同一地图和内容哈希。golden map 测试固定示例 document SHA256 为 `d0e6302a194088d920eaa7fa64803c97abc9c37f9be4cd6bab13721d2c4a50f2`。生成几何或随机抽样规则变化时必须新增生成器版本，不能静默改变 segments-v1。

生成器冻结 gameplay-ai-v2 的 PhysicsConfig，而不是读取可变 gameplay 文件作为生成依据；更换目标物理时应显式新增 profile/版本并重新验证。范围是当前静态地图族的生成边界，不是数学上的必通关保证。

所有段固定 floor=540、world height=600、spawn x=48；随机 approach=540–720、runout=600–760，障碍宽 64。障碍和缺口分别占一个段，按 seed 洗牌，数量独立。地面按缺口切分，松果放在段前与安全落地区。报告记录实际段类型、位置、尺寸、approach/runout，便于审计。

规范化请求 SHA256 写入地图 metadata.source，同时保留 generator_version、seed、theme。改变 level_id 会改变请求/文档哈希，但不能绕过几何去重；旧 v9 哈希算法和清单不受影响。

## CLI 与产物

```bash
.venv/bin/python -m pip install -e '.[content]'

# 只生成与静态检查，无需 Gym 环境依赖
.venv/bin/python -m scripts.generate_level \
  --request game_content/generation_requests/mixed_v1.json \
  --output-dir runs/pcg_rerun/mixed

# 加入真实核心路线搜索、75% 收集目标及独立重放
.venv/bin/python -m pip install -e '.[content,benchmark]'
.venv/bin/python -m scripts.generate_level \
  --request game_content/generation_requests/mixed_v1.json \
  --output-dir runs/pcg_rerun/mixed_verified --verify-route

# 使用输出的路线进行 Pygame 播放
.venv/bin/python -m scripts.play_level_spec \
  --input runs/pcg_rerun/mixed_verified/level.json --controller replay \
  --route runs/pcg_rerun/mixed_verified/route.json
```

也可使用安装入口 `platformer-generate-level`。输出目录必须不存在。先在临时目录写完再原子发布；请求失败时保留 report.json，尚未生成 level.json 时不留下假地图。后续搜索失败时保留候选地图和诊断。

| 文件 | 内容 |
| --- | --- |
| request.json | 填充默认值后的规范化请求 |
| level.json | 具体 LevelSpec v1 地图 |
| report.json | 请求哈希、目标 physics、实际段、数量、静态诊断及全流程状态 |
| route.json | 仅 verify-route 模式；真实路线搜索与重放证据 |

状态含义：candidate 为静态通过且尚未进行路线验证；verified 为通关、达到请求中的 min_coin_ratio 并独立重放通过；unknown 为搜索未找到证据或超预算；invalid 为请求协议拒绝；error 为读取、生成或运行阶段错误。candidate/verified 退出 0，其余结果退出 1；输出冲突/产物写入失败退出 2。

搜索使用冻结 profile、v2 环境、action_repeat=4、step_limit=768；环境 seed=100 与请求中的地图生成 seed 分开。默认 25,000 扩展、15 秒，可用 `--max-expansions`/`--max-seconds` 调整。预算和搜索局限继承 [路线验证 v1](LEVEL_SPEC_ROUTES_V1.md)。未达到全收集目标不能直接判定松果不可达。

仅持有 report 的 verified 状态不是完整证据，消费端仍应加载 level.json 与 route.json 再执行 replay。原子发布或移动产物目录不会改变文档/环境协议，已测试移动后重放仍通过。

## 验收与后续

混合示例生成 3 个障碍、2 个缺口、20 枚松果：coin-jump 找到 206 决策/823 tick 的路线，收集 16/20（80%），满足 75% 目标。独立重放与真实 Pygame headless 播放通过。平地示例要求全部收集，也通过验证。它们是生成管线与规则路线测试，不是 PPO 泛化验收。

测试覆盖固定 golden map、跨 key 顺序复现、不同 seed、全局 RNG 不变、32 个边界种子的静态检查、数量/尺寸边界、请求格式拒绝、去重、无 Gym/Torch 的纯生成、输出冲突、unknown 状态保留和目录迁移后重放。

第一版只生成静态地面、障碍、缺口与松果，不生成敌人、交互方块、传送或自定义美术，也不自动分配训练/保留 split。下一轮 LLM 只需输出 GenerationRequest v1；模型原始响应、失败重试与请求缓存另行接入，全部候选继续经过这套验证管线。

全量测试：155 passed、8 subtests passed，两条既有 Gym 弃用提示。修改 Python 文件 Ruff 与 diff whitespace 检查通过。机器验收报告：[pcg_generation_v1.json](reports/pcg_generation_v1.json)。
