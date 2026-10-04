# LevelSpec v1 与静态校验

日期：2026-10-04。完成计划中的第一轮：严格关卡协议、JSON Schema、与核心的无损转换、结构/静态几何校验和数据集去重。后续已接入新格式加载、播放和真实核心路线搜索/重放，见 [路线验证 v1](LEVEL_SPEC_ROUTES_V1.md)。LLM 和训练数据自动入库尚未接入。

## 协议

[Schema](../schemas/level_spec_v1.schema.json) 由 `LevelSpec.model_json_schema()` 生成并通过快照测试，使用 Draft 2020-12。实现位于 `ai_platformer/content/level_spec.py`，校验位于 `ai_platformer/content/level_validation.py`。

| 字段 | 定义 |
| --- | --- |
| schema_version / level_id | 固定整数版本 1、非空地图标识 |
| world | width、height；units 固定 pixels |
| spawn | x 是玩家左边缘；bottom 是玩家脚底坐标 |
| goal | x 是通关阈值，按当前核心玩家左边缘到达计算 |
| solids | 有序矩形列表；x/y 为左上角，width/height 为正数 |
| collectibles | 松果（core kind=coin），实体 ID、矩形和 score |
| blocks | brick/box，奖励 none/coin/shield |
| enemies | 当前巡逻敌人的矩形、方向、速度与巡逻区间 |
| powerups | shield 道具 |
| metadata | title/theme/source/generator_version/seed；只保存描述，不自动解析为资源路径 |

JSON 保存已展开地图，不保存待解释的生成参数。坐标为 x 向右、y 向下。尺寸/坐标限制到一百万像素，每类数组最多一万项，文件加载最多 8 MB。拒绝未知字段、非法枚举、布尔数值、数值字符串、NaN/Infinity、重复 JSON 键。schema_version 和 direction 必须是整数。

物理、动作和 reward 由环境配置管理，不允许地图覆盖。CLI 默认读取 `config/gameplay_ai_v2.json`；报告记录实际使用的完整 physics。Python API 未传 physics 时采用 `PhysicsConfig()`，因此需要指定与目标环境一致的配置。

## 检查与报告

- 出生玩家完整碰撞盒必须位于地图内，不能与 solids、blocks 或敌人重叠；脚底接触地面不算重叠。
- 终点必须在出生点之后，且玩家左边缘能在地图边界限制内到达。
- 全部实体矩形在界内，动态实体 ID 跨类别唯一；敌人出生点与完整巡逻范围合法。
- 终点后的松果返回 warning；松果完整嵌入不可破坏地形返回 error。悬空松果不直接判错。
- 不普遍禁止地形重叠；地形重叠与交互实体的可达性需要后续检查。
- 松果/地形嵌入检查最多十万个配对；超预算返回 error，阻止将未完成的检查当成通过。

报告含 `structure_valid`、`static_valid`、`reachability` 与 issues。每条 issue 有 code、JSON Pointer 路径、severity、message。静态通过只表示已实现的规则通过，**所有报告的 reachability 均为 not_checked**；不证明存在通关或全收集路线。

CLI 成功退出 0；校验失败退出 1，同时保留报告；输出已存在或写报告失败退出 2。默认不覆盖文件。结构未通过时不进入几何转换；输入 JSON 错误、读文件失败也记录为 input_error。

## 哈希与兼容

1. `document_hash` 包含 metadata、level_id 和有序内容，忽略 JSON 键排列与缩进。
2. `gameplay_hash` 完全沿用旧 `content_hash(LevelDefinition)`，排除 level_id；转换保留原整数/浮点类型与实体顺序，确保 v9 checkpoint 协议哈希不变。
3. `geometry_hash` 用于新数据集去重：忽略地图/实体 ID，规范化整数、浮点与负零；保留实体属性和数组顺序，因为核心遍历顺序可能影响行为。当前不是对任意重排都不变的几何同构检测器。

`validate_dataset({"train": [...], "validation": [...], "test": [...]}, physics=...)` 对所有明确传入的 split 检查重复地图 ID、重复内容和静态规则。空数据集/空 split 不通过。此接口不修改原训练清单，也不自动接纳新地图。

## 使用

安装可选依赖，core 和原训练/部署运行路径不强制依赖 Pydantic：

```bash
.venv/bin/python -m pip install -e '.[content,dev]'

# 校验已提交的平地示例，输出必须不存在
.venv/bin/python -m scripts.validate_level_spec \
  --input game_content/level_specs/flat_v1.json \
  --output runs/level_spec_validation/flat.json

# 导出既有关卡，不改动其来源
.venv/bin/python -m scripts.export_level_spec \
  --manifest config/curriculum_v9.json --level-id v8_full_validation_05 \
  --output runs/level_spec_validation/full.json

.venv/bin/python -m scripts.validate_level_spec \
  --input runs/level_spec_validation/full.json \
  --output runs/level_spec_validation/full_report.json
```

可安装命令为 `platformer-export-level` 和 `platformer-validate-level`。示例：[flat](../game_content/level_specs/flat_v1.json)、[full](../game_content/level_specs/full_v1.json)。

Python 使用：

```python
from ai_platformer.content.level_spec import load_level_spec, to_level_definition
from ai_platformer.content.level_validation import validate_level_spec
from ai_platformer.settings import AI_GAMEPLAY_PATH, load_gameplay_settings

spec = load_level_spec("game_content/level_specs/flat_v1.json")
physics = load_gameplay_settings(AI_GAMEPLAY_PATH).physics
report = validate_level_spec(spec, physics=physics)
if not report["static_valid"]:
    raise ValueError(report["issues"])
level = to_level_definition(spec)
# 转换器本身不是地图准入门槛；环境加载与播放见路线验证文档。
```

## 验收结果与旧内容边界

- v9 全部 645 张地图往返转换一致，旧内容哈希 645/645 保持一致。
- 644 张生成地图静态通过。
- `level_1_main` 严格静态校验失败：solids 索引 3 横跨世界右边界，10 和 38 位于世界右边界之外。这些为 legacy 数据中后续区域残留；核心原本仍能运行该地图。导出保留全部原始内容，报告越界，不裁剪或改写 v9。
- 平地与 full 示例的 CLI 校验通过；主区域 CLI 按预期退出 1 并保存诊断。
- 转换后的 core 固定动作重放一致。全量测试 115 passed、8 subtests passed，仅两条旧 Gym 弃用提示；修改 Python 文件 Ruff 通过。

机器报告：[level_spec_v1.json](reports/level_spec_v1.json)。新关卡应满足严格边界规则；如后续清理 legacy 残留，使用新的地图 ID/内容版本并重新验收，不能覆盖冻结基线。

加载、播放与路线搜索/重放已实现，见 [路线验证](LEVEL_SPEC_ROUTES_V1.md)。下一轮为 LLM + PCG 的受约束生成请求与候选验证。静态通过的新地图暂不自动进入 PPO 训练池。
