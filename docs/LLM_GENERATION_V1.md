# LLM 关卡生成 v1

已接入 Gemini REST 后端，将中文/英文设计意图转换为受约束的 GenerationRequest v1，再复用确定性 PCG、LevelSpec 静态校验、真实游戏核心路线搜索和独立重放。LLM 不输出执行代码或自由几何，不负责判断可通关，也不自动将候选加入训练集。

当前支持 lake 主题、静态障碍、坑、松果布局和收集率目标；范围沿用 [PCG v1](PCG_GENERATION_V1.md)。不支持的设计可返回 `unsupported`，终止生成并保留失败报告。协议校验能验证数值、字段和调用者指定的 seed/level_id；自然语言含义是否被准确理解仍需要查看 `request.json`，本轮尚未开展真实模型语义评测。

## 实际调用

运行前在当前终端配置 `GEMINI_API_KEY`（也支持 `GOOGLE_API_KEY`）和 `GEMINI_MODEL`。模型 ID 应使用账户当前可用、支持 JSON Schema 的 Gemini 模型；也可通过 `--model` 指定。脚本不自动加载 `.env`，不把密钥写入日志、缓存或 URL。接口采用官方 [generateContent API](https://ai.google.dev/api/generate-content) 的 `responseMimeType`/`responseJsonSchema`，并再次使用本地 Pydantic 校验。

```bash
.venv/bin/python -m scripts.generate_level_llm \
  --prompt '生成湖边关卡，三个障碍两个坑，障碍高度48到96，坑宽96到152，短跳收集松果，最低收集率75%' \
  --level-id llm_mixed_001 --seed 20261005 \
  --output-dir runs/llm/live_001 --cache-dir runs/llm/cache \
  --verify-route
```

`platformer-generate-level-llm` 是同一入口，重新安装项目后可用。未提供 `--verify-route` 时结果仅为 `candidate`；只有独立重放成功且达到收集率目标才为 `verified`。`unknown` 表示搜索预算内未找到路线，不代表不可通关。模型未完成输出/阻止输出不会进入 PCG。候选/验收成功退出 0，生成失败或路线 unknown 退出 1，目录已存在或发布失败退出 2。

## 离线复跑

fixture 是显式指定的本地响应样本，不调用 API，也不代表 Gemini 真实生成结果。

```bash
.venv/bin/python -m scripts.generate_level_llm \
  --provider fixture --fixture game_content/llm_fixtures/mixed_v1.json \
  --prompt '三个障碍、两个坑，短跳收集松果，最低收集率75%' \
  --level-id pcg_mixed_v1 --seed 20261005 \
  --output-dir runs/llm/offline_001 --cache-dir runs/llm/cache \
  --verify-route

.venv/bin/python -m scripts.play_level_spec \
  --input runs/llm/offline_001/level.json --controller replay \
  --route runs/llm/offline_001/route.json
```

## 记录、缓存和失败边界

每个新输出目录原子发布，已有目录拒绝覆盖：

- `intent.json`：原始提示、后端、调用者指定 ID/seed。
- `llm.json`：实际模型、提示版本/系统提示哈希、输出 Schema 哈希、逐次结构化响应和 usage、请求哈希、缓存命中状态。
- `request.json`、`level.json`、`report.json`：沿用 PCG 产物。
- `route.json`：启用路线验收时产生，含可重放见证或 unknown 原因。

缓存只保存通过协议校验的请求。键包含 prompt、provider/model、提示及 Schema 版本、ID/seed；fixture 另包含响应哈希。读取缓存重新验证身份、协议和请求哈希。相同缓存可复现请求；无缓存时 temperature=0 也不保证云模型输出逐字确定。改变提示、模型或版本会产生新缓存键。缓存包含提示与模型输出，应当按本地开发数据管理。

最多调用三次（`--max-attempts 1..3`），单次超时 `--timeout` 默认30秒、最高60秒，响应上限1MB。无效结构带校验反馈重试；429/5xx 有有限退避；连接失败可重试；认证/其他 HTTP 错误及明确拒绝立即终止。错误信息不记录原始网络异常或 HTTP body。固定官方域名，禁止跟随重定向携带密钥。无效输出不会强制裁剪或修改 seed 来凑成合法请求。

## 本轮验收与后续

环境未配置 API Key，因此本轮为模拟传输合同测试与显式 fixture 端到端验收，真实云端调用尚未验证。离线 mixed 样本与已有 PCG 基准几何相同：3障碍、2坑、20松果；搜索与独立重放成功，206决策步、823物理tick、16/20松果（80%）。完整测试结果见 [验收记录](reports/llm_generation_v1.json)。

下一步配置 Key 和模型 ID，先用少量实际提示验收 API Schema 兼容、需求表达准确性、拒绝/失败率、成本与延迟，再扩展设计评测集。云端模型可能变化，升级模型/提示须保留版本记录并重新验收。当前不改变 v9 PPO、TinyInfer 或训练保留集。
