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

最多调用三次（`--max-attempts 1..3`），单次超时 `--timeout` 默认30秒、最高60秒，响应上限1MB。无效结构带校验反馈重试；429/5xx 有有限退避；连接失败可重试；认证/其他 HTTP 错误及明确拒绝立即终止。错误信息不记录原始网络异常或完整 HTTP body。HTTP 失败仅提取最多16KB JSON错误体中的 status/message，并对已配置密钥、API Key 参数和 Bearer token 脱敏；非JSON响应体直接丢弃。连接失败单独记录 DNS、TLS证书、TLS握手超时等类别。失败原因和逐次尝试详情会直接打印到终端；缺少模型 ID 时明确提示设置 `--model` 或 `GEMINI_MODEL`。固定官方域名，禁止跟随重定向携带密钥。无效输出不会强制裁剪或修改 seed 来凑成合法请求。

## 本轮验收与后续

初始实现验收时环境未配置 API Key，因此当时仅完成模拟传输合同测试与显式 fixture 端到端验收。2026-10-23 已补充首次真实 API 闭环验收，见下文。离线 mixed 样本与已有 PCG 基准几何相同：3障碍、2坑、20松果；搜索与独立重放成功，206决策步、823物理tick、16/20松果（80%）。完整测试结果见 [验收记录](reports/llm_generation_v1.json)。

下一步基于已通过的连接组合扩展少量真实提示，评估需求表达准确性、拒绝/失败率、成本与延迟，再扩展设计评测集。云端模型可能变化，升级模型/提示须保留版本记录并重新验收。当前不改变 v9 PPO、TinyInfer 或训练保留集。

## Google 连接故障诊断

`SSLEOFError` 表示 TLS 连接异常结束；握手超时、HTTP 503 和模型列表成功曾在不同尝试中出现。这些结果不能单独证明需要付费、模型不可用或客户端未走 VPN。TUN 是流量接管方式，本身不要求存在 HTTP 代理端口。当前排查中一次沙箱外无密钥测试成功建立 TLS 1.3，curl IPv4/HTTP1.1 返回根路径404；仅证明该次连接可达，不代表鉴权和生成接口始终正常。

新增入口在**已配置 Key 的同一终端**运行：

```bash
.venv/bin/python -m scripts.diagnose_llm_connection \
  --model gemini-3.8-flash \
  --comparison-model gemini-3.5-flash-lite \
  --timeout 30 \
  --output runs/llm/connection_check_001.json
```

依次检查直接 TLS、鉴权后的模型列表、目标模型的最小生成请求。只有目标模型失败时才调用对照模型；如果完整列表明确未包含对照模型则跳过。模型 ID 应以账户真实列表为准；此处两者均曾出现在当前账户列表中。每个接口只尝试一次，最多两次256输出token预算的微小生成请求，不生成地图，不缓存业务候选。HTTP成功而 finishReason=MAX_TOKENS 仅说明生成接口已响应，不代表完整输出或关卡验收成功。

密钥仅从环境变量读取用于鉴权，不输出其内容；不会从其他终端或文件提取密钥。诊断报告不包含密钥、代理URL或模型响应正文。输出路径拒绝覆盖；复测使用新文件名。主模型响应正常退出0；缺少Key或主模型失败退出1，即使对照模型成功也保持1。

结果与处理：

| 结果 | 下一步 |
| --- | --- |
| `missing_api_key_in_this_terminal` | 在运行脚本的同一个终端配置 Key；不同进程不会自动继承另一个已打开终端的 export。 |
| TLS失败或所有生成请求均出现握手超时/EOF | 保持其他参数一致，换节点或网络后比较报告；单次失败无法定位具体断开方。 |
| 列表成功、主模型失败、对照模型成功 | 可临时用对照模型完成开发，再重测主模型；不能据一次对照断言主模型服务故障。 |
| HTTP 400/401/403/429/503 | 依据脱敏的 `provider_status`、`provider_message` 区分参数、鉴权、权限、额度或暂时不可用；不要把所有错误归为充值问题。 |
| `minimal_generation_reachable_full_pipeline_still_unverified` | 再运行正式生成命令；如果这时失败，检查结构化协议、输出长度和关卡路线阶段的报告。 |

直接 TLS 探针使用 socket，不读取 HTTP_PROXY；后续 urllib 请求按 Python 的系统/环境代理发现规则运行，因此显式HTTP代理环境下两者可以有不同结果。只应在确认真实 HTTP/Mixed 端口后配置终端代理，不猜测端口。保留证书校验；EOF 本身不是证书验证失败。

### TLS参数对照（保留证书校验）

当直接TLS探针成功但所有urllib请求出现EOF时，可在相同节点、相同终端下比较 TLS 1.2。默认仍使用Python默认TLS策略，不自动降级，也不关闭证书/域名验证。

```bash
.venv/bin/python -m scripts.diagnose_llm_connection \
  --model gemini-3.8-flash \
  --comparison-model gemini-3.5-flash-lite \
  --tls-mode tls12 --timeout 30 \
  --output runs/llm/connection_check_002.json
```

`--tls-mode tls12` 使用受验证的默认SSLContext，限定最小和最大TLS版本均为1.2，并声明HTTP/1.1。设置同时应用于TLS探针、模型列表和最小生成；不能保证修复，成功也只是支持“线路/TLS参数可能相关”的假设，不是对TAG故障的证明。诊断脚本的默认策略与显式策略可能由不同路径实现，证书验证始终开启。

若对照成功，正式生成入口也支持同一参数：

```bash
.venv/bin/python -m scripts.generate_level_llm \
  --model gemini-3.8-flash --tls-mode tls12 \
  --prompt '湖边关卡，三个障碍两个坑，短跳收集松果，最低收集率75%' \
  --level-id llm_test_001 --seed 20261010 \
  --output-dir runs/llm/live_006 --cache-dir runs/llm/cache \
  --max-attempts 1 --timeout 60 --verify-route
```

TLS模式写入intent及非默认模式的缓存身份，避免用默认模式缓存命中来误判新传输已验证。如果两个模式都失败，继续对照节点/网络或确认客户端的真实HTTP/Mixed端口；不把TLS EOF当作充值或模型性能问题。


## 首次真实 API 闭环验收（2026-10-23）

已在用户终端完成真实 Gemini 调用：`gemini-3.5-flash-lite`、`--tls-mode tls12`，首次尝试成功且 `cache_hit=false`。产物位于 `runs/llm/live_006`。模型生成的请求符合三个障碍、两个坑、low_jump 松果和最低75%收集率的意图；PCG生成20个松果，路线211决策步、841物理tick，成功收集17/20（85%）。搜索器独立重放通过，随后再次从已发布地图与见证进行全新重放也通过。

可复查的请求、结果与产物哈希见 [真实API验收记录](reports/llm_live_acceptance_20261023.json)。本次证明一个实际候选的完整链路通过，不代表批量稳定性验收。先前EOF以及3.8最小请求超时的确切根因仍未确定；当前是经过验证的可用组合，默认TLS策略保持原样。

播放已验收路线：

```bash
.venv/bin/python -m scripts.play_level_spec \
  --input runs/llm/live_006/level.json --controller replay \
  --route runs/llm/live_006/route.json
```

后续生成可沿用上文 `gemini-3.5-flash-lite` 和 `--tls-mode tls12`，更换输出目录并保留报告。v9 PPO、TinyInfer与训练保留集不变，本次候选不自动分配训练split。
