# Backend 开发日志

> 每完成一个可测试小步，追加一段。三段结构：**我做了什么** / **关键设计决策** / **你需要操作什么**。
> 不是通用文档，是给下一次会话（或未来自己）快速接手用的。

---

## Step 1 — 项目脚手架

### 我做了什么

- FastAPI 项目骨架：`app/main.py` / `app/config.py`（pydantic-settings）/ `app/db/session.py` / `app/api/health.py`
- SQLAlchemy 2.x 基类：`app/models/base.py`（Base / IdMixin / TimestampMixin）
- Alembic 集成：`alembic.ini` + `app/db/migrations/env.py`
- User / Photo ORM，初始迁移 `0001_init.py`
- `.env.example` 全字段样板

### 关键设计决策

- **配置全走 pydantic-settings + .env**，不硬编码 key
- **软删除**：所有业务表带 `deleted_at`，查询时 filter `deleted_at IS NULL`
- **user_id 用 BigInteger**：为将来接微信 openid 独立表留空间

### 你需要操作什么

（已完成，无待办）

---

## Step 2 — storage_service + 照片上传

### 我做了什么

- `services/storage_service/`：ABC + LocalStorage + HMAC-SHA256 签名 + factory
- `POST /photos` multipart 上传，Pillow 二次校验图像 + 提取宽高
- `GET /files/{key:path}` 校验 sig/exp，供小程序前端展示照片
- 迁移 `0001_init.py` 建 users + photos

### 关键设计决策

- **签名 URL 而非直挂静态目录**：防路径猜测 + 短期过期（默认 15 分钟）
- **接口和 S3/COS 对齐**：`SignedURL` dataclass + `signed_url(key, ttl)`，将来切云存储不改业务
- **种子用户 user_id=1**：未接微信登录前，所有请求挂到 id=1；`_ensure_seed_user` 幂等建

### 你需要操作什么

（已完成。/photos POST 已能上传，GET /photos/{id}/url 已能拿到签名地址）

---

## Step 3 — ai_service 业务接入（真实 LLM + 合规 + 限流）

### 总体计划

3 拆 4 步交付：

- **3a**（已完成）：限流表 + 中间件 + dev 豁免
- **3b**（本次）：`POST /analyses` + 双表持久化 + 图片压缩后 base64 送 LLM
- **3c**（下一步）：schema_guard 严格版 + 合规出参扫描（违禁词/药品名/诊断词）
- **3d**：`POST /chat` + chat_messages 表

---

### Step 3a — 限流表 + 中间件 + dev 豁免

#### 我做了什么

- 新增 `ai_usage_counters` 表 + 迁移 `0002_ai_usage.py`
- `services/ai_gateway/rate_limit.py`：`try_consume` / `peek` / `require`；用 PG `INSERT ... ON CONFLICT DO UPDATE ... WHERE ... RETURNING` 单 SQL 原子占额
- `config.py` 新增 `ai_ratelimit_enforce_in_dev`（默认 false）
- `/ai/debug/quota` GET + `/ai/debug/quota/{kind}/consume` POST，方便调试

#### 关键设计决策

- **原子占额用 UPSERT with WHERE**：达到上限时 UPDATE 分支被 WHERE 挡掉，RETURNING 无返回；避免"先 SELECT 再 UPDATE"的竞态
- **dev 豁免**：`APP_ENV=dev` 且 `AI_RATELIMIT_ENFORCE_IN_DEV=false` 时直接返回 `used=0, allowed=true`，不查库
- **不落在中间件层**：限流是业务动作，让 API handler 显式调 `rl.require(...)`，跳过限流的路径（如 /health）不用配置排除清单

#### 你需要操作什么

（已完成。跑 `alembic upgrade head` 后 `/ai/debug/quota` 能看到 analyze/chat 两条 used=0）

---

### Step 3b — POST /analyses + 双表持久化 + 图片压缩

#### 我做了什么

**新增文件**：
- `models/ai_call_log.py` + `models/analysis.py`
- `db/migrations/versions/0003_analyses.py`（建 ai_call_logs + analyses 两表）
- `services/vision/image_prep.py`（Pillow 压缩 → JPEG q=85 → base64 data URL）
- `services/ai_gateway/prompts.py`（vision system prompt 集中管理 + version 字段）
- `services/analysis_service.py`（业务层：拉图 → 压缩 → gateway → 解析 → 落库）
- `schemas/analysis.py`（AnalyzeRequest / AnalysisOut）
- `api/analyses.py`（`POST /analyses` / `GET /analyses/{id}` / `GET /analyses/by-photo/{photo_id}`）

**修改文件**：
- `models/__init__.py` 追加导入
- `main.py` 挂载 analyses router

#### 关键设计决策

**1. 双表分离（业务表 + 日志表）**
- `analyses` 只存成功结果（前端/趋势查询用）
- `ai_call_logs` 全量落日志（success / llm_failed / parse_failed），供成本核算 + 排障
- 关联：`analyses.ai_call_log_id → ai_call_logs.id`

**2. 幂等（force=false 默认走缓存）**
- 默认命中 `photo_id` 的最近一条成功 analyses 直接返回（`cached=true`）
- `force=true` 时跳过缓存 → 消耗一次配额 → 真调 LLM
- 缓存命中**不消耗配额**（读旧结果不算 AI 调用）

**3. 图片压缩策略**
- 长边 > 1600px → 等比缩放到 1600；JPEG q=85
- **FIXME(step-4)**：接入 vision 模块后应改为"人脸检测 + 裁剪 + 眼部打码"再送 LLM，image_prep 下沉为纯 resize 工具。压缩必然影响细节，只是 MVP 阶段控成本的过渡方案。
- 为什么不上公网 CDN：MVP 本地部署，外网 LLM 拉不到 localhost；上 CDN 是生产方案。

**4. 失败落库分类**
- `llm_failed`：gateway 抛 FatalRequestError / AllProvidersFailedError（HTTP 502）
- `parse_failed`：LLM 返回非 JSON 或 shape 异常（HTTP 422）
- 都写 ai_call_logs 一条，errno + error_message 全存；成功不写 analyses

**5. Prompt 版本化**
- `VISION_ANALYZE_PROMPT_VERSION = "vision-1.0.0"` 写入 `ai_call_logs.input_meta.prompt_version`
- 后面 3c 改 prompt 时递增版本号，就能 SQL 对比"新 prompt 通过率 vs 旧 prompt"

**6. 兼容 gateway 现状**
- 复用 `UnifiedRequest(response_format="json", temperature=0.1)`
- `Message` 中 `image_urls` 塞 base64 data URL；openai_compat 的 `_encode_message` 会自动组装 multipart content
- 不改 gateway 代码

#### 你需要操作什么

**1. 跑迁移**
```bash
cd backend
alembic upgrade head
# 应看到 Running upgrade 0002_ai_usage -> 0003_analyses
```

数据库应新增 `ai_call_logs`、`analyses` 两张表。

**2. 确认 .env 已配 MiniMax key**
```
MINIMAX_API_KEY=sk-xxx（你已配）
MINIMAX_BASE_URL=https://api.minimaxi.com/v1
MINIMAX_MODEL=MiniMax-M3
```

**3. 重启 uvicorn**
```bash
uvicorn app.main:app --reload
```

**4. 端到端验证（在 /docs 或 curl）**

```bash
# a. 先上传一张人脸照片
curl -X POST http://localhost:8000/photos \
     -F "file=@/path/to/face.jpg"
# → 返回 photo_id，记下

# b. 分析这张照片（第一次会真调 LLM）
curl -X POST http://localhost:8000/analyses \
     -H "Content-Type: application/json" \
     -d '{"photo_id": <上一步的id>, "force": false}'
# → 返回完整 parsed_result，cached=false

# c. 再调一次（应走缓存）
curl -X POST http://localhost:8000/analyses \
     -H "Content-Type: application/json" \
     -d '{"photo_id": <同上>, "force": false}'
# → cached=true，秒返回

# d. force 重跑
curl -X POST http://localhost:8000/analyses \
     -d '{"photo_id": <同上>, "force": true}'
# → cached=false，又调一次 LLM

# e. 看历史
curl http://localhost:8000/analyses/by-photo/<photo_id>
# → 返回该照片的所有分析记录（按时间倒序）
```

**5. 观察数据库**
```sql
-- 看每次 AI 调用日志
SELECT id, status, provider, model, input_tokens, output_tokens, latency_ms, error_message
FROM ai_call_logs ORDER BY id DESC LIMIT 10;

-- 看成功的分析结果
SELECT id, photo_id, overall_severity, skin_health_index, needs_doctor, provider
FROM analyses ORDER BY id DESC LIMIT 10;

-- 看 parsed_result 里的痘痘计数
SELECT id, parsed_result->'acne_types' FROM analyses ORDER BY id DESC LIMIT 3;
```

**6. 已知遗留问题（3c 处理）**
- Prompt 里的 schema 是自然语言描述，模型偶尔字段不全时不会拦截
- 违禁词/药品名扫描还没做
- description 字段的合规审计还没做

---

### Step 3b+ — AI 调用可观测性（trace + 每次 provider 落库 + debug API）

#### 我做了什么

**新增文件**：
- `services/ai_gateway/observability.py`
  - `trace_id`（contextvar 存 + `new_trace_id`）
  - `TracedLogger`：结构化日志，格式 `[trace=xxx] event key=val key=val`
  - `sanitize_messages_for_log`：把 base64 data URL 替换成 `<data:image/jpeg;base64,...(N chars)...>`
  - `ProviderCallRecord` dataclass：一次 provider 调用的完整快照
- `db/migrations/versions/0004_ai_call_log_trace.py`

**改造文件**：
- `models/ai_call_log.py`：加 `trace_id / attempt_seq / request_payload` 三列
- `services/ai_gateway/gateway.py`：
  - 新增 `GatewayInvokeResult`（含 response + records）
  - 新增 `invoke_detailed(task, req, *, trace_id, start_attempt_seq, skip_bindings)`
  - `invoke()` 保留旧签名，内部走 `invoke_detailed`
  - 每次 provider 调用/跳过都产生一条 record
  - 关键节点打 trace log
- `services/analysis_service.py`：全部重写
  - 生成 `trace_id` 挂到 contextvar
  - 用 `invoke_detailed` 拿全量 records
  - **B 方案**：gateway 说 ok 但 JSON parse 失败 → 把该 (provider, model) 加入 skip_bindings → 再跑一次 gateway 走下一家；最多 5 轮兜底
  - 每条 record 落一条 `ai_call_logs`
  - `analyses.ai_call_log_id` 关联最终 success 那条
- `api/analyses.py`：response header 加 `X-Trace-Id`；失败 detail 也含 trace_id
- `api/ai_debug.py`：新增
  - `GET /ai/debug/logs?limit=&kind=&status=&provider=`：列最近
  - `GET /ai/debug/logs/{id}`：完整详情
  - `GET /ai/debug/logs/{id}/raw-text`：只看 LLM 原文，方便复制
  - `GET /ai/debug/traces/{trace_id}`：按 trace 聚合，一次业务请求的所有 attempt
- `services/ai_gateway/__init__.py`：导出 `GatewayInvokeResult / ProviderCallRecord / trace_log / new_trace_id / sanitize_messages_for_log`

#### 关键设计决策

**1. 一次业务请求 = 一个 trace_id，一次 provider 调用 = 一条 log**
- 旧版：`/analyses` 一次 → 只落最终一条 log（fallback 时中间的家消失了）
- 新版：minimax parse_failed → glm success → 表里两条 log，同一 trace_id，attempt_seq=1/2
- `analyses.ai_call_log_id` 只指向最终 success 那条，前端看结果不用理会日志

**2. parse 失败的处理归 service，不归 gateway**
- Gateway 只管"网络传输 + 断路"，返回 200 就算 ok
- Service 拿到 response 后 try_parse_json，失败就把这家扔进 `skip_bindings` 再调 gateway
- 好处：gateway 保持职责纯粹；3c 加 schema_guard 时可以复用同一套 skip 机制

**3. request_payload 图片脱敏**
- 直接存 base64 会撑爆 JSONB（180KB 图 × N 条）
- `sanitize_messages_for_log` 把 data URL 换成 `<data:image/jpeg;base64,...(240000 chars)...>` 占位符，长度信息保留
- 排障时能看到 prompt 全文 + 图片元信息，看不到图本身

**4. 兜底解析器 `_extract_json_object`**
- 有些模型（MiniMax M3）习惯在 JSON 前后加散文
- 新 parser 会从文本里抠出第一个平衡的 `{ ... }` 块尝试解析
- 这一步是补救，不是长期方案（3c 会做 schema_guard 严校验）

**5. 保持向后兼容**
- 老的 `gateway.invoke()` 签名不变，别处代码（如 `/ai/debug/invoke`）不需要改
- 内部走 `invoke_detailed`，但只返回 response 或抛异常

#### 你需要操作什么

**1. 跑迁移**
```bash
cd backend
alembic upgrade head
# 应看到 Running upgrade 0003_analyses -> 0004_ai_call_log_trace
```

**2.（可选）智谱 GLM key 现在放上去**
```
# .env
GLM_API_KEY=xxx（你已配）
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
GLM_MODEL=glm-4v-plus
```
factory 会自动把 GLM 加入 `vision_analyze` 的 fallback 链。

**3. 重启 uvicorn**

**4. 端到端验证**

```bash
# 触发一次分析（force=true 跳过缓存）
curl -X POST http://localhost:8000/analyses \
     -H "Content-Type: application/json" \
     -d '{"photo_id": <id>, "force": true}' \
     -i
# 关注 response header 的 X-Trace-Id: xxx
```

**5. Debug API 打点点**

打开 http://localhost:8000/docs，展开 `ai-debug` 标签，重点看：

- `GET /ai/debug/logs?limit=10` → 看最近调用列表，含 status/provider/latency/text_preview
- `GET /ai/debug/logs/{id}` → 完整 request_payload + raw_response
- `GET /ai/debug/logs/{id}/raw-text` → 只看 LLM 原文，方便复制到别处
- `GET /ai/debug/traces/{trace_id}` → 一次 /analyses 的完整 attempt 链（minimax → glm → ...）

**6. 终端看结构化日志**

uvicorn 终端会看到：
```
[trace=abc123] analyze.start photo_id=5 user_id=1
[trace=abc123] analyze.image_prep original=3024x4032 resized=1200x1600 bytes=184320
[trace=abc123] gateway.provider.request seq=1 provider=minimax model=MiniMax-M3 task=vision_analyze
[trace=abc123] gateway.provider.ok seq=1 provider=minimax latency_ms=8321 tokens_in=1500 tokens_out=850 text_preview="The image shows..."
[trace=abc123] analyze.parse.fail provider=minimax text_preview="The image shows..."
[trace=abc123] gateway.provider.request seq=2 provider=glm model=glm-4v-plus
[trace=abc123] gateway.provider.ok seq=2 provider=glm latency_ms=6100 tokens_out=920 text_preview="{\"observation\":..."
[trace=abc123] analyze.parse.ok provider=glm fields=15
[trace=abc123] analyze.done status=success analysis_id=3 log_id=8 provider=glm
```

**7. DBeaver 连接串**

```
Host: localhost
Port: 5432
Database: skin_care
User: skin
Password: skin
```
连上后重点看两张表：`ai_call_logs`（全量日志）+ `analyses`（成功结果）。JSONB 字段 DBeaver 会自动展开。

---

### Step 3b++ — 推理模型 reasoning 分离 + parse_strategy 可观测

#### 我做了什么

**背景**：MiniMax M3 是推理模型，会在正式答案前输出 `<think>...</think>` 块。3b+ 的 `_extract_json_object` 兜底 parser 能从散文里抠出中间的 `{...}` 让业务跑通，但排障时 `text_preview` 显示的是 think 内容，看起来像"输出错乱"。

**新增文件**：
- `services/ai_gateway/parsing.py`：`parse_llm_json(text) -> ParseResult`，返回 `(parsed, reasoning, strategy, stripped_text)`
- `db/migrations/versions/0005_reasoning_fields.py`

**改造文件**：
- `models/ai_call_log.py`：加 `reasoning_text TEXT` + `parse_strategy VARCHAR(16)` 两列
- `services/analysis_service.py`：删除内嵌 parser，改用 `parse_llm_json`；record 落库时并入 `reasoning + strategy`
- `api/ai_debug.py`：
  - `LogRowOut` 加 `parse_strategy` + `has_reasoning`
  - `LogDetailOut` 加 `reasoning_text` + `parse_strategy`
  - `/logs` 和 `/traces/{id}` 支持 `preview_len` query 参数（0-5000，0 表示全文）
  - `/logs/{id}/raw-text` 返回体加 `parse_strategy` + `reasoning_text`
- `services/ai_gateway/prompts.py`：版本升到 `vision-1.2.0`
  - 明确允许 `<think>...</think>` 前置
  - 强调 `</think>` 之后必须只有 JSON
  - 推理块语言不限，正文中文

#### 关键设计决策

**1. parse_strategy 三档：direct / extracted / failed**
- `direct`：模型老实输出 JSON，可以直接 `json.loads`
- `extracted`：从散文里抠出 `{...}` 才成功（MiniMax M3 目前几乎全走这条）
- `failed`：完全没抠到，触发 fallback

意义：将来切模型时，SQL 一句就能出"哪家模型有多少比例是 direct"，量化"结构化能力"。

**2. reasoning 单独存 Text 列，不合并到 raw_response**
- `raw_response` 是 gateway 层原始存档，不该被业务后处理污染
- `reasoning_text` 单独查询、单独展示（未来可能在小程序里"给用户看 AI 是怎么想的"）
- 前端 UI 决策：普通用户看 `parsed_result` + `observation`；高级用户/开发者才展开 `reasoning`

**3. 老数据保留**
- id=1、2、3、4 的 `reasoning_text` 和 `parse_strategy` 都是 NULL
- 保留用于纵向对比"prompt v1.1.0 vs v1.2.0 的 parse_strategy 分布"
- 不清

**4. 兼容非推理模型**
- 没有 `<think>` 标签时 `_extract_reasoning` 返回 `(None, 原文)`
- 后续 GLM-4V / Qwen-VL 直接走 direct 路径，reasoning_text 为空

#### 你需要操作什么

**1. 跑迁移**
```bash
alembic upgrade head
# Running upgrade 0004_ai_call_log_trace -> 0005_reasoning_fields
```

**2. 重启 uvicorn**

**3. 触发一次分析（force=true）**
```bash
curl -X POST http://localhost:8000/analyses \
     -d '{"photo_id": <id>, "force": true}' -i
```

**4. 打点验证**

/docs 里点：

- `GET /ai/debug/logs?limit=5` → 新增 `parse_strategy` 和 `has_reasoning` 字段
  - 应该看到最新一条 `parse_strategy=extracted`（如果 MiniMax 还是 think 模式）或 `direct`（如果它听了新 prompt）
  - `has_reasoning=true` 说明拿到了 think 块
- `GET /ai/debug/logs/{id}` → `reasoning_text` 里能看到完整推理过程
- `GET /ai/debug/logs?preview_len=1000` → 预览拉长到 1000 字符
- `GET /ai/debug/logs/{id}/raw-text` → 返回体多了 `reasoning_text` 和 `parse_strategy`

**5. DBeaver 里跑一句看分布**
```sql
SELECT parse_strategy, COUNT(*)
FROM ai_call_logs
WHERE status = 'success'
GROUP BY parse_strategy;
```

**6. 期望的现象变化**
- 如果 MiniMax M3 遵守新 prompt → `<think>` 里放推理、外面只有 JSON → `parse_strategy=direct`
- 如果 MiniMax 仍然习惯性用散文 → `parse_strategy=extracted`，reasoning 依然被剥离得干干净净
- 两种都能业务成功，区别只是可观测性

---

### Step 3c — schema_guard + 合规出参扫描 + 模板兜底

#### 我做了什么

**新增文件**：
- `services/ai_gateway/schema.py`：pydantic v2 model 定义所有 vision_analyze 字段 + 分级校验
- `services/ai_gateway/compliance.py`：违禁词库（疾病/药品/建议句）+ 扫描 + **B4 模板兜底**
- `services/ai_gateway/validators.py`：一致性 + **needs_doctor 服务端强判**
- `db/migrations/versions/0006_compliance_fields.py`

**改造文件**：
- `models/ai_call_log.py`：加 `schema_errors JSONB` + `compliance_flags JSONB` + `validation_warnings JSONB`
- `services/analysis_service.py`：主循环加两个 fail 分支
  - Parse ok 后跑 schema_guard → 失败也 skip 该 provider + 切 fallback
  - Schema ok 后跑 compliance + validator（不失败，只标记）
  - 最终落 `parsed_result` 用 model.model_dump()（已被模板改写过的合规版本）
- `api/ai_debug.py`：
  - `LogRowOut` 加 `compliance_hit_count / schema_error_count / validation_warning_count / needs_doctor_adjusted`
  - `LogDetailOut` 加 `schema_errors / compliance_flags / validation_warnings`

#### 关键设计决策

**1. 分级 schema（Q1=C）**
- 核心严格：`observation / overall_severity / skin_health_index / needs_doctor / acne_points / acne_types` 缺失或类型不可修复 → schema_failed → 切 fallback
- 边缘宽松：`regions / other_concerns / scars / status_counts` 缺失自动补默认值，未知枚举 fallback 到 unknown
- pydantic v2 `field_validator(mode="before")` 做类型强转（str→int / null→默认值）

**2. B4 模板兜底（Q2）**
- 违禁词命中 → **整字段丢弃**，服务端用模板重生成
- 模板举例：
  - `observation` 命中 → `"共观察到约 X 处皮损特征，主要分布于 Y。"`
  - `regions.{r}.note` 命中 → `"{中文区域}区域可见约 X 处皮损特征。"`
  - `other_concerns.{k}.description` 命中 → 按 severity 拼一句（如 `"该维度呈中度表现。"`）
- 违禁词库分三类：`DISEASE_WORDS`（26 个）/ `DRUG_WORDS`（24 个）/ `ACTION_PATTERNS`（7 个正则）
- 词库集中在 `compliance.py`，合规团队直接编辑列表即可
- 每次命中在 `compliance_flags` 落库：`{field, hits, action, original(前200字), replaced_with}` —— 保留原文用于审计但截断防止表膨胀

**3. needs_doctor 服务端强判（Q3=要）**
- 规则：`overall_severity>=7 OR nodule>0 OR cyst>0 OR broken>=3` → true
- 与 LLM 结果 OR 运算，宁可多提示不可少提示
- 服务端上调时 `validation_warnings.needs_doctor_adjusted=true`，记录原因

**4. schema 失败也切 fallback（用与 parse 相同的 skip_bindings）**
- 复用 3b+ 的 `skip_bindings` 机制，schema 失败也扔进去
- max_parse_retries=5 兜底防死循环

**5. 落库的 parsed_result 是"清洁版"**
- 不再落 LLM 原始 JSON，而是落经过 schema 修正 + 合规重写 + needs_doctor 校准的最终结果
- 想看原文？raw_response.text 仍完整保留
- 用户/前端拿到的永远是合规版本

**6. 老数据兼容**
- id=1~5 的记录 `schema_errors / compliance_flags / validation_warnings` 都是 NULL
- debug API 已处理 None case（compliance_hit_count=0）

#### 你需要操作什么

**1. 跑迁移**
```bash
alembic upgrade head
# Running upgrade 0005_reasoning_fields -> 0006_compliance_fields
```

**2. 重启 uvicorn**

**3. 触发一次分析（force=true）**
```bash
curl -X POST http://localhost:8000/analyses \
     -d '{"photo_id": <id>, "force": true}' -i
```

**4. Debug API 验证**

打开 /docs，看 `GET /ai/debug/logs?limit=5`，最新一条应该有：
- `schema_error_count: 0` （schema 都通过了）
- `compliance_hit_count: N` （0 表示没违规，>0 表示有词被拦下模板改写）
- `validation_warning_count: N`
- `needs_doctor_adjusted: bool`

点 `GET /ai/debug/logs/{id}` 看完整：
- `compliance_flags`：列出被替换的字段和命中的词
- `validation_warnings`：一致性问题 + needs_doctor 校准信息

**5. 端到端测试合规拦截**

想主动触发一次拦截，可以人工构造一次调用（假装 LLM 输出了违规内容）：

方式 A：直接调 `POST /ai/debug/invoke`（跳过 schema/compliance），看模型是否已经会自己不违规
方式 B：临时改 prompt 让 LLM 说"痤疮"，然后跑 /analyses 看 compliance_flags 是否命中

**6. DBeaver 里跑一句看统计**
```sql
-- 各字段被命中的频次
SELECT jsonb_array_elements(compliance_flags)->>'field' AS field,
       COUNT(*)
FROM ai_call_logs
WHERE compliance_flags IS NOT NULL
GROUP BY field
ORDER BY COUNT(*) DESC;

-- 最常见的命中词
SELECT jsonb_array_elements_text(
         jsonb_array_elements(compliance_flags)->'hits'
       ) AS hit_term,
       COUNT(*)
FROM ai_call_logs
WHERE compliance_flags IS NOT NULL
GROUP BY hit_term
ORDER BY COUNT(*) DESC LIMIT 20;
```

**7. 已知局限（不阻塞 MVP）**
- 违禁词库靠字面匹配，"痤 疮"（中间加空格）能绕过——生产建议改为 Aho-Corasick + 正则
- 模板生成的 description 缺少 LLM 观察到的细节（比如"鼻翼旁"这种精细位置）
- 违禁词库需要合规团队定期 review 更新

---

### Step 3d-pre — 痘颗→痘斑（Patch）建模换轨

#### 背景

用户发现"每颗痘单独定位"的设计在重度用户上崩塌：满脸融合成片时无法逐颗计数，跨日追踪无稳定坐标可匹配。

**建模换轨**：
- 主结构：`acne_patch`（一片区域，含 region + bbox + coverage + estimated_count + dominant_type + inflammation + severity + description），**必填**（可为空数组）
- 附加：`acne_point`（单颗痘，含精确定位），**只在轻度可枚举时输出**（<10 颗 且 全为 sparse）

#### 我做了什么

**文档改动**：
- `docs/skin_condition_labels.md`：加 1.4 章"Point vs Patch"，重写 5.1/5.2/5.3（v2 输出格式 + 展示适配 + patch-based 追踪），改 6.2（needs_doctor 加 confluent 触发规则）
- `project_background.md`：核心技术差异点段落改为"区域生命周期追踪 + 单颗痘追踪（轻度可选）"

**代码改动**：
- `schema.py`：新增 `AcnePatch` 类（10 字段，bbox 坐标钳位/coverage/inflammation 枚举校验）；`VisionAnalyzeResult.acne_patches` 必填，`acne_points` 改可选（默认空数组）
- `prompts.py`：升级到 `vision-2.0.0`，重写 system + user prompt 说明 patch-first 规则和 Point 输出条件
- `compliance.py`：
  - 加 `COVERAGE_ZH` / `INFLAMMATION_ZH` 中文映射
  - 加 `_tpl_patch_description`：命中违禁词时用 coverage/type/count/inflammation 拼合规描述
  - `apply_compliance` 主循环加对 `acne_patches[i].description` 的扫描
- `validators.py`：
  - 一致性检查从 `acne_types vs acne_points` 改为 `sum(patch.estimated_count) vs acne_types.total`（±30% 容差）
  - 新加 `point_output_violation` 规则（acne_points > 10 时警告）
  - `_compute_needs_doctor` 加 `任意 patch.coverage == "confluent"` 触发条件
  - `_needs_doctor_reasons` 追加 confluent_patches 原因

**没有 DB 迁移**：`analyses.parsed_result` 是 JSONB，schema 变化不需要 DDL；`ai_call_logs.request_payload.prompt_version` 会自动记为 `vision-2.0.0` 与老数据区分。

#### 关键设计决策

**1. Patch 必填、Point 条件输出**
- v1 用户不确定的问题："这颗痘和上次那颗是同一颗吗？"——重度用户根本没法回答
- v2 用 patch 做主体：即使 estimated_count 抖动，coverage 和 dominant_type 趋势稳定
- Point 只在轻度用户上激活，两条追踪路径并存

**2. bbox_norm 归一化到 [0,1]**
- 相对整张照片而非人脸框：MVP 阶段没有人脸对齐（Task #4 才做）
- 未来 Task #4 上线人脸对齐后，bbox 可以改为相对人脸区域，v3 prompt 升级即可
- 前端画高亮框只需 `img_width * x`

**3. coverage 四档 vs 连续值**
- `sparse/moderate/dense/confluent` 离散化——LLM 更容易稳定输出
- 连续 area_ratio 已经存在，需要更细粒度时可以从这里算

**4. needs_doctor 加 confluent 触发**
- 融合成片临床意义上就是重度信号，不管 severity 数字多少
- 服务端兜底代码里已加，`validation_warnings.reasons` 会记录 `confluent_patches=[p1, p3]`

**5. 一致性容差 ±30%**
- `estimated_count` 本身就是估算，尤其 confluent 情况下
- 强制严等于会频繁误报，反而让日志变噪
- 超过 ±30% 才 warn，代表模型确实自相矛盾

#### 你需要操作什么

**1. 清老数据（Q3=A）**

在 DBeaver 里跑：

```sql
-- 先看当前状态
SELECT COUNT(*) FROM analyses;
SELECT COUNT(*) FROM ai_call_logs;

-- 清 v1 的 analyses 记录（parsed_result 是旧结构）
DELETE FROM analyses;

-- 可选：ai_call_logs 保留（历史 prompt version 对比很有价值）
-- 如果非要清：DELETE FROM ai_call_logs;
```

**2. 重启 uvicorn**

不需要迁移，代码热重载。

**3. 端到端验证**

```bash
# 上传一张照片，force=true 触发真调
curl -X POST http://localhost:8000/analyses \
     -d '{"photo_id": <id>, "force": true}' -i
```

**期望现象**：
- 响应 `parsed_result` 里有 `acne_patches` 数组（可能为空）
- 轻度照片：`acne_points` 也有内容
- 重度照片：`acne_points: []`，`acne_patches` 里有 coverage>=dense 的项

**4. Debug API 检查**

`GET /ai/debug/logs?limit=5`：
- 最新一条 `parse_strategy` 应该是 direct 或 extracted
- `schema_error_count`=0 说明新 schema 通过
- `compliance_hit_count`>0 时说明 patch description 被兜底过

`GET /ai/debug/logs/{id}` 详情看 `parsed_result` 结构。

**5. DBeaver 里看 patch 数据**

```sql
-- 看最新一条 analyses 的 patches
SELECT parsed_result->'acne_patches' FROM analyses ORDER BY id DESC LIMIT 1;

-- 统计新旧 prompt 版本的分布
SELECT input_meta->>'prompt_version' AS version, COUNT(*)
FROM ai_call_logs GROUP BY version ORDER BY version DESC;
```

**6. 已知遗留**
- 前端展示逻辑（sparse→点标 vs dense→高亮框）在小程序阶段处理
- 跨日 patch 匹配算法（Task #6 的 vision.tracker）需要重写：从匈牙利算法+坐标改为 region+特征向量
- 未来 vision 模块（Task #4）上线人脸对齐后，bbox_norm 语义要更新

---

### Step 3d — POST /chat + 合规 + 医疗兜底

#### 我做了什么

**新增文件**：
- `models/chat_message.py`：ChatMessage ORM（问答对业务表）
- `db/migrations/versions/0007_chat_messages.py`
- `schemas/chat.py`：ChatRequest / ChatResponse / ChatMessageOut / ChatContext / ChatHistoryItem
- `services/chat_service.py`：核心业务（医疗兜底 → 拉上下文 → 组装 messages → gateway → 精确删句 → 落库）
- `api/chat.py`：`POST /chat` + `GET /chat/history`

**改造文件**：
- `services/ai_gateway/prompts.py`：加 `CHAT_QA_SYSTEM_PROMPT`（chat-1.0.0）+ `build_chat_context_message`
- `services/ai_gateway/compliance.py`：
  - 加 `MEDICAL_EMERGENCY_KEYWORDS`（20+ 医疗紧急词）+ `detect_medical_emergency`
  - 加 `apply_compliance_to_chat_text`：按中文断句切分，命中违禁词的句子整句删除
  - `MEDICAL_INTERVENTION_MESSAGE`：服务端预设的紧急就医提示文案
- `models/__init__.py`：注册 ChatMessage
- `main.py`：挂载 chat router

#### 关键设计决策（对应 Q1-Q5）

**Q1=B — 前端传上下文**
- `ChatRequest.context.latest_analysis_id`：前端根据当前页面主动传
- 后端拿到 ID 查 `analyses` 表 → `build_chat_context_message` 压缩成短摘要 → 作为额外 system message 注入
- 摘要只包含：observation + severity + skin_health_index + 前 5 个 patches + needs_doctor（防 token 爆炸）

**Q2=A — 单轮（history 可选透传）**
- `ChatRequest.history: list[ChatHistoryItem]` 前端维护多轮
- 服务端零状态，不建 chat_sessions 表
- 需要跨设备同步是 Task #7 之后的事

**Q3=C — 精确删句合规**
- 用 `(?<=[。！？；\.\!\?])\s*|\n+` 中文断句
- 每句独立扫违禁词（复用 vision 那套 DISEASE_WORDS/DRUG_WORDS/ACTION_PATTERNS）
- 命中即整句删，其他句保留 → 用户看到的是自然的短一点的回答
- 兜底：如果所有句子都被删了，返回"抱歉，这个问题涉及无法提供的医疗建议..."的通用回复
- `compliance_flags` 落库：`{field, hits, action:"drop_sentence", original, replaced_with:""}`

**Q4=B — chat_messages 业务表**
- 成功的问答对落 `chat_messages`（干净，前端历史查询用）
- 每次 provider 调用落 `ai_call_logs`（全量，`kind='chat_qa'`，含 fallback/失败）
- 关联：`chat_messages.ai_call_log_id → ai_call_logs.id`
- 医疗兜底的问答也落 `chat_messages`（`medical_intervention=true / provider='server'`），但不落 `ai_call_logs`（根本没调 LLM）

**Q5=B — 医疗关键词兜底**
- `detect_medical_emergency` 命中 → 直接返回 `MEDICAL_INTERVENTION_MESSAGE` 预设文案
- 不调 LLM，成本 0
- 关键词分四类：严重症状（流脓/剧痛/发烧等）、疑似严重疾病（癌/肿瘤）、感染扩散/败血症、特殊风险（怀孕/婴幼儿——涉及用药禁忌）
- 也会占额（防绕限流），落 `chat_messages.medical_intervention=true` 供审计

#### 复用 3b/3b+/3c 基础设施

- Gateway `chat_qa` 链（minimax → deepseek）现成，不改
- 每次 provider 落 `ai_call_logs`：复用 `invoke_detailed`
- trace_id + 结构化日志：复用 observability
- 限流：复用 `rate_limit.require(kind='chat')`
- 合规词库：复用 vision 的 DISEASE_WORDS/DRUG_WORDS/ACTION_PATTERNS

#### 你需要操作什么

**1. 跑迁移**
```bash
alembic upgrade head
# Running upgrade 0006_compliance_fields -> 0007_chat_messages
```

**2. 重启 uvicorn**

**3. 端到端验证**

**场景 A：普通护肤问答**
```bash
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "T 区经常出油怎么办"}' -i
```
期望：assistant_message 给出成分级建议（如水杨酸/烟酰胺）；`medical_intervention: false`；X-Trace-Id 有值。

**场景 B：带上下文问答**
```bash
# 先做一次 analysis 拿到 analysis_id，然后：
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "我这个情况严重吗", "context": {"latest_analysis_id": 5}}'
```
期望：回答里能引用你的当前状态（"你右颊有一片中等密度..."）。

**场景 C：医疗紧急兜底**
```bash
curl -X POST http://localhost:8000/chat \
     -d '{"message": "我脸上流脓了怎么办"}'
```
期望：立即返回 `MEDICAL_INTERVENTION_MESSAGE` 全文；`medical_intervention: true`；`provider: "server"`。响应快（不调 LLM）。

**场景 D：多轮追问**
```bash
curl -X POST http://localhost:8000/chat \
     -d '{
       "message": "那两颊呢",
       "history": [
         {"role": "user", "content": "T 区油怎么办"},
         {"role": "assistant", "content": "..."}
       ]
     }'
```
期望：回答理解"两颊"是相对 T 区的追问。

**4. Debug API 检查**
- `GET /ai/debug/logs?kind=chat_qa&limit=5` 看 chat 调用记录
- 医疗兜底的场景**不会**出现在 ai_call_logs（因为没调 LLM）—— 只在 chat_messages 表里
- `GET /chat/history` 看问答历史

**5. DBeaver 里跑一句看统计**
```sql
-- 医疗兜底触发多少次
SELECT medical_intervention, COUNT(*) FROM chat_messages GROUP BY medical_intervention;

-- 合规删句触发多少次
SELECT jsonb_array_length(compliance_flags) AS dropped_sentences, COUNT(*)
FROM chat_messages WHERE compliance_flags IS NOT NULL
GROUP BY dropped_sentences;

-- 各 provider 分布
SELECT provider, COUNT(*) FROM chat_messages GROUP BY provider;
```

**6. 已知遗留**
- 医疗关键词是字面匹配，"化 脓"（加空格）能绕过——生产改 Aho-Corasick
- 中文断句用简单正则，遇到"..."、"—"等复杂标点可能切错——不阻塞 MVP
- 精确删句在极端情况下（每句都命中）会返回兜底提示，可能让用户感到"AI 拒答"

---

## Step 3 主线完成 ✅

- **3a** 限流 + dev 豁免
- **3b** POST /analyses + 双表 + 图片压缩
- **3b+** trace + 每次 provider 落库 + debug API
- **3b++** reasoning 分离 + parse_strategy
- **3c** schema_guard + 合规 + 模板兜底
- **3d-pre** 痘颗 → 痘斑建模换轨（v2.0.0）
- **3d** POST /chat + 医疗兜底 + 精确删句

**下一步方向**（按用户优先级选）：
- Task #4：vision 模块（MediaPipe 对齐 + 眼部打码 + 标注图渲染）
- Task #6：vision.tracker（跨日 patch 匹配）
- Task #7：趋势/日记/用药 + 小程序对接
- Task #8：合规收尾（免责声明 + checkNeedDoctor 前端提示）





